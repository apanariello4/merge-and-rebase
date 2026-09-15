"""Causal BRACE activation/task-vector swap for single-vector rebase.

The source activation bank used to fit Theseus/BiCo is deliberately separate
from the saved BRACE task-vector bank passed to ``transport``.  This lets a
single 2x2 run distinguish an activation-fit effect from an input-vector
effect without refitting BRACE or changing either captured endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

from merge_and_rebase.utils.helpers import load_json

from ..cli_args import add_config_arg, add_device_dtype_args, merge_non_none
from ..eval.utils import eval_task_top1, to_cpu_fp32
from ..io.ckpt import load_into_model
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..rebase import get_method
from ..rebase.runtime import resolve_rebase_method_config
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from ..utils.alpha_search import PerTaskAlphaTracker
from .datasets.vision8_14_20 import SUITES
from .vision_brace_tv_swap import _expanded_template, state_dict_sha256, validate_artifact_bank
from .vision_rebase import _build_rebase_prepared, _build_task_context, _set_deterministic_seed


DEFAULT_BANKS = ("shared", "skip")
KNOWN_BANKS = ("shared", "skip", "independent")
METHODS = ("theseus", "bico")

# Retained under its historical name: the 2x2 shared/skip crossing is still the
# default, and ``tests`` plus the 20260913 campaign refer to it.
BANKS = DEFAULT_BANKS


def _alpha_values(cfg: Mapping[str, Any]) -> list[float]:
    lo, hi, step = float(cfg["alpha_min"]), float(cfg["alpha_max"]), float(cfg["alpha_step"])
    if step <= 0 or hi < lo:
        raise ValueError("Invalid alpha grid.")
    return [round(lo + index * step, 10) for index in range(int(round((hi - lo) / step)) + 1)]


def _select_alpha(scores: list[tuple[float, float]], patience: int) -> tuple[float, float, list[tuple[float, float]]]:
    """Select alpha with the same early-stopping rule the main runner uses.

    This delegates to ``PerTaskAlphaTracker`` rather than reimplementing the
    rule.  A previous hand-written copy advanced the bad-step counter on any
    non-improving step, including an exact tie, while the tracker resets it on
    a plateau and only counts a genuine decline.  On EuroSAT the validation
    curve plateaus at 0.7741 for seven consecutive alphas; the copy exhausted
    patience=5 there and returned alpha=2.3 after visiting 30 of 101 grid
    points, where the tracker continues past the plateau to the real optimum
    near 6.1.  That single difference accounted for the swap table reading
    79.92% against the in-memory runner's 87.49% on a matched cell, and was
    misread as an artifact-reconstruction defect.
    """
    tracker = PerTaskAlphaTracker(task_names=["cell"], initial_alpha=scores[0][0], patience=int(patience))
    visited: list[tuple[float, float]] = []
    for alpha, score in scores:
        visited.append((alpha, score))
        if not tracker.primary_active[0]:
            visited.pop()
            break
        tracker.update(alpha=float(alpha), indices=[0], primary_accs=[float(score)], secondary_accs=[float(score)])
    return float(tracker.best_primary_alpha[0]), float(tracker.best_primary_acc[0]), visited


def _persist_transported_deltas(
    cfg: Mapping[str, Any],
    *,
    transported: Mapping[tuple[str, str], Mapping[str, torch.Tensor]],
    task: str,
    method_name: str,
    activation_banks: list[str],
    vector_banks: list[str],
) -> dict[str, Any] | None:
    """Save each transported delta so the merge stage need not refit transport.

    Merging the crossed cells requires the same transported vectors this runner
    already produces.  Recomputing them inside a merge runner would refit
    Theseus/BiCo per merger and put the two stages on different numerical
    footings; persisting them keeps one transport fit behind every table that
    quotes it, and the recorded hash lets a later stage prove it consumed the
    exact vector this run evaluated.
    """

    root = cfg.get("save_transported_deltas_root")
    if root in (None, ""):
        return None
    task_dir = Path(str(root)) / method_name / task
    if task_dir.exists():
        raise FileExistsError(f"Refusing to overwrite transported deltas: {task_dir}")
    task_dir.mkdir(parents=True, exist_ok=False)
    records: dict[str, Any] = {}
    for activation_bank in activation_banks:
        for vector_bank in vector_banks:
            delta = transported[(activation_bank, vector_bank)]
            cell = f"{activation_bank}__{vector_bank}"
            torch.save(dict(delta), task_dir / f"{cell}.pt")
            records[cell] = {
                "path": str(task_dir / f"{cell}.pt"),
                "sha256": state_dict_sha256(delta),
                "activation_bank": activation_bank,
                "vector_bank": vector_bank,
            }
    metadata = {
        "task": task, "method": method_name,
        "activation_banks": activation_banks, "vector_banks": vector_banks,
        "campaign": cfg.get("campaign"), "cells": records,
    }
    metadata_path = task_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (task_dir / "COMPLETE").touch(exist_ok=False)
    return records


def _read_bank(root: Path, tasks: list[str], condition: str) -> dict[str, dict[str, Any]]:
    path = root / condition
    if not (path / "COMPLETE").is_file():
        raise FileNotFoundError(f"Incomplete {condition} capture bank: {path}")
    manifest = json.loads((path / "artifact_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("condition") != condition:
        raise ValueError(f"Capture condition mismatch in {path}")
    if list(manifest.get("tasks", [])) != tasks:
        raise ValueError(f"Capture task order differs in {path}")
    return validate_artifact_bank(path, tasks)


def _resolve_banks(cfg: Mapping[str, Any]) -> list[str]:
    """Name the correction banks to cross.

    The historical 2x2 shared/skip crossing stays the default so an existing
    config reproduces its campaign unchanged.  Naming ``independent`` as well
    turns the same runner into the full 3x3 crossing without touching any
    transport, alpha, or evaluation rule.
    """

    return _bank_list(cfg.get("banks", DEFAULT_BANKS), field="banks")


def _bank_list(raw: Any, *, field: str) -> list[str]:
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    banks = [str(item) for item in raw]
    if not banks or len(set(banks)) != len(banks):
        raise ValueError(f"{field} must name one or more distinct correction conditions.")
    unknown = sorted(set(banks) - set(KNOWN_BANKS))
    if unknown:
        raise ValueError(f"Unknown correction banks in {field}: {unknown}; expected a subset of {list(KNOWN_BANKS)}.")
    if "shared" not in banks:
        raise ValueError(f"The shared bank is the alpha anchor and must be present in {field}.")
    return banks


def _resolve_bank_axes(cfg: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Resolve the activation and task-vector axes separately.

    The two axes are not interchangeable.  The activation bank supplies only the
    corrected *base* to the transport fit, and BRACE builds that base the same
    way under shared and independent correction -- the two modes diverge only at
    the fine-tuned endpoint, and their corrected bases are byte-identical on
    every Vision8 task.  Naming ``independent`` on the activation axis therefore
    recomputes the ``shared`` column, while on the task-vector axis it is a
    genuinely different vector.  Splitting the axes lets a config ask for the
    six distinct cells instead of a nine-cell square with three duplicates.

    ``banks`` still sets both axes at once, so existing configs are unchanged.
    """

    default = _resolve_banks(cfg)
    activation = _bank_list(cfg["activation_banks"], field="activation_banks") if "activation_banks" in cfg else default
    vector = _bank_list(cfg["vector_banks"], field="vector_banks") if "vector_banks" in cfg else default
    return activation, vector


def _validate_pair(shared: Mapping[str, Any], skip: Mapping[str, Any]) -> None:
    for task in shared:
        left, right = shared[task], skip[task]
        if set(left["base"]) != set(right["base"]) or set(left["tv"]) != set(right["tv"]):
            raise ValueError(f"Shared/Skip keyspace mismatch for {task}")
        for key in left["base"]:
            if left["base"][key].shape != right["base"][key].shape:
                raise ValueError(f"Shared/Skip base shape mismatch for {task}:{key}")
        for key in left["tv"]:
            if left["tv"][key].shape != right["tv"][key].shape:
                raise ValueError(f"Shared/Skip TV shape mismatch for {task}:{key}")
        for field in ("source_depth", "target_depth", "calibration"):
            if left["metadata"].get(field) != right["metadata"].get(field):
                raise ValueError(f"Shared/Skip capture metadata mismatch for {task}:{field}")


def _apply_determinism(cfg: Mapping[str, Any]) -> bool:
    """Seed and pin deterministic kernels when the config asks for it.

    ``vision_rebase`` does this in its own runner, but this module borrowed only
    its prepare/context helpers, so BiCo has been fitting here under unseeded,
    non-deterministic kernels.  Theseus is forward-only and reproduces anyway;
    BiCo runs a backward pass to populate its hooks and does not.  Returns the
    resolved value so the summary can state which way the run went.
    """

    strict = bool(cfg.get("deterministic_strict", False))
    deterministic = strict or bool(cfg.get("deterministic", False))
    if deterministic:
        _set_deterministic_seed(int(cfg.get("seed", 89)))
    if strict:
        # `_set_deterministic_seed` asks with `warn_only=True`, which lets a
        # non-deterministic kernel run after printing a warning instead of
        # forcing a deterministic one.  BiCo's backward pass reaches the
        # memory-efficient attention backward, which warns and proceeds, so the
        # non-strict switch left four probe runs with four different deltas.
        # PyTorch's own warning says determinism for that kernel requires
        # `warn_only=False`; under it an op with no deterministic implementation
        # raises instead of silently varying, which is the behaviour a
        # reproducibility gate needs.
        torch.use_deterministic_algorithms(True, warn_only=False)
    return deterministic


def run(cfg: dict[str, Any], *, task: str, method_name: str, output_dir: Path) -> dict[str, Any]:
    if method_name not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    deterministic = _apply_determinism(cfg)
    suite = SUITES[str(cfg.get("suite", "vision8"))]
    tasks = list(suite.tasks)
    if task not in tasks:
        raise ValueError(f"Unknown task {task!r}")
    capture_root = Path(str(cfg["artifact_capture_root"]))
    activation_banks, vector_banks = _resolve_bank_axes(cfg)
    # One read per distinct bank: the axes overlap, and a bank costs a full
    # endpoint load plus reconstruction check.
    bank_names = list(dict.fromkeys([*activation_banks, *vector_banks]))
    banks = {bank: _read_bank(capture_root, tasks, bank) for bank in bank_names}
    for bank in bank_names:
        if bank != "shared":
            _validate_pair(banks["shared"], banks[bank])

    source_cfg = OpenClipBuildConfig(
        model_name=str(cfg["source_clip_model"]), pretrained=str(cfg["source_clip_pretrained"]),
        device=str(cfg.get("device", "cuda")), dtype=cfg.get("dtype"),
    )
    target_cfg = OpenClipBuildConfig(
        model_name=str(cfg["target_clip_model"]), pretrained=str(cfg["target_clip_pretrained"]),
        device=str(cfg.get("device", "cuda")), dtype=cfg.get("dtype"),
    )
    clf_source, clf_target = OpenClipClassifier.build(source_cfg), OpenClipClassifier.build(target_cfg)
    target_base = to_cpu_fp32(dict(clf_target.model.state_dict()))
    target_hash = state_dict_sha256(target_base)
    context = _build_task_context(
        task, suite=suite, cfg=cfg, clf_target=clf_target, clf_source=clf_source,
        source_cfg=source_cfg, target_cfg=target_cfg, use_humanized_classnames=False, need_source_loaders=True,
    )
    method_cfg = {**cfg, "method": method_name}
    resolved_method, method_params = resolve_rebase_method_config(method_cfg)
    method = get_method(resolved_method)
    alpha_grid, patience = _alpha_values(cfg), int(cfg.get("alpha_patience", 5))
    if not alpha_grid or alpha_grid[0] != 0.0:
        raise ValueError("The alpha grid must begin at 0.")

    def score(delta: Mapping[str, torch.Tensor], alpha: float, split: str) -> float:
        mounted = {key: value.detach().clone() for key, value in target_base.items()}
        for key, value in delta.items():
            mounted[key] = mounted[key] + float(alpha) * value.to(dtype=mounted[key].dtype)
        load_into_model(clf_target.model, mounted, strict=True)
        return float(eval_task_top1(
            clf=clf_target, loaders=context.loaders, classnames=context.classnames,
            build_cfg_task=context.build_cfg_task, device=source_cfg.device, split=split,
        ))

    transported: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    prepare_records: dict[str, Any] = {}
    source_depth = int(banks["shared"][task]["metadata"]["target_depth"])
    for activation_bank in activation_banks:
        source_model = _expanded_template(clf_source, source_depth, source_cfg.device)
        # Capture banks intentionally contain visual tensors only; the untouched
        # text-side state remains from the source OpenCLIP template.
        load_into_model(source_model, banks[activation_bank][task]["base"], strict=False)
        source_state = to_cpu_fp32(dict(source_model.state_dict()))
        prepared = _build_rebase_prepared(
            method_name=resolved_method, method=method, method_params=method_params, cfg=cfg,
            device=source_cfg.device, grad_batch_size=None, grad_imgs_per_class=None, grad_num_batches=None,
            theseus_like_method=resolved_method.startswith("theseus"), bico_mode=resolved_method.startswith("bico"),
            run_block_extension_prestep=True, clf_source=clf_source, clf_target=clf_target,
            classnames=context.classnames, loaders=context.loaders, source_loaders=context.source_loaders,
            build_cfg_task=context.build_cfg_task, source_build_cfg_task=context.source_build_cfg_task,
            task_source_base_sd=source_state, target_base_sd=target_base,
            task_delta=banks[activation_bank][task]["tv"], source_base_model_task=source_model,
            transfusion_prepared=None,
        )
        prepare_records[activation_bank] = {
            "source_base_sha256": state_dict_sha256(banks[activation_bank][task]["base"]),
            "prepared_reused_for_tvs": list(vector_banks),
        }
        for vector_bank in vector_banks:
            delta = method.transport(
                source_base=source_state, target_base=target_base,
                delta=banks[vector_bank][task]["tv"], strict=True, prepared=prepared, **method_params,
            )
            transported[(activation_bank, vector_bank)] = to_cpu_fp32(delta)

    curves: dict[tuple[str, str], list[tuple[float, float]]] = {}
    selected: dict[tuple[str, str], tuple[float, float, list[tuple[float, float]]]] = {}
    for cell, delta in transported.items():
        curves[cell] = [(alpha, score(delta, alpha, "val")) for alpha in alpha_grid]
        selected[cell] = _select_alpha(curves[cell], patience)
    anchor = selected[("shared", "shared")][0]
    delta_records = _persist_transported_deltas(
        cfg, transported=transported, task=task, method_name=method_name,
        activation_banks=activation_banks, vector_banks=vector_banks,
    )
    rows = []
    for activation_bank in activation_banks:
        for vector_bank in vector_banks:
            cell = (activation_bank, vector_bank)
            selected_alpha, selected_val, visited = selected[cell]
            delta = transported[cell]
            rows.append({
                "activation_bank": activation_bank, "vector_bank": vector_bank,
                "selected_alpha": selected_alpha, "selected_val_accuracy": selected_val,
                "selected_test_accuracy": score(delta, selected_alpha, "test"),
                "shared_anchor_alpha": anchor,
                "shared_anchor_test_accuracy": score(delta, anchor, "test"),
                "visited_validation_curve": [{"alpha": alpha, "accuracy": value} for alpha, value in visited],
                "transported_delta_sha256": state_dict_sha256(delta),
            })
    payload = {
        "campaign": cfg.get("campaign"), "task": task, "method": method_name,
        "deterministic": deterministic,
        "deterministic_strict": bool(cfg.get("deterministic_strict", False)),
        "banks": bank_names, "activation_banks": activation_banks, "vector_banks": vector_banks,
        "transported_delta_artifacts": delta_records,
        "target_base_sha256": target_hash, "prepare_records": prepare_records,
        "capture_root": str(capture_root), "resolved_config": cfg,
        "slurm": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")},
        "alpha_protocol": {"split": "val", "grid": alpha_grid, "patience": patience, "anchor": "shared/shared"},
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    add_device_dtype_args(parser, device_default=None, dtype_default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cfg = merge_non_none(
        load_json(args.config) if args.config else {},
        {"device": args.device, "dtype": args.dtype, "num_workers": args.num_workers},
    )
    logging_cfg = merge_logging_config(cfg.get("logging", {}), {})
    logger = start_run(
        entrypoint="eval.vision_brace_transport_swap", logging_cfg=logging_cfg,
        summary_path=default_summary_path(entrypoint="eval.vision_brace_transport_swap", logging_cfg=logging_cfg,
                                          default_parent=Path(args.output_dir).parent / "run_logs"),
        metadata={"task": args.task, "method": args.method, "resolved_config": cfg},
    )
    try:
        result = run(cfg, task=args.task, method_name=args.method, output_dir=Path(args.output_dir))
        logger.log_summary(result)
        logger.finish("success")
    except Exception as exc:
        finish_with_error(logger, exc)
        raise


if __name__ == "__main__":
    main()
