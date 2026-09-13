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
from .datasets.vision8_14_20 import SUITES
from .vision_brace_tv_swap import _expanded_template, state_dict_sha256, validate_artifact_bank
from .vision_rebase import _build_rebase_prepared, _build_task_context


BANKS = ("shared", "skip")
METHODS = ("theseus", "bico")


def _alpha_values(cfg: Mapping[str, Any]) -> list[float]:
    lo, hi, step = float(cfg["alpha_min"]), float(cfg["alpha_max"]), float(cfg["alpha_step"])
    if step <= 0 or hi < lo:
        raise ValueError("Invalid alpha grid.")
    return [round(lo + index * step, 10) for index in range(int(round((hi - lo) / step)) + 1)]


def _select_alpha(scores: list[tuple[float, float]], patience: int) -> tuple[float, float, list[tuple[float, float]]]:
    """Use the legacy per-task early-stopping rule, including the smaller-alpha tie break."""
    best_alpha, best_score, bad = scores[0][0], scores[0][1], 0
    visited: list[tuple[float, float]] = []
    for alpha, score in scores:
        visited.append((alpha, score))
        if score > best_score:
            best_alpha, best_score, bad = alpha, score, 0
        else:
            bad += 1
            if bad > patience:
                break
    return best_alpha, best_score, visited


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


def run(cfg: dict[str, Any], *, task: str, method_name: str, output_dir: Path) -> dict[str, Any]:
    if method_name not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    suite = SUITES[str(cfg.get("suite", "vision8"))]
    tasks = list(suite.tasks)
    if task not in tasks:
        raise ValueError(f"Unknown task {task!r}")
    capture_root = Path(str(cfg["artifact_capture_root"]))
    banks = {bank: _read_bank(capture_root, tasks, bank) for bank in BANKS}
    _validate_pair(banks["shared"], banks["skip"])

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
    for activation_bank in BANKS:
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
            "prepared_reused_for_tvs": list(BANKS),
        }
        for vector_bank in BANKS:
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
    rows = []
    for activation_bank in BANKS:
        for vector_bank in BANKS:
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
