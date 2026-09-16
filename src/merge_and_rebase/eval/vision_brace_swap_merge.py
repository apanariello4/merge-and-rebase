"""Merge the crossed BRACE activation/task-vector swap cells.

``vision_brace_transport_swap`` answers the single-vector question: it fits
transport on one correction bank's activations and pushes another bank's task
vector through that fit.  Its extension table shows that with the activation
bank held at Shared, a Skip task vector transports about as well as a Shared
one -- 80.87 against 80.50 for Theseus, 90.07 against 90.36 for BiCo.

That equality is what this runner exploits.  The merge correction ablation
compares Shared, Skip, and Independent end to end, so a Shared advantage there
could simply be inherited from better single-vector transport.  Mounting the
already-transported vectors of a fixed activation bank and merging them holds
transport quality fixed and varies only the correction geometry of the vectors
being combined, which isolates the merge stage itself.

The transported deltas are consumed from the bank that
``vision_brace_transport_swap`` persists, and every cell is checked against the
hash recorded there, so one transport fit stands behind both the single-vector
row and the merged row of the same cell.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.utils.helpers import load_json

from ..cli_args import add_config_arg, add_device_dtype_args, merge_non_none
from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import eval_task_top1, to_cpu_fp32
from ..io.ckpt import load_into_model
from ..merge.methods._common import axpy_state_dict
from ..merge.registry import get_method as get_merge_method
from ..merge.tv_conditioning import condition_transported_deltas, spec_from_config
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from .datasets.vision8_14_20 import SUITES
from .vision_brace_transport_swap import _alpha_values, _select_alpha
from .vision_brace_tv_swap import state_dict_sha256

METHODS = ("theseus", "bico")


def _cell_name(activation_bank: str, vector_bank: str) -> str:
    return f"{activation_bank}__{vector_bank}"


def read_transported_cell(
    root: Path, *, method_name: str, tasks: list[str], activation_bank: str, vector_bank: str
) -> dict[str, dict[str, torch.Tensor]]:
    """Load one crossed cell's transported delta for every task.

    The recorded hash is re-checked rather than trusted.  A merged table row and
    its single-vector reference row are only comparable if they consumed the
    same transported vector, and that is exactly what the hash establishes.
    """

    cell = _cell_name(activation_bank, vector_bank)
    deltas: dict[str, dict[str, torch.Tensor]] = {}
    for task in tasks:
        task_dir = root / method_name / task
        if not (task_dir / "COMPLETE").is_file():
            raise FileNotFoundError(f"Incomplete transported-delta bank for '{task}': {task_dir}")
        metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
        record = metadata.get("cells", {}).get(cell)
        if record is None:
            raise KeyError(f"Cell '{cell}' absent from the transported bank for '{task}': {task_dir}")
        delta = torch.load(task_dir / f"{cell}.pt", map_location="cpu", weights_only=True)
        actual = state_dict_sha256(delta)
        if actual != record.get("sha256"):
            raise ValueError(
                f"Transported delta hash mismatch for '{task}' cell '{cell}': "
                f"recorded {record.get('sha256')}, loaded {actual}."
            )
        deltas[task] = delta
    return deltas


def single_vector_reference(
    root: Path | None, *, method_name: str, tasks: list[str], activation_bank: str, vector_bank: str
) -> dict[str, float] | None:
    """Read the matched single-vector test accuracy for this cell, if available.

    Merging is only credited with what it adds: the per-task difference between
    the merged model and the single transported vector of the same cell is the
    quantity that separates a merge-stage effect from an inherited transport
    advantage.
    """

    if root is None:
        return None
    reference: dict[str, float] = {}
    for task in tasks:
        summary_path = root / method_name / task / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing single-vector swap summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = [
            row for row in summary.get("rows", [])
            if row.get("activation_bank") == activation_bank and row.get("vector_bank") == vector_bank
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one {activation_bank}/{vector_bank} row in {summary_path}, found {len(rows)}."
            )
        reference[task] = float(rows[0]["selected_test_accuracy"])
    return reference


def _task_evaluation_context(
    task: str, *, suite: Any, cfg: Mapping[str, Any], clf: OpenClipClassifier, build_cfg: OpenClipBuildConfig
) -> dict[str, Any]:
    hf_path, hf_config, split_map = suite.resolver(task)
    hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
    loaders = build_vision_loaders(
        hf_ds=hf_ds, hf_path=hf_path, preprocess=clf.preprocess, ft_epochs=1, split_map=split_map,
        batch_size=int(cfg.get("batch_size", 32)), num_workers=int(cfg.get("num_workers", 6)),
        pin_memory=True, val_fraction=float(cfg.get("val_fraction", 0.1)), seed=int(cfg.get("seed", 89)),
    )
    classnames = list(loaders.classnames)
    templates = get_templates(task)
    if not templates:
        raise ValueError(f"get_templates('{task}') returned an empty list.")
    build_cfg_task = OpenClipBuildConfig(
        model_name=build_cfg.model_name, pretrained=build_cfg.pretrained,
        device=build_cfg.device, dtype=build_cfg.dtype, prompt_templates=templates,
    )
    # The capture banks carry visual tensors only, so every merged model in the
    # alpha sweep shares this task's text tower.  Building the zero-shot text
    # features once and reusing the tensor keeps the sweep to image forwards.
    clf.build_zeroshot_text_features(classnames, build_cfg_task, cache_dir="src/.cache/zs_cache", force_rebuild=False)
    text_features = clf._zs_text_features.detach().clone()
    return {
        "loaders": loaders, "classnames": classnames,
        "build_cfg_task": build_cfg_task, "text_features": text_features,
    }


def run(
    cfg: dict[str, Any],
    *,
    method_name: str,
    merge_method: str,
    activation_bank: str,
    vector_bank: str,
    output_dir: Path,
) -> dict[str, Any]:
    if method_name not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    suite = SUITES[str(cfg.get("suite", "vision8"))]
    tasks = list(suite.tasks)

    delta_root = Path(str(cfg["transported_deltas_root"]))
    deltas = read_transported_cell(
        delta_root, method_name=method_name, tasks=tasks,
        activation_bank=activation_bank, vector_bank=vector_bank,
    )
    reference_root = cfg.get("swap_summary_root")
    reference = single_vector_reference(
        Path(str(reference_root)) if reference_root else None,
        method_name=method_name, tasks=tasks,
        activation_bank=activation_bank, vector_bank=vector_bank,
    )

    target_cfg = OpenClipBuildConfig(
        model_name=str(cfg["target_clip_model"]), pretrained=str(cfg["target_clip_pretrained"]),
        device=str(cfg.get("device", "cuda")), dtype=cfg.get("dtype"),
    )
    clf_target = OpenClipClassifier.build(target_cfg)
    target_base = to_cpu_fp32(dict(clf_target.model.state_dict()))
    target_hash = state_dict_sha256(target_base)
    eval_clf = OpenClipClassifier(
        model=deepcopy(clf_target.model), tokenizer=clf_target.tokenizer, preprocess=clf_target.preprocess,
        normalize=clf_target.normalize, logit_scale=clf_target.logit_scale,
    )

    contexts = {
        task: _task_evaluation_context(task, suite=suite, cfg=cfg, clf=eval_clf, build_cfg=target_cfg)
        for task in tasks
    }

    strict_load = bool(cfg.get("strict_load", True))
    merge_params = dict(cfg.get("merge_params", {}))

    # Conditioning happens after the hash gate and before the merger, so the
    # summary can name both the transported vector it started from and the
    # operation that was applied to it.  The default spec is a no-op.
    conditioning_spec = spec_from_config(cfg.get("tv_conditioning"))
    transported_delta_sha256 = {task: state_dict_sha256(deltas[task]) for task in tasks}
    deltas, conditioning_diagnostics = condition_transported_deltas(deltas, conditioning_spec)
    conditioned_delta_sha256 = {task: state_dict_sha256(deltas[task]) for task in tasks}
    if conditioning_spec.mode == "off" and conditioned_delta_sha256 != transported_delta_sha256:
        raise AssertionError("tv_conditioning mode 'off' must leave every transported delta untouched.")
    print(f"tv_conditioning: {conditioning_spec.as_dict()}")

    merger = get_merge_method(merge_method)
    tuned = [axpy_state_dict(target_base, deltas[task], alpha=1.0) for task in tasks]
    prepared = merger.prepare(base=target_base, tuned=tuned, strict=strict_load, **merge_params)

    def evaluate(alpha: float, split: str) -> dict[str, float]:
        load_into_model(eval_clf.model, merger.apply(prepared, alpha=float(alpha)), strict=strict_load)
        return {
            task: float(eval_task_top1(
                clf=eval_clf, loaders=contexts[task]["loaders"], classnames=contexts[task]["classnames"],
                build_cfg_task=contexts[task]["build_cfg_task"], device=target_cfg.device, split=split,
                text_features=contexts[task]["text_features"],
            ))
            for task in tasks
        }

    alpha_grid, patience = _alpha_values(cfg), int(cfg.get("alpha_patience", 5))
    if not alpha_grid or alpha_grid[0] != 0.0:
        raise ValueError("The alpha grid must begin at 0.")

    # One shared merge alpha, selected on the validation mean, matching the
    # shared-alpha policy the merge correction tables report.
    validation_curve: list[tuple[float, float]] = []
    validation_per_task: dict[float, dict[str, float]] = {}
    for alpha in alpha_grid:
        per_task = evaluate(alpha, "val")
        validation_per_task[alpha] = per_task
        validation_curve.append((alpha, sum(per_task.values()) / len(per_task)))
        selected_alpha, _, visited = _select_alpha(validation_curve, patience)
        if len(visited) < len(validation_curve):
            break
    selected_alpha, selected_val, visited = _select_alpha(validation_curve, patience)

    test_per_task = evaluate(selected_alpha, "test")
    test_mean = sum(test_per_task.values()) / len(test_per_task)

    merge_gain = None
    if reference is not None:
        per_task_gain = {task: test_per_task[task] - reference[task] for task in tasks}
        merge_gain = {
            "single_vector_test_accuracy": reference,
            "single_vector_test_mean": sum(reference.values()) / len(reference),
            "per_task_merge_gain": per_task_gain,
            "mean_merge_gain": sum(per_task_gain.values()) / len(per_task_gain),
        }

    payload = {
        "campaign": cfg.get("campaign"), "method": method_name, "merge_method": merge_method,
        "merge_params": merge_params, "activation_bank": activation_bank, "vector_bank": vector_bank,
        "cell": _cell_name(activation_bank, vector_bank), "tasks": tasks,
        "target_base_sha256": target_hash,
        "transported_delta_sha256": transported_delta_sha256,
        "conditioned_delta_sha256": conditioned_delta_sha256,
        "tv_conditioning": conditioning_diagnostics,
        "transported_deltas_root": str(delta_root),
        "alpha_protocol": {
            "split": "val", "grid": alpha_grid, "patience": patience, "policy": "shared",
            "selected_alpha": selected_alpha, "selected_val_mean": selected_val,
            "visited_validation_curve": [{"alpha": alpha, "mean_accuracy": value} for alpha, value in visited],
            "visited_validation_per_task": {
                f"{alpha:.10g}": validation_per_task[alpha] for alpha, _ in visited
            },
        },
        "test_per_task": test_per_task, "test_mean": test_mean, "merge_gain": merge_gain,
        "resolved_config": cfg,
        "slurm": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    add_device_dtype_args(parser, device_default=None, dtype_default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--merge-method", required=True)
    parser.add_argument("--activation-bank", required=True)
    parser.add_argument("--vector-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = merge_non_none(
        load_json(args.config) if args.config else {},
        {"device": args.device, "dtype": args.dtype, "num_workers": args.num_workers},
    )
    logging_cfg = merge_logging_config(cfg.get("logging", {}), {})
    logger = start_run(
        entrypoint="eval.vision_brace_swap_merge", logging_cfg=logging_cfg,
        summary_path=default_summary_path(entrypoint="eval.vision_brace_swap_merge", logging_cfg=logging_cfg,
                                          default_parent=Path(args.output_dir).parent / "run_logs"),
        metadata={
            "method": args.method, "merge_method": args.merge_method,
            "activation_bank": args.activation_bank, "vector_bank": args.vector_bank,
            "resolved_config": cfg,
        },
    )
    try:
        result = run(
            cfg, method_name=args.method, merge_method=args.merge_method,
            activation_bank=args.activation_bank, vector_bank=args.vector_bank,
            output_dir=Path(args.output_dir),
        )
        logger.log_summary(result)
        logger.finish("success")
    except Exception as exc:  # noqa: BLE001 - surfaced through run logging then re-raised
        finish_with_error(logger, exc)
        raise


if __name__ == "__main__":
    main()
