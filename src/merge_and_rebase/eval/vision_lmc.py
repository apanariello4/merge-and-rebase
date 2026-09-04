"""Evaluate BRACE endpoint pairs along their parameter-interpolation paths.

This runner deliberately has a small, single purpose: for each task it compares
the original base/FT path with independently, steered, and pairwise-shared
BRACE-corrected paths.  It writes one isolated raw CSV per task;
``scripts/aggregate_lmc_results.py`` combines those files into the
paper-facing tables and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.utils.helpers import load_json, parse_csv

from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import humanize, to_cpu_fp32
from ..io.ckpt import align_to_base_keys, load_ckpt
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from .block_extension import resolve_block_extension_config, run_block_extension, select_loader
from .datasets.vision8_14_20 import SUITES
from .vision_connectivity import _eval_loader_top1_and_loss


MODES = ("original", "independent", "steer", "shared")


def _resolve_tasks(cfg: dict[str, Any]) -> tuple[str, list[str]]:
    suite_name = str(cfg.get("suite", "vision8"))
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
    requested = cfg.get("tasks", "all")
    if requested == "all":
        tasks = list(SUITES[suite_name].tasks)
    elif isinstance(requested, str):
        tasks = parse_csv(requested)
    elif isinstance(requested, list):
        tasks = [str(x) for x in requested]
    else:
        raise ValueError("tasks must be 'all', a comma-separated string, or a list.")
    invalid = [task for task in tasks if task not in SUITES[suite_name].tasks]
    if invalid:
        raise ValueError(f"Tasks are not in {suite_name}: {invalid}")
    return suite_name, tasks


def _resolve_alphas(cfg: dict[str, Any]) -> list[float]:
    raw = cfg.get("alphas", None)
    if raw is not None:
        values = [float(x) for x in raw]
    else:
        step = float(cfg.get("alpha_step", 0.05))
        if step <= 0:
            raise ValueError("alpha_step must be positive.")
        n_steps = round(1.0 / step)
        if abs(n_steps * step - 1.0) > 1e-8:
            raise ValueError("alpha_step must divide [0, 1] exactly.")
        values = [i * step for i in range(n_steps + 1)]
    values = sorted({round(x, 10) for x in values})
    if not values or values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError("The interpolation grid must include exactly alpha=0 and alpha=1.")
    if any(x < 0.0 or x > 1.0 for x in values):
        raise ValueError("LMC alphas must be in [0, 1].")
    return values


def _full_ft_state(*, checkpoint: str, base_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    ckpt_state = load_ckpt(checkpoint)
    aligned = align_to_base_keys(ckpt_state, base_state)
    if not aligned:
        raise ValueError(f"No checkpoint tensors aligned to the source base: {checkpoint}")
    full = to_cpu_fp32(base_state)
    full.update(to_cpu_fp32(aligned))
    return full


def _assert_pair_compatible(*, mode: str, base_state: dict[str, torch.Tensor], ft_state: dict[str, torch.Tensor]) -> None:
    if set(base_state) != set(ft_state):
        only_base = sorted(set(base_state) - set(ft_state))[:5]
        only_ft = sorted(set(ft_state) - set(base_state))[:5]
        raise ValueError(f"{mode}: endpoint keyspaces differ (base-only={only_base}, ft-only={only_ft}).")
    mismatches = [
        f"{key}: {tuple(base_state[key].shape)} != {tuple(ft_state[key].shape)}"
        for key in base_state
        if tuple(base_state[key].shape) != tuple(ft_state[key].shape)
    ]
    if mismatches:
        raise ValueError(f"{mode}: endpoint tensor shapes differ: {mismatches[:5]}")
    non_float = [key for key, value in base_state.items() if not value.is_floating_point()]
    if non_float:
        raise ValueError(f"{mode}: non-floating state tensors cannot be interpolated: {non_float[:5]}")


def _interpolate(
    base_state: dict[str, torch.Tensor], ft_state: dict[str, torch.Tensor], alpha: float
) -> dict[str, torch.Tensor]:
    return {key: torch.lerp(base_value, ft_state[key], float(alpha)) for key, base_value in base_state.items()}


def _state_from_model(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return to_cpu_fp32({key: value.detach().cpu() for key, value in model.state_dict().items()})


def _make_corrected_pair(
    *,
    source_model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    ft_state: dict[str, torch.Tensor],
    calibration_loader: Any,
    target_layers_total: int | None,
    block_config: Any,
    lmc_mode: str,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, torch.nn.Module]:
    base_model = deepcopy(source_model)
    ft_model = deepcopy(source_model)
    base_model.load_state_dict(base_state, strict=True)
    ft_model.load_state_dict(ft_state, strict=True)
    corrected_config = block_config.__class__(**{**block_config.__dict__, "lmc_mode": lmc_mode})
    final_depth = run_block_extension(
        source_base_model=base_model,
        source_ft_model=ft_model,
        calibration_loader=calibration_loader,
        target_layers_total=target_layers_total,
        config=corrected_config,
        device=device,
    )
    return _state_from_model(base_model), _state_from_model(ft_model), int(final_depth), base_model


def _write_raw(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "mode", "alpha", "accuracy", "loss", "campaign_direction", "condition_ridge_identity"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate original, independent, and shared BRACE LMC paths.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", default=None, help="Optional comma-separated task override.")
    parser.add_argument("--output-dir", default=None, help="New directory for this task's raw result.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--alphas", default=None, help="Optional comma-separated alpha override; must include 0 and 1.")
    parser.add_argument("--ridge-identity", type=float, default=None, help="Optional in-memory override for block_extension_params.ridge_identity.")
    args = parser.parse_args()

    cfg = load_json(args.config)
    if args.tasks is not None:
        cfg["tasks"] = args.tasks
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.device is not None:
        cfg["device"] = args.device
    if args.alphas is not None:
        cfg["alphas"] = [float(value) for value in parse_csv(args.alphas)]
        cfg.pop("alpha_step", None)
    if args.ridge_identity is not None:
        block_params = cfg.get("block_extension_params")
        if not isinstance(block_params, dict):
            raise ValueError("--ridge-identity requires block_extension_params to be an object.")
        block_params["ridge_identity"] = float(args.ridge_identity)
    suite_name, tasks = _resolve_tasks(cfg)
    if not tasks:
        raise ValueError("No tasks selected.")
    alphas = _resolve_alphas(cfg)
    output_dir = Path(str(cfg["output_dir"]))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing LMC result directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    enabled, block_config = resolve_block_extension_config(cfg)
    if not enabled:
        raise ValueError("LMC requires block_extension_enabled=true.")
    if str(block_config.lmc_mode) not in {"independent", "steer", "shared"}:
        raise ValueError("block_extension_params.lmc_mode must be independent, steer, or shared.")
    target_layers_total = block_config.target_layers_total
    if target_layers_total is None:
        raise ValueError("Set block_extension_params.target_layers_total explicitly for LMC.")

    device = str(cfg.get("device", "cuda"))
    campaign_direction = str(cfg.get("campaign_direction", "unspecified"))
    condition_ridge_identity = float(block_config.ridge_identity)
    eval_split = str(cfg.get("eval_split", "test"))
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be 'val' or 'test'.")
    tuned_by_task = cfg.get("tuned_ckpts")
    if not isinstance(tuned_by_task, dict):
        raise ValueError("tuned_ckpts must be a task-to-checkpoint object.")
    missing = [task for task in tasks if task not in tuned_by_task]
    if missing:
        raise ValueError(f"Missing tuned checkpoint entries: {missing}")

    source_cfg = OpenClipBuildConfig(
        model_name=str(cfg.get("source_clip_model", "ViT-B-16")),
        pretrained=str(cfg.get("source_clip_pretrained", "openai")),
        device=device,
        dtype=cfg.get("dtype", None),
    )
    clf = OpenClipClassifier.build(source_cfg)
    source_model = clf.model
    source_base_state = _state_from_model(source_model)
    no_humanize = bool(cfg.get("no_humanize", True))
    all_rows: list[dict[str, Any]] = []
    task_metadata: dict[str, Any] = {}

    for task in tasks:
        checkpoint = str(tuned_by_task[task])
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint for {task} does not exist: {checkpoint}")
        print(f"[lmc] task={task} checkpoint={checkpoint}")
        hf_path, hf_config, split_map = SUITES[suite_name].resolver(task)
        hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
        loaders = build_vision_loaders(
            hf_ds=hf_ds,
            hf_path=hf_path,
            preprocess=clf.preprocess,
            ft_epochs=1,
            split_map=split_map,
            batch_size=int(cfg.get("batch_size", 16)),
            num_workers=int(cfg.get("num_workers", 0)),
            pin_memory=True,
            val_fraction=float(cfg.get("val_fraction", 0.1)),
            seed=int(cfg.get("seed", 89)),
        )
        classnames = list(loaders.classnames)
        if not no_humanize:
            classnames = [humanize(name) for name in classnames]
        templates = get_templates(task)
        if not templates:
            raise ValueError(f"No prompt templates for {task}")
        task_cfg = OpenClipBuildConfig(
            model_name=source_cfg.model_name,
            pretrained=source_cfg.pretrained,
            device=source_cfg.device,
            dtype=source_cfg.dtype,
            prompt_templates=templates,
        )
        clf.build_zeroshot_text_features(classnames, task_cfg, cache_dir="src/.cache/zs_cache", force_rebuild=False)
        eval_loader = loaders.val if eval_split == "val" else loaders.test
        calibration_loader = select_loader(
            block_config.calibration_split,
            train_loader=loaders.train,
            test_loader=loaders.test,
            val_loader=loaders.val,
        )
        ft_state = _full_ft_state(checkpoint=checkpoint, base_state=source_base_state)
        pairs: dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.nn.Module]] = {
            "original": (source_base_state, ft_state, source_model)
        }
        depths: dict[str, int] = {"original": len(source_model.visual.transformer.resblocks)}
        for mode in MODES[1:]:
            base_corr, ft_corr, depth, evaluation_model = _make_corrected_pair(
                source_model=source_model,
                base_state=source_base_state,
                ft_state=ft_state,
                calibration_loader=calibration_loader,
                target_layers_total=target_layers_total,
                block_config=block_config,
                lmc_mode=mode,
                device=device,
            )
            pairs[mode] = (base_corr, ft_corr, evaluation_model)
            depths[mode] = depth

        compatibility: dict[str, int] = {}
        for mode, (base_state, tuned_state, evaluation_model) in pairs.items():
            _assert_pair_compatible(mode=mode, base_state=base_state, ft_state=tuned_state)
            compatibility[mode] = len(base_state)
            # Corrected pairs have a different depth from the original pair. Reuse
            # the classifier wrapper and task text features with a compatible model.
            clf.model = evaluation_model
            for alpha in alphas:
                state = _interpolate(base_state, tuned_state, alpha)
                clf.model.load_state_dict(state, strict=True)
                accuracy, loss = _eval_loader_top1_and_loss(
                    clf=clf, loader=eval_loader, device=device, text_features=None
                )
                row = {
                    "task": task,
                    "mode": mode,
                    "alpha": float(alpha),
                    "accuracy": float(accuracy),
                    "loss": float(loss),
                    "campaign_direction": campaign_direction,
                    "condition_ridge_identity": condition_ridge_identity,
                }
                all_rows.append(row)
                print(
                    f"[lmc] task={task} mode={mode} alpha={alpha:.2f} "
                    f"accuracy={accuracy:.6f} loss={loss:.6f}"
                )
        task_metadata[task] = {
            "checkpoint": checkpoint,
            "depths": depths,
            "compatible_tensor_count": compatibility,
        }
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.empty_cache()

    _write_raw(output_dir / "raw.csv", all_rows)
    metadata = {
        "campaign": "vision8_lmc_original_independent_steer_shared",
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "resolved_config": cfg,
        "suite": suite_name,
        "tasks": tasks,
        "alphas": alphas,
        "eval_split": eval_split,
        "modes": list(MODES),
        "campaign_direction": campaign_direction,
        "condition_ridge_identity": condition_ridge_identity,
        "task_metadata": task_metadata,
        "raw_csv": str(output_dir / "raw.csv"),
        "pid": os.getpid(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[lmc] wrote {len(all_rows)} rows to {output_dir / 'raw.csv'}")


if __name__ == "__main__":
    main()
