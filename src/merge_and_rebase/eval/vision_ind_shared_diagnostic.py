"""Run one paired Independent/Shared BRACE endpoint diagnostic.

This entry point deliberately performs no interpolation-grid evaluation and no
transport or merge.  It rebuilds one task/lambda pair, captures fitted maps and
corrected visual endpoints, and emits a small manifest that points to the
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.data.templates import get_templates
from merge_and_rebase.data.vision_loaders import build_vision_loaders, load_hf_splits
from merge_and_rebase.eval.utils import humanize, to_cpu_fp32
from merge_and_rebase.io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from merge_and_rebase.utils.helpers import load_json, parse_csv

from .brace_diagnostics import BRACEDiagnosticCollector
from .block_extension import BlockExtensionConfig, run_block_extension, select_loader
from .datasets.vision8_14_20 import SUITES
from .vision_block_extension import _evaluate_model_top1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic manifest: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve_task(cfg: dict[str, Any], task_arg: str | None) -> str:
    tasks = parse_csv(task_arg) if task_arg else list(cfg.get("tasks", []))
    if len(tasks) != 1:
        raise ValueError("This diagnostic runner requires exactly one task per output directory.")
    task = tasks[0]
    if task not in SUITES["vision8"].tasks:
        raise ValueError(f"Unsupported diagnostic task '{task}'.")
    return task


def _build_extension_config(cfg: dict[str, Any], ridge_identity: float, mode: str, device: str) -> BlockExtensionConfig:
    calibration = cfg.get("calibration", {})
    initialization = str(cfg.get("initialization", "interpolate_per_weight"))
    if initialization not in {"interpolate_per_weight", "duplicate_per_weight"}:
        raise ValueError(f"Unsupported diagnostic initialization '{initialization}'.")
    return BlockExtensionConfig(
        target_layers_total=int(cfg.get("target_depth", 24)),
        extension_strategy=initialization,
        n_batches_act=max(1, int(calibration.get("num_batches", 10))),
        calibration_split="val" if str(calibration.get("split", "validation")).lower() in {"val", "validation"} else str(calibration.get("split", "test")),
        ridge_identity=float(ridge_identity),
        n_cascade_iters=1,
        share_ft_refs=False,
        lmc_mode=mode,
        verbose=True,
        show_progress=False,
    )


def run_one(args: argparse.Namespace) -> Path:
    cfg = load_json(args.config)
    task = _resolve_task(cfg, args.tasks)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty diagnostic output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = str(args.device or cfg.get("device", "cuda"))
    calibration = cfg.get("calibration", {})
    batch_size = int(args.batch_size or calibration.get("batch_size", 16))
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.get("num_workers", 6))
    source_cfg = OpenClipBuildConfig(
        model_name=str(cfg.get("model", "ViT-B-16")),
        pretrained=str(cfg.get("pretrained", "datacomp_xl_s13b_b90k")),
        device=device,
        dtype=args.dtype or cfg.get("dtype", None),
    )
    clf_source = OpenClipClassifier.build(source_cfg)
    source_base_sd = to_cpu_fp32(dict(clf_source.model.state_dict()))

    checkpoint_root = Path(str(cfg.get("source_checkpoint_root", "")))
    checkpoint_path = checkpoint_root / task / "full_best_ep.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing tuned checkpoint for {task}: {checkpoint_path}")
    tuned_sd = align_to_base_keys(load_ckpt(str(checkpoint_path)), source_base_sd)
    if not tuned_sd:
        raise ValueError(f"No tuned checkpoint tensors aligned to base keys: {checkpoint_path}")

    hf_path, hf_config, split_map = SUITES["vision8"].resolver(task)
    hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
    loaders = build_vision_loaders(
        hf_ds=hf_ds,
        hf_path=hf_path,
        preprocess=clf_source.preprocess,
        ft_epochs=1,
        split_map=split_map,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        val_fraction=float(cfg.get("val_fraction", 0.1)),
        seed=int(calibration.get("seed", 89)),
    )
    calibration_loader = select_loader(
        "val" if str(calibration.get("split", "validation")).lower() in {"val", "validation"} else str(calibration.get("split", "test")),
        train_loader=loaders.train,
        test_loader=loaders.test,
        val_loader=loaders.val,
    )

    metadata_base = {
        "experiment": "ind_shared_diagnostic",
        "task": task,
        "lambda_id": float(args.ridge_identity),
        "model": source_cfg.model_name,
        "pretrained": source_cfg.pretrained,
        "initialization": cfg.get("initialization", "interpolate_per_weight"),
        "calibration": {"num_batches": int(calibration.get("num_batches", 10)), "batch_size": batch_size, "split": calibration.get("split", "validation"), "seed": int(calibration.get("seed", 89))},
        "source_checkpoint": str(checkpoint_path),
        "diagnostic_root": str(args.diagnostic_root) if args.diagnostic_root is not None else None,
        "save_activation_banks": False,
    }
    mode_artifacts: dict[str, Any] = {}
    for mode in ("independent", "shared"):
        mode_dir = output_dir / mode
        collector = BRACEDiagnosticCollector(mode_dir, {**metadata_base, "mode": mode})
        base_model = deepcopy(clf_source.model)
        ft_model = deepcopy(clf_source.model)
        load_into_model(base_model, source_base_sd, strict=False)
        load_into_model(ft_model, source_base_sd, strict=False)
        load_into_model(ft_model, tuned_sd, strict=False)
        extension_cfg = _build_extension_config(cfg, float(args.ridge_identity), mode, device)
        final_depth = run_block_extension(
            source_base_model=base_model,
            source_ft_model=ft_model,
            calibration_loader=calibration_loader,
            target_layers_total=extension_cfg.target_layers_total,
            config=extension_cfg,
            device=device,
            diagnostic_collector=collector,
        )
        collector.save_endpoint("base", base_model)
        collector.save_endpoint("ft", ft_model)
        collector.finalize()
        ft_accuracy = None
        try:
            classnames = list(loaders.classnames)
            templates = get_templates(task)
            classnames = [humanize(c) for c in classnames]
            task_cfg = OpenClipBuildConfig(
                model_name=source_cfg.model_name,
                pretrained=source_cfg.pretrained,
                device=device,
                dtype=source_cfg.dtype,
                prompt_templates=templates,
            )
            ft_accuracy = _evaluate_model_top1(
                model=ft_model,
                clf_source=clf_source,
                loaders=loaders,
                classnames=classnames,
                build_cfg_task=task_cfg,
                device=device,
                split="test",
                first_n_batches=args.first_n_eval_batches,
            )
        except Exception as exc:  # Endpoint capture remains valid if optional eval cannot run.
            print(f"[diagnostic] endpoint evaluation skipped: {exc}")
        mode_artifacts[mode] = {
            "directory": str(mode_dir),
            "maps": str(mode_dir / "maps.pt"),
            "endpoint_base": str(mode_dir / "endpoint_base.pt"),
            "endpoint_ft": str(mode_dir / "endpoint_ft.pt"),
            "metadata": str(mode_dir / "metadata.json"),
            "final_depth": final_depth,
            "ft_endpoint_accuracy": ft_accuracy,
        }
        del base_model, ft_model
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.empty_cache()

    files = [Path(item[key]) for item in mode_artifacts.values() for key in ("maps", "endpoint_base", "endpoint_ft", "metadata")]
    manifest = {
        **metadata_base,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": _sha256(Path(args.config)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "artifact_root": str(output_dir),
        "artifacts": mode_artifacts,
        "artifact_sha256": {str(path.relative_to(output_dir)): _sha256(path) for path in files},
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_once(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tasks", default=None, help="Exactly one task name (or comma-separated value of length one).")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--diagnostic-root", default=None, type=Path, help="Recorded for provenance; output-dir remains authoritative.")
    parser.add_argument("--ridge-identity", required=True, type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--first-n-eval-batches", type=int, default=None)
    args = parser.parse_args()
    print(run_one(args))


if __name__ == "__main__":
    main()
