from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.utils.helpers import load_json

from ...cli_args import (
    add_alpha_args,
    add_config_arg,
    add_device_dtype_args,
    add_logging_args,
    add_suite_arg,
    add_tasks_arg,
    build_logging_overrides,
    merge_non_none,
    parse_json_object_arg,
)
from ...io.ckpt import resolve_ckpt_path
from ...merge.registry import list_methods as list_merge_methods
from ...models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ...rebase import list_methods
from ...rebase.run_config import resolve_run_config
from ...run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from ..datasets.vision8_14_20 import SUITES
from .merge import _VALID_MERGE_MODES
from .pipeline import VisionRuntime, run_rebase


def _set_deterministic_seed(seed: int) -> None:
    """Seed torch/cuda and force deterministic kernels for this run.

    BRACE's per-weight reference capture forwards the same pristine model
    through the same frozen calibration batches once per structural step
    (rather than once, globally) to bound memory. Without
    `cudnn.deterministic`/`use_deterministic_algorithms`, repeated forward
    passes over identical weights and inputs are not guaranteed bit-identical
    on GPU, and with a small ridge_weight the resulting ridge fit can amplify
    that noise into visible downstream accuracy drift between otherwise
    identical runs.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    run_logger = None
    try:
        p = argparse.ArgumentParser("Rebase task vectors from source base A to target base B and evaluate")

        add_config_arg(p)
        add_suite_arg(p, choices=sorted(SUITES.keys()))
        add_tasks_arg(p, help_text="Comma-separated task names, or 'all'.")

        p.add_argument("--source-clip-model", type=str, default=None)
        p.add_argument("--source-clip-pretrained", type=str, default=None)
        p.add_argument("--target-clip-model", type=str, default=None)
        p.add_argument("--target-clip-pretrained", type=str, default=None)

        add_device_dtype_args(p, device_default=None, dtype_default=None)

        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--num-workers", type=int, default=None)
        p.add_argument("--val-fraction", type=float, default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--no-humanize", action="store_true", default=None, help="Use raw classnames.")

        p.add_argument("--tuned-ckpts", type=str, nargs="+", default=None)
        p.add_argument("--weights", type=float, nargs="*", default=None)
        p.add_argument("--strict-load", action="store_true", default=None)
        p.add_argument("--method", type=str, choices=list_methods(), default=None)
        p.add_argument("--method-params", type=str, default=None, help="JSON object for rebase-method kwargs.")
        p.add_argument(
            "--merge-mode",
            type=str,
            choices=list(_VALID_MERGE_MODES),
            default=None,
            help="Compose task deltas into one merged model instead of (or after) per-task transport.",
        )
        p.add_argument("--merge-method", type=str, choices=list_merge_methods(), default=None)
        p.add_argument("--merge-params", type=str, default=None, help="JSON object for merge-method kwargs.")

        p.add_argument("--mask-mode", type=str, default=None, choices=["normal", "force"])
        p.add_argument("--vote", type=str, default=None, choices=["mean", "majority", "max"])
        p.add_argument("--grad-batch-size", type=int, default=None)
        p.add_argument("--grad-imgs-per-class", type=int, default=None)
        p.add_argument("--grad-num-batches", type=int, default=None)

        add_alpha_args(
            p,
            alpha_default=None,
            alpha_min_default=None,
            alpha_max_default=None,
            alpha_step_default=None,
            alpha_search_default=None,
            alpha_search_help="Enable linear search over alpha.",
        )
        p.add_argument("--alpha-selection", type=str, choices=["shared", "per_task"], default=None)
        p.add_argument("--alpha-patience", type=int, default=None)
        p.add_argument("--alpha-search-split", type=str, default=None, choices=["val", "test"])
        p.add_argument("--save-merged", type=str, default=None)
        p.add_argument("--save-transported-tvs-dir", type=str, default=None)
        p.add_argument(
            "--eval-before-rebase",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Optionally evaluate source zero-shot/FT on target task dataset before rebase.",
        )
        p.add_argument(
            "--block-extension-enabled",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable block-extension preprocess before task-vector transport when depth mismatch is found.",
        )
        p.add_argument(
            "--block-extension-params",
            type=str,
            default=None,
            help="JSON object for block-extension preprocess kwargs.",
        )
        add_logging_args(p)

        args = p.parse_args()
        method_params_cli = parse_json_object_arg(args.method_params, arg_name="--method-params")
        merge_params_cli = parse_json_object_arg(args.merge_params, arg_name="--merge-params")
        block_extension_params_cli = parse_json_object_arg(
            args.block_extension_params,
            arg_name="--block-extension-params",
        )

        cfg: dict[str, Any] = {}
        if args.config is not None:
            cfg = load_json(args.config)

        cli: dict[str, Any] = {
            "source_clip_model": args.source_clip_model,
            "source_clip_pretrained": args.source_clip_pretrained,
            "target_clip_model": args.target_clip_model,
            "target_clip_pretrained": args.target_clip_pretrained,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "no_humanize": args.no_humanize,
            "suite": getattr(args, "suite", None),
            "tasks": getattr(args, "tasks", None),
            "device": args.device,
            "dtype": args.dtype,
            "tuned_ckpts": args.tuned_ckpts,
            "weights": args.weights,
            "strict_load": args.strict_load,
            "method": args.method,
            "method_params": method_params_cli,
            "mask_mode": args.mask_mode,
            "vote": args.vote,
            "grad_batch_size": args.grad_batch_size,
            "grad_imgs_per_class": args.grad_imgs_per_class,
            "grad_num_batches": args.grad_num_batches,
            "alpha_search": getattr(args, "alpha_search", None),
            "alpha_selection": getattr(args, "alpha_selection", None),
            "merge_mode": args.merge_mode,
            "merge_method": args.merge_method,
            "merge_params": merge_params_cli,
            "alpha_patience": args.alpha_patience,
            "alpha_search_split": args.alpha_search_split,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "alpha_step": args.alpha_step,
            "alpha": args.alpha,
            "save_merged": args.save_merged,
            "save_transported_tvs_dir": args.save_transported_tvs_dir,
            "eval_before_rebase": args.eval_before_rebase,
            "block_extension_enabled": args.block_extension_enabled,
            "block_extension_params": block_extension_params_cli,
        }
        cfg = merge_non_none(cfg, {k: v for k, v in cli.items() if v is not None})
        logging_cfg = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
        cfg["logging"] = logging_cfg
        _set_deterministic_seed(int(cfg.get("seed", 42)))

        if "block_extension_enabled" not in cfg:
            cfg["block_extension_enabled"] = True

        resolved = resolve_run_config(cfg, suites=SUITES)
        suite_name = resolved.suite_name
        tasks = resolved.tasks
        device = resolved.device

        run_summary_path = default_summary_path(
            entrypoint="eval.vision_rebase",
            logging_cfg=logging_cfg,
            default_parent=(Path(str(cfg["save_merged"])).parent if cfg.get("save_merged") else None),
        )
        run_logger = start_run(
            entrypoint="eval.vision_rebase",
            logging_cfg=logging_cfg,
            summary_path=run_summary_path,
            metadata={
                "config_path": args.config,
                "resolved_config": cfg,
                "suite": suite_name,
                "tasks": tasks,
                "summary_path": str(run_summary_path),
            },
        )

        tuned_by_task = cfg.get("tuned_ckpts", None)
        if tuned_by_task is not None:
            tuned_by_task = {t: resolve_ckpt_path(str(p)) for t, p in tuned_by_task.items()}
        if not tuned_by_task:
            raise ValueError("Provide tuned checkpoints via --tuned-ckpts or config 'tuned_ckpts'.")

        merge_weights = cfg.get("weights", None)
        if merge_weights is None:
            merge_weights = [1.0] * len(tasks)
        merge_weights = [float(w) for w in merge_weights]

        source_cfg = OpenClipBuildConfig(
            model_name=cfg.get("source_clip_model", "ViT-B-32"),
            pretrained=cfg.get("source_clip_pretrained", "openai"),
            device=device,
            dtype=cfg.get("dtype", None),
        )
        target_cfg = OpenClipBuildConfig(
            model_name=cfg.get("target_clip_model", "ViT-B-32"),
            pretrained=cfg.get("target_clip_pretrained", "laion2b_s34b_b79k"),
            device=device,
            dtype=cfg.get("dtype", None),
        )

        print(f"Source model (A): {source_cfg.model_name} / {source_cfg.pretrained}")
        print(f"Target model (B): {target_cfg.model_name} / {target_cfg.pretrained}")

        clf_source = OpenClipClassifier.build(source_cfg)
        clf_target = OpenClipClassifier.build(target_cfg)

        source_depth = int(len(clf_source.model.visual.transformer.resblocks))
        target_depth = int(len(clf_target.model.visual.transformer.resblocks))
        runtime = VisionRuntime(
            cfg=cfg,
            plan=resolved.bind(source_depth, target_depth),
            clf_source=clf_source,
            clf_target=clf_target,
            source_cfg=source_cfg,
            target_cfg=target_cfg,
            tuned_by_task=tuned_by_task,
            merge_weights=merge_weights,
            summary_dir=run_summary_path.parent,
        )
        final_summary = run_rebase(resolved, runtime, run_logger)
        run_logger.log_summary(final_summary)
        run_logger.finish("success")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
