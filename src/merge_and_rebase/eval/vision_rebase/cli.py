from __future__ import annotations

import argparse
import itertools  # noqa: F401  (kept importable)
import json
import os
import time
from collections.abc import Mapping, Sequence  # noqa: F401  (kept importable)
from copy import deepcopy
from dataclasses import asdict, dataclass  # noqa: F401  (kept importable)
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: F401  (kept importable)

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
from ...data.balanced_calibration import (  # noqa: F401  (kept importable)
    Vision8TaskContext,
    build_balanced_vision8_calibration_loaders,
)
from ...data.templates import get_templates  # noqa: F401  (kept importable)
from ...data.vision_loaders import (  # noqa: F401  (kept importable)
    build_vision_calibration_loader,
    build_vision_loaders,
    extract_classnames,
    load_hf_splits,
)
from ...eval.utils import (
    eval_task_top1,
    humanize,  # noqa: F401  (kept importable)
    patch_base_for_attn,
    resolve_eval_split_loader,  # noqa: F401  (kept importable)
    to_cpu_fp32,
)
from ...io.ckpt import align_to_base_keys, load_ckpt, load_into_model, resolve_ckpt_path
from ...io.peft_helpers import normalize_attn_patch_cfg
from ...merge.base import PreparedMergeMethod  # noqa: F401  (kept importable)
from ...merge.methods._common import axpy_state_dict
from ...merge.registry import get_method as get_merge_method  # noqa: F401  (kept importable)
from ...merge.registry import list_methods as list_merge_methods
from ...merge.task_vectors import TaskVector
from ...models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ...rebase import list_methods
from ...rebase.discrete_layer_match import DiscreteLayerPairing, build_discrete_indexed_model  # noqa: F401
from ...rebase.methods.ariadne import AriadneRebase, apply_depth_pairing_override  # noqa: F401  (kept importable)
from ...rebase.methods.ariadne.fit import _task_vector_sha256  # noqa: F401  (kept importable)
from ...rebase.methods.theseus import InterpolatedBlockActivations  # noqa: F401  (kept importable)
from ...rebase.orchestration import AriadneRunRecord, CompletionRecord, direct_target_p1_requested
from ...rebase.prestep import StageEnv, TaskInputs
from ...rebase.run_config import _BASE_CONSTRUCTION_MODES, resolve_run_config  # noqa: F401  (kept importable)
from ...run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from ...utils.alpha_search import PerTaskAlphaTracker, average_scores
from ...utils.cost_accounting import PhaseCostRecorder, cost_phase, recording  # noqa: F401  (kept importable)
from ..block_extension import (
    block_extension_protocol,  # noqa: F401  (kept importable)
    calibration_dataset_spec,
    run_block_extension,
    select_loader,  # noqa: F401  (kept importable)
)
from ..datasets.vision8_14_20 import SUITES
from ..print_utils import pretty_print_task_accuracies
from ..rebase_metrics import normalized_accuracy_ratio  # noqa: F401  (kept importable)
from ..target_informed_runtime import (  # noqa: F401  (kept importable)
    capture_residual_references,
    capture_resized_joint_source_inputs,
    complete_direct_p1_shared_correction,
    complete_joint_blockwise,
    complete_residuals,
    complete_residuals_direct,
    projection_transforms,
    scale_completion,
)
from ..target_residual_completion import JointCorrectionConfig, ResidualCompletionConfig  # noqa: F401
from .alpha_search import (  # noqa: F401  (re-exported for tests)
    _average_defined,
    _norm_acc,
)
from .artifacts import (  # noqa: F401  (re-exported for tests)
    _legacy_visual_delta,
    _legacy_visual_key,
    _load_saved_sequential_tv,
    _state_dict_sha256,
)
from .completion import (  # noqa: F401  (re-exported for tests)
    _maybe_capture_target_residual_references,
    _maybe_complete_direct_p1_task_vector,
    _maybe_complete_joint_blockwise_task_vector,
    _maybe_complete_target_residual_task_vector,
    build_completion_stages,
)
from .context import (  # noqa: F401  (re-exported for tests)
    DIRECT_RESIDUAL_TINY_IMAGENET_SPEC,
    TRANSPORT_CALIBRATION_DATA,
    _build_balanced_calibration_context,
    _build_direct_paired_calibration_context,
    _build_direct_residual_calibration,
    _build_task_context,
    _CalibrationLoaders,
    _resolve_transport_calibration_data,
    _select_dedicated_brace_loader,
    _TaskContext,
)
from .merge import (  # noqa: F401  (re-exported for tests)
    _SINGLE_TRANSPORT_MODES,
    _TRANSPORT_THEN_MERGE_MODES,
    _VALID_MERGE_MODES,
    _average_visual_state_dicts,
    _check_untransported_compatibility,
    _ckpt_visual_base_coverage,
    _infer_ckpt_base,
    _merge_direction,
    _pseudo_tuned,
    _relative_visual_state_distance,
    _resolve_merge_mode_config,
    _scale_delta,
    _scale_deltas_by,
    _visual_key_fingerprint,
)
from .method_stages import (  # noqa: F401  (re-exported for tests)
    _build_rebase_prepared,
    _direct_residual_fit_body,
    _run_direct_residual_fit,
    build_method_stage,
)
from .source_lmc import (  # noqa: F401  (re-exported for tests)
    _ZERO_SHOT_CACHE_DIR,
    _evaluate_all_task_star_lmc,
    _evaluate_cross_task_source_lmc,
    _evaluate_source_lmc,
    _evaluate_source_model_top1,
)
from .stages import (  # noqa: F401  (re-exported for tests)
    _resolve_source_activation_plan,
    _visual_only_filter,
    build_prestep,
    build_prestep_observers,
    build_task_models,
)
from .summary import (  # noqa: F401  (re-exported for tests)
    RunRecord,
    assemble_summary,
)


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
        method_name = resolved.method_name
        method_params = resolved.method_params
        method = resolved.method
        method_label = resolved.method_label
        direct_residual_like = resolved.direct_residual_like
        block_extension_enabled = resolved.block_extension_enabled
        block_extension_cfg = resolved.block_extension_cfg
        direct_residual_cfg = resolved.ariadne_cfg
        direct_residual_preset = resolved.ariadne_preset
        theseus_like_method = resolved.theseus_like_method
        blockext_like_method = resolved.blockext_like_method
        transfusion_mode = resolved.transfusion_mode
        bico_mode = resolved.bico_mode
        depth_alignment_mode = resolved.depth_alignment_mode
        source_only = resolved.lmc.source_only
        strict_load = resolved.strict_load
        device = resolved.device
        grad_batch_size = resolved.grad_batch_size
        grad_imgs_per_class = resolved.grad_imgs_per_class
        grad_num_batches = resolved.grad_num_batches
        alpha_patience = resolved.alpha.patience
        alpha_search_split = resolved.alpha.search_split
        alphas = resolved.alpha.alphas
        alpha_selection = resolved.alpha.selection
        merge_mode = resolved.merge.mode
        merge_method_name = resolved.merge.method_name
        merge_params = resolved.merge.params
        global_alpha_search = resolved.merge.global_alpha_search
        base_construction = resolved.merge.base_construction
        suite_name = resolved.suite_name
        suite = resolved.suite
        tasks = resolved.tasks

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
        plan = resolved.bind(source_depth, target_depth)
        run_block_extension_prestep = plan.run_block_extension_prestep
        if blockext_like_method:
            calibration_dataset = calibration_dataset_spec(block_extension_cfg)
            if run_block_extension_prestep:
                print(
                    "Block extension preprocess: enabled "
                    f"(source_depth={source_depth} -> target_depth={target_depth}, "
                    f"split={block_extension_cfg.calibration_split}, "
                    f"dataset={calibration_dataset!r}, "
                    f"n_batches_act={block_extension_cfg.n_batches_act})."
                )
            else:
                reason = "disabled by config"
                if not block_extension_enabled:
                    reason = "disabled by config"
                elif source_depth == target_depth:
                    reason = "source/target depth already match"
                print(
                    "Block extension preprocess: skipped "
                    f"({reason}, source_depth={source_depth}, target_depth={target_depth})."
                )

        block_extension_calibration_loader = None
        calibration_dataset = calibration_dataset_spec(block_extension_cfg)
        if (
            run_block_extension_prestep
            and not block_extension_cfg.skip_correction
            and calibration_dataset is not None
        ):
            block_extension_calibration_loader = build_vision_calibration_loader(
                calibration_dataset,
                resolver=suite.resolver,
                preprocess=clf_source.preprocess,
                calibration_split=block_extension_cfg.calibration_split,
                batch_size=int(cfg.get("batch_size", 128)),
                num_workers=int(cfg.get("num_workers", 6)),
                pin_memory=True,
                val_fraction=float(cfg.get("val_fraction", 0.1)),
                seed=int(cfg.get("seed", 42)),
            )
            print(
                "Block extension preprocess: using one task-independent calibration loader "
                f"from {calibration_dataset!r}."
            )

        attn_patch_cfg_raw = cfg.get("attn_patch_cfg", None)
        if attn_patch_cfg_raw is not None and not isinstance(attn_patch_cfg_raw, dict):
            raise ValueError("config['attn_patch_cfg'] must be a dict when provided.")
        patch_attn_before_rebase = bool(cfg.get("patched_attn", attn_patch_cfg_raw is not None))
        attn_patch_cfg = normalize_attn_patch_cfg(attn_patch_cfg_raw) if patch_attn_before_rebase else None

        if patch_attn_before_rebase:
            print(f"Patching source/target attention before rebase: {attn_patch_cfg}")
            source_base_sd = to_cpu_fp32(
                patch_base_for_attn(
                    clf=clf_source,
                    base_ckpt=None,
                    strict_load=strict_load,
                    attn_patch_cfg=attn_patch_cfg,
                )
            )
            target_base_sd = to_cpu_fp32(
                patch_base_for_attn(
                    clf=clf_target,
                    base_ckpt=None,
                    strict_load=strict_load,
                    attn_patch_cfg=attn_patch_cfg,
                )
            )
        else:
            source_base_sd = to_cpu_fp32({k: v for k, v in clf_source.model.state_dict().items()})
            target_base_sd = to_cpu_fp32({k: v for k, v in clf_target.model.state_dict().items()})
        target_hash_before = _state_dict_sha256(target_base_sd)

        use_humanized_classnames = not bool(cfg.get("no_humanize", True))
        print(f"Classname mode: {'humanized' if use_humanized_classnames else 'raw'}")
        print(f"Rebase method: {method_label}")

        # ---- Mixed-merging pre-pass: classify each tuned checkpoint by its base ----
        native_tasks_requested = [str(t) for t in (cfg.get("native_target_tasks", []) or [])]
        unknown_native_tasks = [t for t in native_tasks_requested if t not in tasks]
        if unknown_native_tasks:
            raise ValueError(f"native_target_tasks contains tasks not in the task list: {unknown_native_tasks}")
        auto_detect_ckpt_base = bool(cfg.get("auto_detect_ckpt_base", True))
        native_tasks: set[str] = set(native_tasks_requested)

        if native_tasks or auto_detect_ckpt_base:
            print("Checkpoint base classification (visual-key coverage vs source/target):")
            for task in tasks:
                if task in native_tasks:
                    print(f"  {task}: native target checkpoint (explicit)")
                    continue
                raw_sd = load_ckpt(str(tuned_by_task[task]))
                inferred = _infer_ckpt_base(raw_sd, source_base_sd=source_base_sd, target_base_sd=target_base_sd)
                if inferred is None:
                    fingerprint = _visual_key_fingerprint(raw_sd)
                    raise ValueError(
                        f"Tuned checkpoint for task '{task}' matches neither the source nor the target "
                        f"visual backbone ({tuned_by_task[task]}). Checkpoint fingerprint: {fingerprint}. "
                        f"Source fingerprint: {_visual_key_fingerprint(source_base_sd)}. "
                        f"Target fingerprint: {_visual_key_fingerprint(target_base_sd)}."
                    )
                if inferred == "target":
                    if not auto_detect_ckpt_base:
                        raise ValueError(
                            f"Tuned checkpoint for task '{task}' matches the target architecture; "
                            "add it to native_target_tasks or set auto_detect_ckpt_base=true."
                        )
                    native_tasks.add(task)
                    print(f"  {task}: native target checkpoint (auto-detected)")
                else:
                    if strict_load:
                        coverage = _ckpt_visual_base_coverage(raw_sd, source_base_sd)
                        if coverage != 1.0:
                            raise ValueError(
                                f"Strict visual checkpoint coverage failed for task '{task}': "
                                f"coverage={coverage:.6f}, expected=1.0 ({tuned_by_task[task]})."
                            )
                    print(f"  {task}: source checkpoint (transport required)")
                del raw_sd

        if native_tasks:
            if merge_mode == "none":
                raise ValueError(
                    "Native target checkpoints require a merge mode; merge_mode='none' evaluates "
                    "per-task transported deltas only. Use merge_mode='rebase_then_merge'."
                )
            if merge_mode in _SINGLE_TRANSPORT_MODES:
                raise ValueError(
                    "Native target checkpoints cannot participate in merge_then_rebase: the merge "
                    "happens on the source base, where native target deltas do not exist."
                )
            if transfusion_mode:
                raise NotImplementedError(
                    "Native target checkpoints with transfusion are not supported: the permutation "
                    "prepare step swaps the target keyspace. Use a theseus/bico transport method."
                )

        if base_construction == "independent_endpoint_average":
            if merge_mode not in _TRANSPORT_THEN_MERGE_MODES:
                raise ValueError(
                    "base_construction='independent_endpoint_average' requires "
                    "merge_mode='brace_transport_then_merge'."
                )
            if alpha_selection != "shared":
                raise ValueError(
                    "base_construction='independent_endpoint_average' requires "
                    "alpha_selection='shared'; per-task alpha search is not part of this baseline."
                )
            if native_tasks:
                raise ValueError(
                    "base_construction='independent_endpoint_average' requires every task to be "
                    "an independently transformed source endpoint; native target tasks are not allowed."
                )

        task_context_by_name: dict[str, _TaskContext] = {}
        per_task: list[dict[str, Any]] = []
        transported_deltas: list[dict[str, torch.Tensor]] = []
        original_deltas: list[dict[str, torch.Tensor]] = []
        transport_timings: dict[str, dict[str, float]] = {}
        transported_artifacts: dict[str, list[str]] = {}
        cross_task_lmc_rows: list[dict[str, Any]] = []
        all_task_lmc_rows: list[dict[str, Any]] = []
        # Last realized per-task extension layout. The insertion schedule
        # depends only on the source/target depths, so every task shares it;
        # the merged-pair transport paths reuse it to address inserted blocks.
        independent_base_average: dict[str, torch.Tensor] | None = None
        independent_base_distance_by_task: dict[str, float] = {}
        independent_base_dispersion: float | None = None
        independent_base_diagnostics_path: str | None = None
        independent_source_merge_param_count: int | None = None
        independent_direct_delta_key_count: dict[str, int] = {}
        brace_calibration_metadata: dict[str, Any] | None = None

        for task in tasks:
            task_ctx = _build_task_context(
                task,
                suite=suite,
                cfg=cfg,
                clf_target=clf_target,
                clf_source=clf_source,
                source_cfg=source_cfg,
                target_cfg=target_cfg,
                use_humanized_classnames=use_humanized_classnames,
                need_source_loaders=bool(
                    (theseus_like_method or transfusion_mode or bico_mode or direct_residual_like)
                    and task not in native_tasks
                ),
            )
            task_context_by_name[task] = task_ctx
            per_task.append(
                {
                    "task": task,
                    "loaders": task_ctx.loaders,
                    "classnames": task_ctx.classnames,
                    "build_cfg_task": task_ctx.build_cfg_task,
                    "source_loaders": task_ctx.source_loaders,
                    "source_build_cfg_task": task_ctx.source_build_cfg_task,
                }
            )

        # Task-independent Direct Residual calibration (direct_residual_params.
        # calibration_data != "task_local"): ONE paired context, built here once
        # and passed to every Direct Residual fit below (per task and
        # merge_in_source_then_fit alike). Only the fit's calibration images
        # change; each task's alpha search and evaluation keep its own splits.
        direct_residual_calibration_ctx: _TaskContext | None = None
        direct_residual_calibration_meta: dict[str, Any] = {"calibration_data": "task_local"}
        if direct_residual_like and direct_residual_cfg.calibration_data != "task_local":
            direct_residual_calibration_ctx, direct_residual_calibration_meta = _build_direct_residual_calibration(
                direct_residual_cfg.calibration_data,
                per_task=[item for item in per_task if item["task"] not in native_tasks],
                suite=suite,
                cfg=cfg,
                clf_source=clf_source,
                clf_target=clf_target,
                source_cfg=source_cfg,
                target_cfg=target_cfg,
                num_batches=int(direct_residual_cfg.num_batches),
                calibration_seed=int(direct_residual_cfg.seed),
            )
            print(f"Direct Residual calibration: {direct_residual_calibration_meta}")

        # Task-independent THESEUS/BiCo calibration (transport_calibration_data):
        # one paired context for every task's prepare; alpha search and
        # evaluation keep each task's own splits.
        transport_calibration_data = _resolve_transport_calibration_data(
            cfg, theseus_like_method=theseus_like_method, bico_mode=bico_mode
        )
        transport_calibration_ctx: _TaskContext | None = None
        transport_calibration_meta: dict[str, Any] = {"transport_calibration_data": transport_calibration_data}
        if transport_calibration_data == "tiny_imagenet":
            transport_calibration_ctx = _build_direct_paired_calibration_context(
                DIRECT_RESIDUAL_TINY_IMAGENET_SPEC,
                suite=suite,
                cfg=cfg,
                clf_source=clf_source,
                clf_target=clf_target,
                source_cfg=source_cfg,
                target_cfg=target_cfg,
            )
            transport_calibration_meta.update(
                **DIRECT_RESIDUAL_TINY_IMAGENET_SPEC,
                num_samples=len(transport_calibration_ctx.loaders.train.dataset),
                num_classes=len(transport_calibration_ctx.classnames),
            )
            print(f"Transport calibration: {transport_calibration_meta}")

        brace_protocol = str(
            (cfg.get("block_extension_params", {}) or {}).get("calibration_protocol", "task_local")
        ).lower()
        if (
            run_block_extension_prestep
            and not block_extension_cfg.skip_correction
            and brace_protocol.startswith("vision8_mix")
        ):
            brace_mix_context, brace_mix_metadata = _build_balanced_calibration_context(
                per_task,
                cfg=cfg,
                clf_source=clf_source,
                clf_target=clf_target,
                n_batches=block_extension_cfg.n_batches_act,
                split=block_extension_cfg.calibration_split,
            )
            block_extension_calibration_loader = brace_mix_context.source_loaders.train
            brace_calibration_metadata = brace_mix_metadata
            run_logger.log_event("brace_calibration_plan", context=brace_mix_metadata)
        stage_env = StageEnv(
            resolved=resolved,
            plan=plan,
            cfg=cfg,
            device=device,
            clf_source=clf_source,
            clf_target=clf_target,
            tuned_by_task=tuned_by_task,
            native_tasks=native_tasks,
            patch_attn_before_rebase=patch_attn_before_rebase,
            source_base_sd=source_base_sd,
            target_base_sd=target_base_sd,
            target_hash_before=target_hash_before,
            block_extension_calibration_loader=block_extension_calibration_loader,
            run_logger=run_logger,
        )
        prestep = build_prestep(plan)
        eval_observer, lmc_observer = build_prestep_observers()
        prestep_observers = (eval_observer, lmc_observer)
        block_extension_eval_rows = eval_observer.rows
        source_lmc_rows = lmc_observer.rows
        method_stage = build_method_stage(
            stage_env,
            transport_calibration_ctx=transport_calibration_ctx,
            task_contexts=task_context_by_name,
            tasks=tasks,
            merge_weights=merge_weights,
            ariadne_calibration_ctx=direct_residual_calibration_ctx,
            ariadne_calibration_meta=direct_residual_calibration_meta,
        )
        ariadne_record = method_stage.record if direct_residual_like else AriadneRunRecord()
        completion_record = CompletionRecord()
        completion_stages = build_completion_stages(plan, block_extension_cfg, completion_record)
        residual_completion_diagnostics = completion_record.residual
        joint_blockwise_diagnostics = completion_record.joint_blockwise
        direct_p1_diagnostics = completion_record.direct_p1
        transfusion_prepared: dict[str, Any] | None = None
        # merge_then_brace_then_transport merges deltas on the native source base first and only
        # then runs its own once-only structural step, so neither prestep fires per-task under it
        # (gating resolved in `ResolvedRunConfig.bind`).
        # Timing/memory brackets (wandb-visible), parallel to transport_timings:
        # alignment_calibration_timings covers whatever depth/width-alignment
        # step runs before any correction is fitted (build_discrete_indexed_model
        # for the discrete-index-match control, or capture_paired_boundary_activations
        # for Direct Residual); correction_fit_timings covers fit_direct_residual
        # only. Both dicts default to {} and are always present in final_summary,
        # even for methods/paths that never populate them, so downstream JSON
        # parsing is uniform across every method.
        alignment_calibration_timings: dict[str, dict[str, float]] = {}
        correction_fit_timings: dict[str, dict[str, float]] = {}
        # Per-task activation_collection / transformation / transport cost split
        # (utils.cost_accounting), for THESEUS/BiCo prepare+transport and for
        # Direct Residual's fit alike.
        cost_phase_timings: dict[str, dict[str, Any]] = {}

        # merge_in_source_then_fit (an Ariadne config field, distinct from the top-level `merge_mode`
        # cfg key) merges every task's native delta ONCE before the per-task loop and fits one shared
        # correction; the stage caches it and the loop reuses it for every task.
        if direct_residual_like:
            method_stage.precompute(stage_env)

        for task in tasks:
            task_ctx = task_context_by_name[task]

            if task in native_tasks:
                print(f"  {task}: native target checkpoint — skipping transport")
                continue

            task_in = TaskInputs(task, task_ctx)
            task_models = build_task_models(stage_env, task)
            for observer in prestep_observers:
                observer.before(stage_env, task_in, task_models)
            pre = prestep.run(stage_env, task_in, task_models)
            # LMC "after" is logged before the target-dataset eval "post" (legacy event order).
            for observer in reversed(prestep_observers):
                observer.after(stage_env, task_in, task_models, pre)
            if pre.completion_note is not None:
                print(pre.completion_note)
            if "alignment_calibration" in pre.timings:
                alignment_calibration_timings[task] = pre.timings["alignment_calibration"]

            if source_only:
                continue

            pre = prestep.load_delta(stage_env, task_in, pre)
            # TransFusion's once-only prepare rebinds these run-level objects (see NoPrestep.load_delta).
            source_base_sd = stage_env.source_base_sd
            target_base_sd = stage_env.target_base_sd
            target_hash_before = stage_env.target_hash_before
            transfusion_prepared = stage_env.transfusion_prepared
            task_delta = pre.task_delta

            direct_target_p1 = direct_target_p1_requested(plan, block_extension_cfg)
            if direct_target_p1:
                print(
                    f"\n--- Direct-target P1 for '{task}' "
                    "(parameter transport skipped) ---"
                )
            else:
                print(f"\n--- Transporting '{task}' with method '{method.name}' ---")
            if merge_mode not in _SINGLE_TRANSPORT_MODES:
                method_result = method_stage.run(stage_env, task_in, pre)
                transported_delta = method_result.transported_delta
                transport_timings[task] = method_result.transport_timing
                cost_phase_timings[task] = method_result.cost_phases
                if method_result.alignment_calibration is not None:
                    alignment_calibration_timings[task] = method_result.alignment_calibration
                if method_result.correction_fit is not None:
                    correction_fit_timings[task] = method_result.correction_fit

                for completion_stage in completion_stages:
                    method_result = completion_stage.run(stage_env, task_in, pre, method_result)
                transported_delta = method_result.transported_delta

                transported_deltas.append(transported_delta)
                original_deltas.append(task_delta)
                print(f"  {task}: transported delta computed for {len(transported_delta)} params")
                run_logger.log_event(
                    "transport_task_end",
                    metrics={f"rebase/{task}/transported_param_count": float(len(transported_delta))},
                    context={"task": task, "method": method.name},
                )

                save_transport_dir = cfg.get("save_transported_tvs_dir", None)
                save_transported_artifacts = bool(cfg.get("save_transported_artifacts", bool(save_transport_dir)))
                if save_transported_artifacts and not save_transport_dir:
                    raise ValueError("save_transported_artifacts=true requires save_transported_tvs_dir.")
                if save_transported_artifacts and save_transport_dir:
                    os.makedirs(save_transport_dir, exist_ok=True)
                    native_path = os.path.join(save_transport_dir, f"{task}_{method.name}_transported_native.pt")
                    if direct_residual_like and direct_residual_cfg.endpoint_construction in {"sequential_source_endpoints", "sequential_delta_on_synthesized_base"}:
                        if os.path.exists(native_path) or os.path.exists(os.path.splitext(native_path)[0] + ".json"):
                            raise FileExistsError(f"refusing to overwrite sequential DR vector: {native_path}")
                    torch.save(to_cpu_fp32(transported_delta), native_path)
                    if direct_residual_like and direct_residual_cfg.endpoint_construction in {"sequential_source_endpoints", "sequential_delta_on_synthesized_base"}:
                        meta_path = os.path.splitext(native_path)[0] + ".json"
                        metadata = {
                            "task": task,
                            "endpoint_construction": direct_residual_cfg.endpoint_construction,
                            "target_base_sha256": _state_dict_sha256(target_base_sd),
                            "vector_sha256": _state_dict_sha256(transported_delta),
                            "calibration_seed": direct_residual_cfg.seed,
                            "num_batches": direct_residual_cfg.num_batches,
                            "direct_residual_config": asdict(direct_residual_cfg),
                        }
                        Path(meta_path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
                    print(f"  {task}: saved transported TV -> {native_path}")
                    transported_artifacts[task] = [native_path]
                    if bool(cfg.get("save_transported_tvs_legacy", False)):
                        legacy_path = os.path.join(save_transport_dir, f"{task}_{method.name}_transported_legacy_visual.pt")
                        legacy_no_conv1_path = os.path.join(
                            save_transport_dir, f"{task}_{method.name}_transported_legacy_visual_no_conv1.pt"
                        )
                        torch.save(_legacy_visual_delta(transported_delta), legacy_path)
                        torch.save(_legacy_visual_delta(transported_delta, drop_conv1=True), legacy_no_conv1_path)
                        print(f"  {task}: saved legacy visual TV -> {legacy_path}")
                        print(f"  {task}: saved legacy visual TV without conv1 -> {legacy_no_conv1_path}")
                        transported_artifacts[task] += [legacy_path, legacy_no_conv1_path]
            else:
                original_deltas.append(task_delta)
                print(f"  {task}: delta collected for merge_then_rebase ({len(task_delta)} params)")

        can_eval_untransported_by_task: list[bool] = []
        single_tv_deltas_for_diagnostic: list[dict[str, torch.Tensor]] | None = None
        single_transport_calibration_metadata: dict[str, Any] | None = None
        if merge_mode == "none":
            rebased_deltas = [_scale_delta(d, w) for d, w in zip(transported_deltas, merge_weights, strict=True)]
            untransported_deltas = [_scale_delta(d, w) for d, w in zip(original_deltas, merge_weights, strict=True)]
            print(f"Prepared {len(tasks)} transported deltas (task-independent alpha mode)")

            for task_name, delta_sd in zip(tasks, untransported_deltas, strict=True):
                enabled, issues = _check_untransported_compatibility(target_base_sd, delta_sd)
                can_eval_untransported_by_task.append(enabled)
                if enabled:
                    print(f"Untransported baseline for '{task_name}': enabled.")
                else:
                    print(f"Untransported baseline for '{task_name}': skipped (incompatible with target model).")
                    for msg in issues[:3]:
                        print(f"  - {msg}")
                    if len(issues) > 3:
                        print(f"  - ... and {len(issues) - 3} more incompatibilities")
        elif merge_mode in _TRANSPORT_THEN_MERGE_MODES:
            native_delta_by_task: dict[str, dict[str, torch.Tensor]] = {}
            for task in sorted(native_tasks):
                path = str(tuned_by_task[task])
                sd = load_ckpt(path)
                aligned = align_to_base_keys(sd, target_base_sd)
                if not aligned:
                    raise ValueError(
                        f"No tensors from native target checkpoint aligned to target base keys "
                        f"for task '{task}': {path}."
                    )
                native_delta_by_task[task] = TaskVector.from_checkpoints(
                    target_base_sd,
                    to_cpu_fp32(aligned),
                    strict=False,
                    key_filter=_visual_only_filter,
                ).delta
                print(
                    f"  {task}: native target delta computed ({len(native_delta_by_task[task])} params)"
                )
                run_logger.log_event(
                    "native_delta_end",
                    metrics={f"rebase/{task}/native_param_count": float(len(native_delta_by_task[task]))},
                    context={"task": task},
                )

            transported_iter = iter(transported_deltas)
            merge_input_deltas = []
            for t in tasks:
                if t in native_delta_by_task:
                    merge_input_deltas.append(native_delta_by_task[t])
                else:
                    merge_input_deltas.append(next(transported_iter))
            single_tv_deltas_for_diagnostic = list(merge_input_deltas)

            if alpha_selection == "per_task":
                # Hierarchical: per-task alphas are searched on the individual
                # deltas (transported + native) first; composition happens after pass 1.
                rebased_deltas = list(merge_input_deltas)
                untransported_deltas = original_deltas
                print(
                    f"Hierarchical mode '{merge_mode}' ({merge_method_name}): per-task alpha "
                    f"search on {len(merge_input_deltas)} deltas before merge"
                )
            else:
                merged_direction = _merge_direction(
                    base_sd=target_base_sd,
                    deltas=merge_input_deltas,
                    merge_method_name=merge_method_name,
                    weights=merge_weights,
                    merge_params=merge_params,
                )
                print(
                    f"Merge mode '{merge_mode}' ({merge_method_name}): composed {len(merge_input_deltas)} "
                    f"deltas (transported + native) -> merged direction with {len(merged_direction)} params"
                )
                run_logger.log_event(
                    "merge_composition_end",
                    metrics={"merge/param_count": float(len(merged_direction))},
                    context={
                        "mode": merge_mode,
                        "merge_method": merge_method_name,
                        "merge_params": merge_params,
                        "n_tasks": len(merge_input_deltas),
                        "n_native": len(native_delta_by_task),
                    },
                )
                # One shared merged model: every task evaluates the same direction at alpha.
                rebased_deltas = [merged_direction] * len(tasks)
                untransported_deltas = original_deltas
        else:
            first_item = per_task[0]
            transport_protocol = str(cfg.get("transport_calibration_protocol", "task_local")).lower()
            calibration_metadata: dict[str, Any] = {"protocol": transport_protocol}
            single_transport_calibration_metadata = calibration_metadata
            if transport_protocol.startswith("tiny"):
                direct_spec = {
                    "path": "zh-plus/tiny-imagenet",
                    "split": "valid",
                    "max_samples": int(cfg.get("transport_calibration_max_samples", 2048)),
                }
                transport_ctx = _build_direct_paired_calibration_context(
                    direct_spec,
                    suite=suite,
                    cfg=cfg,
                    clf_source=clf_source,
                    clf_target=clf_target,
                    source_cfg=source_cfg,
                    target_cfg=target_cfg,
                )
            elif transport_protocol in {"vision8_mix", "vision8_mix_10", "task_local", "task_local_10"}:
                transport_ctx, balanced_meta = _build_balanced_calibration_context(
                    per_task,
                    cfg=cfg,
                    clf_source=clf_source,
                    clf_target=clf_target,
                    n_batches=int(cfg.get("transport_calibration_batches", 10)),
                    split=str(cfg.get("transport_calibration_split", "val")),
                )
                calibration_metadata.update(balanced_meta)
            else:
                raise ValueError(f"Unsupported transport_calibration_protocol: {transport_protocol!r}")

            if merge_mode == "merge_then_rebase":
                # Historical same-depth behavior is intentionally unchanged.
                transport_ctx = _TaskContext(
                    loaders=first_item["loaders"],
                    source_loaders=first_item["source_loaders"],
                    classnames=list(first_item["classnames"]),
                    build_cfg_task=first_item["build_cfg_task"],
                    source_build_cfg_task=first_item["source_build_cfg_task"],
                )
                merged_source_base = source_base_sd
                merged_source_direction = _merge_direction(
                    base_sd=source_base_sd,
                    deltas=original_deltas,
                    merge_method_name=merge_method_name,
                    weights=merge_weights,
                    merge_params=merge_params,
                )
                source_template_once = None
                prepared_has_brace = False
                merged_source_activation_plan = None
            elif merge_mode == "brace_merge_then_transport":
                if stage_env.endpoints.corrected_source_template is None or not stage_env.endpoints.base_by_task:
                    raise RuntimeError("BRACE-then-merge requires corrected source endpoints for every task.")
                average_visual, average_keys = _average_visual_state_dicts(stage_env.endpoints.base_by_task)
                first_base = stage_env.endpoints.base_by_task[sorted(stage_env.endpoints.base_by_task)[0]]
                merged_source_base = dict(first_base)
                merged_source_base.update(average_visual)
                merged_source_direction = _merge_direction(
                    base_sd=merged_source_base,
                    deltas=original_deltas,
                    merge_method_name=merge_method_name,
                    weights=merge_weights,
                    merge_params=merge_params,
                )
                distances = {
                    task: _relative_visual_state_distance(state, average_visual, average_keys)
                    for task, state in stage_env.endpoints.base_by_task.items()
                }
                calibration_metadata.update(
                    {
                        "consensus_source_base": "mean_corrected_source_base",
                        "consensus_visual_key_count": len(average_keys),
                        "source_base_relative_distance_by_task": distances,
                        "source_base_max_relative_distance": max(distances.values()),
                    }
                )
                source_template_once = stage_env.endpoints.corrected_source_template
                prepared_has_brace = True
                merged_source_activation_plan = _resolve_source_activation_plan(
                    block_extension_cfg, stage_env.recorded_extension_layout
                )
            elif merge_mode == "merge_then_brace_then_transport":
                native_merged_direction = _merge_direction(
                    base_sd=source_base_sd,
                    deltas=original_deltas,
                    merge_method_name=merge_method_name,
                    weights=merge_weights,
                    merge_params=merge_params,
                )
                source_base_model_once = deepcopy(clf_source.model)
                source_ft_model_once = deepcopy(clf_source.model)
                load_into_model(source_base_model_once, source_base_sd, strict=True)
                load_into_model(
                    source_ft_model_once,
                    axpy_state_dict(source_base_sd, native_merged_direction, alpha=1.0),
                    strict=True,
                )
                # BRACE and transport have distinct calibration contracts and
                # must not share a bounded loader.  In particular, campaign
                # rows may request 40 BRACE batches but only 10 transport
                # batches.  The dedicated BRACE loader was constructed above
                # from block_extension_cfg; transport_ctx remains exclusively
                # owned by the subsequent transport preparation.
                brace_loader = _select_dedicated_brace_loader(
                    brace_loader=block_extension_calibration_loader,
                    transport_loader=transport_ctx.source_loaders.train,
                    correction_enabled=not block_extension_cfg.skip_correction,
                )
                merged_extension_layout = {}
                final_depth = run_block_extension(
                    source_base_model=source_base_model_once,
                    source_ft_model=source_ft_model_once,
                    calibration_loader=brace_loader,
                    target_layers_total=target_depth,
                    config=block_extension_cfg,
                    device=device,
                    layout_out=merged_extension_layout,
                )
                if final_depth != target_depth:
                    raise RuntimeError(
                        f"Merged-pair BRACE depth mismatch: final_depth={final_depth}, target_depth={target_depth}."
                    )
                merged_source_base = to_cpu_fp32(dict(source_base_model_once.state_dict()))
                merged_source_ft = to_cpu_fp32(dict(source_ft_model_once.state_dict()))
                merged_source_direction = TaskVector.from_checkpoints(
                    merged_source_base,
                    merged_source_ft,
                    strict=True,
                    key_filter=_visual_only_filter,
                ).delta
                source_template_once = deepcopy(source_base_model_once).cpu()
                prepared_has_brace = True
                merged_source_activation_plan = _resolve_source_activation_plan(
                    block_extension_cfg, merged_extension_layout
                )
            else:  # pragma: no cover - validated by _resolve_merge_mode_config
                raise AssertionError(f"Unhandled merge mode: {merge_mode}")

            prepared_once = _build_rebase_prepared(
                method_name=method_name,
                method=method,
                method_params=method_params,
                cfg=cfg,
                device=device,
                grad_batch_size=grad_batch_size,
                grad_imgs_per_class=grad_imgs_per_class,
                grad_num_batches=grad_num_batches,
                theseus_like_method=theseus_like_method,
                bico_mode=bico_mode,
                run_block_extension_prestep=prepared_has_brace,
                clf_source=clf_source,
                clf_target=clf_target,
                classnames=list(transport_ctx.classnames),
                loaders=transport_ctx.loaders,
                source_loaders=transport_ctx.source_loaders,
                build_cfg_task=transport_ctx.build_cfg_task,
                source_build_cfg_task=transport_ctx.source_build_cfg_task,
                task_source_base_sd=merged_source_base,
                target_base_sd=target_base_sd,
                task_delta=merged_source_direction,
                source_base_model_task=source_template_once,
                transfusion_prepared=transfusion_prepared,
                source_text_features=transport_ctx.source_text_features,
                target_text_features=transport_ctx.target_text_features,
                source_activation_plan=merged_source_activation_plan,
            )
            transport_started = time.perf_counter()
            transported_merged_delta = method.transport(
                source_base=merged_source_base,
                target_base=target_base_sd,
                delta=merged_source_direction,
                strict=strict_load,
                prepared=prepared_once,
                **method_params,
            )
            transport_seconds = time.perf_counter() - transport_started
            print(
                f"Merge mode '{merge_mode}' ({merge_method_name}): merged on source -> single transport in "
                f"{transport_seconds:.2f}s ({len(transported_merged_delta)} params)"
            )
            run_logger.log_event(
                "merge_composition_end",
                metrics={
                    "merge/param_count": float(len(transported_merged_delta)),
                    "merge/single_transport_seconds": float(transport_seconds),
                },
                context={
                    "mode": merge_mode,
                    "merge_method": merge_method_name,
                    "merge_params": merge_params,
                    "n_tasks": len(original_deltas),
                    "calibration": calibration_metadata,
                },
            )
            rebased_deltas = [transported_merged_delta] * len(tasks)
            untransported_deltas = original_deltas

        if merge_mode != "none":
            # The per-task "untransported" baseline is meaningless for a single
            # merged model; fall back to the (alpha-independent, cached) target
            # zero-shot baseline so normalized ratios remain defined.
            can_eval_untransported_by_task = [False] * len(tasks)

        def _eval_task(item: dict[str, Any], split: str) -> float:
            return float(
                eval_task_top1(
                    clf=clf_target,
                    loaders=item["loaders"],
                    classnames=list(item["classnames"]),
                    build_cfg_task=item["build_cfg_task"],
                    device=device,
                    split=split,
                )
            )

        def _eval_all_tasks(split: str) -> list[float]:
            return [_eval_task(item, split) for item in per_task]

        if all(can_eval_untransported_by_task):
            baseline_label = "untransported"
        elif any(can_eval_untransported_by_task):
            baseline_label = "mixed_baseline"
        else:
            baseline_label = "target_zeroshot"
        result_label = "rebased"
        task_col = max(max((len(str(item["task"])) for item in per_task), default=4), len("task"), len("avg"))
        metric_col = max(12, len(baseline_label) + 2, len(result_label) + 2, len("norm") + 2)

        baseline_cache_zeroshot: dict[str, list[float]] = {}

        if baseline_label == "untransported":
            print("Using untransported baseline evaluation for all tasks.")
        elif baseline_label == "mixed_baseline":
            print("Using mixed baseline evaluation: untransported where compatible, target zeroshot otherwise.")
        else:
            print("Using target zeroshot baseline for all tasks.")

        def _load_into_target_model(sd: dict[str, torch.Tensor]) -> None:
            if transfusion_mode:
                method.load_into_target_visual(clf_target, sd, strict=False)
            else:
                load_into_model(clf_target.model, sd, strict=strict_load)

        def _eval_zeroshot_all_tasks(split: str) -> list[float]:
            if split not in baseline_cache_zeroshot:
                _load_into_target_model(target_base_sd)
                baseline_cache_zeroshot[split] = _eval_all_tasks(split)
            return list(baseline_cache_zeroshot[split])

        def _eval_baseline_task(split: str, idx: int, alpha: float) -> float:
            if can_eval_untransported_by_task[idx]:
                baseline_sd = axpy_state_dict(target_base_sd, untransported_deltas[idx], alpha=float(alpha))
                _load_into_target_model(baseline_sd)
                del baseline_sd
                return _eval_task(per_task[idx], split)
            return _eval_zeroshot_all_tasks(split)[idx]

        def _eval_baseline_task_indices(split: str, indices: list[int], alpha: float) -> dict[int, float]:
            return {idx: _eval_baseline_task(split, idx, alpha) for idx in indices}

        def _eval_rebased_task_indices(split: str, indices: list[int], alpha: float) -> dict[int, float]:
            out: dict[int, float] = {}
            for idx in indices:
                rebase_sd_task = axpy_state_dict(target_base_sd, rebased_deltas[idx], alpha=float(alpha))
                _load_into_target_model(rebase_sd_task)
                del rebase_sd_task
                out[idx] = _eval_task(per_task[idx], split)
            return out

        def _eval_single_tv_task_indices(split: str, indices: list[int], alpha: float) -> dict[int, float]:
            if single_tv_deltas_for_diagnostic is None:
                raise RuntimeError("Single-TV diagnostic requires merge_mode='rebase_then_merge'.")
            out: dict[int, float] = {}
            for idx in indices:
                single_sd = axpy_state_dict(
                    target_base_sd,
                    single_tv_deltas_for_diagnostic[idx],
                    alpha=float(alpha),
                )
                _load_into_target_model(single_sd)
                del single_sd
                out[idx] = _eval_task(per_task[idx], split)
            return out

        hierarchical = bool(merge_mode in _TRANSPORT_THEN_MERGE_MODES and alpha_selection == "per_task")
        single_tv_diagnostic_enabled = single_tv_deltas_for_diagnostic is not None
        single_tv_val_best_acc: list[float] | None = (
            [float("-inf")] * len(per_task) if single_tv_diagnostic_enabled else None
        )
        single_tv_val_best_alpha: list[float] | None = (
            [float(alphas[0])] * len(per_task) if single_tv_diagnostic_enabled else None
        )
        single_tv_alpha_protocol = (
            "per_task_premerge_alpha" if hierarchical else "single_tv_validation_oracle"
        )
        per_task_premerge_alphas: list[float] | None = None
        hierarchical_premerge_alpha_curve: list[dict[str, Any]] | None = None
        global_alpha_curve: list[dict[str, Any]] | None = None
        selected_validation_results: dict[str, Any] | None = None

        if alpha_selection == "shared" or hierarchical:
            if hierarchical:
                # ---------------- PASS 1: per-task alpha on individual deltas ----------------
                hierarchical_premerge_alpha_curve = []
                tracker = PerTaskAlphaTracker(
                    task_names=[str(item["task"]) for item in per_task],
                    initial_alpha=float(alphas[0]),
                    patience=alpha_patience,
                )
                # Merge-mode baselines are the alpha-independent target zero-shot:
                # pre-seed the secondary stream inactive for every task.
                for idx in range(len(per_task)):
                    baseline_val = _eval_baseline_task(alpha_search_split, idx, 0.0)
                    tracker.best_secondary_alpha[idx] = 0.0
                    tracker.best_secondary_acc[idx] = baseline_val
                    tracker.secondary_active[idx] = False

                for alpha in alphas:
                    eval_indices = tracker.eval_active_indices()
                    if not eval_indices:
                        print("\nAll tasks have early-stopped; ending hierarchical pass-1 alpha sweep.")
                        break

                    primary_indices = set(tracker.primary_active_indices())
                    print(
                        f"\n=== alpha {alpha:.3f} — {method_label} "
                        f"(split: {alpha_search_split}, mode: per_task pass 1/2, hierarchical) ==="
                    )

                    baseline_by_idx = _eval_baseline_task_indices(alpha_search_split, eval_indices, float(alpha))
                    rebase_eval_indices = [idx for idx in eval_indices if idx in primary_indices]
                    rebase_by_idx_active = _eval_rebased_task_indices(alpha_search_split, rebase_eval_indices, float(alpha))
                    rebase_by_idx: dict[int, float] = {}
                    for idx in eval_indices:
                        rebase_by_idx[idx] = rebase_by_idx_active[idx] if idx in primary_indices else float("-inf")
                    baseline_accs = [baseline_by_idx[idx] for idx in eval_indices]
                    rebase_accs = [rebase_by_idx[idx] for idx in eval_indices]

                    for idx, baseline_acc, rebase_acc in zip(eval_indices, baseline_accs, rebase_accs, strict=True):
                        task_name = per_task[idx]["task"]
                        if idx in primary_indices:
                            display_rebase = rebase_acc
                            marker = " "
                        else:
                            frozen = float(tracker.best_primary_acc[idx])
                            display_rebase = frozen if frozen != float("-inf") else 0.0
                            marker = "*"
                        norm = _norm_acc(display_rebase, baseline_acc)
                        print(
                            f" {marker}{task_name}: {baseline_label}={baseline_acc:.6f}  "
                            f"per-task={display_rebase:.6f}  norm={norm:.6f}"
                        )

                    hierarchical_premerge_alpha_curve.append(
                        {
                            "alpha": float(alpha),
                            "per_task_rebased": {
                                per_task[idx]["task"]: float(rebase_by_idx[idx])
                                for idx in rebase_eval_indices
                            },
                            "per_task_baseline": {
                                per_task[idx]["task"]: float(baseline_by_idx[idx])
                                for idx in eval_indices
                            },
                        }
                    )
                    run_logger.log_event(
                        "hierarchical_premerge_alpha_eval_end",
                        metrics={"alpha/value": float(alpha)},
                        context=hierarchical_premerge_alpha_curve[-1],
                    )

                    stopped_primary, _ = tracker.update(
                        alpha=float(alpha),
                        indices=eval_indices,
                        primary_accs=rebase_accs,
                        secondary_accs=baseline_accs,
                    )
                    if stopped_primary:
                        stopped_names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_primary)
                        print(f"  Early-stopping per-task alphas at alpha={alpha:.3f}: {stopped_names}")

                per_task_premerge_alphas = [float(tracker.best_primary_alpha[idx]) for idx in range(len(per_task))]
                single_tv_val_best_acc = [float(tracker.best_primary_acc[idx]) for idx in range(len(per_task))]
                single_tv_val_best_alpha = list(per_task_premerge_alphas)
                print("\n=== Hierarchical pass-1 summary (per-task alphas) ===")
                for item, a in zip(per_task, per_task_premerge_alphas, strict=True):
                    print(f"  {item['task']}: premerge_alpha={a:.3f}")
                run_logger.log_event(
                    "hierarchical_pass1_end",
                    metrics={
                        "hierarchical/avg_premerge_alpha": float(
                            sum(per_task_premerge_alphas) / max(1, len(per_task_premerge_alphas))
                        )
                    },
                    context={
                        "per_task_premerge_alphas": {
                            item["task"]: float(per_task_premerge_alphas[i]) for i, item in enumerate(per_task)
                        }
                    },
                )

                # ---------------- PASS 2: scale by per-task alphas, compose once ----------------
                scaled_input = _scale_deltas_by(merge_input_deltas, per_task_premerge_alphas)
                merged_direction = _merge_direction(
                    base_sd=target_base_sd,
                    deltas=scaled_input,
                    merge_method_name=merge_method_name,
                    weights=merge_weights,
                    merge_params=merge_params,
                )
                rebased_deltas = [merged_direction] * len(per_task)
                print(
                    f"Hierarchical merge ({merge_method_name}): composed {len(scaled_input)} scaled deltas "
                    f"-> merged direction with {len(merged_direction)} params"
                )
                run_logger.log_event(
                    "merge_composition_end",
                    metrics={"merge/param_count": float(len(merged_direction))},
                    context={
                        "mode": merge_mode,
                        "merge_method": merge_method_name,
                        "merge_params": merge_params,
                        "n_tasks": len(scaled_input),
                        "per_task_premerge_alphas": {
                            item["task"]: float(per_task_premerge_alphas[i]) for i, item in enumerate(per_task)
                        },
                    },
                )

            sweep_alphas = list(alphas)
            if hierarchical and not global_alpha_search:
                sweep_alphas = [1.0]
                print("\nglobal_alpha_search=false: evaluating the merged model at gamma=1.0 only.")
            sweep_positive_alphas = [float(a) for a in sweep_alphas if float(a) > 0.0]
            sweep_mode_label = "global (gamma)" if hierarchical else "shared"

            best_rebase_avg = float("-inf")
            best_baseline_avg = float("-inf")
            best_alpha = float(sweep_alphas[0])
            # When every task's baseline is target_zeroshot (untransported infeasible),
            # the baseline is alpha-independent — keep best_baseline_alpha at 0.0 so
            # the summary does not report a spurious non-zero value.
            has_untransported = any(can_eval_untransported_by_task)
            best_baseline_alpha = (
                float(sweep_positive_alphas[0] if sweep_positive_alphas else sweep_alphas[0])
                if has_untransported
                else 0.0
            )
            sweep_results: list[dict[str, Any]] = []
            shared_bad_steps = 0

            for alpha in sweep_alphas:
                print(f"\n=== alpha {alpha:.3f} — {method_label} (split: {alpha_search_split}, mode: {sweep_mode_label}) ===")

                idxs = list(range(len(per_task)))
                baseline_by_idx = _eval_baseline_task_indices(alpha_search_split, idxs, float(alpha))
                rebase_by_idx = _eval_rebased_task_indices(alpha_search_split, idxs, float(alpha))
                if (
                    single_tv_diagnostic_enabled
                    and single_tv_val_best_acc is not None
                    and single_tv_val_best_alpha is not None
                ):
                    single_tv_by_idx = _eval_single_tv_task_indices(alpha_search_split, idxs, float(alpha))
                    for idx in idxs:
                        if single_tv_by_idx[idx] > single_tv_val_best_acc[idx]:
                            single_tv_val_best_acc[idx] = float(single_tv_by_idx[idx])
                            single_tv_val_best_alpha[idx] = float(alpha)
                baseline_accs = [baseline_by_idx[i] for i in idxs]
                rebase_accs = [rebase_by_idx[i] for i in idxs]

                print(
                    f"  {'task':<{task_col}}  {baseline_label:>{metric_col}}  {result_label:>{metric_col}}  {'norm':>{metric_col}}"
                )
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                for i, item in enumerate(per_task):
                    task_name = item["task"]
                    baseline_acc = baseline_accs[i]
                    rebase_acc = rebase_accs[i]
                    norm = _norm_acc(rebase_acc, baseline_acc)
                    print(
                        f"  {task_name:<{task_col}}  {baseline_acc:>{metric_col}.6f}  {rebase_acc:>{metric_col}.6f}  {norm:>{metric_col}.6f}"
                    )

                avg_rebase = average_scores(rebase_accs)
                avg_baseline = _average_defined(baseline_accs)
                avg_norm = _average_defined([_norm_acc(r, b) for r, b in zip(rebase_accs, baseline_accs, strict=True)])
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                print(
                    f"  {'avg':<{task_col}}  {avg_baseline:>{metric_col}.6f}  {avg_rebase:>{metric_col}.6f}  {avg_norm:>{metric_col}.6f}"
                )

                sweep_results.append(
                    {
                        "alpha": float(alpha),
                        "baseline_accs": baseline_accs,
                        "rebase_accs": rebase_accs,
                    }
                )
                run_logger.log_event(
                    "alpha_eval_end",
                    metrics={
                        "alpha/value": float(alpha),
                        "alpha/avg_acc": float(avg_rebase),
                        "alpha/avg_norm_acc": float(avg_norm),
                    },
                    context={
                        "baseline_label": baseline_label,
                        "per_task_baseline": {item["task"]: float(baseline_accs[i]) for i, item in enumerate(per_task)},
                        "per_task_rebased": {item["task"]: float(rebase_accs[i]) for i, item in enumerate(per_task)},
                    },
                )

                eps = 1e-12
                # Track baseline best alpha independently of rebased best alpha,
                # but only when there is at least one untransported baseline task
                # (otherwise the baseline is target_zeroshot and alpha-independent).
                if has_untransported:
                    if avg_baseline != avg_baseline:  # NaN guard
                        avg_baseline_for_track = float("-inf")
                    else:
                        avg_baseline_for_track = float(avg_baseline)
                    if avg_baseline_for_track > best_baseline_avg + eps:
                        best_baseline_avg = avg_baseline_for_track
                        best_baseline_alpha = float(alpha)

                if avg_rebase > best_rebase_avg + eps:
                    best_rebase_avg = avg_rebase
                    best_alpha = float(alpha)
                    shared_bad_steps = 0
                elif avg_rebase + eps >= best_rebase_avg:
                    shared_bad_steps = 0
                elif len(sweep_alphas) > 1:
                    shared_bad_steps += 1
                    print(
                        f"  (alpha={alpha:.3f} fell below best shared avg {best_rebase_avg:.6f}; "
                        f"bad_steps={shared_bad_steps}/{alpha_patience + 1})"
                    )
                    if shared_bad_steps > alpha_patience:
                        break

            print("\n=== Alpha search summary (shared) ===")
            for r in sweep_results:
                a = r["alpha"]
                avg_r = average_scores(r["rebase_accs"])
                avg_b = _average_defined(r["baseline_accs"])
                print(f"  alpha={a:.3f}  {baseline_label}={avg_b:.6f}  {result_label}={avg_r:.6f}")
            global_alpha_curve = [
                {
                    "alpha": float(row["alpha"]),
                    "avg_rebased": float(average_scores(row["rebase_accs"])),
                    "avg_baseline": float(_average_defined(row["baseline_accs"])),
                    "per_task_rebased": {
                        item["task"]: float(row["rebase_accs"][idx]) for idx, item in enumerate(per_task)
                    },
                    "per_task_baseline": {
                        item["task"]: float(row["baseline_accs"][idx]) for idx, item in enumerate(per_task)
                    },
                }
                for row in sweep_results
            ]
            print(
                f"\nBest alpha: rebase={best_alpha:.3f} (avg rebased val acc={best_rebase_avg:.6f}) | "
                f"baseline={best_baseline_alpha:.3f} (avg baseline val acc={best_baseline_avg:.6f})"
            )

            print(
                f"\n(Re-running on test split: rebase at alpha={best_alpha:.3f}, baseline at alpha={best_baseline_alpha:.3f})"
            )
            all_indices = list(range(len(per_task)))
            baseline_test_by_idx = _eval_baseline_task_indices("test", all_indices, float(best_baseline_alpha))
            rebase_test_by_idx = _eval_rebased_task_indices("test", all_indices, float(best_alpha))
            baseline_test_accs = [baseline_test_by_idx[i] for i in all_indices]
            rebase_test_accs = [rebase_test_by_idx[i] for i in all_indices]
            selected_alpha_by_task = [float(best_alpha)] * len(per_task)
            selected_baseline_alpha_by_task = [float(best_baseline_alpha)] * len(per_task)

        else:
            tracker = PerTaskAlphaTracker(
                task_names=[str(item["task"]) for item in per_task],
                initial_alpha=float(alphas[0]),
                patience=alpha_patience,
            )
            # For tasks where the untransported baseline is infeasible (different
            # architecture / shape), the baseline is target_zeroshot and
            # alpha-independent. Pre-seed the secondary tracker so
            # best_secondary_alpha stays at 0.0 and the secondary stream never
            # participates in alpha optimization for these tasks.
            for idx in range(len(per_task)):
                if not can_eval_untransported_by_task[idx]:
                    baseline_val = _eval_baseline_task(alpha_search_split, idx, 0.0)
                    tracker.best_secondary_alpha[idx] = 0.0
                    tracker.best_secondary_acc[idx] = baseline_val
                    tracker.secondary_active[idx] = False
            sweep_results = []

            for alpha in alphas:
                # The baseline may keep being swept after the rebased has early-stopped
                # a task, so we evaluate the union of primary- and secondary-active tasks
                # and decouple the two streams' early stopping.
                eval_indices = tracker.eval_active_indices()
                if not eval_indices:
                    print("\nAll tasks have early-stopped on both streams; ending per-task alpha sweep.")
                    break

                primary_indices = set(tracker.primary_active_indices())
                print(f"\n=== alpha {alpha:.3f} — {method_label} (split: {alpha_search_split}, mode: per_task) ===")

                # Baseline must be evaluated for every index in the union (baseline may
                # still be active for tasks where the rebased already early-stopped).
                baseline_by_idx = _eval_baseline_task_indices(alpha_search_split, eval_indices, float(alpha))
                # Rebased is only evaluated for primary-active tasks; for tasks where it
                # has already stopped, we pass -inf as a no-op placeholder.
                rebase_eval_indices = [idx for idx in eval_indices if idx in primary_indices]
                rebase_by_idx_active = _eval_rebased_task_indices(alpha_search_split, rebase_eval_indices, float(alpha))
                rebase_by_idx: dict[int, float] = {}
                for idx in eval_indices:
                    if idx in primary_indices:
                        rebase_by_idx[idx] = rebase_by_idx_active[idx]
                    else:
                        rebase_by_idx[idx] = float("-inf")

                baseline_accs = [baseline_by_idx[idx] for idx in eval_indices]
                rebase_accs = [rebase_by_idx[idx] for idx in eval_indices]

                print(
                    f"  {'task':<{task_col}}  {baseline_label:>{metric_col}}  {result_label:>{metric_col}}  {'norm':>{metric_col}}"
                )
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                for idx, baseline_acc, rebase_acc in zip(eval_indices, baseline_accs, rebase_accs, strict=True):
                    task_name = per_task[idx]["task"]
                    # During the val sweep, norm uses the baseline's own best-so-far
                    # accuracy (not the same-alpha baseline), so the per-step ratio
                    # already reflects the final normalization semantics.
                    best_secondary = float(tracker.best_secondary_acc[idx])
                    norm_baseline = best_secondary if best_secondary != float("-inf") else baseline_acc
                    # For tasks whose rebased stream already early-stopped, rebase_acc
                    # is -inf (not evaluated this step); display the frozen best rebased
                    # accuracy instead so the reported number tracks the rebased peak.
                    if idx in primary_indices:
                        display_rebase = rebase_acc
                    else:
                        frozen = float(tracker.best_primary_acc[idx])
                        display_rebase = frozen if frozen != float("-inf") else 0.0
                    norm = _norm_acc(display_rebase, norm_baseline)
                    marker = " " if idx in primary_indices else "*"
                    print(
                        f" {marker}{task_name:<{task_col - 1}}  {baseline_acc:>{metric_col}.6f}  {display_rebase:>{metric_col}.6f}  {norm:>{metric_col}.6f}"
                    )

                # Average rebased across ALL tasks: use the current value for active tasks
                # and the frozen best for stopped tasks, so the avg does not
                # collapse to 0.0 once every rebased task has early-stopped.
                all_rebase_vals: list[float] = []
                for idx in range(len(per_task)):
                    if idx in primary_indices:
                        all_rebase_vals.append(float(rebase_by_idx[idx]))
                    else:
                        frozen = float(tracker.best_primary_acc[idx])
                        all_rebase_vals.append(frozen if frozen != float("-inf") else 0.0)
                avg_rebase = average_scores(all_rebase_vals)
                avg_baseline = _average_defined(baseline_accs)
                avg_norm = _average_defined(
                    [_norm_acc(
                        float(rebase_by_idx[idx]) if idx in primary_indices else max(float(tracker.best_primary_acc[idx]), 0.0),
                        baseline_accs[i],
                    ) for i, idx in enumerate(eval_indices)]
                )
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                print(
                    f"  {'avg':<{task_col}}  {avg_baseline:>{metric_col}.6f}  {avg_rebase:>{metric_col}.6f}  {avg_norm:>{metric_col}.6f}"
                )

                stopped_primary: list[int] = []
                stopped_secondary: list[int] = []
                stopped_primary, stopped_secondary = tracker.update(
                    alpha=float(alpha),
                    indices=eval_indices,
                    primary_accs=rebase_accs,
                    secondary_accs=baseline_accs,
                )
                if stopped_primary:
                    stopped_names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_primary)
                    print(f"  Early-stopping REBASED tasks at alpha={alpha:.3f}: {stopped_names}")
                if stopped_secondary:
                    stopped_names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_secondary)
                    print(f"  Early-stopping BASELINE tasks at alpha={alpha:.3f}: {stopped_names}")

                run_logger.log_event(
                    "alpha_eval_end",
                    metrics={
                        "alpha/value": float(alpha),
                        "alpha/avg_acc": float(avg_rebase),
                        "alpha/avg_norm_acc": float(avg_norm),
                    },
                    context={
                        "active_tasks": [per_task[idx]["task"] for idx in eval_indices],
                        "per_task_baseline": {per_task[idx]["task"]: float(baseline_by_idx[idx]) for idx in eval_indices},
                        "per_task_rebased": {per_task[idx]["task"]: float(rebase_by_idx[idx]) for idx in eval_indices},
                        "stopped_primary": [int(idx) for idx in stopped_primary],
                        "stopped_secondary": [int(idx) for idx in stopped_secondary],
                    },
                )

                sweep_results.append(
                    {
                        "alpha": float(alpha),
                        "active_indices": list(eval_indices),
                        "baseline_accs": baseline_accs,
                        "rebase_accs": rebase_accs,
                        "primary_active": [int(idx) for idx in eval_indices if idx in primary_indices],
                    }
                )

            print("\n=== Alpha search summary (per-task) ===")
            for idx, item in enumerate(per_task):
                print(
                    f"  {item['task']}: rebase_alpha={tracker.best_primary_alpha[idx]:.3f}  "
                    f"rebase_val={tracker.best_primary_acc[idx]:.6f} | "
                    f"baseline_alpha={tracker.best_secondary_alpha[idx]:.3f}  "
                    f"baseline_val={tracker.best_secondary_acc[idx]:.6f}"
                )
            print(f"\nAvg per-task best rebase val acc: {tracker.best_avg():.6f}")
            if alpha_search_split == "val":
                # Preserve the scores used for selection separately from test
                # metrics so campaign selection never needs a test fallback.
                selected_validation_results = {
                    "split": "val",
                    "per_task_rebased": {
                        item["task"]: float(tracker.best_primary_acc[i])
                        for i, item in enumerate(per_task)
                    },
                    "avg_rebased": float(tracker.best_avg()),
                }
            best_baseline_vals = [float(v) for v in tracker.best_secondary_acc if v != float("-inf")]
            if best_baseline_vals:
                print(f"Avg per-task best baseline val acc: {sum(best_baseline_vals) / len(best_baseline_vals):.6f}")

            print("\n(Re-running per-task best alphas on test split — decoupled per stream)")
            baseline_test_accs: list[float] = []
            rebase_test_accs: list[float] = []
            selected_alpha_by_task: list[float] = []
            selected_baseline_alpha_by_task: list[float] = []
            for idx, item in enumerate(per_task):
                rebase_alpha = float(tracker.best_primary_alpha[idx])
                baseline_alpha = float(tracker.best_secondary_alpha[idx])
                selected_alpha_by_task.append(rebase_alpha)
                selected_baseline_alpha_by_task.append(baseline_alpha)
                print(f"  {item['task']}: rebase_alpha={rebase_alpha:.3f}  baseline_alpha={baseline_alpha:.3f}")

                baseline_test_accs.append(_eval_baseline_task("test", idx, baseline_alpha))

                rebase_sd_task = axpy_state_dict(target_base_sd, rebased_deltas[idx], alpha=rebase_alpha)
                _load_into_target_model(rebase_sd_task)
                del rebase_sd_task
                rebase_test_accs.append(_eval_task(item, "test"))
            best_alpha = float(sum(selected_alpha_by_task) / max(1, len(selected_alpha_by_task)))
            best_baseline_alpha = float(sum(selected_baseline_alpha_by_task) / max(1, len(selected_baseline_alpha_by_task)))

        norm_accs = [_norm_acc(r, b) for r, b in zip(rebase_test_accs, baseline_test_accs, strict=True)]

        single_tv_test_accs: list[float] | None = None
        single_tv_test_alpha_by_task: list[float] | None = None
        if single_tv_diagnostic_enabled and single_tv_val_best_alpha is not None:
            single_tv_test_alpha_by_task = [float(a) for a in single_tv_val_best_alpha]
            single_tv_test_accs = []
            print("\nSingle transported task-vector test diagnostic:")
            for idx, item in enumerate(per_task):
                alpha = single_tv_test_alpha_by_task[idx]
                single_acc = _eval_single_tv_task_indices("test", [idx], alpha)[idx]
                single_tv_test_accs.append(float(single_acc))
                print(f"  {item['task']}: alpha={alpha:.3f}  single_tv_test={single_acc:.6f}")
            single_avg = sum(single_tv_test_accs) / len(single_tv_test_accs)
            merged_avg = sum(rebase_test_accs) / len(rebase_test_accs)
            print(
                f"  avg single_tv_test={single_avg:.6f}  merged_test={merged_avg:.6f} "
                f"merge_gap={single_avg - merged_avg:+.6f}"
            )
            run_logger.log_event(
                "single_tv_test_diagnostic_end",
                metrics={
                    "single_tv/avg_test_accuracy": float(single_avg),
                    "single_tv/merged_test_accuracy": float(merged_avg),
                    "single_tv/merge_gap": float(single_avg - merged_avg),
                },
                context={
                    "alpha_protocol": single_tv_alpha_protocol,
                    "per_task_alpha": {
                        item["task"]: float(single_tv_test_alpha_by_task[i])
                        for i, item in enumerate(per_task)
                    },
                    "per_task_test_accuracy": {
                        item["task"]: float(single_tv_test_accs[i])
                        for i, item in enumerate(per_task)
                    },
                },
            )

        alpha_display_label = "hierarchical(per_task+global)" if hierarchical else alpha_selection
        pretty_print_task_accuracies(
            suite_name,
            f"{method_label}, alpha={alpha_display_label}",
            f"A={source_cfg.pretrained} → B={target_cfg.pretrained}",
            per_task,
            rebase_test_accs,
            norm_accs,
            single_accs=baseline_test_accs,
            baseline_label=baseline_label,
            result_label=result_label,
        )

        if hierarchical and per_task_premerge_alphas is not None:
            print("\nHierarchical alphas (per-task premerge + global merge alpha):")
            for item, a in zip(per_task, per_task_premerge_alphas, strict=True):
                print(f"  {item['task']}: premerge={a:.3f}  global={best_alpha:.3f}")
        elif alpha_selection == "per_task":
            print("\nSelected test-time alpha by task:")
            for item, r_a, b_a in zip(per_task, selected_alpha_by_task, selected_baseline_alpha_by_task, strict=True):
                print(f"  {item['task']}: rebase={r_a:.3f}  baseline={b_a:.3f}")

        saved_merged_path: str | None = None
        if cfg.get("save_merged"):
            if merge_mode == "none":
                print(
                    "save_merged was requested, but task-independent alpha mode does not produce a single merged checkpoint; skipping save."
                )
            else:
                best_merged_state = axpy_state_dict(
                    target_base_sd,
                    rebased_deltas[0],
                    alpha=float(best_alpha),
                )
                out_path = str(cfg["save_merged"])
                out_parent = Path(out_path).parent
                if str(out_parent):
                    out_parent.mkdir(parents=True, exist_ok=True)
                torch.save(to_cpu_fp32(best_merged_state), out_path)
                saved_merged_path = out_path
                print(f"Saved merged model (alpha={best_alpha:.3f}) -> {out_path}")

        target_hash_after = _state_dict_sha256(target_base_sd)
        if target_hash_after != target_hash_before:
            raise RuntimeError(
                "Native target base was mutated during merge/transport preparation: "
                f"before={target_hash_before}, after={target_hash_after}."
            )

        final_summary = assemble_summary(
            RunRecord(
                suite_name=suite_name,
                tasks=tasks,
                method_label=method_label,
                merge_mode=merge_mode,
                target_hash_before=target_hash_before,
                target_hash_after=target_hash_after,
                single_transport_calibration_metadata=single_transport_calibration_metadata,
                brace_calibration_metadata=brace_calibration_metadata,
                base_construction=base_construction,
                hierarchical_premerge_alpha_curve=hierarchical_premerge_alpha_curve,
                global_alpha_curve=global_alpha_curve,
                alpha_selection=alpha_selection,
                selected_validation_results=selected_validation_results,
                baseline_label=baseline_label,
                block_extension_eval_rows=block_extension_eval_rows,
                source_lmc_rows=source_lmc_rows,
                cross_task_lmc_rows=cross_task_lmc_rows,
                all_task_lmc_rows=all_task_lmc_rows,
                transported_artifacts=transported_artifacts,
                transport_timings=transport_timings,
                transport_calibration_meta=transport_calibration_meta,
                cost_phase_timings=cost_phase_timings,
                alignment_calibration_timings=alignment_calibration_timings,
                correction_fit_timings=correction_fit_timings,
                depth_alignment_mode=depth_alignment_mode,
                saved_merged_path=saved_merged_path,
                method=method,
                merge_method_name=merge_method_name,
                merge_params=merge_params,
                block_extension_cfg=block_extension_cfg,
                native_tasks=native_tasks,
                hierarchical=hierarchical,
                global_alpha_search=global_alpha_search,
                best_alpha=best_alpha,
                best_baseline_alpha=best_baseline_alpha,
                direct_residual_like=direct_residual_like,
                independent_base_dispersion=independent_base_dispersion,
                independent_base_distance_by_task=independent_base_distance_by_task,
                independent_source_merge_param_count=independent_source_merge_param_count,
                independent_base_diagnostics_path=independent_base_diagnostics_path,
                independent_direct_delta_key_count=independent_direct_delta_key_count,
                single_tv_test_accs=single_tv_test_accs,
                single_tv_alpha_protocol=single_tv_alpha_protocol,
                direct_residual_calibration_meta=ariadne_record.calibration_meta,
                direct_residual_diagnostics=ariadne_record.diagnostics,
                direct_residual_realization=ariadne_record.realization,
                direct_residual_task_vector_stats=ariadne_record.task_vector_stats,
                direct_residual_alignment_diagnostics=ariadne_record.alignment_diagnostics,
                direct_residual_calibration_by_task=ariadne_record.calibration_by_task,
                direct_residual_tv_scaling=ariadne_record.tv_scaling,
                direct_residual_pairing_record=ariadne_record.pairing,
                direct_residual_fidelity_holdout=ariadne_record.fidelity_holdout,
                direct_residual_sequential_endpoints=ariadne_record.sequential_endpoints,
                loaded_direct_residual_tvs=ariadne_record.loaded_vectors,
                residual_completion_diagnostics=residual_completion_diagnostics,
                joint_blockwise_diagnostics=joint_blockwise_diagnostics,
                direct_p1_diagnostics=direct_p1_diagnostics,
                per_task_premerge_alphas=per_task_premerge_alphas,
                selected_alpha_by_task=selected_alpha_by_task,
                per_task=per_task,
                selected_baseline_alpha_by_task=selected_baseline_alpha_by_task,
                direct_residual_cfg=direct_residual_cfg,
                method_name=method_name,
                independent_base_average=independent_base_average,
                baseline_test_accs=baseline_test_accs,
                rebase_test_accs=rebase_test_accs,
                norm_accs=norm_accs,
                direct_residual_preset=direct_residual_preset,
                single_tv_val_best_alpha=single_tv_val_best_alpha,
                single_tv_val_best_acc=single_tv_val_best_acc,
            )
        )
        run_logger.log_summary(final_summary)
        run_logger.finish("success")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
