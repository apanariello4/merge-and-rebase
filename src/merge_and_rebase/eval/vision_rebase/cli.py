from __future__ import annotations

import argparse
import itertools  # noqa: F401  (kept importable)
import json
import os
import time  # noqa: F401  (kept importable)
from collections.abc import Mapping, Sequence  # noqa: F401  (kept importable)
from copy import deepcopy  # noqa: F401  (kept importable)
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
    eval_task_top1,  # noqa: F401  (kept importable)
    humanize,  # noqa: F401  (kept importable)
    patch_base_for_attn,
    resolve_eval_split_loader,  # noqa: F401  (kept importable)
    to_cpu_fp32,
)
from ...io.ckpt import (  # noqa: F401  (kept importable)
    align_to_base_keys,
    load_ckpt,
    load_into_model,
    resolve_ckpt_path,
)
from ...io.peft_helpers import normalize_attn_patch_cfg
from ...merge.base import PreparedMergeMethod  # noqa: F401  (kept importable)
from ...merge.methods._common import axpy_state_dict
from ...merge.registry import get_method as get_merge_method  # noqa: F401  (kept importable)
from ...merge.registry import list_methods as list_merge_methods
from ...merge.task_vectors import TaskVector  # noqa: F401  (kept importable)
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
from ...utils.alpha_search import PerTaskAlphaTracker, average_scores  # noqa: F401  (kept importable)
from ...utils.cost_accounting import PhaseCostRecorder, cost_phase, recording  # noqa: F401  (kept importable)
from ..block_extension import (
    block_extension_protocol,  # noqa: F401  (kept importable)
    calibration_dataset_spec,
    run_block_extension,  # noqa: F401  (kept importable)
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
    AlphaSearchSpec,
    TargetEvaluator,
    _average_defined,
    _norm_acc,
    run_alpha_search,
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
    compose_rebased_deltas,
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
        method_params = resolved.method_params  # noqa: F841
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
        grad_batch_size = resolved.grad_batch_size  # noqa: F841
        grad_imgs_per_class = resolved.grad_imgs_per_class  # noqa: F841
        grad_num_batches = resolved.grad_num_batches  # noqa: F841
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
            transfusion_prepared = stage_env.transfusion_prepared  # noqa: F841
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

        merge_plan = compose_rebased_deltas(
            stage_env,
            per_task=per_task,
            transported_deltas=transported_deltas,
            original_deltas=original_deltas,
            merge_weights=merge_weights,
            source_cfg=source_cfg,
            target_cfg=target_cfg,
        )
        single_transport_calibration_metadata = merge_plan.calibration_metadata

        evaluator = TargetEvaluator.from_plan(stage_env, merge_plan, per_task=per_task)
        alpha_result = run_alpha_search(
            AlphaSearchSpec.from_run(
                resolved,
                per_task=per_task,
                merge_weights=merge_weights,
                method_label=method_label,
                run_logger=run_logger,
            ),
            evaluator,
            merge_plan,
        )
        baseline_label = alpha_result.baseline_label
        result_label = alpha_result.result_label
        hierarchical = alpha_result.hierarchical
        best_alpha = alpha_result.best_alpha
        best_baseline_alpha = alpha_result.best_baseline_alpha
        selected_alpha_by_task = alpha_result.selected_alpha_by_task
        selected_baseline_alpha_by_task = alpha_result.selected_baseline_alpha_by_task
        rebase_test_accs = alpha_result.rebase_test_accs
        baseline_test_accs = alpha_result.baseline_test_accs
        norm_accs = alpha_result.norm_accs
        per_task_premerge_alphas = alpha_result.per_task_premerge_alphas
        hierarchical_premerge_alpha_curve = alpha_result.hierarchical_premerge_alpha_curve
        global_alpha_curve = alpha_result.global_alpha_curve
        selected_validation_results = alpha_result.selected_validation_results
        single_tv_test_accs = alpha_result.single_tv_test_accs
        single_tv_alpha_protocol = alpha_result.single_tv_alpha_protocol
        single_tv_val_best_alpha = alpha_result.single_tv_val_best_alpha
        single_tv_val_best_acc = alpha_result.single_tv_val_best_acc
        rebased_deltas = alpha_result.rebased_deltas

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
