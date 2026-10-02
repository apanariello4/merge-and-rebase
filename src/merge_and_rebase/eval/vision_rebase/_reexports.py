# ruff: noqa: F401
"""Names that were importable from ``merge_and_rebase.eval.vision_rebase`` when it was one module.

Nothing here is used by the package itself; ``__init__`` re-exports these so existing imports keep working.
"""

from __future__ import annotations

import argparse
import itertools  # noqa: F401  (kept importable)
import json  # noqa: F401  (kept importable)
import os  # noqa: F401  (kept importable)
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
from ...rebase.methods._ariadne.fit import _task_vector_sha256  # noqa: F401  (kept importable)
from ...rebase.methods.ariadne import AriadneRebase, apply_depth_pairing_override  # noqa: F401  (kept importable)
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
    TransportedTvSaver,
    TransportedTvSaveSpec,
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
    build_run_calibration,
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
from .pipeline import VisionRuntime, run_rebase  # noqa: F401  (re-exported)
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
