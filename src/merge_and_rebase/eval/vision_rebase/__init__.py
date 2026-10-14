# ruff: noqa: F401
"""Vision rebase CLI and compatibility imports from their owning modules."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
from ...data.balanced_calibration import Vision8TaskContext, build_balanced_vision8_calibration_loaders
from ...data.templates import get_templates
from ...data.vision_loaders import (
    build_vision_calibration_loader,
    build_vision_loaders,
    extract_classnames,
    load_hf_splits,
)
from ...eval.utils import eval_task_top1, humanize, patch_base_for_attn, resolve_eval_split_loader, to_cpu_fp32
from ...io.ckpt import align_to_base_keys, load_ckpt, load_into_model, resolve_ckpt_path
from ...io.peft_helpers import normalize_attn_patch_cfg
from ...merge.base import PreparedMergeMethod
from ...merge.methods._common import axpy_state_dict
from ...merge.registry import get_method as get_merge_method
from ...merge.registry import list_methods as list_merge_methods
from ...merge.task_vectors import TaskVector
from ...models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ...rebase import list_methods
from ...rebase.block_extension.config import block_extension_protocol, calibration_dataset_spec, select_loader
from ...rebase.block_extension.vision import run_block_extension
from ...rebase.discrete_layer_match import DiscreteLayerPairing, build_discrete_indexed_model
from ...rebase.methods._ariadne.fit import _task_vector_sha256
from ...rebase.methods.ariadne import AriadneRebase, apply_depth_pairing_override
from ...rebase.methods.theseus import InterpolatedBlockActivations
from ...rebase.run_config import _BASE_CONSTRUCTION_MODES, resolve_run_config
from ...run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from ...utils.alpha_search import PerTaskAlphaTracker, average_scores
from ...utils.cost_accounting import PhaseCostRecorder, cost_phase, recording
from ..datasets.vision8_14_20 import SUITES
from ..print_utils import pretty_print_task_accuracies
from ..rebase_metrics import normalized_accuracy_ratio
from .alpha_search import _average_defined, _norm_acc
from .artifacts import _legacy_visual_delta, _legacy_visual_key, _load_saved_sequential_tv, _state_dict_sha256
from .cli import _set_deterministic_seed, main
from .context import (
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
from .merge import (
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
from .method_stages import _build_rebase_prepared, _direct_residual_fit_body, _run_direct_residual_fit
from .source_eval import _ZERO_SHOT_CACHE_DIR, _evaluate_source_model_top1
from .stages import _resolve_source_activation_plan, _visual_only_filter
from .summary import RunRecord, assemble_summary
