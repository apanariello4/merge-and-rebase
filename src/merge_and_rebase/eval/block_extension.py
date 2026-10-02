"""Compatibility shim: the OpenCLIP BRACE block extender lives in :mod:`merge_and_rebase.rebase.block_extension`.

Every name this module used to define is re-exported (by identity) so existing importers keep working.
"""

from __future__ import annotations

import logging

from ..models.vision_utils import _encode_image  # noqa: F401
from ..rebase.block_extension.adapters import _InProjCapture  # noqa: F401
from ..rebase.block_extension.config import (  # noqa: F401
    _ANNOTATION_PARAMS,
    _MISPLACED_TOP_LEVEL_KEYS,
    BlockExtensionConfig,
    TargetSharedCorrection,
    _as_correction_scope,
    _as_inserted_block_mode,
    _as_optional_calibration_dataset,
    _as_optional_dict_float,
    _as_optional_int,
    _as_optional_str,
    _as_reference_capture,
    _as_target_shared_correction,
    _as_transport_activation_mode,
    _warn_unknown_block_extension_params,
    block_extension_protocol,
    calibration_dataset_spec,
    resolve_block_extension_config,
    select_loader,
)
from ..rebase.block_extension.core import (  # noqa: F401
    BlockExtenderCore,
    _deterministic_calibration_loader,
    _iter_with_progress,
)
from ..rebase.block_extension.schedules import (  # noqa: F401
    balanced_collapse_spans,
    build_extension_layout,
    build_reduction_layout,
    disjoint_collapse_schedule,
    plan_inserted_positions,
    spread_anchor_schedule,
    vision_collapse_schedule,
    vision_locate_collapse_pos,
)
from ..rebase.block_extension.vision import BlockExtender, run_block_extension  # noqa: F401

logger = logging.getLogger(__name__)
