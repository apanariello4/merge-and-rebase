"""Compatibility shim: the HF-decoder BRACE block extender lives in :mod:`merge_and_rebase.rebase.block_extension`.

Every name this module used to define is re-exported (by identity) so existing importers keep working.
"""

from __future__ import annotations

import logging

from ..rebase.block_extension.adapters import _get_final_norm, _get_layers, _run_decoder_forward  # noqa: F401
from ..rebase.block_extension.config import BlockExtensionConfig  # noqa: F401
from ..rebase.block_extension.core import (  # noqa: F401
    BlockExtenderCore,
    _deterministic_calibration_loader,
    _iter_with_progress,
)
from ..rebase.block_extension.decoder import (  # noqa: F401
    _DECODER_COMPONENTS,
    DecoderBlockExtender,
    run_block_extension_llm,
)
from ..rebase.block_extension.schedules import (  # noqa: F401
    build_extension_layout,
    build_reduction_layout,
    decoder_collapse_schedule,
    decoder_locate_collapse_pos,
)

logger = logging.getLogger(__name__)
