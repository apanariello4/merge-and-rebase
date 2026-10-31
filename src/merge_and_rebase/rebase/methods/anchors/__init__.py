"""Shared-anchor rebasin (``anchors``): one anchor frame for every model, experts as anchor coordinates.

See ``method.py`` for the pipeline, ``core.py`` for the algebra and ``state.py`` for the on-disk state.
"""

from __future__ import annotations

from ...registry import register
from .config import AnchorsConfig, parse_anchors_config, resolve_anchors_decoder_config
from .method import AnchorsPrepared, AnchorsRebase, compile_desired_effects, decode_desired_effects, transport_rows
from .state import AnchorState, RegisteredModel

__all__ = [
    "AnchorState",
    "AnchorsConfig",
    "AnchorsPrepared",
    "AnchorsRebase",
    "RegisteredModel",
    "compile_desired_effects",
    "decode_desired_effects",
    "parse_anchors_config",
    "resolve_anchors_decoder_config",
    "transport_rows",
]

register(AnchorsRebase())
