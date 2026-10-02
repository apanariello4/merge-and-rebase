"""Reference capture and completion stages for the ARIADNE target-informed corrections.

``_maybe_capture_*`` / ``_maybe_complete_*`` are re-exported from ``eval.vision_rebase`` (tests import them).
"""

from __future__ import annotations

from typing import Any

import torch

from ..target_informed_runtime import capture_residual_references
from ..target_residual_completion import ResidualCompletionConfig


def _maybe_capture_target_residual_references(
    *,
    config: ResidualCompletionConfig,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_loader: Any,
    target_loader: Any,
    seed: int,
    device: str,
) -> dict[str, Any] | None:
    """Capture ARIADNE proposal-1 native reference banks, or no-op when disabled.

    Must be called before ``run_block_extension`` structurally resizes
    ``source_base_model``/``source_ft_model``: the native references are the
    un-resized source model's own boundary activations, paired against the
    pretrained target model at the doubled positions those source blocks will
    be inserted at. Returns ``None`` when the option is disabled, so callers
    that thread the result through unconditionally get a byte-identical no-op.
    """
    if not config.enabled:
        return None
    return capture_residual_references(
        source_base_model,
        source_ft_model,
        target_model,
        source_loader,
        target_loader,
        num_batches=config.num_batches,
        seed=seed,
        device=device,
        target_scope=config.target_scope,
    )
