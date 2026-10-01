"""Ariadne (formerly Direct Residual): a standalone, transport-free rebase method.

Direct Residual writes the desired effect a source fine-tune had on its boundary activations directly
into a *different* target model's residual-writing projections, with no parameter transport. Source and
target depths are paired by the flat closed-form ``DiscreteLayerPairing`` (one target position <-> exactly
one source position), uniformly across extend, shrink and same-arch.

The per-position ridge solve is shared with ARIADNE's independent-mode path via
``fit._fit_all_positions_independent`` (see ``tests/test_direct_residual_extend_anchor.py``). Nothing
cascades: every position is fit against the pristine target base (``E_j == D_j``), so all positions share
one target forward sweep; ``DirectResidualConfig.cascade_order`` is a documented no-op kept for
config parity with ``ResidualCompletionConfig``.
"""

from __future__ import annotations

from .ablations import apply_tv_scaling
from .alignment import (
    apply_depth_pairing_override,
    centered_rectangular_procrustes,
    centered_ridge_alignment,
    compute_alignment_diagnostics,
    compute_desired_effects,
)
from .capture import (
    capture_block_gradients,
    capture_paired_boundary_activations,
    capture_tokens,
    iter_capture_block_gradients,
    iter_capture_tokens,
    paired_calibration,
)
from .config import (
    COMPONENT_FORWARD_ORDER,
    DirectResidualConfig,
    order_components,
    parse_direct_residual_config,
    resolve_direct_residual_preset,
)
from .diagnostics import (
    compute_direct_residual_task_vector_stats,
    compute_fidelity_holdout_diagnostics,
    draw_fidelity_holdout_calibration,
    measure_direct_residual_realization,
    measure_direct_residual_realization_streaming,
)
from .fit import (
    ResidualSufficientStatistics,
    fit_direct_residual,
    fit_sequential_source_endpoints,
)
from .method import AriadnePrepared, AriadneRebase
from .streaming import (
    compute_alignment_diagnostics_streaming,
    fit_direct_residual_streaming,
    measure_streaming_realization_for,
    prepare_direct_residual_streaming,
)

__all__ = [
    "AriadnePrepared",
    "AriadneRebase",
    "COMPONENT_FORWARD_ORDER",
    "DirectResidualConfig",
    "ResidualSufficientStatistics",
    "apply_depth_pairing_override",
    "apply_tv_scaling",
    "capture_block_gradients",
    "capture_paired_boundary_activations",
    "capture_tokens",
    "centered_rectangular_procrustes",
    "centered_ridge_alignment",
    "compute_alignment_diagnostics",
    "compute_alignment_diagnostics_streaming",
    "compute_desired_effects",
    "compute_direct_residual_task_vector_stats",
    "compute_fidelity_holdout_diagnostics",
    "draw_fidelity_holdout_calibration",
    "fit_direct_residual",
    "fit_direct_residual_streaming",
    "fit_sequential_source_endpoints",
    "iter_capture_block_gradients",
    "iter_capture_tokens",
    "measure_direct_residual_realization",
    "measure_direct_residual_realization_streaming",
    "measure_streaming_realization_for",
    "order_components",
    "paired_calibration",
    "parse_direct_residual_config",
    "prepare_direct_residual_streaming",
    "resolve_direct_residual_preset",
]
