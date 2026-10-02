"""Deprecated alias of :mod:`merge_and_rebase.rebase.methods.ariadne`.

Direct Residual was renamed Ariadne and now lives in
``merge_and_rebase.rebase.methods.ariadne``. This module only re-exports its names
(including the private ones historically imported from here) so that old configs,
local scripts and tests keep working. No warning is emitted; import from the new
package in new code.
"""

from __future__ import annotations

from ..rebase.discrete_layer_match import DiscreteLayerPairing  # noqa: F401  (legacy import path)
from ..rebase.methods._ariadne.ablations import (  # noqa: F401
    _TV_SCALING_D_NORM_EPS,
    _fit_block_boundary_backfit,
    _fit_block_boundary_joint,
    _position_delta,
    _tau_frobenius_norm,
    apply_tv_scaling,
)
from ..rebase.methods._ariadne.alignment import (  # noqa: F401
    _derive_block_seed,
    _fit_activation_map,
    _procrustes_from_cross,
    _random_isometry_map,
    _validate_alignment_options,
    apply_depth_pairing_override,
    centered_rectangular_procrustes,
    centered_ridge_alignment,
    compute_alignment_diagnostics,
    compute_desired_effects,
)
from ..rebase.methods._ariadne.capture import (  # noqa: F401
    capture_block_gradients,
    capture_paired_boundary_activations,
    capture_tokens,
    iter_capture_block_gradients,
    iter_capture_tokens,
    paired_calibration,
)
from ..rebase.methods._ariadne.config import (  # noqa: F401
    COMPONENT_FORWARD_ORDER,
    DirectResidualConfig,
    order_components,
    parse_direct_residual_config,
    resolve_direct_residual_preset,
)
from ..rebase.methods._ariadne.diagnostics import (  # noqa: F401
    compute_direct_residual_task_vector_stats,
    compute_fidelity_holdout_diagnostics,
    draw_fidelity_holdout_calibration,
    measure_direct_residual_realization,
    measure_direct_residual_realization_streaming,
)
from ..rebase.methods._ariadne.fit import (  # noqa: F401
    ResidualSufficientStatistics,
    _finalize_independent_component,
    _fit_all_positions_independent,
    _task_vector_sha256,
    fit_direct_residual,
    fit_sequential_source_endpoints,
)
from ..rebase.methods._ariadne.layouts import (  # noqa: F401
    COMPONENT_INPUT_KIND,
    _aligned,
    _component_effective_out,
    _family_bias_key,
    _layout_for,
    _rows,
)
from ..rebase.methods._ariadne.streaming import (  # noqa: F401
    _streaming_desired,
    _streaming_source_iters,
    _StreamingCrossCovariance,
    compute_alignment_diagnostics_streaming,
    fit_direct_residual_streaming,
    measure_streaming_realization_for,
    prepare_direct_residual_streaming,
)
from ..rebase.methods.ariadne import (  # noqa: F401
    AriadnePrepared,
    AriadneRebase,
)

__all__ = ["DirectResidualConfig", "apply_depth_pairing_override", "apply_tv_scaling", "capture_paired_boundary_activations", "compute_alignment_diagnostics", "compute_alignment_diagnostics_streaming", "compute_desired_effects", "compute_direct_residual_task_vector_stats", "compute_fidelity_holdout_diagnostics", "draw_fidelity_holdout_calibration", "fit_direct_residual", "fit_sequential_source_endpoints", "fit_direct_residual_streaming", "measure_direct_residual_realization", "measure_direct_residual_realization_streaming", "parse_direct_residual_config", "prepare_direct_residual_streaming"]
