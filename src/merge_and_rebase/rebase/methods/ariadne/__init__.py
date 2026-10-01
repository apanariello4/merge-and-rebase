"""Ariadne (formerly Direct Residual): a standalone, transport-free rebase method.

Direct Residual answers the same local-functional-effect question ARIADNE's
Proposal 1 ``direct_target`` mode does -- can the desired effect a source
fine-tune had on its own boundary activations be written directly into a
*different* target model's residual-writing projections, with no parameter
transport at all -- but reaches it through a depth/width-alignment mechanism
that has nothing to do with ARIADNE's block-extension apparatus.

Where ARIADNE only ever aligns a target that is exactly ``2x`` a source's
depth (via its depth-doubling protocol gate, and the
bottom-top spread/duplicate insertion that produces an ``ancestry`` map), this
module pairs an arbitrary source depth with an arbitrary target depth through
the flat, closed-form ``DiscreteLayerPairing`` (``i(j) = round(j*(D_A-1)/
(D_B-1))``, `merge_and_rebase.rebase.discrete_layer_match`). That one
cardinality change -- one target position maps to exactly one source
position, never many -- is what lets this module drop ARIADNE's two-code-path
ancestry bookkeeping (one-to-many groups for extend, many-to-one span
tracking for shrink) entirely, and is why it works uniformly across extend,
shrink, and same-arch without any structural gate.

The actual per-position ridge solve is NOT reimplemented here. It is the
exact same code `target_informed_runtime.complete_residuals_direct` has
always executed for `cascade_order="independent"` -- shared, via the private
`merge_and_rebase.rebase.methods.ariadne.independent._fit_all_positions_independent` helper, so this
module and ARIADNE's own `target_scope="all"` independent-mode path are
structurally guaranteed to agree whenever they are handed the same alignment
and the same captured banks (see `tests/test_direct_residual_extend_anchor.py`).

Direct Residual never mounts a correction before fitting the next one: it
never cascades, by construction, at either the cross-position or
intra-position (attn.out_proj -> mlp.c_proj) level. Every position is fit
against the pristine, untouched target base, so its `E_j == D_j` identically
-- there is no upstream state a later fit could see. `DirectResidualConfig
.cascade_order` is kept only for config-parity with
`ResidualCompletionConfig` (so a campaign generator can reuse one code path
to build both configs' JSON) and is a documented no-op here; see
`tests/test_direct_residual_cascade_order.py`. Because every position is
independent by construction, all of them are fit from one shared target
forward sweep rather than one sweep per position -- see
`_fit_all_positions_independent`'s docstring for why that is exact, not an
approximation.

"""

from __future__ import annotations

from .alignment import (
    apply_depth_pairing_override,
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
from .components import (
    COMPONENT_FORWARD_ORDER,
    order_components,
)
from .config import (
    DirectResidualConfig,
    parse_direct_residual_config,
)
from .diagnostics import (
    compute_direct_residual_task_vector_stats,
    compute_fidelity_holdout_diagnostics,
    draw_fidelity_holdout_calibration,
    measure_direct_residual_realization,
    measure_direct_residual_realization_streaming,
)
from .fit import (
    fit_direct_residual,
    fit_sequential_source_endpoints,
)
from .linalg import (
    ResidualSufficientStatistics,
    centered_rectangular_procrustes,
    centered_ridge_alignment,
)
from .scaling import (
    apply_tv_scaling,
)
from .streaming import (
    compute_alignment_diagnostics_streaming,
    fit_direct_residual_streaming,
    measure_streaming_realization_for,
    prepare_direct_residual_streaming,
)

__all__ = [
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
]
