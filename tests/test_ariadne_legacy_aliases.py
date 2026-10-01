"""Old import paths (``eval.direct_residual`` / ``eval.target_informed_runtime`` /
``eval.target_residual_completion``) must expose the very same objects as the
Ariadne package, not copies."""

from __future__ import annotations

from merge_and_rebase.eval import direct_residual as legacy_dr
from merge_and_rebase.eval import target_informed_runtime as legacy_runtime
from merge_and_rebase.eval import target_residual_completion as legacy_completion
from merge_and_rebase.rebase.methods import ariadne
from merge_and_rebase.rebase.methods.ariadne import (
    blockwise,
    capture,
    components,
    hashing,
    independent,
    layouts,
    linalg,
)


def test_direct_residual_shim_exposes_same_objects():
    for name in ariadne.__all__:
        assert getattr(legacy_dr, name) is getattr(ariadne, name), name
    assert legacy_dr._fit_all_positions_independent is independent._fit_all_positions_independent
    assert legacy_dr._fit_block_boundary_backfit is blockwise._fit_block_boundary_backfit
    assert legacy_dr._fit_block_boundary_joint is blockwise._fit_block_boundary_joint
    assert legacy_dr._layout_for is layouts._layout_for
    assert legacy_dr._task_vector_sha256 is hashing._task_vector_sha256


def test_runtime_and_completion_reexport_moved_names():
    assert legacy_runtime.capture_tokens is capture.capture_tokens
    assert legacy_runtime.paired_calibration is capture.paired_calibration
    assert legacy_runtime._fit_all_positions_independent is independent._fit_all_positions_independent
    assert legacy_runtime._task_vector_sha256 is hashing._task_vector_sha256
    assert legacy_completion.centered_rectangular_procrustes is linalg.centered_rectangular_procrustes
    assert legacy_completion.ResidualSufficientStatistics is linalg.ResidualSufficientStatistics
    assert legacy_completion.order_components is components.order_components
    assert legacy_completion.COMPONENT_FORWARD_ORDER is components.COMPONENT_FORWARD_ORDER
