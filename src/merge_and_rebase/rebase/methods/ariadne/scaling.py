"""Task-vector strength scaling for Ariadne."""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import median
from typing import Any

import torch

from ....utils.cost_accounting import cost_phase_decorator
from .components import order_components
from .config import DirectResidualConfig
from .diagnostics import measure_direct_residual_realization
from .hashing import _task_vector_sha256
from .layouts import _family_bias_key, _layout_for

Tensor = torch.Tensor


# Below-epsilon ||D_j|| positions have no reliable ratio r_j = ||delta T_j|| /
# ||D_j||; per_block's guard keeps s_j frozen for that position/iteration
# rather than dividing by (near) zero.
_TV_SCALING_D_NORM_EPS = 1e-8


def _tau_frobenius_norm(sd: Mapping[str, Tensor]) -> float:
    total_sq = 0.0
    for value in sd.values():
        total_sq += float((value.detach().double() ** 2).sum().item())
    return total_sq**0.5


def _position_delta(
    shim, position: int, components: tuple[str, ...], target_corrections: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """The subset of ``target_corrections`` (weight + bias, unsliced) that
    belongs to block-boundary position ``position``.

    Each ``(position, component)`` pair maps to a distinct physical
    state-dict key (one set of projection matrices per block), so, unlike
    ``ariadne.diagnostics._family_delta_state`` (which slices packed
    q/k/v rows to isolate one COMPONENT out of several sharing one physical
    parameter), no row-slicing is needed here to isolate one POSITION: a
    key present in ``target_corrections`` belongs to exactly one position.
    """
    out: dict[str, Tensor] = {}
    for component in components:
        key = shim.component_key(position, component, prefixed=True)
        if key in target_corrections:
            out[key] = target_corrections[key]
        bias_key = _family_bias_key(key)
        if bias_key is not None and bias_key in target_corrections:
            out[bias_key] = target_corrections[bias_key]
    return out


@cost_phase_decorator("transformation", exclusive=True)
def apply_tv_scaling(
    target_model,
    target_base_state: Mapping[str, Tensor],
    target_corrections: Mapping[str, Tensor],
    positions: list[int],
    captured: Mapping[str, Any],
    desired: Mapping[int, list[Tensor]],
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
    measure_fn=None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Label-free rescaling of a unit-strength Direct Residual task vector.

    ``measure_fn(delta) -> {j: realization row}`` replaces the resident measurement when
    given (the streaming path passes `measure_direct_residual_realization_streaming`
    bound to its own calibration sweep); ``captured``/``desired`` are then unused.

    Reuses ``measure_direct_residual_realization`` (mount, capture
    block-boundary outputs over the SAME calibration batches
    ``capture_paired_boundary_activations`` collected for the fit, compare to
    the pristine target base and to the block-boundary desired effect
    ``D_j``, restore-and-hash-verify) as the sole measurement primitive:
    ``r_j = joint_delta_norm_over_desired`` is ``||delta T_j||_F /
    (||D_j||_F + eps)`` for whatever delta is currently mounted.

    ``config.tv_scaling == "none"`` is a strict no-op (returns
    ``target_corrections`` unchanged, by value) -- callers must gate on this
    themselves for the golden-hash-pinned default path, but calling this
    function directly with ``tv_scaling="none"`` is also safe.

    Returns ``(scaled_corrections, diagnostics)``. ``diagnostics`` is
    intended to be recorded verbatim (or nested) under the Direct Residual
    run-summary section's additive ``tv_scaling_by_task`` key.
    """
    if config.tv_scaling == "none":
        return dict(target_corrections), {"mode": "none"}
    if config.block_split != "none":
        # parse_direct_residual_config should already have rejected this
        # combination; re-check here so a caller that builds a
        # DirectResidualConfig by hand (bypassing the parser) cannot reach
        # the unverified interaction either.
        raise ValueError(
            f"apply_tv_scaling: tv_scaling={config.tv_scaling!r} requires block_split='none' "
            f"(got block_split={config.block_split!r})"
        )
    shim = _layout_for(family_adapter)
    components = order_components(config.components)

    tau_stats_before = {
        "frobenius_norm": _tau_frobenius_norm(target_corrections),
        "sha256": _task_vector_sha256(target_corrections),
    }

    def measure(delta: Mapping[str, Tensor]) -> dict[int, dict[str, Any]]:
        if measure_fn is not None:
            return measure_fn(delta)
        return measure_direct_residual_realization(
            target_model,
            target_base_state,
            delta,
            positions,
            captured["target_batches"],
            captured["target_base_outputs_by_position"],
            desired,
            device=device,
            components=components,
            family_adapter=family_adapter,
        )

    if config.tv_scaling == "global":
        realization = measure(target_corrections)
        r_j = {j: float(realization[j]["joint_delta_norm_over_desired"]) for j in positions}
        c = float(median(r_j[j] for j in positions))
        if not math.isfinite(c) or c == 0.0:
            raise ValueError(
                f"tv_scaling='global' produced a degenerate scale c={c} "
                "(median of r_j across positions); cannot rescale tau"
            )
        final = {key: value / c for key, value in target_corrections.items()}
        post_realization = measure(final)
        diagnostics = {
            "mode": "global",
            "c": c,
            "r_j": r_j,
            "post_scaling_r_j": {j: float(post_realization[j]["joint_delta_norm_over_desired"]) for j in positions},
            "tau_stats_before": tau_stats_before,
            "tau_stats_after": {
                "frobenius_norm": _tau_frobenius_norm(final),
                "sha256": _task_vector_sha256(final),
            },
        }
        return final, diagnostics

    # config.tv_scaling == "per_block": per-position scalars s_j, found by a
    # simultaneous (Jacobi-style) fixed-point update -- every s_j is updated
    # from the SAME mounted-combination measurement sweep, never sequentially
    # (Gauss-Seidel) against a partially-updated combination.
    pos_delta = {j: _position_delta(shim, j, components, target_corrections) for j in positions}
    s = {j: 1.0 for j in positions}
    r_traces: list[dict[int, float | None]] = []
    s_traces: list[dict[int, float]] = []
    max_dev_trace: list[float] = []
    guard_log: list[dict[str, Any]] = []
    for iteration in range(int(config.tv_scaling_iters)):
        combined: dict[str, Tensor] = {}
        for j in positions:
            for key, value in pos_delta[j].items():
                combined[key] = value * s[j]
        realization = measure(combined)
        r_j: dict[int, float | None] = {}
        for j in positions:
            d_norm = float(realization[j]["desired_norm"])
            r = float(realization[j]["joint_delta_norm_over_desired"])
            if d_norm < _TV_SCALING_D_NORM_EPS or not math.isfinite(r):
                r_j[j] = None
                guard_log.append(
                    {
                        "iteration": iteration,
                        "position": j,
                        "reason": "desired_norm_near_zero" if d_norm < _TV_SCALING_D_NORM_EPS else "non_finite_r",
                        "desired_norm": d_norm,
                        "r_j": r,
                        "s_j_kept": s[j],
                    }
                )
            else:
                r_j[j] = r
        r_traces.append(dict(r_j))
        new_s = dict(s)
        for j in positions:
            if r_j[j] is not None:
                new_s[j] = s[j] / r_j[j]
        max_dev = max(abs((r_j[j] if r_j[j] is not None else 1.0) - 1.0) for j in positions)
        max_dev_trace.append(max_dev)
        s = new_s
        s_traces.append(dict(s))
    final = {}
    for j in positions:
        for key, value in pos_delta[j].items():
            final[key] = value * s[j]
    post_realization = measure(final)
    diagnostics = {
        "mode": "per_block",
        "iters": int(config.tv_scaling_iters),
        "r_traces": r_traces,
        "s_traces": s_traces,
        # max_j|r_j - 1| per iteration; "did it decrease vs the previous
        # iteration" is derivable from this trace directly (max_dev_trace[k]
        # < max_dev_trace[k-1]), reported as its own list rather than
        # collapsed to one bool so an oscillating trace is visible verbatim.
        "max_dev_trace": max_dev_trace,
        "guard_log": guard_log,
        "tau_stats_before": tau_stats_before,
        "tau_stats_after": {
            "frobenius_norm": _tau_frobenius_norm(final),
            "sha256": _task_vector_sha256(final),
        },
        "post_scaling_r_j": {j: float(post_realization[j]["joint_delta_norm_over_desired"]) for j in positions},
    }
    return final, diagnostics
