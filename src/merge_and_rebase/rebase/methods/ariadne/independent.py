"""Shared independent-position ridge kernel.

Used by Ariadne's ``fit_direct_residual`` and by target-informed
``complete_residuals_direct`` so both are structurally guaranteed to agree.
"""

from __future__ import annotations

from typing import Any

import torch

from .capture import capture_tokens
from .layouts import COMPONENT_INPUT_KIND, _component_effective_out, _layout_for
from .linalg import ResidualSufficientStatistics


def _realization_diagnostic_fields(
    h_batches, correction, bias_for_pred, effective_out, weight_before, desired_norm, residual_norm_after, *, stats=None
):
    """Analysis-only per-component fit fields, gated on ``realization_
    diagnostics`` (never read by any fit -- see the callers). Shared by every
    block_boundary fit path (``_fit_all_positions_independent``,
    ``_fit_block_boundary_backfit``) with the fields
    ``fit_relative_residual``, ``target_norm``, ``update_norm``,
    ``relative_update_norm`` (denominator = the pre-correction weight slice),
    ``realized_target_norm_ratio``.

    ``realized_target_norm_ratio`` is computed EXACTLY from the accumulated
    fit (``H @ correction^T + bias``, pushed through the SAME ``effective_out``
    -- LayerScale -- the solve itself used), reusing the already-resident
    ``h_batches`` bank rather than a second capture sweep.
    """
    update_norm = float(torch.linalg.norm(correction).item())
    weight_norm = float(torch.linalg.norm(weight_before).item())
    if h_batches is None:
        # Streaming path: no resident input bank, so the prediction norm comes from the
        # solver's accumulated statistics (exact up to floating-point summation order).
        if stats is None:
            raise ValueError("realization fields need either h_batches or the accumulated stats")
        pred_sq = _realized_pred_sq_from_stats(stats, correction, bias_for_pred, effective_out)
    else:
        pred_sq = 0.0
        for h in h_batches:
            pred = h.reshape(-1, h.shape[-1]).double() @ correction.double().T + bias_for_pred.double()
            pred = pred @ effective_out.double()
            pred_sq += float((pred**2).sum().item())
    return {
        "fit_relative_residual": (residual_norm_after / desired_norm) if desired_norm else 0.0,
        "target_norm": desired_norm,
        "update_norm": update_norm,
        "relative_update_norm": update_norm / (weight_norm + 1e-12),
        "realized_target_norm_ratio": (pred_sq**0.5) / (desired_norm + 1e-12),
    }


@torch.no_grad()
def _fit_all_positions_independent(
    target_model,
    current_state: dict[str, torch.Tensor],
    positions: list[int],
    source_coordinates: dict[int, float],
    desired_batches: dict[int, list],
    target_output_batches: dict[int, list],
    batches: list,
    components: tuple,
    config,
    device,
    family_adapter=None,
) -> dict[int, tuple[dict, list]]:
    """Fit every ``(position, component)`` pair for ``cascade_order='independent'``
    from ONE shared target forward sweep instead of one sweep per pair.

    Under ``cascade_order="independent"`` the target model is never mutated
    between fits -- the only mount site
    (``if config.cascade_order != "independent": target_model.load_state_dict(...)``
    in ``_fit_direct_target_position``) is unconditionally skipped, both across
    positions and across components within one position. Every ``(position,
    component)`` pair therefore observes the identical, pristine
    ``target_model`` state that ``current_state`` already describes, which
    makes ``_fit_direct_target_position``'s per-pair ``capture_tokens`` calls
    (one full calibration sweep each, up to ``len(positions) *
    len(components)`` of them) redundant: they all capture from the same
    model. This function captures once and reuses the banks for every solve.

    ``capture_tokens`` already accepts a combined ``requests`` dict spanning
    many ``(position, kind)`` pairs and turns it into one set of hooks fired
    by one sweep over ``batches`` (see its docstring/implementation); nothing
    below it needed to change; this function is only a restructuring of the
    *caller* side. One ``"out"`` (boundary) request is registered per
    position, shared by every component at that position -- the block's own
    boundary output does not depend on which component's input is being
    fitted -- and one ``"h"`` request per ``(position, component)`` pair for
    that component's regression features
    (``COMPONENT_INPUT_KIND[component]``).

    Per-``(position, component)`` solving is otherwise byte-identical to
    ``_fit_direct_target_position``'s independent-mode body: same
    ``ResidualSufficientStatistics`` online accumulation (already a streaming
    accumulator; nothing here changes how it accumulates, only how many
    forward sweeps feed it), same ridge solve, same ``missing_bias`` handling,
    same ``block_rows`` diagnostic fields -- including the (under independent
    mode, vacuous but historically populated) ``measured_residual_norm_after``
    field on every component but the position's last: since nothing is ever
    mounted, that field is simply the next component's own pre-fit residual,
    identical in both the old per-pair-capture path and this one.

    The pristine-effect assertion is strengthened relative to
    ``_fit_direct_target_position``: that function can only assert it for
    ``component_index == 0`` of whichever position the caller nominates as
    "first" (a real cascade only ever *starts* pristine). Under true
    independent semantics there is no privileged "first" position -- every
    ``(position, component)`` pair sees the pristine base -- so this function
    asserts it unconditionally for all of them. This is a strictly stronger
    correctness check with no effect on the fitted values themselves: it can
    only ever raise on a bug (a stale reference bank or a mutated base), never
    change a number that was going to be returned.

    Intra-position "replay" (the user's brief step 6, mounting attn.out_proj
    locally before fitting mlp.c_proj) is deliberately NOT implemented: under
    independent mode there is no intra-position mount to replay in the first
    place (the same skipped-mount conditional gates it), so there is no
    sequential attention->MLP effect for a replay to preserve. Building it
    would change independent-mode's numerics, not merely speed it up.

    Returns ``{position: (position_corrections, block_rows)}``, matching what
    ``len(positions)`` separate ``_fit_direct_target_position`` calls would
    each have returned for the identical inputs.
    """
    if config.cascade_order != "independent":
        raise ValueError("_fit_all_positions_independent requires cascade_order='independent'")
    shim = _layout_for(family_adapter)
    requests: dict[str, tuple[int, str]] = {}
    for pos in positions:
        requests[f"{pos}.out"] = (pos, "boundary")
        for component in components:
            requests[f"{pos}.{component}.h"] = (pos, COMPONENT_INPUT_KIND[component])
    captured = capture_tokens(target_model, batches, requests, device, family_adapter=family_adapter)

    results: dict[int, tuple[dict, list]] = {}
    for pos in positions:
        out_batches = captured[f"{pos}.out"]
        position_corrections: dict[str, torch.Tensor] = {}
        block_rows: list[dict[str, Any]] = []
        for component in components:
            key = shim.component_key(pos, component, prefixed=True)
            h_batches = captured[f"{pos}.{component}.h"]
            width = int(current_state[key].shape[0])
            effective_out = _component_effective_out(shim, target_model, pos, component, width)
            stats = ResidualSufficientStatistics(device=device)
            desired_sq = 0.0
            effect_sq = 0.0
            for h, out, desired_batch, base_out in zip(
                h_batches, out_batches, desired_batches[pos], target_output_batches[pos], strict=True
            ):
                effect = out - base_out
                error = desired_batch - effect
                desired_sq += float((desired_batch.double() ** 2).sum().item())
                effect_sq += float((effect.double() ** 2).sum().item())
                stats.update(h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), None, effective_out)
            _finalize_independent_component(
                stats,
                desired_sq,
                effect_sq,
                pos,
                component,
                key,
                current_state,
                effective_out,
                config,
                source_coordinates,
                position_corrections,
                block_rows,
                h_batches=h_batches,
            )
        results[pos] = (position_corrections, block_rows)
    return results


def _finalize_independent_component(
    stats,
    desired_sq,
    effect_sq,
    pos,
    component,
    key,
    current_state,
    effective_out,
    config,
    source_coordinates,
    position_corrections,
    block_rows,
    h_batches=None,
):
    """Everything after one component's accumulation loop in
    ``_fit_all_positions_independent``: the pristine-effect check, the ridge
    solve, state/bookkeeping updates, missing-bias handling, and the block_row
    diagnostic (plus optional realization diagnostics). Mutates
    ``position_corrections``, ``current_state`` and ``block_rows`` in place.
    """
    # See the docstring: under independent mode this holds for every
    # (position, component) pair, not only a historically-first one.
    if effect_sq > 1e-12 * max(desired_sq, 1.0):
        raise RuntimeError(
            "Direct completion started from a target model that is not the native "
            f"base: nonzero pre-fit effect at position {pos} (||T-T0||^2={effect_sq:.3e})"
        )
    correction, diag = stats.solve(
        ridge_relative=config.ridge_relative,
        ridge_estimator=config.ridge_estimator,
        exact_form=config.exact_form,
    )
    correction = correction.cpu()
    diag["bias_correction"] = diag["bias_correction"].cpu()
    weight_before = current_state[key].detach().clone()
    if block_rows:
        # Same bookkeeping as _fit_direct_target_position: the previous
        # component's row records this component's own pre-fit residual
        # under its historical name. Under independent mode nothing
        # mounted in between, so this is not a "post-mount" measurement
        # -- see the docstring -- but it is the identical value the old
        # per-pair-capture path would have recorded.
        block_rows[-1]["measured_residual_norm_after"] = diag["residual_norm_before"]
    if correction.shape != current_state[key].shape or not torch.isfinite(correction).all():
        raise RuntimeError("Direct residual completion produced an invalid projection")
    position_corrections[key] = correction
    current_state[key] = current_state[key] + correction.to(current_state[key])
    bias_key = f"{key[: -len('.weight')]}.bias"
    bias_correction = diag["bias_correction"]
    skip_bias = False
    if bias_key not in current_state:
        if config.missing_bias == "materialize":
            raise RuntimeError(
                f"missing_bias='materialize' requires {bias_key} to exist on the target "
                "before residual completion runs; call "
                "materialize_missing_projection_biases() on the target model and its "
                "base state dict first"
            )
        elif config.missing_bias == "skip":
            if torch.count_nonzero(bias_correction):
                raise RuntimeError(
                    "missing_bias='skip' would discard a nonzero intercept at "
                    f"{bias_key}; the weight was fitted on centered banks and is "
                    "not valid without it"
                )
            skip_bias = True
        else:
            raise RuntimeError(
                f"Target model is missing the expected bias parameter {bias_key}. "
                "Decoder MLP projections are bias-free; set "
                "target_residual_completion.missing_bias to 'materialize' "
                "(exact, adds the parameter) or 'skip' with exact_form=false."
            )
    if not skip_bias:
        bias_delta = bias_correction.to(current_state[bias_key])
        if bias_delta.shape != current_state[bias_key].shape or not torch.isfinite(bias_delta).all():
            raise RuntimeError("Direct residual completion produced an invalid bias")
        position_corrections[bias_key] = bias_correction
        current_state[bias_key] = current_state[bias_key] + bias_delta
    desired_norm = desired_sq**0.5
    block_row = {
        "mode": "direct_target",
        "component": component,
        "position": pos,
        "source_coordinate": float(source_coordinates[pos]),
        "desired_norm": desired_norm,
        "effect_before_norm": effect_sq**0.5,
        "relative_residual_before": (diag["residual_norm_before"] / desired_norm) if desired_norm else 0.0,
        "relative_residual_after": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
        "correction_rank": int(torch.linalg.matrix_rank(correction.double()).item()),
        **diag,
    }
    if bool(getattr(config, "realization_diagnostics", False)):
        bias_for_pred = bias_correction if not skip_bias else torch.zeros(correction.shape[0])
        block_row.update(
            _realization_diagnostic_fields(
                h_batches,
                correction,
                bias_for_pred,
                effective_out,
                weight_before,
                desired_norm,
                diag["residual_norm_after"],
                stats=stats,
            )
        )
    block_rows.append(block_row)


def _realized_pred_sq_from_stats(stats, correction, bias_for_pred, effective_out) -> float:
    """``||(H C^T + 1 b^T) E||_F^2`` from `ResidualSufficientStatistics` alone.

    With ``A = H`` (``t_in=None``, the direct-target convention), the solver accumulates
    ``s = A^T A``, ``sum_a = A^T 1`` and ``n_rows``, so
    ``P^T P = C s C^T + (C sum_a) b^T + b (C sum_a)^T + n b b^T`` and the value is
    ``trace(E^T P^T P E)``. Equals `_realization_diagnostic_fields`'s bank-based sum up to
    floating-point summation order; used where no activation bank is resident (streaming).
    """
    if stats.s is None or stats.sum_a is None:
        raise ValueError("realization fields from stats require an accumulated ResidualSufficientStatistics")
    c = correction.double().to(stats.s.device)
    b = bias_for_pred.double().to(stats.s.device)
    e_out = effective_out.double().to(stats.s.device)
    c_sum_a = c @ stats.sum_a
    ptp = (
        c @ stats.s @ c.T + torch.outer(c_sum_a, b) + torch.outer(b, c_sum_a) + float(stats.n_rows) * torch.outer(b, b)
    )
    return float(torch.trace(e_out.T @ ptp @ e_out).item())
