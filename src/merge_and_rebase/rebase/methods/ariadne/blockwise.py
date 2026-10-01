"""Intra-block backfit and joint solvers (kept verbatim; not part of the headline method)."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn

from .capture import (
    _assert_layerscale_identity,
    _stock_mha_out_proj_input,
    _verify_recomputed_attention_input,
    capture_tokens,
)
from .components import COMPONENT_FORWARD_ORDER, order_components
from .independent import _realization_diagnostic_fields
from .layouts import COMPONENT_INPUT_KIND, _component_weight_bias, _layout_for
from .linalg import ResidualSufficientStatistics


def _replay_block_components(shim, local_block, x_batches, components, device):
    """Run ``local_block`` directly (no full-model forward) over ``x_batches``,
    hooking each of ``components`` (``attn.out_proj`` / ``mlp.c_proj`` only) for
    its own input, plus the block's own output.

    ``local_block`` is a standalone (already-unwrapped) block module -- e.g. a
    ``copy.deepcopy`` of ``shim.block_module(...)`` -- called with a single
    positional tensor, mirroring how ``_VisionLayout.forward``/``_encode_image``
    invoke every block during an ordinary full-model sweep. Returns
    ``(component_h_batches, out_batches)``, both CPU float32, same convention as
    ``capture_tokens``.
    """
    handles = []
    component_h: dict[str, list[torch.Tensor]] = {c: [] for c in components}
    try:
        if "attn.out_proj" in components:
            attn = shim.attn_module(local_block)
            if isinstance(attn, nn.MultiheadAttention):
                # A stock nn.MultiheadAttention applies out_proj functionally
                # (see capture_tokens' own docstring): hook the attention
                # itself and recompute the rows it fed the projection.
                def attn_hook(mod, args, kwargs, value, *, store=component_h["attn.out_proj"]):
                    rows = _stock_mha_out_proj_input(mod, args, kwargs)
                    _verify_recomputed_attention_input(mod, rows, value)
                    store.append(rows.detach().float().cpu().clone())

                handles.append(attn.register_forward_hook(attn_hook, with_kwargs=True))
            else:
                # A plain (non-MHA) attention wrapper calls out_proj as an
                # ordinary submodule, so a direct forward hook on it fires
                # normally -- same fallback capture_tokens itself takes.
                proj = shim.attn_proj_module(local_block)

                def proj_hook(_m, inputs, _value, *, store=component_h["attn.out_proj"]):
                    store.append(inputs[0].detach().float().cpu().clone())

                handles.append(proj.register_forward_hook(proj_hook))
        if "mlp.c_proj" in components:
            proj = shim.proj_module(local_block)

            def proj_hook(_m, inputs, _value, *, store=component_h["mlp.c_proj"]):
                store.append(inputs[0].detach().float().cpu().clone())

            handles.append(proj.register_forward_hook(proj_hook))
        out_batches = []
        for x in x_batches:
            out = local_block(x.to(device))
            out_batches.append(out.detach().float().cpu().clone())
        if any(len(v) != len(x_batches) for v in component_h.values()):
            raise RuntimeError("A backfit component hook did not fire exactly once per batch")
        return component_h, out_batches
    finally:
        for h in handles:
            h.remove()


def _mount_component(shim, local_block, component, weight, bias):
    module = shim.attn_proj_module(local_block) if component == "attn.out_proj" else shim.proj_module(local_block)
    module.weight.data.copy_(weight.to(module.weight.dtype).to(module.weight.device))
    if bias is not None:
        if module.bias is None:
            raise RuntimeError(f"{component} has no bias parameter to mount a fitted bias correction onto")
        module.bias.data.copy_(bias.to(module.bias.dtype).to(module.bias.device))


def _mount_all_deltas(shim, local_block, order, base, deltas):
    """Mount ``base[c] + deltas[c]`` for every ``c in order`` onto ``local_block``."""
    for c in order:
        w, b = deltas[c]
        _mount_component(shim, local_block, c, base[c][0] + w, None if base[c][1] is None else base[c][1] + b)


def _backfit_data_fit_sq(shim, local_block64, device, x_batches, d_batches, t0_batches, order, base, deltas):
    """``||D_j - (block_j(X_j^0; deltas mounted) - T_j^0)||_F^2`` -- the (un-linearized,
    measured on the actual local block replay) data-fit term of the safeguarded
    backfit objective ``J(Delta)``. See ``_fit_block_boundary_backfit``'s docstring.

    ``local_block64`` is a dedicated float64 replica of the block (see
    ``_fit_block_boundary_backfit``), used ONLY for this measurement -- never
    for the candidate sub-solves, whose CPU-float32 delta convention is
    unaffected. The accept/backtrack decision this feeds compares two J
    values that can be arbitrarily close near a fixed point (a near-singular
    Gauss-Seidel design, e.g., can leave genuine per-sweep improvements far
    below float32's ~1e-7 relative precision); evaluating in the module's own
    float32 would let ordinary float32 rounding noise in the forward pass
    flip the accept/reject decision and stall the sweep well short of
    convergence -- exactly the kind of numerical noise a *safeguard*
    (whose entire job is a reliable ``<`` comparison) must not be sensitive
    to. ``_mount_component``'s own ``.to(module.weight.dtype)`` upcasts the
    float32 base/delta tensors to float64 automatically since
    ``local_block64``'s parameters are float64.
    """
    _mount_all_deltas(shim, local_block64, order, base, deltas)
    t_all = [local_block64(x.to(device).double()).detach().cpu().clone() for x in x_batches]
    return sum(
        float(((d.double() - (t - t0.double())) ** 2).sum().item())
        for d, t, t0 in zip(d_batches, t_all, t0_batches, strict=True)
    )


def _backfit_ridge_penalty(order, deltas, lambdas):
    """``sum_c lambda_c * ||Delta W_c||_F^2`` -- the exact penalty
    ``ResidualSufficientStatistics.solve`` minimizes for the weight (the bias is
    fit unpenalized; see ``_fit_block_boundary_backfit``'s docstring), evaluated
    with each component's FROZEN round-1 ``lambda_c`` from ``lambdas``. A
    component with no ``lambdas`` entry yet (never solved) contributes 0, which
    is always exact since its delta is still zero at that point.
    """
    total = 0.0
    for c in order:
        w, _ = deltas[c]
        lam = lambdas.get(c)
        if lam:
            total += lam * float((w.double() ** 2).sum().item())
    return total


def _backfit_objective(shim, local_block64, device, x_batches, d_batches, t0_batches, order, base, deltas, lambdas):
    """Returns ``(J, data_fit_sq)`` -- the full safeguarded objective and its
    data-fit term alone (the latter is what feeds the diagnostic ``r`` trace).
    """
    data_fit_sq = _backfit_data_fit_sq(
        shim, local_block64, device, x_batches, d_batches, t0_batches, order, base, deltas
    )
    return data_fit_sq + _backfit_ridge_penalty(order, deltas, lambdas), data_fit_sq


@torch.no_grad()
def _fit_block_boundary_backfit(
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
    """Intra-block Gauss-Seidel backfitting for ``component_target='block_boundary'``,
    ``block_split='backfit'`` (Direct Residual only; see ``DirectResidualConfig``).

    Every position is independent (the upstream target model is pristine, as
    everywhere else under independent mode), and every component within one
    position shares the SAME block-boundary target ``D_j`` -- exactly as
    ``_fit_all_positions_independent`` -- but instead of regressing each
    component independently against the full ``D_j`` (leaving a "double
    target" when more than one component is requested), this refits each
    component against the RESIDUAL left over once every other component's
    current fit is actually mounted and the block is replayed:

        E_c = D_j - (block_j(X_j^0; others mounted, c at its native base) - T_j^0)

    Gauss-Seidel over ``components`` (canonical forward order), from scratch
    every sweep (no warm start -- ``ResidualSufficientStatistics`` solves a
    fresh ridge system each time, exactly as every other direct-target fit
    does).

    Monotone safeguarded backfitting. Each Gauss-Seidel sub-step refits
    component ``c`` against ``E_c`` above, but ``c``'s update also changes
    the OTHER component's effect through the block's own nonlinear path
    (``attn.out_proj -> ln_2 -> GELU -> mlp.c_proj``'s input, on ViT), so the
    sub-step candidate is not an exact block-coordinate minimizer of the true
    objective and nothing prevents it from increasing that objective. This is
    fixed by measuring the true, un-linearized per-block objective on the
    local block replay,

        J(Delta) = ||D_j - (block_j(X_j^0; Delta mounted) - T_j^0)||_F^2
                   + sum_c lambda_c * ||Delta W_c||_F^2,

    and only ever accepting a change that decreases it. ``lambda_c`` is each
    component's ridge coefficient -- ``ResidualSufficientStatistics.solve``'s
    ``diag["ridge"]`` -- FROZEN at the value its round-1 (first sweep) solve
    returns, not recomputed every sweep: the whole point of a monotone
    descent objective is that it is a fixed function of ``Delta``, so a
    ridge that itself drifts sweep to sweep (as it does inside ``solve``,
    since it is a function of that sweep's own H_c statistics, which change
    as other components' deltas move) would make "J decreased" incomparable
    across sweeps. ``lambda_c`` penalizes ``Delta W_c`` (the weight only) at
    the SAME scale ``solve`` itself minimizes: its normal equations are
    ``S_c X G + lambda X = B_c``, i.e. the stationarity condition of
    ``||A X L - E||_F^2 + lambda ||X||_F^2`` with the bias fit unpenalized
    (``solve``'s ``beta`` is derived with no ridge term) -- so
    ``lambda_c * ||Delta W_c||_F^2`` is exactly what that component's own
    sub-solve minimizes, with ``Delta W_c`` the returned weight correction
    itself (``solve`` returns ``x.T``, and the penalty ``lambda ||x||_F^2``
    is transpose-invariant).

    Each sub-step computes the candidate ``Delta_c^new`` exactly as the
    unsafeguarded rule did, then accepts it only if ``J`` decreases;
    otherwise backtracks ``Delta_c = Delta_c^old + eta (Delta_c^new -
    Delta_c^old)`` (weight AND bias together) for ``eta = 1, 1/2, 1/4, ...``
    down to ``1/256`` (an initial full step plus up to 8 halvings); if no
    ``eta`` decreases ``J``, ``Delta_c`` is left at ``Delta_c^old``
    (recorded as an accepted ``eta`` of 0). ``J`` is therefore non-increasing
    by construction, at every sub-step and therefore every sweep. Sweeping
    stops when the relative decrease of ``J`` over a full sweep (measured
    once, with every current delta mounted, after each sweep's Gauss-Seidel
    pass) drops below ``config.backfit_tol``, or at
    ``config.backfit_max_iters`` sweeps. The plain relative residual
    ``r = ||D_j - (block_j(X_j^0; ALL current deltas mounted) - T_j^0)||_F /
    ||D_j||_F`` is still measured and logged every sweep as a diagnostic
    (``backfit_residual_trace``), but no longer drives the stopping rule.

    With a single component, the first sweep's ``E_c`` reduces exactly to
    ``D_j`` (no other component is mounted, so the replay term is
    ``T_j^0 - T_j^0 = 0``), so the fit is byte-for-byte the single-component
    call of ``_fit_all_positions_independent`` PROVIDED the replayed
    ``H_c``/output from the local block copy on the captured pristine
    ``X_j^0`` bitwise reproduce what a direct hook on the live target model
    would have captured -- asserted below (see ``block_replay_bitwise`` in the
    returned diagnostics) rather than assumed. With one component there is
    also nothing to backtrack against on later sweeps: the candidate is
    always accepted at ``eta=1`` (see the docstring of
    ``_fit_block_boundary_backfit``'s test coverage), since a single
    component's own sub-solve is an exact minimizer of ``J`` restricted to
    that component with every OTHER (nonexistent) component fixed.

    The full target model is never mutated: every mount happens on a
    ``copy.deepcopy`` of the block, discarded at the end of each position.
    """
    if family_adapter is not None:
        raise NotImplementedError("block_split='backfit' is vision-only")
    shim = _layout_for(family_adapter)
    residual_writers = set(COMPONENT_FORWARD_ORDER)
    if set(components) - residual_writers:
        raise ValueError(
            "block_split='backfit' only supports residual-writing components "
            f"{sorted(residual_writers)}; internal components (q/k/v/c_fc) act on the block "
            "output nonlinearly and have no linear regression onto a block-boundary target"
        )
    order = order_components(components)
    if not order:
        raise ValueError("components must not be empty")

    # Capture the pristine block input X_j^0 for every position in ONE target
    # sweep -- new capture kind, the forward-hook INPUT of the block module
    # itself (the same module "boundary" hooks for its OUTPUT).
    block_input_requests = {f"{pos}.block_input": (pos, "block_input") for pos in positions}
    block_inputs = capture_tokens(target_model, batches, block_input_requests, device, family_adapter=family_adapter)

    results: dict[int, tuple[dict, list]] = {}
    for pos in positions:
        x_batches = block_inputs[f"{pos}.block_input"]
        t0_batches = target_output_batches[pos]
        d_batches = desired_batches[pos]
        block = shim.blocks(target_model)[pos]
        local_block = copy.deepcopy(shim.block_module(block)).to(device).eval()

        # Device convention: every "delta" tensor this function accumulates
        # (base[c], deltas[c], and everything derived from them) lives on CPU
        # float32, exactly like ResidualSufficientStatistics.solve()'s own
        # output (`correction = correction.cpu()` below) and like
        # _fit_all_positions_independent's `current_state` bookkeeping.
        # `local_block` itself is moved to `device` (mirroring how the live
        # target model would sit on a training device at runtime), so its
        # parameters -- and therefore _component_weight_bias's raw read of
        # them -- are on `device`. Without the explicit `.cpu()` here, `base[c]`
        # would silently inherit that device, and `base[c2][0] + w2` below (w2
        # is always a CPU delta) would mix a CUDA tensor with a CPU one -- a
        # RuntimeError on CUDA that a CPU-only run can never surface, since
        # cpu + cpu never errors regardless of provenance. _mount_component
        # is the only place a base/delta tensor is moved back onto `device`
        # (via its own `.to(module.weight.device)`), so mounting stays correct
        # regardless of what device `local_block` lives on.
        base: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        for c in order:
            w, b, row_slice = _component_weight_bias(shim, local_block, c)
            if row_slice is not None:
                raise RuntimeError("block_split='backfit' components must not be packed (row_slice must be None)")
            base[c] = (w.detach().cpu().clone(), None if b is None else b.detach().cpu().clone())
        scale_modules = {c: shim.component_scale_module(local_block, c) for c in order}
        # A dedicated float64 replica, used ONLY by the monotone safeguard's own
        # J(Delta) measurement (see _backfit_data_fit_sq's docstring) -- never
        # for candidate generation, so the returned corrections' CPU-float32
        # convention is untouched. Deepcopied here while `local_block` is still
        # pristine (nothing has been mounted onto it yet).
        local_block64 = copy.deepcopy(local_block).double().eval()

        def reset_all(local_block=local_block, order=order, base=base):
            for c in order:
                w, b = base[c]
                _mount_component(shim, local_block, c, w, b)

        # Pristine-replay check: X_j^0 through the untouched local copy must
        # reproduce the captured boundary output T_j^0.
        reset_all()
        t_replayed = []
        for x in x_batches:
            t_replayed.append(local_block(x.to(device)).detach().float().cpu().clone())
        block_replay_bitwise = all(torch.equal(rep, ref) for rep, ref in zip(t_replayed, t0_batches, strict=True))
        for rep, ref in zip(t_replayed, t0_batches, strict=True):
            if not torch.allclose(rep, ref, atol=1e-4, rtol=1e-4):
                raise RuntimeError(
                    f"Block replay at position {pos} does not reproduce the pristine boundary "
                    "output within tolerance; X_j^0/local-block-copy mismatch"
                )

        deltas: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {
            c: (torch.zeros_like(base[c][0]), None if base[c][1] is None else torch.zeros_like(base[c][1]))
            for c in order
        }
        d_sq_total = sum(float((d.double() ** 2).sum().item()) for d in d_batches)
        desired_norm = d_sq_total**0.5
        residual_trace: list[float] = []
        j_trace: list[float] = []
        converged = False
        n_sweeps = 0
        j_prev = None
        round1_j: float | None = None
        round1_r: float | None = None
        # J(Delta=0) = ||D_j - (T_j^0 - T_j^0)||^2 + 0 = ||D_j||^2 (no ridge penalty
        # at the all-zero start).
        lambdas: dict[str, float] = {}
        per_component_diag: dict[str, dict] = {}
        per_component_h: dict[str, list] = {}
        per_component_effective_out: dict[str, torch.Tensor] = {}
        per_component_eta_history: dict[str, list[float]] = {c: [] for c in order}
        # Backtracking line-search factors: an initial full step, then up to 8
        # halvings (see the docstring). eta=0 (keep the old delta) is the
        # implicit fallback when none of these decrease J.
        backtrack_etas = [1.0] + [1.0 / (2**k) for k in range(1, 9)]
        for sweep in range(1, int(config.backfit_max_iters) + 1):
            n_sweeps = sweep
            for c in order:
                reset_all()
                for c2 in order:
                    if c2 == c:
                        continue
                    w2, b2 = deltas[c2]
                    _mount_component(
                        shim, local_block, c2, base[c2][0] + w2, None if base[c2][1] is None else base[c2][1] + b2
                    )
                component_h, out_batches = _replay_block_components(shim, local_block, x_batches, [c], device)
                h_batches = component_h[c]
                e_batches = [d - (t - t0) for d, t, t0 in zip(d_batches, out_batches, t0_batches, strict=True)]
                width = int(base[c][0].shape[0])
                identity_out = torch.eye(width, dtype=torch.float32)
                scale_module = scale_modules[c]
                effective_out = identity_out
                if not isinstance(scale_module, nn.Identity):
                    scale = getattr(scale_module, "gamma", None)
                    if scale is None or scale.ndim != 1 or scale.shape[0] != width:
                        raise ValueError("Unsupported non-diagonal target LayerScale")
                    effective_out = identity_out * scale.detach().cpu().float().unsqueeze(0)
                stats = ResidualSufficientStatistics(device=device)
                for h, e in zip(h_batches, e_batches, strict=True):
                    stats.update(h.reshape(-1, h.shape[-1]), e.reshape(-1, e.shape[-1]), None, effective_out)
                correction, diag = stats.solve(
                    ridge_relative=config.ridge_relative,
                    ridge_estimator=config.ridge_estimator,
                    exact_form=config.exact_form,
                )
                correction = correction.cpu()
                diag["bias_correction"] = diag["bias_correction"].cpu()
                if correction.shape != base[c][0].shape or not torch.isfinite(correction).all():
                    raise RuntimeError("block_split='backfit' produced an invalid projection")
                per_component_diag[c] = diag
                per_component_h[c] = h_batches
                per_component_effective_out[c] = effective_out

                # Freeze lambda_c at its round-1 (first-solve) value: J must stay
                # a FIXED function of Delta across the whole backfit for "J
                # decreased" to be comparable sweep to sweep (see the docstring).
                # solve()'s own internal ridge is recomputed every sweep from
                # that sweep's H_c -- that only shapes the CANDIDATE proposed
                # below, never the objective the safeguard accepts or rejects
                # against.
                if sweep == 1:
                    lambdas[c] = float(diag["ridge"])

                old_w, old_b = deltas[c]
                candidate_w, candidate_b = correction, diag["bias_correction"]
                trial_deltas = dict(deltas)
                j_old, _ = _backfit_objective(
                    shim, local_block64, device, x_batches, d_batches, t0_batches, order, base, trial_deltas, lambdas
                )
                accepted_eta = 0.0
                accepted_delta = (old_w, old_b)
                for eta in backtrack_etas:
                    trial_w = old_w + eta * (candidate_w - old_w)
                    trial_b = None if old_b is None else old_b + eta * (candidate_b - old_b)
                    trial_deltas[c] = (trial_w, trial_b)
                    j_trial, _ = _backfit_objective(
                        shim,
                        local_block64,
                        device,
                        x_batches,
                        d_batches,
                        t0_batches,
                        order,
                        base,
                        trial_deltas,
                        lambdas,
                    )
                    if j_trial < j_old:
                        accepted_eta = eta
                        accepted_delta = (trial_w, trial_b)
                        break
                deltas[c] = accepted_delta
                per_component_eta_history[c].append(accepted_eta)
            # Measure the full-sweep objective and diagnostic residual with every
            # current delta mounted -- the same mount _backfit_objective performs,
            # done once more here only because we also want the plain (ridge-free)
            # data-fit norm `r` for the diagnostic trace.
            j_now, data_fit_sq = _backfit_objective(
                shim, local_block64, device, x_batches, d_batches, t0_batches, order, base, deltas, lambdas
            )
            r = (data_fit_sq**0.5) / (desired_norm + 1e-12)
            residual_trace.append(r)
            j_trace.append(j_now)
            if sweep == 1:
                round1_j, round1_r = j_now, r
            if j_prev is not None:
                rel_decrease = (j_prev - j_now) / j_prev if j_prev > 0 else 0.0
                if rel_decrease < float(config.backfit_tol):
                    converged = True
                    break
            j_prev = j_now

        position_corrections: dict[str, torch.Tensor] = {}
        block_rows: list[dict[str, Any]] = []
        for c in order:
            key = shim.component_key(pos, c, prefixed=True)
            correction, bias_correction = deltas[c]
            position_corrections[key] = correction
            bias_key = f"{key[: -len('.weight')]}.bias"
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
                        "Set target_residual_completion.missing_bias to 'materialize' (exact, "
                        "adds the parameter) or 'skip' with exact_form=false."
                    )
            if not skip_bias:
                bias_delta = bias_correction.to(current_state[bias_key])
                if bias_delta.shape != current_state[bias_key].shape or not torch.isfinite(bias_delta).all():
                    raise RuntimeError("block_split='backfit' produced an invalid bias")
                position_corrections[bias_key] = bias_delta
            diag = per_component_diag[c]
            block_row = {
                "mode": "direct_target",
                "component": c,
                "component_target": "block_boundary",
                "block_split": "backfit",
                "position": pos,
                "source_coordinate": float(source_coordinates[pos]),
                "desired_norm": desired_norm,
                "relative_residual_before": (diag["residual_norm_before"] / desired_norm) if desired_norm else 0.0,
                "relative_residual_after": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                "correction_rank": int(torch.linalg.matrix_rank(correction.double()).item()),
                "backfit_n_sweeps": n_sweeps,
                "backfit_converged": converged,
                "backfit_residual_trace": list(residual_trace),
                "backfit_j_trace": list(j_trace),
                "backfit_round1_j": round1_j,
                "backfit_round1_r": round1_r,
                "backfit_ridge_lambda": lambdas.get(c),
                "backfit_eta_history": list(per_component_eta_history[c]),
                "block_replay_bitwise": block_replay_bitwise,
                **diag,
            }
            if bool(getattr(config, "realization_diagnostics", False)):
                bias_for_pred = bias_correction if not skip_bias else torch.zeros(correction.shape[0])
                block_row.update(
                    _realization_diagnostic_fields(
                        per_component_h[c],
                        correction,
                        bias_for_pred,
                        per_component_effective_out[c],
                        base[c][0],
                        desired_norm,
                        diag["residual_norm_after"],
                    )
                )
            block_rows.append(block_row)
        results[pos] = (position_corrections, block_rows)
    return results


class _JointBlockRidgeStatistics:
    """Streaming Gram/cross accumulator for the closed-form joint (O, D)-stacked
    ridge fit ``block_split='joint'`` uses (see ``_fit_block_boundary_joint``).

    Mirrors ``ResidualSufficientStatistics`` at ``t_in=None, t_out=I`` (the only
    configuration ``block_split='joint'`` ever needs -- LayerScale is asserted
    ``nn.Identity`` before this class is ever touched), generalized from one
    ridge scalar to a per-block-diagonal ridge vector over the STACKED feature
    dimension ``sum(dims)``. Accumulates the ``(d_total, d_total)`` Gram and
    ``(d_total, d_out)`` cross statistics batch by batch in float64, never
    materializing a design matrix with as many rows as calibration tokens (the
    Gram/cross tensors are the only ``O(d^2)``-sized state this class holds --
    ``d_total = d_O + d_D`` is a few thousand for ViT-L/14, not the token
    count).
    """

    def __init__(self, dims: list[int], d_out: int, device=None) -> None:
        if len(dims) < 1 or any(d <= 0 for d in dims) or d_out <= 0:
            raise ValueError("dims and d_out must be positive")
        self.device = device
        self.dims = list(dims)
        self.d_total = sum(dims)
        self.d_out = d_out
        self.gram = torch.zeros(self.d_total, self.d_total, dtype=torch.float64, device=device)
        self.cross = torch.zeros(self.d_total, d_out, dtype=torch.float64, device=device)
        self.sum_a = torch.zeros(self.d_total, dtype=torch.float64, device=device)
        self.sum_e = torch.zeros(d_out, dtype=torch.float64, device=device)
        self.sum_e2 = 0.0
        self.n_rows = 0

    def update(self, h_list: list[torch.Tensor], e: torch.Tensor) -> None:
        if len(h_list) != len(self.dims):
            raise ValueError("h_list must supply one feature bank per stacked component")
        if self.device is not None:
            h_list = [h.to(self.device) for h in h_list]
            e = e.to(self.device)
        for h, d in zip(h_list, self.dims, strict=True):
            if h.ndim != 2 or h.shape[1] != d:
                raise ValueError("component feature bank has an unexpected shape")
        if e.ndim != 2 or e.shape[1] != self.d_out:
            raise ValueError("target bank has an unexpected shape")
        rows = {h.shape[0] for h in h_list} | {e.shape[0]}
        if len(rows) != 1:
            raise ValueError("component feature banks and the target bank must share the row count")
        a = torch.cat([h.to(torch.float64) for h in h_list], dim=1)
        er = e.to(torch.float64)
        if not torch.isfinite(a).all() or not torch.isfinite(er).all():
            raise ValueError("solver inputs must be finite")
        self.gram += a.T @ a
        self.cross += a.T @ er
        self.sum_a += a.sum(dim=0)
        self.sum_e += er.sum(dim=0)
        self.sum_e2 += float((er * er).sum().item())
        self.n_rows += int(a.shape[0])

    def solve(self, lambdas: list[float]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Exact centered normal-equation solve with a block-diagonal ridge.

        Returns ``(x, beta, diag)`` with ``x`` shaped ``(d_total, d_out)`` (the
        stacked, UNTRANSPOSED weight -- callers split it by ``self.dims`` and
        transpose each block to the usual ``(out, in)`` weight convention) and
        ``beta`` the shared ``(d_out,)`` intercept solved jointly with ``x``
        (not ridge-penalized, matching ``ResidualSufficientStatistics.solve``'s
        own convention).

        Equivalent to rescaling features by ``1/sqrt(lambda_c)`` per block and
        solving with unit ridge (the textbook reduction of a block-diagonal
        ridge to a scalar one), but implemented directly on the centered normal
        equations ``(S_c + diag(lambda)) X = B_c`` -- exact in float64, no
        rescale/un-rescale round trip.
        """
        if len(lambdas) != len(self.dims):
            raise ValueError("lambdas must supply one ridge coefficient per stacked component")
        if self.n_rows == 0:
            raise ValueError("cannot solve empty joint block statistics")
        if any((not math.isfinite(float(lam))) or float(lam) < 0 for lam in lambdas):
            raise ValueError("lambdas must be finite and non-negative")
        n = float(self.n_rows)
        mu_a = self.sum_a / n
        mu_e = self.sum_e / n
        s = (self.gram + self.gram.T) * 0.5
        sc = s - torch.outer(self.sum_a, self.sum_a) / n
        bc = self.cross - torch.outer(self.sum_a, mu_e)
        sc = (sc + sc.T) * 0.5
        lam_vec = torch.cat(
            [
                torch.full((d,), float(lam), dtype=torch.float64, device=sc.device)
                for d, lam in zip(self.dims, lambdas, strict=True)
            ]
        )
        reg = sc + torch.diag(lam_vec)
        x = torch.linalg.solve(reg, bc)
        beta = mu_e - x.T @ mu_a
        # Exact total ||A x + 1 beta^T - E||_F^2 from raw (uncentered) sufficient
        # statistics -- same decomposition as ResidualSufficientStatistics.
        # _residual_sq, specialized to t_out=I (g=I, lmat=I).
        predicted_sq = torch.trace(x.T @ s @ x).item()
        cross_term = 2.0 * torch.sum(x * self.cross).item()
        resid_no_bias = self.sum_e2 - cross_term + predicted_sq
        pred_mean = mu_a @ x
        bias_term = 2.0 * n * float((beta @ (pred_mean - mu_e)).item()) + n * float((beta @ beta).item())
        residual_sq = max(0.0, resid_no_bias + bias_term)
        diag = {
            "n_rows": self.n_rows,
            "residual_norm_before": self.sum_e2**0.5,
            "residual_norm_after": residual_sq**0.5,
        }
        return x, beta, diag


@torch.no_grad()
def _fit_block_boundary_joint(
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
    """Closed-form joint ridge fit for ``component_target='block_boundary'``,
    ``block_split='joint'`` (Direct Residual only; see ``DirectResidualConfig``).

    Where ``block_split='none'`` fits every requested component INDEPENDENTLY
    against the same block-boundary target ``D_j`` (double-counting the target
    when more than one component is requested) and ``block_split='backfit'``
    resolves that by iterative Gauss-Seidel replay of the block's own
    nonlinearity, ``'joint'`` instead solves ONE closed-form ridge in a single
    linear-algebra step, under the explicit first-order approximation that the
    MLP does not respond to a change in ``attn.out_proj`` (``J_M = 0``): with
    ``H_O``/``H_D`` the pristine ``attn.out_proj``/``mlp.c_proj`` inputs (the
    SAME captures ``_fit_all_positions_independent`` uses) and
    ``D_j`` the block-boundary desired effect,

        min_{Wo,Wd,beta} ||H_O Wo^T + H_D Wd^T + 1 beta^T - D_j||_F^2
                          + lambda_O ||Wo||_F^2 + lambda_D ||Wd||_F^2.

    Only valid for ``components`` a non-empty subset of
    ``{"attn.out_proj", "mlp.c_proj"}`` (enforced by
    ``config.parse_direct_residual_config``).

    **lambda_c convention.** Each component's ridge coefficient is EXACTLY the
    value its OWN standalone ``block_split='none'`` fit would use --
    ``ResidualSufficientStatistics.solve(...)``'s own ``diag["ridge"]``,
    computed from that component's own ``H_c`` and the ridge_relative/
    ridge_estimator/exact_form config, penalizing ``Delta W_c`` (the weight
    only; the joint intercept below is unpenalized) at the identical scale
    ``solve`` itself would use it at alone -- see
    ``tests/test_direct_residual_joint.py``'s ``test_lambda_matches_single_
    component_solver`` for the regression pinning this.

    **Bias convention.** The joint intercept is not identifiable between the
    two components' biases (only their sum enters the objective). The WHOLE
    fitted intercept is assigned to ``mlp.c_proj.bias`` -- the block's LAST
    residual writer -- and ``attn.out_proj.bias`` is left untouched (a zero
    delta): ``attn.out_proj``'s bias also feeds the MLP's input on the real
    (nonlinear) block, which this first-order joint model ignores by
    construction, so it must not absorb any share of the block-level
    intercept a purely-linear model derived. With a single requested component
    this convention is moot -- see below.

    **LayerScale.** Asserted ``nn.Identity`` on both ``ls_1``/``ls_2`` before
    any solve (``_assert_layerscale_identity``): a nontrivial LayerScale would
    give the two writers different output gammas, which the stacked-feature
    derivation above assumes away.

    **Single-component reduction.** With one requested component, this
    function does not build a (degenerate, one-block) joint system at all --
    it returns that component's own standalone ``block_split='none'`` solve
    directly (the same ``ResidualSufficientStatistics`` call this function
    computes ``lambda_c`` from in the first place), so single-component
    ``block_split='joint'`` is not merely numerically close to
    ``block_split='none'`` but literally the same function call.

    Every position is independent and pristine (nothing is ever mounted
    between fits, matching every other Direct Residual path), so ``E_j ==
    D_j`` identically and this reuses ``capture_tokens``'s combined-request,
    one-sweep capture exactly like ``_fit_all_positions_independent``.
    """
    shim = _layout_for(family_adapter)
    residual_writers = set(COMPONENT_FORWARD_ORDER)
    if set(components) - residual_writers:
        raise ValueError(
            "block_split='joint' only supports residual-writing components "
            f"{sorted(residual_writers)}; internal components (q/k/v/c_fc) act on the block "
            "output nonlinearly and have no linear regression onto a block-boundary target"
        )
    order = order_components(components)
    if not order:
        raise ValueError("components must not be empty")

    requests: dict[str, tuple[int, str]] = {}
    for pos in positions:
        requests[f"{pos}.out"] = (pos, "boundary")
        for component in order:
            requests[f"{pos}.{component}.h"] = (pos, COMPONENT_INPUT_KIND[component])
    captured = capture_tokens(target_model, batches, requests, device, family_adapter=family_adapter)

    results: dict[int, tuple[dict, list]] = {}
    for pos in positions:
        block = shim.blocks(target_model)[pos]
        _assert_layerscale_identity(shim, block, context=f"position {pos}", requirement="block_split='joint'")
        out_batches = captured[f"{pos}.out"]
        d_batches = desired_batches[pos]
        t0_batches = target_output_batches[pos]

        # The block hasn't been touched yet at this position (independent
        # mode, pristine by construction): the pre-fit effect is the SAME for
        # every component, so it is measured once rather than per component.
        desired_sq = 0.0
        effect_sq = 0.0
        error_batches = []
        for out, desired_batch, base_out in zip(out_batches, d_batches, t0_batches, strict=True):
            effect = out - base_out
            desired_sq += float((desired_batch.double() ** 2).sum().item())
            effect_sq += float((effect.double() ** 2).sum().item())
            error_batches.append(desired_batch - effect)
        if effect_sq > 1e-12 * max(desired_sq, 1.0):
            raise RuntimeError(
                "Direct completion started from a target model that is not the native "
                f"base: nonzero pre-fit effect at position {pos} (||T-T0||^2={effect_sq:.3e})"
            )
        desired_norm = desired_sq**0.5

        h_batches_by_component: dict[str, list[torch.Tensor]] = {}
        widths: dict[str, int] = {}
        lambdas: dict[str, float] = {}
        single_component_fit: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}
        for component in order:
            key = shim.component_key(pos, component, prefixed=True)
            h_batches = captured[f"{pos}.{component}.h"]
            h_batches_by_component[component] = h_batches
            width = int(current_state[key].shape[0])
            widths[component] = width
            scale_module = shim.component_scale_module(block, component)
            if not isinstance(scale_module, nn.Identity):
                raise ValueError(f"block_split='joint' requires an identity LayerScale on {component}'s output")
            effective_out = torch.eye(width, dtype=torch.float32)
            stats = ResidualSufficientStatistics(device=device)
            for h, error in zip(h_batches, error_batches, strict=True):
                stats.update(h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), None, effective_out)
            correction, diag = stats.solve(
                ridge_relative=config.ridge_relative,
                ridge_estimator=config.ridge_estimator,
                exact_form=config.exact_form,
            )
            correction = correction.cpu()
            diag["bias_correction"] = diag["bias_correction"].cpu()
            lambdas[component] = float(diag["ridge"])
            single_component_fit[component] = (correction, diag)

        position_corrections: dict[str, torch.Tensor] = {}
        block_rows: list[dict[str, Any]] = []
        if len(order) == 1:
            # See the docstring: the bias-split convention below is only
            # meaningful with both writers present, so a single requested
            # component just IS the standalone block_split='none' fit -- the
            # identical ResidualSufficientStatistics call computed above for
            # lambda_c, reused verbatim rather than resolved.
            (component,) = order
            correction, diag = single_component_fit[component]
            key = shim.component_key(pos, component, prefixed=True)
            weight_before = current_state[key].detach().clone()
            if correction.shape != current_state[key].shape or not torch.isfinite(correction).all():
                raise RuntimeError("block_split='joint' produced an invalid projection")
            position_corrections[key] = correction
            current_state[key] = current_state[key] + correction.to(current_state[key])
            bias_key = f"{key[: -len('.weight')]}.bias"
            bias_correction = diag["bias_correction"]
            skip_bias = _apply_bias_correction(config, current_state, bias_key, bias_correction, position_corrections)
            block_row = {
                "mode": "direct_target",
                "component": component,
                "component_target": "block_boundary",
                "block_split": "joint",
                "position": pos,
                "source_coordinate": float(source_coordinates[pos]),
                "desired_norm": desired_norm,
                "effect_before_norm": effect_sq**0.5,
                "relative_residual_before": (diag["residual_norm_before"] / desired_norm) if desired_norm else 0.0,
                "relative_residual_after": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                "correction_rank": int(torch.linalg.matrix_rank(correction.double()).item()),
                "joint_lambda": {component: lambdas[component]},
                "joint_residual_relative": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                **diag,
            }
            if bool(getattr(config, "realization_diagnostics", False)):
                bias_for_pred = bias_correction if not skip_bias else torch.zeros(correction.shape[0])
                block_row.update(
                    _realization_diagnostic_fields(
                        h_batches_by_component[component],
                        correction,
                        bias_for_pred,
                        torch.eye(widths[component], dtype=torch.float32),
                        weight_before,
                        desired_norm,
                        diag["residual_norm_after"],
                    )
                )
            block_rows.append(block_row)
            results[pos] = (position_corrections, block_rows)
            continue

        # Genuine joint (>=2 component) stacked solve. `dims` is each
        # component's own INPUT feature width (H_c's last dim -- e.g.
        # mlp.c_proj's input is d_model*mlp_ratio, NOT its output width
        # `widths[c]`, which is d_model like every other residual writer's
        # OUTPUT). d_out is read off the block's own boundary output bank
        # (the shared regression target every component's H_c writes into).
        dims = [int(h_batches_by_component[c][0].shape[-1]) for c in order]
        d_out = int(out_batches[0].shape[-1])
        joint_stats = _JointBlockRidgeStatistics(dims, d_out, device=device)
        for rows in zip(*(h_batches_by_component[c] for c in order), error_batches, strict=True):
            *h_rows, error = rows
            joint_stats.update([h.reshape(-1, h.shape[-1]) for h in h_rows], error.reshape(-1, error.shape[-1]))
        lambda_list = [lambdas[c] for c in order]
        x, beta, joint_diag = joint_stats.solve(lambda_list)
        x = x.cpu()
        beta = beta.cpu().to(torch.float32)

        offsets = [0]
        for d in dims:
            offsets.append(offsets[-1] + d)
        last_writer = order[-1]
        for idx, component in enumerate(order):
            key = shim.component_key(pos, component, prefixed=True)
            correction = x[offsets[idx] : offsets[idx + 1], :].T.to(torch.float32).contiguous()
            weight_before = current_state[key].detach().clone()
            if correction.shape != current_state[key].shape or not torch.isfinite(correction).all():
                raise RuntimeError("block_split='joint' produced an invalid projection")
            position_corrections[key] = correction
            current_state[key] = current_state[key] + correction.to(current_state[key])
            bias_key = f"{key[: -len('.weight')]}.bias"
            # See the docstring: the whole joint intercept goes to the LAST
            # residual writer (mlp.c_proj in the historical two-writer case);
            # every other component's bias gets an exact zero delta.
            bias_correction = beta if component == last_writer else torch.zeros_like(beta)
            skip_bias = _apply_bias_correction(config, current_state, bias_key, bias_correction, position_corrections)
            _single_correction, single_diag = single_component_fit[component]
            block_row = {
                "mode": "direct_target",
                "component": component,
                "component_target": "block_boundary",
                "block_split": "joint",
                "position": pos,
                "source_coordinate": float(source_coordinates[pos]),
                "desired_norm": desired_norm,
                "effect_before_norm": effect_sq**0.5,
                "relative_residual_before": (joint_diag["residual_norm_before"] / desired_norm)
                if desired_norm
                else 0.0,
                "relative_residual_after": (joint_diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                "correction_rank": int(torch.linalg.matrix_rank(correction.double()).item()),
                "n_rows": joint_diag["n_rows"],
                "residual_norm_before": joint_diag["residual_norm_before"],
                "residual_norm_after": joint_diag["residual_norm_after"],
                "exact_form": bool(config.exact_form),
                "ridge_estimator": config.ridge_estimator,
                "configured_ridge_relative": float(config.ridge_relative),
                "effective_ridge_relative": single_diag["effective_ridge_relative"],
                "ridge": lambdas[component],
                "correction_norm": float(torch.linalg.norm(correction).item()),
                "bias_norm": float(torch.linalg.norm(bias_correction).item()),
                "bias_correction": bias_correction,
                "joint_lambda": dict(lambdas),
                "joint_residual_relative": (joint_diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
            }
            if bool(getattr(config, "realization_diagnostics", False)):
                bias_for_pred = bias_correction if not skip_bias else torch.zeros(correction.shape[0])
                block_row.update(
                    _realization_diagnostic_fields(
                        h_batches_by_component[component],
                        correction,
                        bias_for_pred,
                        torch.eye(widths[component], dtype=torch.float32),
                        weight_before,
                        desired_norm,
                        joint_diag["residual_norm_after"],
                    )
                )
            block_rows.append(block_row)
        results[pos] = (position_corrections, block_rows)
    return results


def _apply_bias_correction(config, current_state, bias_key, bias_correction, position_corrections) -> bool:
    """Shared ``missing_bias`` handling for a single component's bias delta
    (extracted from the ``block_split in {'none', 'backfit'}`` paths so
    ``block_split='joint'`` follows the exact same ``error``/``materialize``/
    ``skip`` contract). Mutates ``position_corrections`` in place with the
    accepted bias delta (unless skipped) and returns whether the bias was
    skipped.
    """
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
            return True
        else:
            raise RuntimeError(
                f"Target model is missing the expected bias parameter {bias_key}. "
                "Set target_residual_completion.missing_bias to 'materialize' (exact, "
                "adds the parameter) or 'skip' with exact_form=false."
            )
    bias_delta = bias_correction.to(current_state[bias_key])
    if bias_delta.shape != current_state[bias_key].shape or not torch.isfinite(bias_delta).all():
        raise RuntimeError("block_split='joint' produced an invalid bias")
    position_corrections[bias_key] = bias_delta
    current_state[bias_key] = current_state[bias_key] + bias_delta
    return False
