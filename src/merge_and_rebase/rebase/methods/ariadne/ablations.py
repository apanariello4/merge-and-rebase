"""Ablation arms: intra-block backfit/joint solvers and task-vector strength scaling (not the headline method)."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from statistics import median
from typing import Any

import torch
from torch import nn

from ....utils.cost_accounting import cost_phase_decorator
from .capture import (
    _assert_layerscale_identity,
    _stock_mha_out_proj_input,
    _verify_recomputed_attention_input,
    capture_tokens,
)
from .config import COMPONENT_FORWARD_ORDER, DirectResidualConfig, order_components
from .diagnostics import measure_direct_residual_realization
from .fit import ResidualSufficientStatistics, _realization_diagnostic_fields, _task_vector_sha256
from .layouts import COMPONENT_INPUT_KIND, _component_weight_bias, _family_bias_key, _layout_for

Tensor = torch.Tensor


def _replay_block_components(shim, local_block, x_batches, components, device):
    """Run ``local_block`` directly over ``x_batches``, hooking the input of each of ``components``
    (``attn.out_proj`` / ``mlp.c_proj``) plus the block output.

    ``local_block`` is a standalone block called with one positional tensor, as in a full-model sweep.
    Returns ``(component_h_batches, out_batches)``, CPU float32, same convention as ``capture_tokens``.
    """
    handles = []
    component_h: dict[str, list[torch.Tensor]] = {c: [] for c in components}
    try:
        if "attn.out_proj" in components:
            attn = shim.attn_module(local_block)
            if isinstance(attn, nn.MultiheadAttention):
                # Stock nn.MultiheadAttention applies out_proj functionally: hook the attention
                # and recompute the rows it fed the projection.
                def attn_hook(mod, args, kwargs, value, *, store=component_h["attn.out_proj"]):
                    rows = _stock_mha_out_proj_input(mod, args, kwargs)
                    _verify_recomputed_attention_input(mod, rows, value)
                    store.append(rows.detach().float().cpu().clone())

                handles.append(attn.register_forward_hook(attn_hook, with_kwargs=True))
            else:
                # Plain attention wrapper: out_proj is an ordinary submodule, hook it directly
                # (same fallback as capture_tokens).
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
    """``||D_j - (block_j(X_j^0; deltas mounted) - T_j^0)||_F^2``: data-fit term of the backfit objective J(Delta).

    ``local_block64`` is a dedicated float64 replica used ONLY for this measurement: the accept/backtrack
    comparison of two nearly equal J values must not be flipped by float32 forward-pass rounding noise.
    ``_mount_component`` upcasts the float32 base/delta tensors to float64 automatically.
    """
    _mount_all_deltas(shim, local_block64, order, base, deltas)
    t_all = [local_block64(x.to(device).double()).detach().cpu().clone() for x in x_batches]
    return sum(
        float(((d.double() - (t - t0.double())) ** 2).sum().item())
        for d, t, t0 in zip(d_batches, t_all, t0_batches, strict=True)
    )


def _backfit_ridge_penalty(order, deltas, lambdas):
    """``sum_c lambda_c * ||Delta W_c||_F^2``: the exact weight penalty ``ResidualSufficientStatistics.solve``
    minimizes (bias unpenalized), with each component's FROZEN round-1 ``lambda_c`` from ``lambdas``.
    A component with no entry yet contributes 0 (exact: its delta is still zero).
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
    """Intra-block Gauss-Seidel backfit for ``component_target='block_boundary'``, ``block_split='backfit'``.

    Ablation arm (not used in current experiments; kept); vision-only. Positions are independent (pristine upstream
    target). Within a position all components share the block target ``D_j``; each component is refit against
    the residual left with the other components mounted on a replayed local block copy:

        E_c = D_j - (block_j(X_j^0; others mounted, c at its native base) - T_j^0)

    Safeguarded (monotone) objective, measured on the actual block replay:

        J(Delta) = ||D_j - (block_j(X_j^0; Delta mounted) - T_j^0)||_F^2 + sum_c lambda_c ||Delta W_c||_F^2.

    ``lambda_c`` is ``solve``'s ``diag["ridge"]`` FROZEN at its first-sweep value so J is a fixed function of
    Delta (bias unpenalized). Each sub-step computes the candidate from scratch (no warm start), accepts it only
    if J decreases, else backtracks Delta_c = old + eta (new - old) (weight and bias) for eta = 1, 1/2, ..., 1/256;
    if none decreases J, Delta_c keeps its old value (eta recorded as 0). Sweeps stop when the relative J decrease
    over a sweep is below ``config.backfit_tol`` or at ``config.backfit_max_iters``. The relative residual ``r`` is
    logged every sweep (``backfit_residual_trace``) but does not drive stopping.

    With one component the first-sweep ``E_c`` equals ``D_j``, so the fit equals the single-component
    ``_fit_all_positions_independent`` call PROVIDED the replayed ``H_c``/output bitwise reproduce the live
    target-model capture; this is checked, not assumed (``block_replay_bitwise`` in the diagnostics).
    The target model is never mutated: mounts happen on a ``copy.deepcopy`` of the block, discarded per position.
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

    # Capture the pristine block input X_j^0 for every position in ONE target sweep
    # (forward-hook INPUT of the block module).
    block_input_requests = {f"{pos}.block_input": (pos, "block_input") for pos in positions}
    block_inputs = capture_tokens(target_model, batches, block_input_requests, device, family_adapter=family_adapter)

    results: dict[int, tuple[dict, list]] = {}
    for pos in positions:
        x_batches = block_inputs[f"{pos}.block_input"]
        t0_batches = target_output_batches[pos]
        d_batches = desired_batches[pos]
        block = shim.blocks(target_model)[pos]
        local_block = copy.deepcopy(shim.block_module(block)).to(device).eval()

        # Device convention: base/deltas (and everything derived) live on CPU float32, like
        # ResidualSufficientStatistics.solve() output and _fit_all_positions_independent's current_state.
        # base[c] needs the explicit .cpu(): otherwise it inherits local_block's device and
        # `base[c2][0] + w2` (CPU delta) would raise on CUDA, which a CPU-only run never surfaces.
        # _mount_component is the only place tensors move back onto `device`.
        base: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        for c in order:
            w, b, row_slice = _component_weight_bias(shim, local_block, c)
            if row_slice is not None:
                raise RuntimeError("block_split='backfit' components must not be packed (row_slice must be None)")
            base[c] = (w.detach().cpu().clone(), None if b is None else b.detach().cpu().clone())
        scale_modules = {c: shim.component_scale_module(local_block, c) for c in order}
        # Float64 replica for the J(Delta) measurement only (see _backfit_data_fit_sq); copied
        # while local_block is still pristine.
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
        # Backtracking factors: full step, then up to 8 halvings; eta=0 (keep old delta) is the implicit fallback.
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

                # Freeze lambda_c at its first-sweep value so J stays a FIXED function of Delta;
                # solve()'s per-sweep ridge only shapes the candidate, not the accept test.
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
            # Full-sweep J and ridge-free data-fit norm r (diagnostic) with every current delta mounted.
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
    """Streaming Gram/cross accumulator for the joint (O, D)-stacked ridge fit of ``block_split='joint'``.

    Mirrors ``ResidualSufficientStatistics`` at ``t_in=None, t_out=I`` (LayerScale is asserted Identity before
    use) with a block-diagonal ridge vector over the stacked feature dim ``sum(dims)``. Accumulates the
    ``(d_total, d_total)`` Gram and ``(d_total, d_out)`` cross statistics in float64 batch by batch; no
    token-sized design matrix is materialized.
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
        """Exact centered normal-equation solve ``(S_c + diag(lambda)) X = B_c`` with a block-diagonal ridge.

        Returns ``(x, beta, diag)``: ``x`` is ``(d_total, d_out)``, stacked and UNTRANSPOSED (callers split it by
        ``self.dims`` and transpose each block to ``(out, in)``); ``beta`` is the shared ``(d_out,)`` intercept,
        not ridge-penalized (as in ``ResidualSufficientStatistics.solve``). Float64 throughout.
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
        # Exact ||A x + 1 beta^T - E||_F^2 from raw sufficient statistics, as in
        # ResidualSufficientStatistics._residual_sq with t_out=I (g=I, lmat=I).
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
    """Closed-form joint ridge fit for ``component_target='block_boundary'``, ``block_split='joint'`` (ablation).

    Solves ONE ridge under the first-order approximation that the MLP does not respond to a change in
    ``attn.out_proj``; with ``H_O``/``H_D`` the pristine ``attn.out_proj``/``mlp.c_proj`` inputs (same captures
    as ``_fit_all_positions_independent``) and ``D_j`` the block-boundary desired effect,

        min_{Wo,Wd,beta} ||H_O Wo^T + H_D Wd^T + 1 beta^T - D_j||_F^2 + lambda_O ||Wo||_F^2 + lambda_D ||Wd||_F^2.

    ``components`` must be a non-empty subset of ``{"attn.out_proj", "mlp.c_proj"}`` (enforced by
    ``config.parse_direct_residual_config``). Positions are independent and pristine, so ``E_j == D_j``.

    lambda_c: exactly the value the component's standalone ``block_split='none'`` fit would use
    (``ResidualSufficientStatistics.solve`` ``diag["ridge"]``); pinned by
    ``tests/test_direct_residual_joint.py::test_joint_lambda_matches_single_component_solver_ridge``.
    Bias: the intercept is not identifiable between the two biases; the WHOLE intercept goes to the LAST residual
    writer (``mlp.c_proj.bias``) and ``attn.out_proj.bias`` gets a zero delta (its bias feeds the MLP on the real
    nonlinear block, which this first-order model ignores).
    LayerScale: asserted ``nn.Identity`` on ``ls_1``/``ls_2`` (``_assert_layerscale_identity``).
    Single component: returns that component's standalone ``block_split='none'`` solve verbatim (the same
    ``ResidualSufficientStatistics`` call), not merely a numerically close fit.
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

        # Block is pristine at this position (independent mode): the pre-fit effect is shared by
        # all components, so it is measured once.
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
            # Single component: just the standalone block_split='none' fit (see docstring), reusing
            # the solve above.
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

        # Genuine joint (>=2 components) solve. dims = each component's INPUT feature width (H_c last dim;
        # mlp.c_proj: d_model*mlp_ratio, not its output width); d_out from the block boundary output bank.
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
            # Whole joint intercept goes to the LAST residual writer; others get an exact zero bias delta.
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
    """Shared ``missing_bias`` handling for one component's bias delta (error/materialize/skip contract).

    Updates ``position_corrections`` and ``current_state`` with the accepted bias delta (unless skipped);
    returns whether the bias was skipped.
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


# ||D_j|| below this has no reliable ratio r_j; per_block keeps s_j frozen for that position/iteration.
_TV_SCALING_D_NORM_EPS = 1e-8


def _tau_frobenius_norm(sd: Mapping[str, Tensor]) -> float:
    total_sq = 0.0
    for value in sd.values():
        total_sq += float((value.detach().double() ** 2).sum().item())
    return total_sq**0.5


def _position_delta(
    shim, position: int, components: tuple[str, ...], target_corrections: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """The subset of ``target_corrections`` (weight + bias, unsliced) belonging to block position ``position``.

    Each (position, component) maps to a distinct state-dict key, so no row-slicing is needed (unlike
    ``ariadne.diagnostics._family_delta_state`` for packed q/k/v rows).
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
    """Label-free rescaling of a unit-strength Direct Residual task vector (``config.tv_scaling``).

    Measurement primitive: ``measure_direct_residual_realization`` (mount, capture block-boundary outputs over the
    SAME calibration batches as the fit, compare to the pristine base and to ``D_j``, restore-and-hash-verify);
    ``r_j = joint_delta_norm_over_desired = ||delta T_j||_F / (||D_j||_F + eps)`` for the mounted delta.
    ``measure_fn(delta) -> {j: realization row}`` replaces it when given (streaming path, bound to its own
    calibration sweep); ``captured``/``desired`` are then unused.

    ``tv_scaling == "none"`` is a strict no-op (returns ``target_corrections`` by value); callers must still gate
    on it themselves for the golden-hash-pinned default path. Other modes require ``block_split == 'none'``.
    Returns ``(scaled_corrections, diagnostics)``; diagnostics are recorded verbatim (or nested) under the run
    summary's additive ``tv_scaling_by_task`` key.
    """
    if config.tv_scaling == "none":
        return dict(target_corrections), {"mode": "none"}
    if config.block_split != "none":
        # Also rejected by parse_direct_residual_config; re-checked for hand-built configs.
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

    # per_block: scalars s_j via a simultaneous (Jacobi) fixed-point update: every s_j comes from the
    # SAME mounted-combination measurement, never sequentially (Gauss-Seidel).
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
        # max_j|r_j - 1| per iteration, kept as a list so an oscillating trace is visible.
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
