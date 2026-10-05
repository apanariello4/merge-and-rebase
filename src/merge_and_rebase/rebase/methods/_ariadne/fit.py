"""Ariadne fits: ``fit_direct_residual`` and sequential source endpoints."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from ....utils.cost_accounting import cost_phase_decorator
from ...discrete_layer_match import DiscreteLayerPairing
from .alignment import _check_rows, centered_rectangular_procrustes
from .capture import capture_tokens
from .config import DirectResidualConfig, order_components
from .layouts import COMPONENT_INPUT_KIND, _aligned, _component_effective_out, _layout_for, _rows

Tensor = torch.Tensor


@torch.no_grad()
def fit_direct_residual(
    target_model,
    target_base_state: Mapping[str, Tensor],
    captured: Mapping[str, Any],
    desired: Mapping[int, list[Tensor]],
    pairing: DiscreteLayerPairing,
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Fit every target position's residual-writing components independently.

    Wraps `_fit_all_positions_independent` for all ``j in range(pairing.target_depth)`` at once
    (``source_coordinate = pairing.pairing[j]``) from one shared target forward sweep. Always runs with independent
    (no-cascade) semantics regardless of ``config.cascade_order`` (a schema-parity no-op here; see
    ``tests/test_direct_residual_cascade_order.py``): nothing is mounted between fits.

    Fits at unit strength (gamma=1) and returns the raw correction; the caller applies
    `target_informed_runtime.scale_completion`, so ``strength=0`` is an exact native-target-base control.

    Returns ``(target_corrections, diagnostics)``; ``target_corrections`` only holds the weight+bias keys of
    `config.components`.
    """
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    components = order_components(config.components)
    batches = captured["target_batches"]
    target_outputs_by_position = captured["target_base_outputs_by_position"]
    if set(desired) != set(range(pairing.target_depth)):
        raise ValueError(
            "desired effects do not exactly match the pairing's target positions: "
            f"expected={sorted(range(pairing.target_depth))}, found={sorted(desired)}"
        )
    # Force independent (no-mount) semantics for the shared solver regardless of config.cascade_order.
    solver_config = replace(config, cascade_order="independent") if config.cascade_order != "independent" else config
    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        positions = list(range(pairing.target_depth))
        for j in positions:
            if j not in target_outputs_by_position:
                raise ValueError(f"Captured target reference is missing for position {j}")
        source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
        # Local import: ablations.py imports this module (stats, kernel helpers), so a top-level import would cycle.
        from .ablations import _fit_block_boundary_backfit, _fit_block_boundary_joint

        if config.block_split == "backfit":
            fitted = _fit_block_boundary_backfit(
                target_model,
                current_state,
                positions,
                source_coordinates,
                desired,
                target_outputs_by_position,
                batches,
                components,
                solver_config,
                device,
                family_adapter=family_adapter,
            )
        elif config.block_split == "joint":
            fitted = _fit_block_boundary_joint(
                target_model,
                current_state,
                positions,
                source_coordinates,
                desired,
                target_outputs_by_position,
                batches,
                components,
                solver_config,
                device,
                family_adapter=family_adapter,
            )
        else:
            # All positions are pristine (nothing is mounted between fits): share one target forward sweep.
            fitted = _fit_all_positions_independent(
                target_model,
                current_state,
                positions,
                source_coordinates,
                desired,
                target_outputs_by_position,
                batches,
                components,
                solver_config,
                device,
                family_adapter=family_adapter,
            )
        for j in positions:
            position_corrections, block_rows = fitted[j]
            target_corrections.update(position_corrections)
            for row in block_rows:
                row["source_position"] = row.pop("source_coordinate")
                # Analysis-only: which statistic Q_j was fit on; never read by any fit (golden hashes cover corrections).
                row["procrustes_source"] = config.procrustes_source
                diagnostics.append(row)
    finally:
        target_model.load_state_dict(original_state, strict=True)
    return target_corrections, diagnostics


def fit_sequential_source_endpoints(
    target_model,
    target_base_state: Mapping[str, Tensor],
    captured: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Fit the source base, mount its synthesis, then fit the fine-tuned stage.

    Q and its means are fitted once on source base/native target boundaries.
    Both endpoint modes retain the mounted synthesized base as the stage-two
    design. ``sequential_source_endpoints`` fits the mapped FT endpoint against
    that network's outputs; ``sequential_delta_on_synthesized_base`` instead
    fits the mapped source update directly on that network's activations.
    """
    if config.endpoint_construction not in {"sequential_source_endpoints", "sequential_delta_on_synthesized_base"}:
        raise ValueError("sequential endpoint fit requires a sequential endpoint_construction")
    native_outputs = captured["target_base_outputs_by_position"]
    source_base = captured["source_base_outputs"]
    source_ft = captured["source_ft_outputs"]
    mapped_base: dict[int, list[Tensor]] = {}
    mapped_ft: dict[int, list[Tensor]] = {}
    base_desired: dict[int, list[Tensor]] = {}
    alignment: dict[int, dict[str, float]] = {}
    for j in range(pairing.target_depth):
        i = pairing.pairing[j]
        native = native_outputs[j]
        s0 = _aligned(source_base[i], native)
        s1 = _aligned(source_ft[i], native)
        q, mu_s, mu_t = centered_rectangular_procrustes(_rows(s0).double(), _rows(native).double())
        q, mu_s, mu_t = q.float(), mu_s.float(), mu_t.float()
        mapped_base[j] = [(b - mu_s) @ q + mu_t for b in s0]
        mapped_ft[j] = [(f - mu_s) @ q + mu_t for f in s1]
        base_desired[j] = [m - t for m, t in zip(mapped_base[j], native, strict=True)]
        alignment[j] = {
            "pretrained_residual_norm": float(_rows(base_desired[j]).double().norm()),
            "source_update_norm": float(
                _rows([f - b for b, f in zip(mapped_base[j], mapped_ft[j], strict=True)]).double().norm()
            ),
        }
    entry_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    try:
        base_correction, base_rows = fit_direct_residual(
            target_model,
            target_base_state,
            captured,
            base_desired,
            pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
        synthesized_base = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        for key, correction in base_correction.items():
            synthesized_base[key] = synthesized_base[key] + correction.to(synthesized_base[key])
        target_model.load_state_dict(synthesized_base, strict=True)
        requests = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
        synthesized_raw = capture_tokens(
            target_model, captured["target_batches"], requests, device, family_adapter=family_adapter
        )
        synthesized_outputs = {int(j): batches for j, batches in synthesized_raw.items()}
        if config.endpoint_construction == "sequential_delta_on_synthesized_base":
            # Keep the stage-one synthesized base as the stage-two design H_syn and fit only the mapped source update;
            # the affine Procrustes means cancel between the two mapped endpoints (zero-update invariant).
            ft_desired = {
                j: [ft - base for base, ft in zip(mapped_base[j], mapped_ft[j], strict=True)]
                for j in range(pairing.target_depth)
            }
        else:
            # Historical endpoint mode: fit the mapped FT endpoint against the synthesized network's output.
            ft_desired = {
                j: [m - t for m, t in zip(mapped_ft[j], synthesized_outputs[j], strict=True)]
                for j in range(pairing.target_depth)
            }
        ft_captured = dict(captured)
        ft_captured["target_base_outputs_by_position"] = synthesized_outputs
        task_vector, ft_rows = fit_direct_residual(
            target_model,
            synthesized_base,
            ft_captured,
            ft_desired,
            pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
        # Confirm that subtracting the two *mounted* endpoints represents the
        # returned correction, subject to fp32 rounding of full model weights.
        max_endpoint_error = 0.0
        for key, correction in task_vector.items():
            ft_value = synthesized_base[key] + correction.to(synthesized_base[key])
            max_endpoint_error = max(
                max_endpoint_error,
                float(((ft_value - synthesized_base[key]).float() - correction.float()).abs().max()),
            )
        if not all(torch.isfinite(value).all() for value in task_vector.values()):
            raise RuntimeError("sequential endpoint fit produced a non-finite task vector")
        for row in base_rows:
            row["endpoint_stage"] = "pretrained"
        for row in ft_rows:
            row["endpoint_stage"] = "finetuned"
        diagnostics = {
            "alignment_by_position": alignment,
            "pretrained_correction_norm": float(
                sum(v.double().square().sum() for v in base_correction.values()).sqrt()
            ),
            "task_vector_norm": float(sum(v.double().square().sum() for v in task_vector.values()).sqrt()),
            "max_endpoint_subtraction_error": max_endpoint_error,
        }
        return task_vector, base_rows + ft_rows, diagnostics
    finally:
        target_model.load_state_dict(entry_state, strict=True)


# Above this 2-norm float64 condition number, ridge_estimator="none" refuses to solve rather than return a
# numerically meaningless "exact" fit (a conventional float64 well-posedness threshold).
_RIDGE_NONE_CONDITION_THRESHOLD = 1e8


def _clamped_eigh_inverse(sym: Tensor, *, eps: float = 1e-12) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(eigvals_clamped, eigvecs, pseudo_inverse)`` of a symmetric PSD matrix."""
    vals, vecs = torch.linalg.eigh(sym)
    vals = vals.clamp_min(0.0)
    inv = vecs @ torch.diag(torch.where(vals > eps, 1.0 / vals, torch.zeros_like(vals))) @ vecs.T
    return vals, vecs, inv


class ResidualSufficientStatistics:
    """Streaming statistics for the source-space Sylvester solve.

    ``device``, when given, moves each batch onto it before the float64 Gram accumulation in ``update()``, so the
    statistics (and ``solve()``) live there; ``None`` (default) moves nothing (historical CPU behaviour).
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = device
        self.s: Tensor | None = None
        self.g: Tensor | None = None
        self.b: Tensor | None = None
        self.sum_a: Tensor | None = None
        self.sum_e: Tensor | None = None
        self.sum_e2 = 0.0
        self.n_rows = 0
        self.m_source: int | None = None
        self.d_source: int | None = None

    @cost_phase_decorator("transformation")
    def update(self, h: Tensor, e: Tensor, t_in: Tensor | None, t_out: Tensor, row_mask: Tensor | None = None) -> None:
        """Accumulate one batch.

        ``row_mask`` (bool ``[N]``, optional) keeps only the selected rows, applied before any finiteness check;
        a batch with no selected row is a no-op. ``None`` is the unchanged (vision) behaviour.

        ``t_in=None`` means an identity input map (``direct_target`` mode): the weight lives in the target input
        coordinates of ``h`` and the normal equations reduce to ``A = H``. It is spelled ``None`` so the
        ``d_mlp``-sized identity matmul (4096x4096 per batch on ViT-L/14) is never materialized.
        """
        if row_mask is not None:
            row_mask = row_mask.to(device=h.device, dtype=torch.bool)
            if h.ndim != 2 or e.ndim != 2 or row_mask.shape != (h.shape[0],) or e.shape[0] != h.shape[0]:
                raise ValueError("row_mask must be a bool vector matching the rows of h and e")
            h, e = h[row_mask], e[row_mask.to(e.device)]
            if h.shape[0] == 0:
                return
        _check_rows(h, e, "h", "e")
        if h.ndim != 2 or e.ndim != 2 or t_out.ndim != 2 or (t_in is not None and t_in.ndim != 2):
            raise ValueError("activations and transport maps must be matrices")
        if t_in is not None and h.shape[1] != t_in.shape[1]:
            raise ValueError("transport maps do not match target activation dimensions")
        if e.shape[1] != t_out.shape[1]:
            raise ValueError("transport maps do not match target activation dimensions")
        tensors = (h, e, t_out) if t_in is None else (h, e, t_in, t_out)
        if not all(torch.isfinite(x).all() for x in tensors):
            raise ValueError("solver inputs must be finite")
        if self.device is not None:
            h = h.to(self.device)
            e = e.to(self.device)
            t_in = None if t_in is None else t_in.to(self.device)
            t_out = t_out.to(self.device)
        # C_target = t_out.T C_source t_in; X = Delta_C_source.T.
        a = h.to(torch.float64) if t_in is None else h.to(torch.float64) @ t_in.to(torch.float64).T
        lmat = t_out.to(torch.float64).T
        er = e.to(torch.float64)
        s = a.T @ a
        g = lmat.T @ lmat
        b = a.T @ er @ lmat
        sum_a = a.sum(dim=0)
        sum_e = er.sum(dim=0)
        if self.s is None:
            self.s, self.g, self.b = s, g, b
            self.sum_a, self.sum_e = sum_a, sum_e
            self.m_source, self.d_source = a.shape[1], lmat.shape[1]
            self._t_in = None if t_in is None else t_in.detach().clone()
            self._t_out = t_out.detach().clone()
        else:
            if (s.shape, g.shape, b.shape) != (self.s.shape, self.g.shape, self.b.shape):
                raise ValueError("inconsistent source dimensions across updates")
            same_in = (t_in is None) == (self._t_in is None) and (t_in is None or torch.equal(t_in, self._t_in))
            if not same_in or not torch.equal(t_out, self._t_out):
                raise ValueError("transport maps must remain fixed across streaming updates")
            self.s += s
            self.b += b
            self.sum_a += sum_a
            self.sum_e += sum_e
        self.sum_e2 += float((er * er).sum().item())
        self.n_rows += int(h.shape[0])

    def _residual_sq(self, x: Tensor, beta: Tensor, t_out64: Tensor, mu_a: Tensor, mu_e: Tensor) -> float:
        """Exact total ||(A X + 1 beta^T) t_out - E||_F^2, from raw sufficient statistics.

        No centering shortcut is used here: the reduction of the intercept
        optimum to a plain centered sum-of-squares only holds when ``t_out``
        is surjective onto its output space, which does not hold in general
        (e.g. ``d_source < d_target``). This formula is exact for any
        ``t_out`` rank.
        """
        predicted_sq = torch.trace(x.T @ self.s @ x @ self.g).item()
        cross = 2.0 * torch.sum(x * self.b).item()
        resid_no_bias = self.sum_e2 - cross + predicted_sq
        c_vec = t_out64.T @ beta
        pred_mean = mu_a @ x @ t_out64
        n = float(self.n_rows)
        bias_term = 2.0 * n * float((c_vec @ (pred_mean - mu_e)).item()) + n * float((c_vec @ c_vec).item())
        return max(0.0, resid_no_bias + bias_term)

    @cost_phase_decorator("transformation")
    def solve(
        self,
        *,
        ridge_relative: float,
        ridge_estimator: str = "fixed_relative",
        exact_form: bool = True,
        ridge_mode: str = "trace_normalized",
        ridge_absolute: float | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        if self.s is None or self.g is None or self.b is None or self.n_rows == 0:
            raise ValueError("cannot solve empty residual statistics")
        # ridge_relative must be finite and positive even for ridge_estimator="none" (which never reads it): the
        # field is shared schema surface, and relaxing the check would let an invalid value pass silently.
        if isinstance(ridge_relative, bool) or ridge_relative <= 0 or not math.isfinite(float(ridge_relative)):
            raise ValueError("ridge_relative must be finite and > 0")
        if ridge_estimator not in {"fixed_relative", "empirical_bayes", "none"}:
            raise ValueError("ridge_estimator must be 'fixed_relative', 'empirical_bayes' or 'none'")
        if ridge_mode not in {"trace_normalized", "absolute"}:
            raise ValueError("ridge_mode must be 'trace_normalized' or 'absolute'")
        if ridge_mode == "absolute":
            if isinstance(ridge_absolute, bool) or not isinstance(ridge_absolute, (int, float)):
                raise ValueError("ridge_absolute must be a finite real number for absolute ridge")
            if not math.isfinite(float(ridge_absolute)) or float(ridge_absolute) <= 0:
                raise ValueError("ridge_absolute must be finite and > 0 for absolute ridge")
            if ridge_estimator != "fixed_relative":
                raise ValueError("ridge_mode='absolute' is only valid with ridge_estimator='fixed_relative'")
        s = (self.s + self.s.T) * 0.5
        g = (self.g + self.g.T) * 0.5
        n = float(self.n_rows)
        mu_a = self.sum_a / n
        mu_e = self.sum_e / n
        t_out64 = self._t_out.to(torch.float64)
        lmat64 = t_out64.T

        if exact_form:
            # Center the sufficient statistics (Eq. 10 pattern) so the ridge-
            # penalized slope solve is unaffected by the residual/feature means.
            sc = s - torch.outer(self.sum_a, self.sum_a) / n
            bc = self.b - torch.outer(self.sum_a, mu_e @ lmat64)
            sc = (sc + sc.T) * 0.5
        else:
            sc, bc = s, self.b

        trace_sc = float(torch.trace(sc).item())
        trace_g = float(torch.trace(g).item())
        condition_number: float | None = None
        if ridge_estimator == "none":
            # Exact least squares (lambda = 0) on the centered normal equations. Unlike the ridge estimators
            # (pseudo-inverse, defined for singular sc/g), a singular or ill-conditioned system must fail loudly
            # here: the zeroed near-null directions would masquerade as a valid exact solve.
            effective_ridge_relative = 0.0
            base = 0.0
            cond_sc = float(torch.linalg.cond(sc).item()) if sc.shape[0] > 0 else 1.0
            cond_g = float(torch.linalg.cond(g).item()) if g.shape[0] > 0 else 1.0
            condition_number = max(cond_sc, cond_g)
            if not math.isfinite(condition_number) or condition_number > _RIDGE_NONE_CONDITION_THRESHOLD:
                raise ValueError(
                    f"ridge_estimator='none': the normal-equations system is singular or "
                    f"ill-conditioned (condition number {condition_number:.6e} exceeds the "
                    f"{_RIDGE_NONE_CONDITION_THRESHOLD:.0e} threshold for an exact solve); use "
                    "ridge_estimator='fixed_relative' or 'empirical_bayes' instead"
                )
        elif ridge_estimator == "empirical_bayes":
            if self.n_rows <= 1:
                raise ValueError("empirical_bayes ridge requires at least two activation rows")
            # Empirical Bayes: with Sigma_hat = S_c / (N - 1) the precision denominator is S_c + trace(Sigma_hat) I,
            # so the scalar ridge is trace(S_c) / (N - 1) (relative parameterization with d_in / (N - 1)).
            effective_ridge_relative = float(self.m_source) / float(self.n_rows - 1)
            base = trace_sc / float(self.n_rows - 1)
        else:
            effective_ridge_relative = float(ridge_relative)
            base = effective_ridge_relative * trace_sc / float(self.m_source)
        trace_normalized_lam = base * (trace_g / float(self.d_source)) if exact_form else base
        # ridge_mode="absolute" is validated to require ridge_estimator="fixed_relative".
        lam = float(ridge_absolute) if ridge_mode == "absolute" else trace_normalized_lam

        es, us, sc_inv = _clamped_eigh_inverse(sc)
        eg, ug, g_inv = _clamped_eigh_inverse(g)

        if trace_sc == 0.0 or trace_g == 0.0:
            x = torch.zeros(self.m_source, self.d_source, dtype=torch.float64, device=t_out64.device)
        else:
            denom = es[:, None] * eg[None, :] + lam
            rhs = us.T @ bc @ ug
            xhat = torch.where(denom > 0, rhs / denom, torch.zeros_like(rhs))
            x = us @ xhat @ ug.T

        if exact_form:
            beta = g_inv @ (t_out64 @ mu_e - g @ (x.T @ mu_a))
        else:
            # Explicit device: bare torch.zeros is CPU, but t_out64/x/mu_a may be CUDA (device_transform="gpu") and
            # _residual_sq does t_out64.T @ beta; only exact_form=False reaches this branch.
            beta = torch.zeros(self.d_source, dtype=torch.float64, device=t_out64.device)

        residual_sq = self._residual_sq(x, beta, t_out64, mu_a, mu_e)

        # Ridge-free (best possible) reference solve, for the reachable/unreachable split.
        x0 = sc_inv @ bc @ g_inv
        if exact_form:
            beta0 = g_inv @ (t_out64 @ mu_e - g @ (x0.T @ mu_a))
        else:
            beta0 = torch.zeros(self.d_source, dtype=torch.float64, device=t_out64.device)
        best_possible_sq = self._residual_sq(x0, beta0, t_out64, mu_a, mu_e)
        reachable_sq = max(0.0, self.sum_e2 - best_possible_sq)
        unreachable_sq = best_possible_sq

        diag: dict[str, Any] = {
            "n_rows": self.n_rows,
            "ridge": lam,
            "ridge_estimator": ridge_estimator,
            # Only populated for ridge_estimator="none"; None for the (always well-posed) regularized estimators.
            "condition_number": condition_number,
            "configured_ridge_relative": float(ridge_relative),
            "effective_ridge_relative": effective_ridge_relative,
            "ridge_mode": ridge_mode,
            "ridge_absolute": float(ridge_absolute) if ridge_absolute is not None else None,
            "ridge_trace_normalized": trace_normalized_lam,
            "trace_centered_feature_gram": trace_sc,
            "trace_output_transport_gram": trace_g,
            "residual_norm_before": self.sum_e2**0.5,
            "residual_norm_after": residual_sq**0.5,
            "reachable_residual_norm": reachable_sq**0.5,
            "unreachable_residual_norm": unreachable_sq**0.5,
            "correction_norm": float(torch.linalg.norm(x).item()),
            "bias_norm": float(torch.linalg.norm(beta).item()),
            "exact_form": exact_form,
            # Source-coordinate c_proj.bias delta, transported like the weight's output side
            # (t_out.T @ bias_correction, matching theseus._transport_bias's `delta_vec @ t_out`).
            "bias_correction": beta.to(torch.float32),
        }
        return x.T.to(torch.float32), diag


def _realization_diagnostic_fields(
    h_batches, correction, bias_for_pred, effective_out, weight_before, desired_norm, residual_norm_after, *, stats=None
):
    """Analysis-only per-component fit fields, gated on ``realization_diagnostics`` (never read by any fit).

    Shared by every block_boundary fit path. Fields: ``fit_relative_residual``, ``target_norm``, ``update_norm``,
    ``relative_update_norm`` (denominator = pre-correction weight slice) and ``realized_target_norm_ratio``,
    computed EXACTLY from the accumulated fit (``H @ correction^T + bias`` pushed through the SAME
    ``effective_out`` -- LayerScale -- the solve used) from the resident ``h_batches`` bank, with no second
    capture sweep.
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
    """Fit every ``(position, component)`` pair for ``cascade_order='independent'`` from ONE shared target sweep.

    Under independent mode the target model is never mutated between fits (the mount in
    ``_fit_direct_target_position`` is skipped), so every pair sees the same pristine ``target_model`` that
    ``current_state`` describes and a single ``capture_tokens`` sweep serves all solves: one ``"out"`` (boundary)
    request per position and one ``"h"`` request per pair (``COMPONENT_INPUT_KIND[component]``).

    Per-pair solving is byte-identical to ``_fit_direct_target_position``'s independent-mode body (same
    ``ResidualSufficientStatistics`` accumulation, ridge solve, ``missing_bias`` handling and ``block_rows`` fields,
    including ``measured_residual_norm_after`` on every component but the position's last, which is the next
    component's own pre-fit residual). The pristine-effect assertion runs for ALL pairs; it can only raise on a bug
    (stale reference bank or mutated base), never change a returned number. Intra-position "replay" (mounting
    attn.out_proj before mlp.c_proj) is deliberately NOT implemented: it would change independent-mode numerics.

    Returns ``{position: (position_corrections, block_rows)}``, as ``len(positions)`` separate
    ``_fit_direct_target_position`` calls would for identical inputs.
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
    """Everything after one component's accumulation loop in ``_fit_all_positions_independent``: pristine-effect
    check, ridge solve, state/bookkeeping updates, missing-bias handling and the block_row diagnostic (plus optional
    realization diagnostics). Mutates ``position_corrections``, ``current_state`` and ``block_rows`` in place.
    """
    # Pristine-effect check holds for every (position, component) pair under independent mode.
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
        # As in _fit_direct_target_position: the previous component's row records this component's pre-fit residual
        # as ``measured_residual_norm_after`` (not a post-mount measurement; nothing is mounted).
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


def _task_vector_sha256(sd: Mapping[str, torch.Tensor]) -> str:
    """Stable CPU hash of a task-vector-shaped tensor mapping.

    Algorithm: sorted keys, dtype, shape, raw bytes. This is the single
    implementation; ``vision_rebase._state_dict_sha256`` is an alias of it, so any
    caller hashing the same dict through either name gets the same digest.
    """
    digest = hashlib.sha256()
    for key in sorted(sd):
        value = sd[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(memoryview(value.numpy()))
    return digest.hexdigest()
