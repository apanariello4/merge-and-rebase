"""Streaming (low-memory) Ariadne fit path and its diagnostics."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from ....utils.cost_accounting import cost_phase_decorator
from ...discrete_layer_match import DiscreteLayerPairing
from .alignment import (
    _cross_rank,
    _derive_block_seed,
    _procrustes_from_cross,
    _random_isometry_map,
    _validate_alignment_options,
    procrustes_rank_diagnostics,
)
from .capture import iter_capture_block_gradients, iter_capture_tokens, paired_calibration
from .config import DirectResidualConfig, order_components
from .diagnostics import measure_direct_residual_realization_streaming
from .fit import ResidualSufficientStatistics, _finalize_independent_component
from .layouts import COMPONENT_INPUT_KIND, _aligned, _component_effective_out, _layout_for

Tensor = torch.Tensor


class _StreamingCrossCovariance:
    """Chan's pairwise online accumulator for a centered cross-covariance.

    Equivalent to ``(X - mean_x).T @ (Y - mean_y)`` over all rows at once (as `centered_rectangular_procrustes`
    computes it) in O(1)-in-batches memory. Kept on ``device`` in float64; inputs must already be float64.
    """

    def __init__(self, device=None, *, track_source_gram: bool = False) -> None:
        self.device = device
        self.track_source_gram = track_source_gram
        self.n = 0
        self.mean_x: Tensor | None = None
        self.mean_y: Tensor | None = None
        self.c: Tensor | None = None
        self.xx: Tensor | None = None
        self.weight = 0.0

    @cost_phase_decorator("transformation")
    def update(self, x: Tensor, y: Tensor, weights: Tensor | None = None, row_mask: Tensor | None = None) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError("cross-covariance update requires matching row counts")
        if row_mask is not None:
            row_mask = row_mask.to(dtype=torch.bool)
            if row_mask.shape != (x.shape[0],):
                raise ValueError("row_mask must be a bool vector over the rows of x")
            x, y = x[row_mask.to(x.device)], y[row_mask.to(y.device)]
            weights = None if weights is None else weights[row_mask.to(weights.device)]
        n_b = int(x.shape[0])
        if n_b == 0:
            return
        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)
        if weights is None:
            # Preserve historical uniform streaming arithmetic exactly.
            mean_x_b = x.mean(dim=0)
            mean_y_b = y.mean(dim=0)
            c_b = (x - mean_x_b).T @ (y - mean_y_b)
            if self.n == 0:
                self.n, self.weight, self.mean_x, self.mean_y, self.c = n_b, float(n_b), mean_x_b, mean_y_b, c_b
                if self.track_source_gram:
                    self.xx = (x - mean_x_b).T @ (x - mean_x_b)
                return
            n_a = self.n
            n = n_a + n_b
            dx = mean_x_b - self.mean_x
            dy = mean_y_b - self.mean_y
            self.c = self.c + c_b + (n_a * n_b / n) * torch.outer(dx, dy)
            if self.track_source_gram:
                xx_b = (x - mean_x_b).T @ (x - mean_x_b)
                self.xx = self.xx + xx_b + (n_a * n_b / n) * torch.outer(dx, dx)
            self.mean_x = self.mean_x + dx * (n_b / n)
            self.mean_y = self.mean_y + dy * (n_b / n)
            self.n = n
            self.weight = float(n)
            return
        w_b_rows = weights.to(device=x.device, dtype=x.dtype)
        if w_b_rows.shape != (n_b,) or (w_b_rows < 0).any() or not torch.isfinite(w_b_rows).all():
            raise ValueError("cross-covariance weights must be finite nonnegative row weights")
        w_b = float(w_b_rows.sum().item())
        if w_b <= 0:
            return
        mean_x_b = (x * w_b_rows[:, None]).sum(0) / w_b
        mean_y_b = (y * w_b_rows[:, None]).sum(0) / w_b
        c_b = (x - mean_x_b).T @ ((y - mean_y_b) * w_b_rows[:, None])
        xx_b = (x - mean_x_b).T @ ((x - mean_x_b) * w_b_rows[:, None]) if self.track_source_gram else None
        if self.weight == 0:
            self.n, self.weight, self.mean_x, self.mean_y, self.c, self.xx = n_b, w_b, mean_x_b, mean_y_b, c_b, xx_b
            return
        w_total = self.weight + w_b
        dx = mean_x_b - self.mean_x
        dy = mean_y_b - self.mean_y
        self.c = self.c + c_b + (self.weight * w_b / w_total) * torch.outer(dx, dy)
        if self.track_source_gram:
            self.xx = self.xx + xx_b + (self.weight * w_b / w_total) * torch.outer(dx, dx)
        self.mean_x = self.mean_x + dx * (w_b / w_total)
        self.mean_y = self.mean_y + dy * (w_b / w_total)
        self.weight = w_total
        self.n += n_b

    def cross(self) -> Tensor:
        if self.c is None:
            raise ValueError("cannot compute cross-covariance of zero batches")
        return self.c


def prepare_direct_residual_streaming(
    source_base_model,
    target_base_model,
    source_loader,
    target_loader,
    pairing: DiscreteLayerPairing,
    *,
    num_batches: int,
    seed: int | None,
    device,
    family_adapter=None,
    procrustes_source: str = "activation",
    source_recipe=None,
    target_recipe=None,
    source_ft_model=None,
    alignment_map: str = "polar",
    alignment_row_weighting: str = "uniform",
    alignment_seed: int = 0,
    fidelity_alignment_diagnostics: bool = False,
) -> dict[str, Any]:
    """Streaming (Pass A) equivalent of ``capture_paired_boundary_activations`` + ``compute_desired_effects``.

    Accumulates each position's Procrustes cross-covariance batch by batch (no resident boundary banks) and solves
    once per position at the end. Uses the identical `paired_calibration` call (same ``num_batches``/``seed`` ->
    identical sample IDs) and `_aligned` helper as the resident path, so the only difference is when the map is
    extracted from the (mathematically identical) centered cross-covariance.

    ``procrustes_source="gradient"`` also runs the `iter_capture_block_gradients` generators in lockstep and fits
    ``Q_j`` on the accumulated centered GRADIENT cross-covariance, as the resident path does. The activation
    accumulators always run: they give the target fingerprints, the means ``mu_s``/``mu_t``
    (``residual_target="transported_endpoint"``) and the activation-space map used by the diagnostics.

    Returns a dict consumed by `fit_direct_residual_streaming`: ``q_by_position``, per-(batch, position)
    ``fingerprints`` of the target boundary activations (Pass B uses them to detect a target model mutated between
    passes), activation-space maps and means, the replayed calibration batches and the calibration metadata.
    ``procrustes_diagnostics`` is the scalar-only per-position dict merged into the diagnostics rows: always the
    rank diagnostics of the cross-covariance solved from and ``procrustes_source``; in gradient mode also the
    activation-vs-gradient overlap and map distance; with ``fidelity_alignment_diagnostics`` also the alignment_*
    scalars. The q/mu_s/mu_t tensors are not duplicated into rows (unlike the resident path); they stay in
    ``q_by_position`` / ``source_mean_by_position`` / ``target_mean_by_position``.
    """
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    if pairing.source_depth < 1:
        raise ValueError("pairing.source_depth must be positive")
    if procrustes_source not in {"activation", "gradient"}:
        raise ValueError("procrustes_source must be 'activation' or 'gradient'")
    _validate_alignment_options(alignment_map, alignment_row_weighting, procrustes_source)
    gradient_mode = procrustes_source == "gradient"
    if gradient_mode and (source_recipe is None or target_recipe is None):
        raise ValueError("procrustes_source='gradient' requires source_recipe and target_recipe")
    source_batches, target_batches, metadata = paired_calibration(
        source_loader, target_loader, num_batches=num_batches, seed=seed, family_adapter=family_adapter
    )
    distinct_source_indices = sorted(set(pairing.pairing))
    src_requests = {str(i): (i, "boundary") for i in distinct_source_indices}
    tgt_requests = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
    accumulators = {
        j: _StreamingCrossCovariance(device=device, track_source_gram=alignment_map == "ridge")
        for j in range(pairing.target_depth)
    }
    grad_accumulators = (
        {j: _StreamingCrossCovariance(device=device) for j in range(pairing.target_depth)} if gradient_mode else {}
    )
    fingerprints: dict[tuple[int, int], tuple[float, float]] = {}

    if alignment_row_weighting == "delta_magnitude" and source_ft_model is None:
        raise ValueError("delta_magnitude streaming alignment requires source_ft_model")

    gens = {
        "src": iter_capture_tokens(
            source_base_model, source_batches, src_requests, device, family_adapter=family_adapter, store_device=device
        ),
        "tgt": iter_capture_tokens(
            target_base_model, target_batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
        ),
    }
    if alignment_row_weighting == "delta_magnitude":
        gens["ft"] = iter_capture_tokens(
            source_ft_model, source_batches, src_requests, device, family_adapter=family_adapter, store_device=device
        )
    if gradient_mode:
        gens["src_grad"] = iter_capture_block_gradients(
            source_base_model,
            source_batches,
            {str(i): i for i in distinct_source_indices},
            source_recipe,
            device,
            family_adapter=family_adapter,
        )
        gens["tgt_grad"] = iter_capture_block_gradients(
            target_base_model,
            target_batches,
            {str(j): j for j in range(pairing.target_depth)},
            target_recipe,
            device,
            family_adapter=family_adapter,
        )
    names = list(gens)
    with contextlib.ExitStack() as stack:
        for gen in gens.values():
            stack.enter_context(contextlib.closing(gen))
        for k, values in enumerate(zip(*(gens[n] for n in names), strict=True)):
            by_name = dict(zip(names, values, strict=True))
            src, tgt = by_name["src"], by_name["tgt"]
            for j in range(pairing.target_depth):
                i = pairing.pairing[j]
                t = tgt[str(j)]
                x = _aligned([src[str(i)]], [t])[0]
                x_rows = x.reshape(-1, x.shape[-1]).double()
                y_rows = t.reshape(-1, t.shape[-1]).double()
                row_weights = None
                if alignment_row_weighting == "cls_balanced":
                    if x.shape[1] < 2:
                        raise ValueError("cls_balanced alignment requires a CLS token and at least one patch token")
                    row_weights = torch.full(x.shape[:2], 0.5 / (x.shape[1] - 1), dtype=torch.float64, device=x.device)
                    row_weights[:, 0] = 0.5
                elif alignment_row_weighting == "delta_magnitude":
                    sf = by_name["ft"][str(i)]
                    sf = _aligned([sf], [t])[0]
                    delta_norms = torch.linalg.vector_norm(sf.double() - x.double(), dim=-1)
                    means = delta_norms.mean(dim=1, keepdim=True)
                    scale = torch.where(
                        means > 0,
                        (delta_norms / means.clamp_min(torch.finfo(delta_norms.dtype).tiny)).clamp(0.25, 4.0),
                        torch.ones_like(delta_norms),
                    )
                    row_weights = scale / scale.sum(dim=1, keepdim=True)
                accumulators[j].update(x_rows, y_rows, None if row_weights is None else row_weights.reshape(-1))
                t64 = t.double()
                fingerprints[(k, j)] = (float(t64.sum().item()), float((t64**2).sum().item()))
                if gradient_mode:
                    g_t = by_name["tgt_grad"][str(j)]
                    g_s = _aligned([by_name["src_grad"][str(i)]], [g_t])[0]
                    grad_accumulators[j].update(
                        g_s.reshape(-1, g_s.shape[-1]).double(), g_t.reshape(-1, g_t.shape[-1]).double()
                    )

    def solve_map(acc, position):
        """Return ``(q, rank, polar_derived, map_scalars)`` for one position."""
        cross = acc.cross()
        if alignment_map == "polar":
            q, rank = _procrustes_from_cross(cross, return_rank=True)
            scalars: dict[str, Any] = {"alignment_map": "polar", "alignment_row_weighting": alignment_row_weighting}
            if alignment_row_weighting != "uniform":
                # Resident normalises the row weights to total mass 1; Chan's weights sum to the image count.
                scalars["weighted_cross_frobenius"] = float(torch.linalg.norm(cross).item() / acc.weight)
            return q, rank, True, scalars
        if alignment_map == "random_isometry":
            # Same per-block seed derivation as the resident path, so streaming and resident produce the identical
            # random map for the same alignment_seed; only the direction is randomized.
            q = _random_isometry_map(tuple(cross.shape), seed=_derive_block_seed(alignment_seed, position))
            scalars = {
                "alignment_map": "random_isometry",
                "alignment_row_weighting": "uniform",
                "alignment_seed_used": int(_derive_block_seed(alignment_seed, position)),
            }
            return q, _cross_rank(cross), False, scalars
        assert acc.xx is not None
        trace = float(torch.trace(acc.xx).item())
        lam = trace / max(1, acc.n - 1)
        scalars = {
            "alignment_map": "ridge",
            "alignment_row_weighting": "uniform",
            "ridge": lam,
            "source_trace": trace,
        }
        if trace == 0:
            return torch.zeros_like(cross), _cross_rank(cross), False, scalars
        eye = torch.eye(acc.xx.shape[0], dtype=acc.xx.dtype, device=acc.xx.device)
        return torch.linalg.solve(acc.xx + lam * eye, cross), _cross_rank(cross), False, scalars

    solved = {j: solve_map(acc, j) for j, acc in accumulators.items()}
    activation_q64 = {j: v[0].cpu() for j, v in solved.items()}
    procrustes_diagnostics: dict[int, dict[str, Any]] = {}
    for j, (q64, rank, polar_derived, scalars) in solved.items():
        row = {
            "procrustes_source": "activation",
            **procrustes_rank_diagnostics(rank, q64.shape[0], q64.shape[1], polar_derived=polar_derived),
        }
        if fidelity_alignment_diagnostics:
            row.update(
                {
                    "alignment_q_frobenius": float(torch.linalg.norm(activation_q64[j]).item()),
                    "alignment_q_rank": int(torch.linalg.matrix_rank(activation_q64[j])),
                    **scalars,
                }
            )
        procrustes_diagnostics[j] = row
    gradient_q64: dict[int, Tensor] = {}
    if gradient_mode:
        q_by_position = {}
        for j, acc in grad_accumulators.items():
            cross = acc.cross()
            q64, grad_rank = _procrustes_from_cross(cross, return_rank=True)
            q64 = q64.cpu()
            gradient_q64[j] = q64
            q_by_position[j] = q64.float()
            d_min = min(q64.shape)
            procrustes_diagnostics[j] = {
                "procrustes_source": "gradient",
                **procrustes_rank_diagnostics(grad_rank, q64.shape[0], q64.shape[1]),
                "activation_gradient_procrustes_overlap": float(((activation_q64[j].T @ q64).norm() ** 2) / d_min),
                "activation_gradient_map_distance": float(
                    torch.linalg.norm(activation_q64[j] - q64) / (2.0 * d_min) ** 0.5
                ),
            }
    else:
        q_by_position = {j: q.float() for j, q in activation_q64.items()}
    return {
        "q_by_position": q_by_position,
        "procrustes_source": procrustes_source,
        "alignment_map": alignment_map,
        "alignment_row_weighting": alignment_row_weighting,
        "procrustes_diagnostics": procrustes_diagnostics,
        "activation_q64_by_position": activation_q64,
        "gradient_q64_by_position": gradient_q64,
        "source_mean_by_position": {j: acc.mean_x.cpu() for j, acc in accumulators.items()},
        "target_mean_by_position": {j: acc.mean_y.cpu() for j, acc in accumulators.items()},
        "fingerprints": fingerprints,
        "source_batches": source_batches,
        "target_batches": target_batches,
        "calibration": metadata,
        "distinct_source_indices": distinct_source_indices,
    }


@torch.no_grad()
def fit_direct_residual_streaming(
    target_model,
    target_base_state: Mapping[str, Tensor],
    source_base_model,
    source_ft_model,
    prepared: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Streaming (Pass B) equivalent of ``fit_direct_residual``.

    Fits every position's residual-writing components from chunked capture sweeps that accumulate
    `ResidualSufficientStatistics` batch by batch. Requires the config already validated by
    `parse_direct_residual_config` for ``activation_storage='streaming'`` (``component_target='block_boundary'``,
    ``block_split='none'``). Realization diagnostics are measured separately by the caller (``method.py``).

    Mirrors ``fit_direct_residual``'s wrapper: same position checks, forced ``cascade_order='independent'``,
    save/load/restore-in-``finally`` of the target model's state, and ``source_coordinate`` -> ``source_position``
    renaming in position order. Each chunk of ``config.streaming_position_chunk`` positions (``None`` -> all) gets
    a lockstep sweep over source_base / source_ft / target `iter_capture_tokens` generators, re-deriving
    ``desired = (f - b) @ q_j`` per batch from `prepared`'s maps and checking the captured target boundary against
    `prepared`'s fingerprint before using it as ``base_out`` (no resident base-output bank to diff against).
    """
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    components = order_components(config.components)
    batches = prepared["target_batches"]
    source_batches = prepared["source_batches"]
    q_by_position = prepared["q_by_position"]
    fingerprints = prepared["fingerprints"]
    residual_target = config.residual_target
    if residual_target == "transported_endpoint":
        if prepared.get("procrustes_source", "activation") != "activation":
            raise ValueError("residual_target='transported_endpoint' requires procrustes_source='activation'")
        mu_s_by_position = {j: m.float() for j, m in prepared["source_mean_by_position"].items()}
        mu_t_by_position = {j: m.float() for j, m in prepared["target_mean_by_position"].items()}
    positions = list(range(pairing.target_depth))
    for j in positions:
        if j not in q_by_position:
            raise ValueError(f"Missing prepared Procrustes map for position {j}")
    source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
    solver_config = replace(config, cascade_order="independent") if config.cascade_order != "independent" else config
    shim = _layout_for(family_adapter)

    chunk_size = config.streaming_position_chunk or len(positions)
    chunks = [positions[start : start + chunk_size] for start in range(0, len(positions), chunk_size)]

    unique_models: dict[int, Any] = {}
    for m in (target_model, source_base_model, source_ft_model):
        unique_models.setdefault(id(m), m)
    originals = {mid: (next(m.parameters()).device, m.training) for mid, m in unique_models.items()}

    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        for m in unique_models.values():
            m.to(device).eval()

        for chunk in chunks:
            distinct_i_needed = sorted({pairing.pairing[pos] for pos in chunk})
            src_requests = {str(i): (i, "boundary") for i in distinct_i_needed}
            tgt_requests: dict[str, tuple[int, str]] = {}
            for pos in chunk:
                tgt_requests[f"{pos}.out"] = (pos, "boundary")
                for component in components:
                    tgt_requests[f"{pos}.{component}.h"] = (pos, COMPONENT_INPUT_KIND[component])

            stats_by: dict[tuple[int, str], ResidualSufficientStatistics] = {}
            desired_sq: dict[tuple[int, str], float] = {}
            effect_sq: dict[tuple[int, str], float] = {}
            effective_out: dict[tuple[int, str], Tensor] = {}
            for pos in chunk:
                for component in components:
                    key = shim.component_key(pos, component, prefixed=True)
                    width = int(current_state[key].shape[0])
                    pair = (pos, component)
                    effective_out[pair] = _component_effective_out(shim, target_model, pos, component, width)
                    stats_by[pair] = ResidualSufficientStatistics(device=device)
                    desired_sq[pair] = 0.0
                    effect_sq[pair] = 0.0

            sb_gen = iter_capture_tokens(
                source_base_model,
                source_batches,
                src_requests,
                device,
                family_adapter=family_adapter,
                store_device=device,
            )
            sf_gen = iter_capture_tokens(
                source_ft_model,
                source_batches,
                src_requests,
                device,
                family_adapter=family_adapter,
                store_device=device,
            )
            tgt_gen = iter_capture_tokens(
                target_model, batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
            )
            with contextlib.closing(sb_gen), contextlib.closing(sf_gen), contextlib.closing(tgt_gen):
                for k, (sb, sf, tgt) in enumerate(zip(sb_gen, sf_gen, tgt_gen, strict=True)):
                    for pos in chunk:
                        i = pairing.pairing[pos]
                        out = tgt[f"{pos}.out"]
                        out64 = out.double()
                        fp_sum = float(out64.sum().item())
                        fp_sumsq = float((out64**2).sum().item())
                        exp_sum, exp_sumsq = fingerprints[(k, pos)]
                        scale = math.sqrt(exp_sumsq) if exp_sumsq > 0 else 1.0
                        sumsq_ok = math.isclose(fp_sumsq, exp_sumsq, rel_tol=1e-9, abs_tol=1e-9)
                        sum_ok = abs(fp_sum - exp_sum) <= 1e-9 * max(scale, 1.0)
                        if not (sumsq_ok and sum_ok):
                            raise RuntimeError(
                                "Direct completion started from a target model that is not the "
                                f"native base: target boundary fingerprint mismatch between "
                                f"pass A and pass B at position {pos} (batch {k})"
                            )
                        # Align on CPU, as the resident path does on its CPU banks, so D_j stays bit-comparable.
                        out_cpu = out.cpu()
                        b = _aligned([sb[str(i)].cpu()], [out_cpu])[0]
                        f = _aligned([sf[str(i)].cpu()], [out_cpu])[0]
                        base_out = out_cpu
                        q = q_by_position[pos]
                        if residual_target == "transported_delta":
                            desired = (f - b) @ q
                        else:
                            desired = (f - mu_s_by_position[pos]) @ q + mu_t_by_position[pos] - out_cpu
                        effect = out_cpu - base_out
                        error = desired - effect
                        desired_sq_val = float((desired.double() ** 2).sum().item())
                        effect_sq_val = float((effect.double() ** 2).sum().item())
                        for component in components:
                            pair = (pos, component)
                            h = tgt[f"{pos}.{component}.h"].cpu()
                            desired_sq[pair] += desired_sq_val
                            effect_sq[pair] += effect_sq_val
                            stats_by[pair].update(
                                h.reshape(-1, h.shape[-1]),
                                error.reshape(-1, error.shape[-1]),
                                None,
                                effective_out[pair],
                            )

            for pos in chunk:
                position_corrections: dict[str, Tensor] = {}
                block_rows: list[dict[str, Any]] = []
                for component in components:
                    key = shim.component_key(pos, component, prefixed=True)
                    pair = (pos, component)
                    _finalize_independent_component(
                        stats_by[pair],
                        desired_sq[pair],
                        effect_sq[pair],
                        pos,
                        component,
                        key,
                        current_state,
                        effective_out[pair],
                        solver_config,
                        source_coordinates,
                        position_corrections,
                        block_rows,
                        h_batches=None,
                    )
                target_corrections.update(position_corrections)
                for row in block_rows:
                    row["source_position"] = row.pop("source_coordinate")
                    # Same analysis-only tag the resident path sets in fit_direct_residual.
                    row["procrustes_source"] = prepared.get("procrustes_source", "activation")
                    diagnostics.append(row)
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
    return target_corrections, diagnostics


def _streaming_source_iters(source_base_model, source_ft_model, prepared, pairing, device, family_adapter=None):
    """Fresh lockstep source generators (base + FT boundary at the paired indices) over the
    streaming path's replayed source calibration batches."""
    requests = {str(i): (i, "boundary") for i in sorted(set(pairing.pairing))}
    batches = prepared["source_batches"]
    return {
        "sb": iter_capture_tokens(
            source_base_model, batches, requests, device, family_adapter=family_adapter, store_device=device
        ),
        "sf": iter_capture_tokens(
            source_ft_model, batches, requests, device, family_adapter=family_adapter, store_device=device
        ),
    }


def _streaming_desired(prepared, pairing, residual_target, source_values, t0, positions):
    """Pass B's per-batch ``D_j`` (same arithmetic as `fit_direct_residual_streaming`)."""
    out: dict[int, Tensor] = {}
    for pos in positions:
        i = pairing.pairing[pos]
        out_cpu = t0[str(pos)].cpu()
        b = _aligned([source_values["sb"][str(i)].cpu()], [out_cpu])[0]
        f = _aligned([source_values["sf"][str(i)].cpu()], [out_cpu])[0]
        q = prepared["q_by_position"][pos]
        if residual_target == "transported_delta":
            out[pos] = (f - b) @ q
        else:
            mu_s = prepared["source_mean_by_position"][pos].float()
            mu_t = prepared["target_mean_by_position"][pos].float()
            out[pos] = (f - mu_s) @ q + mu_t - out_cpu
    return out


def measure_streaming_realization_for(
    target_model,
    target_base_state,
    source_base_model,
    source_ft_model,
    prepared,
    pairing,
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
):
    """Bind `measure_direct_residual_realization_streaming` to one streaming run: returns
    ``measure(delta) -> {j: row}`` (the `apply_tv_scaling` ``measure_fn`` signature)."""
    positions = list(range(pairing.target_depth))
    components = order_components(config.components)

    def measure(delta):
        return measure_direct_residual_realization_streaming(
            target_model,
            target_base_state,
            delta,
            positions,
            prepared["target_batches"],
            lambda _k, source_values, t0: _streaming_desired(
                prepared, pairing, config.residual_target, source_values, t0, positions
            ),
            lambda: _streaming_source_iters(
                source_base_model, source_ft_model, prepared, pairing, device, family_adapter=family_adapter
            ),
            device=device,
            components=components,
            family_adapter=family_adapter,
        )

    return measure


@contextlib.contextmanager
def _pristine_target_sweep(
    source_base_model, source_ft_model, target_model, target_base_state, prepared, pairing, *, device, family_adapter
):
    """Lockstep sweep (source base ``sb``, source FT ``sf``, pristine target ``tgt``) over the calibration batches.

    Yields an iterator of per-batch ``{"sb": ..., "sf": ..., "tgt": ...}`` capture dicts (boundary activations
    keyed by str(source index) for ``sb``/``sf`` and str(target position) for ``tgt``). The target model is put in
    its pristine base state for the sweep and its entry state is restored on exit.
    """
    positions = list(range(pairing.target_depth))
    entry_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    try:
        target_model.load_state_dict({k: v.detach().cpu().clone() for k, v in target_base_state.items()}, strict=True)
        requests = {str(j): (j, "boundary") for j in positions}
        gens = _streaming_source_iters(source_base_model, source_ft_model, prepared, pairing, device, family_adapter)
        gens["tgt"] = iter_capture_tokens(
            target_model,
            prepared["target_batches"],
            requests,
            device,
            family_adapter=family_adapter,
            store_device=device,
        )
        names = list(gens)
        with contextlib.ExitStack() as stack:
            for gen in gens.values():
                stack.enter_context(contextlib.closing(gen))
            yield (dict(zip(names, values, strict=True)) for values in zip(*(gens[n] for n in names), strict=True))
    finally:
        target_model.load_state_dict(entry_state, strict=True)


@torch.no_grad()
def compute_gradient_delta_disagreement_streaming(
    source_base_model,
    source_ft_model,
    target_model,
    target_base_state: Mapping[str, Tensor],
    prepared: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    device,
    family_adapter=None,
) -> dict[int, float]:
    """Streaming counterpart of the resident ``activation_gradient_delta_disagreement`` (gradient mode).

    ``||dS (Q_act - Q_grad)||_F / (||dS Q_act||_F + 1e-12)`` per position, ``dS = S_1 - S_0`` over all
    calibration rows, with both float64 maps taken from `prepared` (known after Pass A, so the norms are
    accumulated per batch as sums of squares: no Gram matrix, O(1) host memory). Analysis-only; call it
    outside the timed brackets. The target model's entry state is restored.
    """
    positions = list(range(pairing.target_depth))
    q_act = prepared["activation_q64_by_position"]
    q_grad = prepared["gradient_q64_by_position"]
    sums = {j: [0.0, 0.0] for j in positions}
    with _pristine_target_sweep(
        source_base_model,
        source_ft_model,
        target_model,
        target_base_state,
        prepared,
        pairing,
        device=device,
        family_adapter=family_adapter,
    ) as sweep:
        for by_name in sweep:
            for j in positions:
                i = pairing.pairing[j]
                t_cpu = by_name["tgt"][str(j)].cpu()
                b = _aligned([by_name["sb"][str(i)].cpu()], [t_cpu])[0]
                f = _aligned([by_name["sf"][str(i)].cpu()], [t_cpu])[0]
                delta = (f - b).reshape(-1, b.shape[-1]).double()
                d_act = delta @ q_act[j].double()
                d_grad = delta @ q_grad[j].double()
                sums[j][0] += float(((d_act - d_grad) ** 2).sum().item())
                sums[j][1] += float((d_act**2).sum().item())
    return {j: float(num**0.5 / (den**0.5 + 1e-12)) for j, (num, den) in sums.items()}


@torch.no_grad()
def compute_alignment_diagnostics_streaming(
    source_base_model,
    source_ft_model,
    target_model,
    target_base_state: Mapping[str, Tensor],
    prepared: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    device,
    family_adapter=None,
) -> dict[int, dict[str, float]]:
    """Streaming counterpart of `compute_alignment_diagnostics` (same keys, same meaning).

    One lockstep sweep (source base, source FT, pristine target) over the calibration
    batches, reusing Pass A's activation-space map and means (``Q_j``, ``mu_s``, ``mu_t``
    in float64), and accumulating every norm as a per-batch sum of squares: equal to the
    resident diagnostics up to floating-point summation order (and the Chan-accumulated
    Procrustes map). Analysis-only; like the resident function it must be called outside
    the timed brackets. The target model's entry state is restored.
    """
    positions = list(range(pairing.target_depth))
    q64 = prepared["activation_q64_by_position"]
    mu_s = prepared["source_mean_by_position"]
    mu_t = prepared["target_mean_by_position"]
    sums = {j: {"e": 0.0, "t": 0.0, "delta": 0.0, "in": 0.0, "out": 0.0, "src_dim": 0, "tgt_dim": 0} for j in positions}
    with _pristine_target_sweep(
        source_base_model,
        source_ft_model,
        target_model,
        target_base_state,
        prepared,
        pairing,
        device=device,
        family_adapter=family_adapter,
    ) as sweep:
        for by_name in sweep:
            for j in positions:
                i = pairing.pairing[j]
                t_cpu = by_name["tgt"][str(j)].cpu()
                b = _aligned([by_name["sb"][str(i)].cpu()], [t_cpu])[0]
                f = _aligned([by_name["sf"][str(i)].cpu()], [t_cpu])[0]
                s0 = b.reshape(-1, b.shape[-1]).double()
                s1 = f.reshape(-1, f.shape[-1]).double()
                t0 = t_cpu.reshape(-1, t_cpu.shape[-1]).double()
                q = q64[j].double()
                e = (s0 - mu_s[j]) @ q - (t0 - mu_t[j])
                e_in = (e @ q.T) @ q
                acc = sums[j]
                acc["e"] += float((e**2).sum().item())
                acc["t"] += float(((t0 - mu_t[j]) ** 2).sum().item())
                acc["delta"] += float((((s1 - s0) @ q) ** 2).sum().item())
                acc["in"] += float((e_in**2).sum().item())
                acc["out"] += float(((e - e_in) ** 2).sum().item())
                acc["src_dim"], acc["tgt_dim"] = int(s0.shape[-1]), int(t0.shape[-1])
    diagnostics: dict[int, dict[str, float]] = {}
    for j in positions:
        acc = sums[j]
        error_norm = acc["e"] ** 0.5
        target_norm = acc["t"] ** 0.5
        delta_norm = acc["delta"] ** 0.5
        diagnostics[j] = {
            "procrustes_error_norm": error_norm,
            "procrustes_relative_error": error_norm / target_norm if target_norm > 0 else 0.0,
            "delta_target_norm": delta_norm,
            "endpoint_minus_delta_over_delta": error_norm / delta_norm if delta_norm > 0 else 0.0,
            "procrustes_error_in_range_norm": acc["in"] ** 0.5,
            "procrustes_error_out_of_range_norm": acc["out"] ** 0.5,
            "mean_offset_norm": float(torch.linalg.norm(mu_s[j] @ q64[j].double() - mu_t[j])),
            "source_dim": acc["src_dim"],
            "target_dim": acc["tgt_dim"],
        }
    return diagnostics
