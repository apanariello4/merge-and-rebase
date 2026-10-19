"""Streaming (low-memory) Ariadne fit path and its diagnostics."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping
from copy import deepcopy
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
    _truncated_procrustes_from_cross,
    _validate_alignment_options,
    procrustes_rank_diagnostics,
)
from .capture import (
    _assert_layerscale_identity,
    iter_capture_block_gradients,
    iter_capture_tokens,
    paired_calibration,
)
from .config import DirectResidualConfig, annotate_target_split, order_components, resolve_fit_blocks, target_groups
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
    procrustes_rank: int | None = None,
    procrustes_energy: float | None = None,
    exclude_first_token: bool = False,
    mid_alignment: bool = False,
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
    scalars. ``procrustes_rank`` / ``procrustes_energy`` (opt-in, polar + activation only, mutually exclusive) keep only
    the top-k singular pairs of the cross-covariance (``procrustes_energy`` thresholds the cumulative PLAIN singular
    value share) and add ``procrustes_rank_used`` / ``procrustes_kept_singular_mass`` /
    ``procrustes_kept_singular_energy`` to the rows. The q/mu_s/mu_t tensors are not duplicated into rows (unlike the resident path); they stay in
    ``q_by_position`` / ``source_mean_by_position`` / ``target_mean_by_position``.

    ``mid_alignment`` (opt-in, vision only, polar + activation + no truncation; for ``sequential_points="mid_end"``)
    additionally accumulates, per target position ``j``, the centered cross-covariance of the source MID-block residual
    at ``pairing[j]`` against the target mid residual at ``j`` (``x_mid = block_input + attention branch output``, both
    captured on the same block; LayerScale ``ls_1`` must be the identity) and solves it with the same polar map. It adds
    ``q_mid_by_position`` (float32) and ``mid_fingerprints`` (per-(batch, position) ``(sum, sumsq)`` of the target mid
    residual); with ``mid_alignment=False`` nothing else is requested or returned.
    """
    if mid_alignment:
        if family_adapter is not None:
            # HF decoder layers receive hidden_states as a keyword argument, which the block_input capture (positional
            # inputs[0]) does not see; the mid residual is therefore only defined on the vision layout.
            raise NotImplementedError("mid_end is vision-only")
        if alignment_map != "polar" or alignment_row_weighting != "uniform" or procrustes_source != "activation":
            raise ValueError(
                "mid_alignment requires alignment_map='polar', alignment_row_weighting='uniform' and "
                "procrustes_source='activation'"
            )
        if procrustes_rank is not None or procrustes_energy is not None:
            raise ValueError("mid_alignment does not support procrustes_rank/procrustes_energy")
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    if pairing.source_depth < 1:
        raise ValueError("pairing.source_depth must be positive")
    if procrustes_source not in {"activation", "gradient"}:
        raise ValueError("procrustes_source must be 'activation' or 'gradient'")
    _validate_alignment_options(alignment_map, alignment_row_weighting, procrustes_source)
    truncate = procrustes_rank is not None or procrustes_energy is not None
    if truncate and (alignment_map != "polar" or procrustes_source != "activation"):
        raise ValueError(
            "procrustes_rank/procrustes_energy require alignment_map='polar' and procrustes_source='activation'"
        )
    if procrustes_rank is not None and procrustes_energy is not None:
        raise ValueError("procrustes_rank and procrustes_energy are mutually exclusive")
    truncation_info: dict[int, dict[str, Any]] = {}
    gradient_mode = procrustes_source == "gradient"
    if gradient_mode and (source_recipe is None or target_recipe is None):
        raise ValueError("procrustes_source='gradient' requires source_recipe and target_recipe")
    source_batches, target_batches, metadata = paired_calibration(
        source_loader,
        target_loader,
        num_batches=num_batches,
        seed=seed,
        family_adapter=family_adapter,
        exclude_first_token=exclude_first_token,
    )
    distinct_source_indices = sorted(set(pairing.pairing))
    src_requests = {str(i): (i, "boundary") for i in distinct_source_indices}
    tgt_requests = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
    if mid_alignment:
        for i in distinct_source_indices:
            _assert_layerscale_identity(
                _layout_for(None),
                _layout_for(None).blocks(source_base_model)[i],
                context=f"source block {i}",
                requirement="mid_alignment",
            )
            src_requests[f"{i}.mid_in"] = (i, "block_input")
            src_requests[f"{i}.mid_attn"] = (i, "attn_proj")
        for j in range(pairing.target_depth):
            _assert_layerscale_identity(
                _layout_for(None),
                _layout_for(None).blocks(target_base_model)[j],
                context=f"target block {j}",
                requirement="mid_alignment",
            )
            tgt_requests[f"{j}.mid_in"] = (j, "block_input")
            tgt_requests[f"{j}.mid_attn"] = (j, "attn_proj")
    mid_accumulators = (
        {j: _StreamingCrossCovariance(device=device) for j in range(pairing.target_depth)} if mid_alignment else {}
    )
    mid_fingerprints: dict[tuple[int, int], tuple[float, float]] = {}
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
                if mid_alignment:
                    t_mid = _mid_residual(tgt, f"{j}.mid_in", f"{j}.mid_attn")
                    x_mid = _aligned([_mid_residual(src, f"{i}.mid_in", f"{i}.mid_attn")], [t_mid])[0]
                    mid_accumulators[j].update(
                        x_mid.reshape(-1, x_mid.shape[-1]).double(), t_mid.reshape(-1, t_mid.shape[-1]).double()
                    )
                    t_mid64 = t_mid.double()
                    mid_fingerprints[(k, j)] = (float(t_mid64.sum().item()), float((t_mid64**2).sum().item()))
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
            if truncate:
                q, k_used, mass, kept_energy = _truncated_procrustes_from_cross(
                    cross, rank=procrustes_rank, energy=procrustes_energy
                )
                rank = _cross_rank(cross)
                truncation_info[position] = {
                    "procrustes_rank_used": int(k_used),
                    "procrustes_kept_singular_mass": mass,
                    "procrustes_kept_singular_energy": kept_energy,
                }
            else:
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
        row.update(truncation_info.get(j, {}))
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
    prepared = {
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
    if mid_alignment:
        prepared["q_mid_by_position"] = {
            j: _procrustes_from_cross(acc.cross()).cpu().float() for j, acc in mid_accumulators.items()
        }
        prepared["mid_fingerprints"] = mid_fingerprints
    return prepared


def _mid_residual(captured, in_key: str, attn_key: str) -> Tensor:
    """``x_mid = block_input + attention branch output`` (identity ``ls_1``) from two captures of one block."""
    block_input, attn_out = captured[in_key], captured[attn_key]
    if block_input.shape != attn_out.shape:
        raise ValueError(
            f"mid residual needs matching block-input and attention-output shapes; got {tuple(block_input.shape)} "
            f"and {tuple(attn_out.shape)}"
        )
    return block_input + attn_out


def _none_stream(n: int):
    """``n`` Nones: a stand-in generator for a capture stream the current target does not need."""
    for _ in range(n):
        yield None


def _check_boundary_fingerprint(out, expected, config, pos, k) -> None:
    """Raise if the captured target boundary ``out`` does not match Pass A's ``(sum, sumsq)`` fingerprint."""
    out64 = out.double()
    fp_sum = float(out64.sum().item())
    fp_sumsq = float((out64**2).sum().item())
    exp_sum, exp_sumsq = expected
    scale = math.sqrt(exp_sumsq) if exp_sumsq > 0 else 1.0
    tol = float(config.streaming_fingerprint_tol)
    sumsq_ok = math.isclose(fp_sumsq, exp_sumsq, rel_tol=tol, abs_tol=tol)
    sum_ok = abs(fp_sum - exp_sum) <= tol * max(scale, 1.0)
    if not (sumsq_ok and sum_ok):
        drift_sumsq = abs(fp_sumsq - exp_sumsq) / max(abs(exp_sumsq), 1e-300)
        drift_sum = abs(fp_sum - exp_sum) / max(scale, 1.0)
        raise RuntimeError(
            "Direct completion started from a target model that is not the "
            f"native base: target boundary fingerprint mismatch between "
            f"pass A and pass B at position {pos} (batch {k}); relative drift "
            f"sumsq={drift_sumsq:.3e}, sum={drift_sum:.3e} vs "
            f"streaming_fingerprint_tol={tol:.3e}"
        )


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
    stats_factory=None,
    native_model=None,
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

    ``stats_factory(position, component, device)`` (analysis only, default ``None``) builds the per-(position,
    component) accumulator in place of ``ResidualSufficientStatistics(device=device)``; it must return a subclass
    that leaves ``update``/``solve`` numerically unchanged, so the fit and its hash do not depend on it.

    ``residual_target="native_oracle"`` (diagnostic) replaces ``D_j`` by ``D*_j = T^ft_j - T^base_j``: the block-``j``
    output change of ``native_model`` (the NATIVE fine-tuned target, its own full forward) against the target base on the
    same target calibration batches. No source model and no Procrustes map enter that target. ``config.fit_blocks``
    restricts the fit to those target blocks (every other block gets no update).
    """
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    components = order_components(config.components)
    oracle = config.residual_target == "native_oracle"
    if oracle and native_model is None:
        raise ValueError("residual_target='native_oracle' needs native_model (the native fine-tuned target)")
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
    positions = resolve_fit_blocks(config, pairing.target_depth)
    groups = target_groups(pairing.pairing)
    share = config.pair_target_split == "per_block_share"
    source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
    solver_config = replace(config, cascade_order="independent") if config.cascade_order != "independent" else config
    shim = _layout_for(family_adapter)

    chunk_size = config.streaming_position_chunk or len(positions)
    chunks = [positions[start : start + chunk_size] for start in range(0, len(positions), chunk_size)]
    post_mlp_norm = config.post_mlp_norm
    if post_mlp_norm is not None and not callable(getattr(family_adapter, "post_mlp_norm", None)):
        raise ValueError("post_mlp_norm needs a family adapter exposing post_mlp_norm(block) (e.g. gemma2)")

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
                if post_mlp_norm is not None:
                    # y = W h of the target base's down_proj (pre-norm), for the per-row RMS r of its post-MLP norm.
                    tgt_requests[f"{pos}.mlp.c_proj.y"] = (pos, "c_proj")

            stats_by: dict[tuple[int, str], ResidualSufficientStatistics] = {}
            desired_sq: dict[tuple[int, str], float] = {}
            effect_sq: dict[tuple[int, str], float] = {}
            effective_out: dict[tuple[int, str], Tensor] = {}
            norm_gain: dict[int, tuple[Tensor, float]] = {}
            for pos in chunk:
                if post_mlp_norm is not None:
                    norm_gain[pos] = family_adapter.post_mlp_norm(shim.blocks(target_model)[pos])
                for component in components:
                    key = shim.component_key(pos, component, prefixed=True)
                    width = int(current_state[key].shape[0])
                    pair = (pos, component)
                    effective_out[pair] = _component_effective_out(shim, target_model, pos, component, width)
                    if post_mlp_norm == "linearized":
                        # The norm is part of the output map: d(residual) = (1/r) diag(g) dW h (rank-one term dropped).
                        effective_out[pair] = effective_out[pair] @ torch.diag(norm_gain[pos][0])
                    stats_by[pair] = (
                        ResidualSufficientStatistics(device=device, store_device=config.stats_device)
                        if stats_factory is None
                        else stats_factory(pos, component, device)
                    )
                    desired_sq[pair] = 0.0
                    effect_sq[pair] = 0.0

            if oracle:
                # The oracle target needs no source model (no D_j, no Procrustes map): only the native fine-tuned target.
                sb_gen, sf_gen = _none_stream(len(batches)), _none_stream(len(batches))
                nat_gen = iter_capture_tokens(
                    native_model,
                    batches,
                    {f"{pos}.out": (pos, "boundary") for pos in chunk},
                    device,
                    family_adapter=family_adapter,
                    store_device=device,
                )
            else:
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
                nat_gen = _none_stream(len(batches))
            tgt_gen = iter_capture_tokens(
                target_model, batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
            )
            with (
                contextlib.closing(sb_gen),
                contextlib.closing(sf_gen),
                contextlib.closing(tgt_gen),
                contextlib.closing(nat_gen),
            ):
                for k, (sb, sf, tgt, nat) in enumerate(zip(sb_gen, sf_gen, tgt_gen, nat_gen, strict=True)):
                    for pos in chunk:
                        i = pairing.pairing[pos]
                        out = tgt[f"{pos}.out"]
                        _check_boundary_fingerprint(out, fingerprints[(k, pos)], config, pos, k)
                        # Align on CPU, as the resident path does on its CPU banks, so D_j stays bit-comparable.
                        out_cpu = out.cpu()
                        base_out = out_cpu
                        if oracle:
                            # Same images, same architecture: no alignment, no Q_j. D*_j = T^ft_j - T^base_j.
                            desired = nat[f"{pos}.out"].cpu() - out_cpu
                        else:
                            b = _aligned([sb[str(i)].cpu()], [out_cpu])[0]
                            f = _aligned([sf[str(i)].cpu()], [out_cpu])[0]
                            q = q_by_position[pos]
                            if residual_target == "transported_delta":
                                desired = (f - b) @ q
                                if share and groups[pos][0] > 1:
                                    desired = desired / groups[pos][0]
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
                            h_rows = h.reshape(-1, h.shape[-1])
                            e_rows = error.reshape(-1, error.shape[-1])
                            if post_mlp_norm is not None:
                                gain, eps = norm_gain[pos]
                                y = tgt[f"{pos}.mlp.c_proj.y"].cpu().reshape(-1, e_rows.shape[-1]).float()
                                rms = torch.sqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
                                if post_mlp_norm == "pre_norm":
                                    e_rows = e_rows * rms / gain
                                else:
                                    h_rows = h_rows / rms
                            stats_by[pair].update(h_rows, e_rows, None, effective_out[pair])

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
                    # Release this position's statistics now (not at the next chunk): with stats_device the solve
                    # moved them to the compute device, and only one position should be resident there at a time.
                    del stats_by[pair]
                target_corrections.update(position_corrections)
                for row in block_rows:
                    row["source_position"] = row.pop("source_coordinate")
                    # Same analysis-only tag the resident path sets in fit_direct_residual.
                    row["procrustes_source"] = prepared.get("procrustes_source", "activation")
                    if oracle:
                        row["residual_target"] = "native_oracle"
                        row["procrustes_used"] = False
                    diagnostics.append(row)
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
    annotate_target_split(diagnostics, config, pairing)
    return target_corrections, diagnostics


@torch.no_grad()
def fit_direct_residual_sequential_streaming(
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
    """Sequential-across-depth variant of `fit_direct_residual_streaming` (``depth_fit="sequential"``).

    Pass A (`prepare_direct_residual_streaming`) is unchanged. Blocks are then fitted in order ``j = 0..N-1``; before
    block ``j`` the target holds the base plus the already-fitted unit-strength updates of blocks ``0..j-1``. Per
    calibration batch (source base, source FT, current target and a pristine target copy swept in lockstep) it
    forms ``D_j`` exactly as Pass B does, the upstream effect ``U_j = out_current_j - out_pristine_j`` at boundary
    ``j`` and the residual target ``E_j = D_j - U_j``, accumulates ``ResidualSufficientStatistics(h_current, E_j)``
    and solves with the same ridge configuration. Block ``j``'s own update moves boundary ``j`` by exactly
    ``H_j dW_j + 1 beta_j^T`` (its projection input does not depend on that projection) and later blocks do not
    affect boundary ``j``, so after the full fit the mounted realized error at boundary ``j`` equals block ``j``'s
    fit residual. Block 0 (no upstream) is bit-identical to the independent fit.

    Rows follow the independent schema plus ``depth_fit="sequential"`` and ``upstream_effect_norm``
    (``sqrt(sum ||U_j||^2)``; ``effect_before_norm`` stays 0 because the solver is called with a zero pre-fit
    effect). Returns ``(corrections, rows)``. The target model's entry state and placement are restored in a
    ``finally``. A pristine deep copy of the target model is held for the duration (extra model memory).
    """
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    if config.depth_fit != "sequential":
        raise ValueError("fit_direct_residual_sequential_streaming requires depth_fit='sequential'")
    if config.sequential_points != "single":
        return _fit_sequential_two_points_streaming(
            target_model,
            target_base_state,
            source_base_model,
            source_ft_model,
            prepared,
            pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
    if config.block_split == "backfit":
        return _fit_sequential_backfit_streaming(
            target_model,
            target_base_state,
            source_base_model,
            source_ft_model,
            prepared,
            pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
    components = order_components(config.components)
    groups = target_groups(pairing.pairing)
    if len(components) != 1:
        raise ValueError("depth_fit='sequential' requires exactly one component")
    component = components[0]
    batches = prepared["target_batches"]
    source_batches = prepared["source_batches"]
    q_by_position = prepared["q_by_position"]
    fingerprints = prepared["fingerprints"]
    positions = list(range(pairing.target_depth))
    for j in positions:
        if j not in q_by_position:
            raise ValueError(f"Missing prepared Procrustes map for position {j}")
    source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
    shim = _layout_for(family_adapter)

    unique_models: dict[int, Any] = {}
    for m in (target_model, source_base_model, source_ft_model):
        unique_models.setdefault(id(m), m)
    originals = {mid: (next(m.parameters()).device, m.training) for mid, m in unique_models.items()}

    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    pristine_model = None
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        # Frozen copy holding exactly the base state: the reference for U_j, and (when a source model is the same
        # object as the target) the source model, which must not see the mounted updates.
        pristine_model = deepcopy(target_model)
        pristine_model.load_state_dict({k: v.detach().cpu().clone() for k, v in target_base_state.items()}, strict=True)
        sb_model = pristine_model if source_base_model is target_model else source_base_model
        sf_model = pristine_model if source_ft_model is target_model else source_ft_model
        for m in {id(x): x for x in (target_model, sb_model, sf_model, pristine_model)}.values():
            m.to(device).eval()

        fit_set = set(resolve_fit_blocks(config, pairing.target_depth))
        for pos in positions:
            if pos not in fit_set:
                continue  # unfitted block: nothing mounted, later blocks see the true drift U_j
            i = pairing.pairing[pos]
            q = q_by_position[pos]
            key = shim.component_key(pos, component, prefixed=True)
            width = int(current_state[key].shape[0])
            effective_out = _component_effective_out(shim, target_model, pos, component, width)
            stats = ResidualSufficientStatistics(device=device)
            desired_sq = 0.0
            upstream_sq = 0.0
            src_requests = {str(i): (i, "boundary")}
            group_size, group_index, group_start = groups[pos]
            # pair_target_split='interpolate': the last block of a group keeps D_j; earlier ones move from D_prev (the
            # unsplit change of block group_start - 1, own Q; zero before block 0) towards D_j.
            interp = config.pair_target_split == "interpolate" and group_index < group_size - 1
            interp_weight = (group_index + 1) / group_size
            if interp and group_start > 0:
                i_prev = pairing.pairing[group_start - 1]
                q_prev = q_by_position[group_start - 1]
                src_requests[str(i_prev)] = (i_prev, "boundary")
            out_key, h_key = f"{pos}.out", f"{pos}.{component}.h"
            tgt_requests = {out_key: (pos, "boundary"), h_key: (pos, COMPONENT_INPUT_KIND[component])}
            ref_requests = {out_key: (pos, "boundary")}
            gens = (
                iter_capture_tokens(
                    sb_model, source_batches, src_requests, device, family_adapter=family_adapter, store_device=device
                ),
                iter_capture_tokens(
                    sf_model, source_batches, src_requests, device, family_adapter=family_adapter, store_device=device
                ),
                iter_capture_tokens(
                    target_model, batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
                ),
                iter_capture_tokens(
                    pristine_model, batches, ref_requests, device, family_adapter=family_adapter, store_device=device
                ),
            )
            with contextlib.ExitStack() as stack:
                for gen in gens:
                    stack.enter_context(contextlib.closing(gen))
                for k, (sb, sf, tgt, ref) in enumerate(zip(*gens, strict=True)):
                    # The pristine copy's boundary must match Pass A's fingerprint (detects a mutated base).
                    _check_boundary_fingerprint(ref[out_key], fingerprints[(k, pos)], config, pos, k)
                    out_cpu = tgt[out_key].cpu()
                    b = _aligned([sb[str(i)].cpu()], [out_cpu])[0]
                    f = _aligned([sf[str(i)].cpu()], [out_cpu])[0]
                    desired = (f - b) @ q
                    if interp:
                        if group_start > 0:
                            b_prev = _aligned([sb[str(i_prev)].cpu()], [out_cpu])[0]
                            f_prev = _aligned([sf[str(i_prev)].cpu()], [out_cpu])[0]
                            d_prev = (f_prev - b_prev) @ q_prev
                        else:
                            d_prev = torch.zeros_like(desired)
                        desired = d_prev + interp_weight * (desired - d_prev)
                    upstream = out_cpu - ref[out_key].cpu()
                    error = desired - upstream
                    desired_sq += float((desired.double() ** 2).sum().item())
                    upstream_sq += float((upstream.double() ** 2).sum().item())
                    h = tgt[h_key].cpu()
                    stats.update(h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), None, effective_out)

            position_corrections: dict[str, Tensor] = {}
            block_rows: list[dict[str, Any]] = []
            # effect_sq=0.0: the upstream effect is intentional here (it is part of the residual target), so the
            # pristine-effect check does not apply; the true value is recorded as ``upstream_effect_norm``.
            _finalize_independent_component(
                stats,
                desired_sq,
                0.0,
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
            )
            target_corrections.update(position_corrections)
            for row in block_rows:
                row["source_position"] = row.pop("source_coordinate")
                row["procrustes_source"] = prepared.get("procrustes_source", "activation")
                row["depth_fit"] = "sequential"
                row["upstream_effect_norm"] = upstream_sq**0.5
                diagnostics.append(row)
            # Mount this block's update (weight and bias, as updated in current_state) before the next block.
            target_model.load_state_dict({name: current_state[name] for name in position_corrections}, strict=False)
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
        del pristine_model
    annotate_target_split(diagnostics, config, pairing)
    return target_corrections, diagnostics


@torch.no_grad()
def _fit_sequential_backfit_streaming(
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
    """``depth_fit="sequential"`` with ``block_split="backfit"`` (vision only; attn.out_proj + mlp.c_proj).

    Same depth loop as `fit_direct_residual_sequential_streaming` (current target holding the fitted blocks < j, a
    pristine copy for ``U_j``), but per block ``j`` the batches ``X_j`` (block input on the CURRENT target),
    ``E_j = D_j - U_j`` and ``T_j = out_current_j`` are collected (CPU float32) and the block is fitted by the
    intra-block Gauss-Seidel backfit (`ablations._backfit_one_block`) with ``E_j`` as the shared block target. Block 0
    (``U_0 = 0``) equals the resident independent backfit of position 0. Both components' updates (weights and
    biases) are mounted before block ``j + 1``. Rows: backfit diagnostics plus ``depth_fit`` / ``upstream_effect_norm``.
    """
    if family_adapter is not None:
        raise NotImplementedError("depth_fit='sequential' with block_split='backfit' is vision-only")
    from .ablations import _backfit_one_block  # local import: ablations imports the fit modules

    components = order_components(config.components)
    batches = prepared["target_batches"]
    source_batches = prepared["source_batches"]
    q_by_position = prepared["q_by_position"]
    fingerprints = prepared["fingerprints"]
    positions = list(range(pairing.target_depth))
    for j in positions:
        if j not in q_by_position:
            raise ValueError(f"Missing prepared Procrustes map for position {j}")
    source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
    shim = _layout_for(family_adapter)

    unique_models: dict[int, Any] = {}
    for m in (target_model, source_base_model, source_ft_model):
        unique_models.setdefault(id(m), m)
    originals = {mid: (next(m.parameters()).device, m.training) for mid, m in unique_models.items()}

    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    pristine_model = None
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        pristine_model = deepcopy(target_model)
        pristine_model.load_state_dict({k: v.detach().cpu().clone() for k, v in target_base_state.items()}, strict=True)
        sb_model = pristine_model if source_base_model is target_model else source_base_model
        sf_model = pristine_model if source_ft_model is target_model else source_ft_model
        for m in {id(x): x for x in (target_model, sb_model, sf_model, pristine_model)}.values():
            m.to(device).eval()

        fit_set = set(resolve_fit_blocks(config, pairing.target_depth))
        for pos in positions:
            if pos not in fit_set:
                continue
            i = pairing.pairing[pos]
            q = q_by_position[pos]
            upstream_sq = 0.0
            src_requests = {str(i): (i, "boundary")}
            out_key, in_key = f"{pos}.out", f"{pos}.block_input"
            tgt_requests = {out_key: (pos, "boundary"), in_key: (pos, "block_input")}
            ref_requests = {out_key: (pos, "boundary")}
            gens = (
                iter_capture_tokens(
                    sb_model, source_batches, src_requests, device, family_adapter=family_adapter, store_device=device
                ),
                iter_capture_tokens(
                    sf_model, source_batches, src_requests, device, family_adapter=family_adapter, store_device=device
                ),
                iter_capture_tokens(
                    target_model, batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
                ),
                iter_capture_tokens(
                    pristine_model, batches, ref_requests, device, family_adapter=family_adapter, store_device=device
                ),
            )
            x_batches: list[Tensor] = []
            e_batches: list[Tensor] = []
            t0_batches: list[Tensor] = []
            with contextlib.ExitStack() as stack:
                for gen in gens:
                    stack.enter_context(contextlib.closing(gen))
                for k, (sb, sf, tgt, ref) in enumerate(zip(*gens, strict=True)):
                    _check_boundary_fingerprint(ref[out_key], fingerprints[(k, pos)], config, pos, k)
                    out_cpu = tgt[out_key].cpu()
                    b = _aligned([sb[str(i)].cpu()], [out_cpu])[0]
                    f = _aligned([sf[str(i)].cpu()], [out_cpu])[0]
                    desired = (f - b) @ q
                    upstream = out_cpu - ref[out_key].cpu()
                    upstream_sq += float((upstream.double() ** 2).sum().item())
                    x_batches.append(tgt[in_key].cpu())
                    e_batches.append(desired - upstream)
                    t0_batches.append(out_cpu)

            local_block = deepcopy(shim.block_module(shim.blocks(target_model)[pos])).to(device).eval()
            position_corrections, block_rows = _backfit_one_block(
                shim,
                local_block,
                x_batches,
                e_batches,
                t0_batches,
                pos,
                source_coordinates[pos],
                components,
                current_state,
                config,
                device,
            )
            del local_block, x_batches, e_batches, t0_batches
            target_corrections.update(position_corrections)
            for name, delta in position_corrections.items():
                current_state[name] = current_state[name] + delta.to(current_state[name])
            for row in block_rows:
                row["source_position"] = row.pop("source_coordinate")
                row["procrustes_source"] = prepared.get("procrustes_source", "activation")
                row["depth_fit"] = "sequential"
                row["upstream_effect_norm"] = upstream_sq**0.5
                diagnostics.append(row)
            target_model.load_state_dict({name: current_state[name] for name in position_corrections}, strict=False)
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
        del pristine_model
    annotate_target_split(diagnostics, config, pairing)
    return target_corrections, diagnostics


def _check_mid_fingerprint(mid, expected, config, pos, k) -> None:
    """Raise if the captured pristine mid residual ``mid`` does not match Pass A's ``(sum, sumsq)`` fingerprint."""
    try:
        _check_boundary_fingerprint(mid, expected, config, pos, k)
    except RuntimeError as exc:
        raise RuntimeError(str(exc).replace("target boundary fingerprint", "target mid-block fingerprint")) from None


@torch.no_grad()
def _fit_sequential_two_points_streaming(
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
    """``depth_fit="sequential"`` with two correction points per block (``sequential_points`` ``end_end`` / ``mid_end``).

    Same machinery as `fit_direct_residual_sequential_streaming` (blocks fitted in order, blocks ``< j`` mounted, source
    base / source FT / current target / pristine target swept in lockstep), but block ``j`` runs two calibration passes:

    1. component ``attn.out_proj`` (O), then mounted. ``end_end``: against the block-end target ``D_j = (f - b) Q_j``
       with upstream ``U_j = out_current - out_pristine`` at the block boundary. ``mid_end``: against the mid-block
       target ``M_j = (f_mid - b_mid) Q^mid_j`` (``x_mid = block_input + attention branch output``, ``Q^mid_j`` the polar
       map of Pass A's mid cross-covariance) with upstream ``x_mid_current - x_mid_pristine``.
    2. component ``mlp.c_proj`` (D) with O_j mounted: against ``D_j`` at the block end, upstream ``out_current -
       out_pristine``, features the current target's ``c_proj`` input.

    Each solve is an exact linear least squares because the component writes straight into the residual stream, so
    after the full fit the block-end error at boundary ``j`` equals D's fit residual and, for ``mid_end``, the mid error
    after O_j (and blocks ``< j``) equals O's fit residual. For ``end_end`` the O substep is the same least squares
    with O's own write as the only unknown, but the MLP reacts to O's change of ``x_mid``, so its realized end error is
    not the O residual; D then fits what is left at the block end on the current (O-mounted) target.

    Rows follow the single-point schema plus ``sequential_points``, ``correction_point`` (``"mid"`` / ``"end"``) and
    ``upstream_effect_norm`` per substep, O's row first. The target model's entry state is restored in a ``finally``.
    """
    mid_end = config.sequential_points == "mid_end"
    substeps = [
        ("attn.out_proj", "mid" if mid_end else "end"),
        ("mlp.c_proj", "end"),
    ]
    if set(order_components(config.components)) != {"attn.out_proj", "mlp.c_proj"}:
        raise ValueError(
            f"sequential_points={config.sequential_points!r} requires components attn.out_proj and mlp.c_proj"
        )
    if mid_end and family_adapter is not None:
        raise NotImplementedError("mid_end is vision-only")
    batches = prepared["target_batches"]
    source_batches = prepared["source_batches"]
    q_by_position = prepared["q_by_position"]
    fingerprints = prepared["fingerprints"]
    q_mid_by_position = prepared.get("q_mid_by_position")
    mid_fingerprints = prepared.get("mid_fingerprints")
    positions = list(range(pairing.target_depth))
    for j in positions:
        if j not in q_by_position:
            raise ValueError(f"Missing prepared Procrustes map for position {j}")
        if mid_end and (q_mid_by_position is None or j not in q_mid_by_position or mid_fingerprints is None):
            raise ValueError(f"Missing prepared mid-block Procrustes map for position {j} (prepare with mid_alignment)")
    source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
    shim = _layout_for(family_adapter)

    unique_models: dict[int, Any] = {}
    for m in (target_model, source_base_model, source_ft_model):
        unique_models.setdefault(id(m), m)
    originals = {mid: (next(m.parameters()).device, m.training) for mid, m in unique_models.items()}

    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    pristine_model = None
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        pristine_model = deepcopy(target_model)
        pristine_model.load_state_dict({k: v.detach().cpu().clone() for k, v in target_base_state.items()}, strict=True)
        sb_model = pristine_model if source_base_model is target_model else source_base_model
        sf_model = pristine_model if source_ft_model is target_model else source_ft_model
        for m in {id(x): x for x in (target_model, sb_model, sf_model, pristine_model)}.values():
            m.to(device).eval()
        if mid_end:
            for pos in positions:
                _assert_layerscale_identity(
                    shim, shim.blocks(target_model)[pos], context=f"target block {pos}", requirement="mid_end"
                )

        for pos in positions:
            i = pairing.pairing[pos]
            for component, point in substeps:
                key = shim.component_key(pos, component, prefixed=True)
                width = int(current_state[key].shape[0])
                effective_out = _component_effective_out(shim, target_model, pos, component, width)
                stats = ResidualSufficientStatistics(device=device)
                desired_sq = 0.0
                upstream_sq = 0.0
                out_key, h_key = f"{pos}.out", f"{pos}.{component}.h"
                in_key, attn_key = f"{pos}.mid_in", f"{pos}.mid_attn"
                src_requests = {str(i): (i, "boundary")}
                tgt_requests = {out_key: (pos, "boundary"), h_key: (pos, COMPONENT_INPUT_KIND[component])}
                ref_requests = {out_key: (pos, "boundary")}
                if point == "mid":
                    src_requests = {f"{i}.mid_in": (i, "block_input"), f"{i}.mid_attn": (i, "attn_proj")}
                    mid_requests = {in_key: (pos, "block_input"), attn_key: (pos, "attn_proj")}
                    tgt_requests.update(mid_requests)
                    ref_requests.update(mid_requests)
                gens = (
                    iter_capture_tokens(
                        sb_model,
                        source_batches,
                        src_requests,
                        device,
                        family_adapter=family_adapter,
                        store_device=device,
                    ),
                    iter_capture_tokens(
                        sf_model,
                        source_batches,
                        src_requests,
                        device,
                        family_adapter=family_adapter,
                        store_device=device,
                    ),
                    iter_capture_tokens(
                        target_model, batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
                    ),
                    iter_capture_tokens(
                        pristine_model,
                        batches,
                        ref_requests,
                        device,
                        family_adapter=family_adapter,
                        store_device=device,
                    ),
                )
                with contextlib.ExitStack() as stack:
                    for gen in gens:
                        stack.enter_context(contextlib.closing(gen))
                    for k, (sb, sf, tgt, ref) in enumerate(zip(*gens, strict=True)):
                        # The pristine copy must match Pass A's fingerprints (detects a mutated base).
                        _check_boundary_fingerprint(ref[out_key], fingerprints[(k, pos)], config, pos, k)
                        if point == "mid":
                            ref_mid = _mid_residual(ref, in_key, attn_key)
                            _check_mid_fingerprint(ref_mid, mid_fingerprints[(k, pos)], config, pos, k)
                            cur = _mid_residual(tgt, in_key, attn_key).cpu()
                            b = _aligned([_mid_residual(sb, f"{i}.mid_in", f"{i}.mid_attn").cpu()], [cur])[0]
                            f = _aligned([_mid_residual(sf, f"{i}.mid_in", f"{i}.mid_attn").cpu()], [cur])[0]
                            desired = (f - b) @ q_mid_by_position[pos]
                            upstream = cur - ref_mid.cpu()
                        else:
                            cur = tgt[out_key].cpu()
                            b = _aligned([sb[str(i)].cpu()], [cur])[0]
                            f = _aligned([sf[str(i)].cpu()], [cur])[0]
                            desired = (f - b) @ q_by_position[pos]
                            upstream = cur - ref[out_key].cpu()
                        error = desired - upstream
                        desired_sq += float((desired.double() ** 2).sum().item())
                        upstream_sq += float((upstream.double() ** 2).sum().item())
                        h = tgt[h_key].cpu()
                        stats.update(
                            h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), None, effective_out
                        )

                position_corrections: dict[str, Tensor] = {}
                block_rows: list[dict[str, Any]] = []
                _finalize_independent_component(
                    stats,
                    desired_sq,
                    0.0,
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
                )
                target_corrections.update(position_corrections)
                for row in block_rows:
                    row["source_position"] = row.pop("source_coordinate")
                    row["procrustes_source"] = prepared.get("procrustes_source", "activation")
                    row["depth_fit"] = "sequential"
                    row["sequential_points"] = config.sequential_points
                    row["correction_point"] = point
                    row["upstream_effect_norm"] = upstream_sq**0.5
                    diagnostics.append(row)
                # Mount this substep (weight and bias) before the next one sees the target.
                target_model.load_state_dict({name: current_state[name] for name in position_corrections}, strict=False)
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
        del pristine_model
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
