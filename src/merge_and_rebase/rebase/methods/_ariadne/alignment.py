"""Activation alignment maps, desired effects and alignment diagnostics."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch

from ....utils.cost_accounting import cost_phase_decorator
from ...discrete_layer_match import DiscreteLayerPairing
from .layouts import _aligned, _rows

Tensor = torch.Tensor


def apply_depth_pairing_override(
    pairing: DiscreteLayerPairing, depth_pairing: str, *, ancestry: DiscreteLayerPairing | None = None
) -> DiscreteLayerPairing:
    """Ablation: rewrite ``pairing.pairing`` per ``DirectResidualConfig.depth_pairing``.

    Called right after ``DiscreteLayerPairing.compute`` and before any capture. Every consumer
    (capture, ``compute_desired_effects``, fit, and their streaming equivalents) reads
    ``pairing.pairing[j]`` as the single source of truth for both which source block defines
    ``D_j = (S_1 - S_0) Q_j`` and which one ``Q_j`` aligns target ``j`` with, so one swap here
    changes pi(j) consistently everywhere.

    ``"relative"`` returns ``pairing`` unchanged (bit-for-bit; the default, golden-hash-pinned
    path). The other modes derive from the ORIGINAL relative pairing, not from each other:
    ``"reversed"``: ``source_depth - 1 - pi(j)``; ``"shift_plus1"``: ``min(source_depth - 1, pi(j) + 1)``;
    ``"shift_minus1"``: ``max(0, pi(j) - 1)``; ``"spread_duplicate"``: the BRACE ancestor of ``j`` (``ancestry``).
    Only valid for ``component_target="block_boundary"``
    (validated by ``parse_direct_residual_config``, not here).
    """
    if depth_pairing == "relative":
        return pairing
    if depth_pairing == "brace_ancestry":
        raise ValueError("depth_pairing 'brace_ancestry' was renamed to 'spread_duplicate'")
    if depth_pairing == "spread_duplicate":
        # pi(j) = the BRACE ancestor of target position j (rebase.depth_pairing.spread_duplicate_pairing).
        if ancestry is None:
            raise ValueError("depth_pairing='spread_duplicate' needs the ancestry pairing (spread_duplicate_pairing)")
        if (ancestry.source_depth, ancestry.target_depth) != (pairing.source_depth, pairing.target_depth):
            raise ValueError("ancestry pairing depths do not match the source/target depths")
        return ancestry
    source_depth = pairing.source_depth
    if depth_pairing == "reversed":
        new_pairing = tuple(source_depth - 1 - i for i in pairing.pairing)
    elif depth_pairing == "shift_plus1":
        new_pairing = tuple(min(source_depth - 1, i + 1) for i in pairing.pairing)
    elif depth_pairing == "shift_minus1":
        new_pairing = tuple(max(0, i - 1) for i in pairing.pairing)
    else:
        raise ValueError(
            f"depth_pairing must be 'relative', 'reversed', 'shift_plus1', 'shift_minus1' or 'spread_duplicate', got {depth_pairing!r}"
        )
    return DiscreteLayerPairing(
        source_depth=pairing.source_depth, target_depth=pairing.target_depth, pairing=new_pairing
    )


def _derive_block_seed(alignment_seed: int, position: int) -> int:
    """Deterministic per-block seed for ``alignment_map='random_isometry'``.

    Hashed (not ``alignment_seed + position``) so nearby positions' draws are decorrelated.
    """
    digest = hashlib.sha256(f"direct_residual_random_isometry:{int(alignment_seed)}:{int(position)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _random_isometry_map(shape: tuple[int, int], *, seed: int) -> Tensor:
    """A random partial isometry of ``shape``, deterministic given ``seed``.

    Built like ``_procrustes_from_cross`` (SVD, ``U @ Vh``) on a seeded standard-normal matrix, so
    shape and orthonormal structure match the polar map for the same widths. Returned in float64.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gaussian = torch.randn(shape, generator=generator, dtype=torch.float64)
    return _procrustes_from_cross(gaussian)


@cost_phase_decorator("transformation")
def compute_desired_effects(
    captured: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    residual_target: str = "transported_delta",
    procrustes_source: str = "activation",
    alignment_map: str = "polar",
    alignment_row_weighting: str = "uniform",
    alignment_seed: int = 0,
    diagnostics_out: dict[int, dict[str, Any]] | None = None,
    rank_out: dict[int, dict[str, Any]] | None = None,
) -> dict[int, list[Tensor]]:
    """``D_j`` = Procrustes-aligned target effect at ``pairing.pairing[j]``.

    Fits the centered Procrustes ``Q_j, mu_s, mu_t = centered_rectangular_procrustes(S_{j,0} -> T_j^0)``
    per target position and returns per-batch ``D_j`` lists (one flat loop; one source position per
    target position, for extend, shrink and same-arch alike).

    ``residual_target``: ``"transported_delta"`` (default) gives ``D_j = (S_1 - S_0) Q_j``, byte-identical
    to the pre-ablation code (expression and op order); ``"transported_endpoint"`` gives
    ``D_j = (S_1 - mu_s) Q_j + mu_t - T_j^0``. They differ by the Procrustes residual ``E_j``, whose
    diagnostics live in ``compute_alignment_diagnostics`` (kept out of this function's cost).

    ``procrustes_source="activation"`` (default) is bit-identical to the pre-ablation code.
    ``"gradient"`` fits ``Q_j`` on the block-boundary gradient banks (``"source_base_gradients"`` /
    ``"target_base_gradients"``) with the same helper, centering and ``_aligned`` interpolation; ``D_j`` keeps
    the same form (gradients are only the alignment statistic, never the regression target).

    ``diagnostics_out`` (caller-owned, empty dict) receives analysis-only per-position fields, never read by
    any fit; in gradient mode this includes activation-vs-gradient ``Q`` comparisons. ``rank_out`` (same
    contract) receives only the scalar ``RANK_DIAGNOSTIC_KEYS`` in every mode; the rank comes from the polar
    SVD itself (no extra decomposition on the default path) and ``Q`` is bit-identical.
    """
    if procrustes_source not in {"activation", "gradient"}:
        raise ValueError(f"procrustes_source must be 'activation' or 'gradient', got {procrustes_source!r}")
    _validate_alignment_options(alignment_map, alignment_row_weighting, procrustes_source)
    if (alignment_map != "polar" or alignment_row_weighting != "uniform") and residual_target != "transported_delta":
        raise ValueError("non-default alignment options require residual_target='transported_delta'")
    source_base = captured["source_base_outputs"]
    source_ft = captured["source_ft_outputs"]
    target_by_position = captured["target_base_outputs_by_position"]
    if set(target_by_position) != set(range(pairing.target_depth)):
        raise ValueError(
            "Captured target references do not exactly match the pairing's target positions: "
            f"expected={sorted(range(pairing.target_depth))}, found={sorted(target_by_position)}"
        )
    if residual_target not in {"transported_delta", "transported_endpoint"}:
        raise ValueError("residual_target must be 'transported_delta' or 'transported_endpoint'")
    source_base_grad = captured.get("source_base_gradients")
    target_base_grad = captured.get("target_base_gradients")
    if procrustes_source == "gradient" and (source_base_grad is None or target_base_grad is None):
        raise ValueError(
            "procrustes_source='gradient' requires 'source_base_gradients' and 'target_base_gradients' in "
            "captured; call capture_paired_boundary_activations with procrustes_source='gradient'"
        )
    if residual_target == "transported_endpoint" and procrustes_source != "activation":
        raise ValueError("residual_target='transported_endpoint' requires procrustes_source='activation'")
    desired: dict[int, list[Tensor]] = {}
    for j in range(pairing.target_depth):
        i = pairing.pairing[j]
        if i not in source_base or i not in source_ft:
            raise ValueError(f"Missing captured source reference for pairing index {i} at target position {j}")
        targets = target_by_position[j]
        source_base_batches = _aligned(source_base[i], targets)
        source_ft_batches = _aligned(source_ft[i], targets)
        if procrustes_source == "gradient":
            if i not in source_base_grad or j not in target_base_grad:
                raise ValueError(f"Missing captured gradient reference for pairing index {i} at target position {j}")
            aligned_source_grad = _aligned(source_base_grad[i], target_base_grad[j])
            grad_source_rows = _rows(aligned_source_grad).double()
            grad_target_rows = _rows(target_base_grad[j]).double()
            q, _mu_gs, _mu_gt, grad_rank = centered_rectangular_procrustes(
                grad_source_rows, grad_target_rows, return_rank=True
            )
            grad_rank_diag = procrustes_rank_diagnostics(grad_rank, q.shape[0], q.shape[1])
            if rank_out is not None:
                rank_out[j] = grad_rank_diag
            if diagnostics_out is not None:
                q_act, _mu_s, _mu_t = centered_rectangular_procrustes(
                    _rows(source_base_batches).double(), _rows(targets).double()
                )
                d_min = min(q.shape)
                overlap = float(((q_act.T @ q).norm() ** 2) / d_min)
                map_distance = float(torch.linalg.norm(q_act - q) / (2.0 * d_min) ** 0.5)
                delta_act = (
                    torch.cat(
                        [(f - b).double() for b, f in zip(source_base_batches, source_ft_batches, strict=True)], 0
                    )
                    @ q_act
                )
                delta_grad = (
                    torch.cat(
                        [(f - b).double() for b, f in zip(source_base_batches, source_ft_batches, strict=True)], 0
                    )
                    @ q
                )
                delta_disagreement = float(
                    torch.linalg.norm(delta_act - delta_grad) / (torch.linalg.norm(delta_act) + 1e-12)
                )
                diagnostics_out[j] = {
                    "procrustes_source": "gradient",
                    # Rank from the SVD that produced q.
                    **grad_rank_diag,
                    "activation_gradient_procrustes_overlap": overlap,
                    "activation_gradient_map_distance": map_distance,
                    "activation_gradient_delta_disagreement": delta_disagreement,
                    # Kept so compute_fidelity_holdout_diagnostics reuses the SAME fitted Q_j.
                    "q": q.detach().float().clone(),
                }
            q = q.float()
            desired[j] = [(f - b) @ q for b, f in zip(source_base_batches, source_ft_batches, strict=True)]
            continue
        q, mu_s, mu_t, alignment_diag = _fit_activation_map(
            source_base_batches,
            targets,
            source_ft_batches,
            alignment_map=alignment_map,
            row_weighting=alignment_row_weighting,
            random_isometry_seed=_derive_block_seed(alignment_seed, j) if alignment_map == "random_isometry" else None,
        )
        if rank_out is not None:
            rank_out[j] = {k: alignment_diag[k] for k in RANK_DIAGNOSTIC_KEYS}
        if diagnostics_out is not None:
            diagnostics_out[j] = {
                "procrustes_source": "activation",
                "alignment_q_frobenius": float(torch.linalg.norm(q).item()),
                "alignment_q_rank": int(torch.linalg.matrix_rank(q)),
                # Fitted map/means, kept so held-out diagnostics re-apply the SAME Q_j/mu (never a
                # refit). Never read by any fit.
                "q": q.detach().float().clone(),
                "mu_s": mu_s.detach().float().clone(),
                "mu_t": mu_t.detach().float().clone(),
                **alignment_diag,
            }
        if residual_target == "transported_delta":
            q = q.float()
            desired[j] = [(f - b) @ q for b, f in zip(source_base_batches, source_ft_batches, strict=True)]
        else:
            q = q.float()
            mu_s = mu_s.float()
            mu_t = mu_t.float()
            desired[j] = [(f - mu_s) @ q + mu_t - t for f, t in zip(source_ft_batches, targets, strict=True)]
    return desired


def _validate_alignment_options(alignment_map, row_weighting, procrustes_source="activation"):
    if alignment_map not in {"polar", "ridge", "random_isometry"}:
        raise ValueError("alignment_map must be 'polar', 'ridge' or 'random_isometry'")
    if row_weighting not in {"uniform", "cls_balanced", "delta_magnitude"}:
        raise ValueError("alignment_row_weighting must be 'uniform', 'cls_balanced' or 'delta_magnitude'")
    if alignment_map != "polar" and row_weighting != "uniform":
        raise ValueError(f"alignment_map={alignment_map!r} requires uniform row weighting")
    if (alignment_map != "polar" or row_weighting != "uniform") and procrustes_source != "activation":
        raise ValueError("non-default alignment options require procrustes_source='activation'")


def _fit_activation_map(
    source_batches,
    target_batches,
    ft_batches,
    *,
    alignment_map="polar",
    row_weighting="uniform",
    random_isometry_seed: int | None = None,
):
    """Fit an activation map, optionally weighting each image's token rows."""
    xs, ys = [], []
    for x, y in zip(source_batches, target_batches, strict=True):
        xs.append(x.double())
        ys.append(y.double())
    # Keep the historical uniform polar call, including its exact reduction
    # order, so default configurations retain their golden hashes.
    if alignment_map == "polar" and row_weighting == "uniform":
        q, mx, my, rank = centered_rectangular_procrustes(_rows(xs), _rows(ys), return_rank=True)
        return (
            q,
            mx,
            my,
            {
                "alignment_map": "polar",
                "alignment_row_weighting": "uniform",
                **procrustes_rank_diagnostics(rank, q.shape[0], q.shape[1]),
            },
        )
    if alignment_map == "random_isometry":
        # Requires uniform weighting (validated), so centering matches the uniform-polar path;
        # only Q is replaced by a random partial isometry of the same shape.
        if random_isometry_seed is None:
            raise ValueError("alignment_map='random_isometry' requires random_isometry_seed")
        q_polar, mx, my, rank = centered_rectangular_procrustes(_rows(xs), _rows(ys), return_rank=True)
        q = _random_isometry_map(tuple(q_polar.shape), seed=random_isometry_seed)
        return (
            q,
            mx,
            my,
            {
                "alignment_map": "random_isometry",
                "alignment_row_weighting": "uniform",
                "alignment_seed_used": int(random_isometry_seed),
                # Rank of the cross-covariance the polar factor WOULD use (polar_derived=False).
                **procrustes_rank_diagnostics(rank, q.shape[0], q.shape[1], polar_derived=False),
            },
        )
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    weights = torch.ones(x.shape[:2], dtype=torch.float64, device=x.device)
    if row_weighting == "cls_balanced":
        if x.shape[1] < 2:
            raise ValueError("cls_balanced alignment requires a CLS token and at least one patch token")
        weights[:, 0] = 0.5
        weights[:, 1:] = 0.5 / (x.shape[1] - 1)
    elif row_weighting == "delta_magnitude":
        if ft_batches is None:
            raise ValueError("delta_magnitude alignment requires source_ft_batches")
        ds = torch.cat([(f.double() - b.double()) for b, f in zip(source_batches, ft_batches, strict=True)], 0)
        norms = torch.linalg.vector_norm(ds, dim=-1)
        means = norms.mean(dim=1, keepdim=True)
        image_scale = torch.where(
            means > 0, (norms / means.clamp_min(torch.finfo(norms.dtype).tiny)).clamp(0.25, 4.0), torch.ones_like(norms)
        )
        weights = image_scale
    # Each image receives equal total mass; within-image token weights sum to 1.
    weights = weights / weights.sum(dim=1, keepdim=True)
    weights = weights / weights.sum()
    mu_x = (x * weights[..., None]).sum((0, 1))
    mu_y = (y * weights[..., None]).sum((0, 1))
    xc, yc = x - mu_x, y - mu_y
    xf, yf, wf = xc.reshape(-1, x.shape[-1]), yc.reshape(-1, y.shape[-1]), weights.reshape(-1)
    cross = xf.T @ (yf * wf[:, None])
    if alignment_map == "ridge":
        q, _, _, diag = centered_ridge_alignment(xf, yf)
        return (
            q,
            mu_x,
            mu_y,
            {
                "alignment_map": "ridge",
                "alignment_row_weighting": "uniform",
                **diag,
                **procrustes_rank_diagnostics(_cross_rank(cross), q.shape[0], q.shape[1], polar_derived=False),
            },
        )
    q, rank = _procrustes_from_cross(cross, return_rank=True)
    return (
        q,
        mu_x,
        mu_y,
        {
            "alignment_map": "polar",
            "alignment_row_weighting": row_weighting,
            "weighted_cross_frobenius": float(torch.linalg.norm(cross).item()),
            **procrustes_rank_diagnostics(rank, q.shape[0], q.shape[1]),
        },
    )


def compute_alignment_diagnostics(
    captured: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    alignment_map: str = "polar",
    alignment_row_weighting: str = "uniform",
    alignment_seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Per-position alignment diagnostics of the CONFIGURED activation map. Analysis-only.

    Refits the SAME deterministic activation map ``compute_desired_effects`` uses (a second call, not a
    cache) and reports the residual it minimizes; nothing here is fed back into a fit.

    ``alignment_map`` / ``alignment_row_weighting`` / ``alignment_seed`` must be the values the fit used. For
    the default (``polar``, ``uniform``) the historical uniform centered-Procrustes expression is kept verbatim
    (bit-identical); otherwise ``Q_j`` and the weighted means come from ``_fit_activation_map``, as in
    ``compute_alignment_diagnostics_streaming``. Error / delta norms are plain (unweighted) Frobenius norms.

    Deliberately NOT called from ``compute_desired_effects``: the caller times and peak-memory-profiles that
    step as its own "alignment_calibration" bracket, and this function's float64 N x d_t temporaries would
    inflate it and contaminate cross-code-generation cost comparisons. Call it as a separate step.

    Per position ``j``, float64 on all-batch rows, with ``E_j = (S_0 - mu_s) Q_j - (T^0 - mu_t)``:
    ``procrustes_error_norm`` = ``||E_j||``; ``procrustes_relative_error`` = ``||E_j|| / ||T^0 - mu_t||``;
    ``delta_target_norm`` = ``||(S_1 - S_0) Q_j||``; ``endpoint_minus_delta_over_delta`` =
    ``||E_j|| / delta_target_norm`` (both ratios are 0 if the denominator is 0);
    ``procrustes_error_in_range_norm`` / ``procrustes_error_out_of_range_norm`` = ``||E_j Q_j^T Q_j||`` /
    ``||E_j (I - Q_j^T Q_j)||`` (the latter is exactly 0 when ``d_t <= d_s``); ``mean_offset_norm`` =
    ``||mu_s Q_j - mu_t||``; plus ``source_dim``, ``target_dim``.
    """
    source_base = captured["source_base_outputs"]
    source_ft = captured["source_ft_outputs"]
    target_by_position = captured["target_base_outputs_by_position"]
    if set(target_by_position) != set(range(pairing.target_depth)):
        raise ValueError(
            "Captured target references do not exactly match the pairing's target positions: "
            f"expected={sorted(range(pairing.target_depth))}, found={sorted(target_by_position)}"
        )
    _validate_alignment_options(alignment_map, alignment_row_weighting)
    default_map = alignment_map == "polar" and alignment_row_weighting == "uniform"
    alignment_diagnostics: dict[int, dict[str, float]] = {}
    for j in range(pairing.target_depth):
        i = pairing.pairing[j]
        if i not in source_base or i not in source_ft:
            raise ValueError(f"Missing captured source reference for pairing index {i} at target position {j}")
        targets = target_by_position[j]
        source_base_batches = _aligned(source_base[i], targets)
        source_ft_batches = _aligned(source_ft[i], targets)
        s0_rows = _rows(source_base_batches).double()
        s1_rows = _rows(source_ft_batches).double()
        t0_rows = _rows(targets).double()
        if default_map:
            q64, mu_s64, mu_t64 = centered_rectangular_procrustes(s0_rows, t0_rows)
        else:
            q64, mu_s64, mu_t64, _ = _fit_activation_map(
                source_base_batches,
                targets,
                source_ft_batches,
                alignment_map=alignment_map,
                row_weighting=alignment_row_weighting,
                random_isometry_seed=_derive_block_seed(alignment_seed, j)
                if alignment_map == "random_isometry"
                else None,
            )

        e = (s0_rows - mu_s64) @ q64 - (t0_rows - mu_t64)
        procrustes_error_norm = float(torch.linalg.norm(e))
        target_centered_norm = float(torch.linalg.norm(t0_rows - mu_t64))
        procrustes_relative_error = procrustes_error_norm / target_centered_norm if target_centered_norm > 0 else 0.0
        delta_target_norm = float(torch.linalg.norm((s1_rows - s0_rows) @ q64))
        endpoint_minus_delta_over_delta = procrustes_error_norm / delta_target_norm if delta_target_norm > 0 else 0.0
        # E @ Q^T @ Q rather than forming the d_t x d_t projector explicitly.
        e_in_range = (e @ q64.T) @ q64
        e_out_of_range = e - e_in_range
        procrustes_error_in_range_norm = float(torch.linalg.norm(e_in_range))
        procrustes_error_out_of_range_norm = float(torch.linalg.norm(e_out_of_range))
        mean_offset_norm = float(torch.linalg.norm(mu_s64 @ q64 - mu_t64))
        alignment_diagnostics[j] = {
            "procrustes_error_norm": procrustes_error_norm,
            "procrustes_relative_error": procrustes_relative_error,
            "delta_target_norm": delta_target_norm,
            "endpoint_minus_delta_over_delta": endpoint_minus_delta_over_delta,
            "procrustes_error_in_range_norm": procrustes_error_in_range_norm,
            "procrustes_error_out_of_range_norm": procrustes_error_out_of_range_norm,
            "mean_offset_norm": mean_offset_norm,
            "source_dim": int(s0_rows.shape[-1]),
            "target_dim": int(t0_rows.shape[-1]),
        }
    return alignment_diagnostics


@cost_phase_decorator("transformation")
def centered_rectangular_procrustes(
    source_rows: Tensor, target_rows: Tensor, *, eps: float = 1e-8, return_rank: bool = False
):
    """Return the polar factor of the centered source/target cross-covariance.

    ``source_rows`` is ``[N, d_source]`` and ``target_rows`` is
    ``[N, d_target]``; the returned map is ``[d_source, d_target]``.
    For same-width maps and source-to-target extensions this also minimizes
    ``||X Q - Y||_F`` under the corresponding orthogonality constraint. For
    shrink maps it maximizes cross-covariance alignment, but generally does
    not minimize that least-squares objective because ``||X Q||`` varies with
    the selected source subspace.

    With ``return_rank=True`` a fourth element, the numerical rank of the centered cross-covariance
    (from the same SVD), is appended; ``q`` and the means are unchanged bit for bit.
    """
    _check_rows(source_rows, target_rows, "source_rows", "target_rows")
    if source_rows.shape[0] == 0:
        raise ValueError("Procrustes requires at least one row")
    if not torch.isfinite(source_rows).all() or not torch.isfinite(target_rows).all():
        raise ValueError("Procrustes inputs must be finite")
    if eps <= 0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be finite and > 0")
    source_rows = source_rows.to(torch.float64)
    target_rows = target_rows.to(torch.float64)
    src_mean = source_rows.mean(dim=0)
    tgt_mean = target_rows.mean(dim=0)
    cross = (source_rows - src_mean).T @ (target_rows - tgt_mean)
    if return_rank:
        q, rank = _procrustes_from_cross(cross, return_rank=True)
        return q, src_mean, tgt_mean, rank
    q = _procrustes_from_cross(cross)
    return q, src_mean, tgt_mean


@cost_phase_decorator("transformation")
def centered_ridge_alignment(
    source_rows: Tensor, target_rows: Tensor, *, ridge: float | None = None
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    """Fit a centered, ridge-regularized linear source-to-target map.

    With no explicit ridge, uses ``trace(X.T @ X) / (N - 1)``. Returns the
    map, row means and compact solver diagnostics. The zero-covariance case
    maps to zero; one-row inputs are accepted and also map to zero.
    """
    _check_rows(source_rows, target_rows, "source_rows", "target_rows")
    if source_rows.shape[0] == 0:
        raise ValueError("ridge alignment requires at least one row")
    if not torch.isfinite(source_rows).all() or not torch.isfinite(target_rows).all():
        raise ValueError("ridge alignment inputs must be finite")
    x, y = source_rows.to(torch.float64), target_rows.to(torch.float64)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    gram, cross = xc.T @ xc, xc.T @ yc
    tr = float(torch.trace(gram).item())
    lam = tr / float(x.shape[0] - 1) if ridge is None and x.shape[0] > 1 else (0.0 if ridge is None else float(ridge))
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("ridge must be finite and >= 0")
    if tr == 0.0:
        mapping = torch.zeros((x.shape[1], y.shape[1]), dtype=x.dtype, device=x.device)
    else:
        mapping = torch.linalg.solve(gram + lam * torch.eye(gram.shape[0], dtype=x.dtype, device=x.device), cross)
    return mapping, mx, my, {"ridge": lam, "source_trace": tr}


def _numerical_rank(singular_values: Tensor, shape: tuple[int, int]) -> int:
    """Numerical rank with ``torch.linalg.matrix_rank``'s default tolerance (``max(shape) * eps * s_max``)."""
    if singular_values.numel() == 0:
        return 0
    tol = singular_values.max() * max(shape) * torch.finfo(singular_values.dtype).eps
    return int((singular_values > tol).sum().item())


def _cross_rank(cross: Tensor) -> int:
    """Numerical rank of a cross-covariance (one extra ``svdvals``; used where no polar SVD is run)."""
    return _numerical_rank(torch.linalg.svdvals(cross), tuple(cross.shape))


def procrustes_rank_diagnostics(rank: int, source_dim: int, target_dim: int, *, polar_derived: bool = True) -> dict:
    """Per-position rank diagnostics of the cross-covariance a Procrustes map was solved from.

    ``procrustes_q_non_unique`` is True when the polar factor ``U V^T`` is not unique, i.e. the
    cross-covariance has numerical rank below ``min(d_source, d_target)`` (the null directions of
    ``U``/``V`` can be rotated freely without changing the SVD). It is only meaningful (and only
    ever True) when the fitted map actually is that polar factor (``polar_derived``): ridge and
    random-isometry maps are reported with their rank but ``procrustes_q_non_unique=False``.
    Analysis-only: never read by any fit, and the fitted Q is NOT altered by a non-unique flag.
    """
    min_dim = int(min(source_dim, target_dim))
    return {
        "procrustes_rank": int(rank),
        "procrustes_min_dim": min_dim,
        "procrustes_q_non_unique": bool(polar_derived and int(rank) < min_dim),
    }


RANK_DIAGNOSTIC_KEYS = ("procrustes_rank", "procrustes_min_dim", "procrustes_q_non_unique")


@cost_phase_decorator("transformation")
def _procrustes_from_cross(cross: Tensor, *, return_rank: bool = False):
    """Orthogonal Procrustes map from a cross-covariance matrix via SVD.

    With ``return_rank=True`` returns ``(q, numerical_rank)`` from the SAME SVD (``q`` is bit-identical
    to the default return value).
    """
    u, s, vh = torch.linalg.svd(cross, full_matrices=False)
    q = u @ vh
    if return_rank:
        return q, _numerical_rank(s, tuple(cross.shape))
    return q


def _check_rows(h: Tensor, e: Tensor, h_name: str, e_name: str) -> None:
    if h.ndim != 2 or e.ndim != 2:
        raise ValueError(f"{h_name} and {e_name} must be rank-2")
    if h.shape[0] != e.shape[0]:
        raise ValueError("activation and residual row counts must match")
