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


def apply_depth_pairing_override(pairing: DiscreteLayerPairing, depth_pairing: str) -> DiscreteLayerPairing:
    """Ablation: rewrite ``pairing.pairing`` per ``DirectResidualConfig.depth_pairing``.

    Called by the caller (``vision_rebase._direct_residual_fit_body`` and its
    ``merge_in_source_then_fit`` sibling) immediately after
    ``DiscreteLayerPairing.compute(source_depth, target_depth)``, before any
    capture. Every Direct Residual consumer -- ``capture_paired_boundary_
    activations``, ``compute_desired_effects``, ``fit_direct_residual``, and
    their streaming equivalents -- reads ``pairing.pairing[j]`` as the single
    source of truth for BOTH which source block's activations define
    ``D_j = (S_1 - S_0) Q_j`` and which source block ``Q_j`` itself aligns
    target position ``j`` with (``compute_desired_effects`` fits ``Q_j`` from
    ``source_base[pairing.pairing[j]]`` against ``target_base[j]``). So this
    one swap, applied once at construction, changes pi(j) consistently
    everywhere downstream without touching any of those call sites.

    ``depth_pairing="relative"`` returns ``pairing`` unchanged (identity,
    bit-for-bit -- the default, golden-hash-pinned path never calls this with
    anything else).  The other three modes derive a new pairing tuple from
    the ORIGINAL ``pairing.pairing`` (the closed-form relative pairing), not
    from each other:

      * ``"reversed"``:     ``pi_rev(j) = source_depth - 1 - pairing.pairing[j]``.
      * ``"shift_plus1"``:  ``min(source_depth - 1, pairing.pairing[j] + 1)``.
      * ``"shift_minus1"``: ``max(0, pairing.pairing[j] - 1)``.

    Only valid for ``component_target="block_boundary"`` -- validated by
    ``parse_direct_residual_config``, not here (this function has no config
    to check against and is usable standalone, e.g. by tests).
    """
    if depth_pairing == "relative":
        return pairing
    source_depth = pairing.source_depth
    if depth_pairing == "reversed":
        new_pairing = tuple(source_depth - 1 - i for i in pairing.pairing)
    elif depth_pairing == "shift_plus1":
        new_pairing = tuple(min(source_depth - 1, i + 1) for i in pairing.pairing)
    elif depth_pairing == "shift_minus1":
        new_pairing = tuple(max(0, i - 1) for i in pairing.pairing)
    else:
        raise ValueError(
            f"depth_pairing must be 'relative', 'reversed', 'shift_plus1' or 'shift_minus1', got {depth_pairing!r}"
        )
    return DiscreteLayerPairing(
        source_depth=pairing.source_depth, target_depth=pairing.target_depth, pairing=new_pairing
    )


def _derive_block_seed(alignment_seed: int, position: int) -> int:
    """Deterministic per-block seed for ``alignment_map='random_isometry'``.

    A plain ``alignment_seed + position`` would work too, but hashing keeps
    nearby positions' draws decorrelated (no shared low-order-bit structure
    across an entire depth sweep) and keeps the derivation obviously
    collision-free across the ``(alignment_seed, position)`` product space
    used across a sweep of many runs.
    """
    digest = hashlib.sha256(f"direct_residual_random_isometry:{int(alignment_seed)}:{int(position)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _random_isometry_map(shape: tuple[int, int], *, seed: int) -> Tensor:
    """A random partial isometry of ``shape``, deterministic given ``seed``.

    Uses the exact same construction ``_procrustes_from_cross`` uses to turn
    a cross-covariance into the polar factor (SVD, then ``U @ Vh``) -- applied
    to a seeded standard-normal matrix instead of a cross-covariance -- so
    the result has the identical shape, orientation, and orthonormal-row/
    -column structure ``centered_rectangular_procrustes``'s polar map would
    have for the same (source, target) activation widths; only the direction
    it points in is randomized. Returned in float64, matching every other
    alignment map's fitting precision.
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
) -> dict[int, list[Tensor]]:
    """``D_j`` = Procrustes-aligned target effect at ``pairing.pairing[j]``.

    Generalizes `target_informed_runtime._capture_residual_references`'s
    inner Procrustes computation from ARIADNE's two-code-path ancestry-group
    iteration (one-to-many groups for extend, many-to-one span tracking for
    shrink -- both needed because ARIADNE tracks *which* source block a
    position descends from, for provenance) to a single flat loop. Direct
    Residual doesn't need that bookkeeping: one cardinality (one target
    position <- exactly one source position, for every regime) handles
    extend, shrink, and same-arch uniformly.

    ``residual_target`` selects what ``D_j`` is built from, given the SAME
    centered Procrustes fit ``Q_j, mu_s, mu_t = centered_rectangular_procrustes
    (S_{j,0} -> T_j^0)``:

      * ``"transported_delta"`` (default): ``D_j = (S_1 - S_0) Q_j``, exactly
        the historical expression and op order -- byte-identical to the
        pre-ablation code.
      * ``"transported_endpoint"``: ``D_j = (S_1 - mu_s) Q_j + mu_t - T_j^0``
        per batch, applying the centered map to the fine-tuned source
        endpoint and subtracting the target zero-shot endpoint.

    The two differ by exactly the Procrustes residual ``E_j = (S_0 - mu_s)
    Q_j - (T_j^0 - mu_t)`` -- the thing the fit minimizes; see
    `compute_alignment_diagnostics` for that residual's diagnostics, kept
    deliberately separate (and out of this function's cost) -- see its
    docstring for why.

    ``procrustes_source="activation"`` (default) is bit-identical to the
    pre-ablation code: ``Q_j`` is fit on the (source_base, target_base)
    ACTIVATION banks, exactly as before. ``procrustes_source="gradient"``
    changes only the statistic ``Q_j`` is fit on -- to the block-boundary
    GRADIENT banks `capture_paired_boundary_activations` captured under
    ``"source_base_gradients"``/``"target_base_gradients"`` -- using the same
    helper, the same centering and the same ``_aligned`` token interpolation.
    ``D_j = (source_ft_i - source_base_i) @ Q_j`` is unchanged in form in both
    modes: a gradient difference is never used as the regression target,
    only as the alignment statistic.

    When ``diagnostics_out`` is provided (a caller-owned, initially-empty
    dict), it is populated per position with analysis-only fields -- never
    read by any fit. In gradient mode this includes the activation-space
    ``Q`` computed purely for comparison (``"activation_gradient_procrustes_
    overlap"`` = :math:`\\lVert Q_{act}^\\top Q_{grad}\\rVert_F^2 / d_{\\min}`)
    and ``"procrustes_rank"`` (the numerical rank of the centered
    cross-covariance the gradient ``Q`` was solved from).
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
            q, _mu_gs, _mu_gt = centered_rectangular_procrustes(grad_source_rows, grad_target_rows)
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
                gs_centered = grad_source_rows - grad_source_rows.mean(dim=0)
                gt_centered = grad_target_rows - grad_target_rows.mean(dim=0)
                cross = gs_centered.T @ gt_centered
                diagnostics_out[j] = {
                    "procrustes_source": "gradient",
                    "procrustes_rank": int(torch.linalg.matrix_rank(cross)),
                    "activation_gradient_procrustes_overlap": overlap,
                    "activation_gradient_map_distance": map_distance,
                    "activation_gradient_delta_disagreement": delta_disagreement,
                    # See the activation branch below: kept for
                    # compute_fidelity_holdout_diagnostics to reuse the SAME
                    # fitted Q_j on held-out images without refitting.
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
        if diagnostics_out is not None:
            diagnostics_out[j] = {
                "procrustes_source": "activation",
                "alignment_q_frobenius": float(torch.linalg.norm(q).item()),
                "alignment_q_rank": int(torch.linalg.matrix_rank(q)),
                # The fitted map/means themselves, kept for callers that need
                # to re-apply the SAME Q_j/mu without refitting (e.g.
                # compute_fidelity_holdout_diagnostics, which must evaluate
                # D_j on held-out images using the fit's own Q_j, never a
                # freshly refit one). Never read by any fit.
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
        q, mx, my = centered_rectangular_procrustes(_rows(xs), _rows(ys))
        return q, mx, my, {"alignment_map": "polar", "alignment_row_weighting": "uniform"}
    if alignment_map == "random_isometry":
        # random_isometry always requires uniform row weighting (validated by
        # _validate_alignment_options), so the mean/centering is identical to
        # the plain uniform-polar path above -- only the map Q itself, whose
        # shape/orientation the fast path's centered_rectangular_procrustes
        # call also determines, is replaced by a random partial isometry of
        # that same shape.
        if random_isometry_seed is None:
            raise ValueError("alignment_map='random_isometry' requires random_isometry_seed")
        q_polar, mx, my = centered_rectangular_procrustes(_rows(xs), _rows(ys))
        q = _random_isometry_map(tuple(q_polar.shape), seed=random_isometry_seed)
        return (
            q,
            mx,
            my,
            {
                "alignment_map": "random_isometry",
                "alignment_row_weighting": "uniform",
                "alignment_seed_used": int(random_isometry_seed),
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
        return q, mu_x, mu_y, {"alignment_map": "ridge", "alignment_row_weighting": "uniform", **diag}
    q = _procrustes_from_cross(cross)
    return (
        q,
        mu_x,
        mu_y,
        {
            "alignment_map": "polar",
            "alignment_row_weighting": row_weighting,
            "weighted_cross_frobenius": float(torch.linalg.norm(cross).item()),
        },
    )


def compute_alignment_diagnostics(
    captured: Mapping[str, Any], pairing: DiscreteLayerPairing
) -> dict[int, dict[str, float]]:
    """Per-position centered-Procrustes alignment diagnostics. Analysis-only.

    Recomputes the SAME deterministic centered Procrustes fit ``compute_
    desired_effects`` computes internally (`centered_rectangular_procrustes`
    is a pure function of the captured rows, so this is a second, independent
    call, not a cached one), then reports the residual it minimizes and a few
    derived quantities -- never anything fed back into a fit.

    Deliberately NOT called from ``compute_desired_effects`` or folded into
    its cost: the caller (`vision_rebase._run_direct_residual_fit`) times and
    peak-memory-profiles the delta/endpoint construction as its own
    "alignment_calibration" bracket, and this function's float64 N x d_t
    temporaries (N in the tens of thousands of rows) would otherwise inflate
    that bracket's recorded seconds/peak-memory even in the default
    ``residual_target="transported_delta"`` path -- contaminating any
    cross-code-generation cost comparison for a number this function's own
    diagnostics never influence. Call it as a separate, untimed (or
    separately timed) step instead.

    Per position ``j``, all in float64 on the concatenated (all-batch) rows:
    ``procrustes_error_norm`` = ``||E_j||`` where ``E_j = (S_0 - mu_s) Q_j -
    (T^0 - mu_t)``; ``procrustes_relative_error`` = ``||E_j|| / ||T^0 -
    mu_t||`` (0 if the denominator is 0); ``delta_target_norm`` = ``||(S_1 -
    S_0) Q_j||``; ``endpoint_minus_delta_over_delta`` = ``||E_j|| /
    delta_target_norm`` (0 if the denominator is 0) -- how the transported-
    delta and transported-endpoint targets actually differ, scaled against
    the delta itself (not against ``T^0``, which is typically much larger
    than a fine-tuning delta); ``procrustes_error_in_range_norm`` /
    ``procrustes_error_out_of_range_norm`` = ``||E_j Q_j^T Q_j||`` /
    ``||E_j (I - Q_j^T Q_j)||`` (the latter is exactly 0 when ``d_t <=
    d_s``, since ``Q_j^T Q_j`` is only a proper projector -- rank ``d_s`` --
    when ``d_t > d_s``); ``mean_offset_norm`` = ``||mu_s Q_j - mu_t||``
    (documents what the literal, non-affine ``S Q`` form would have added).
    Also ``source_dim``, ``target_dim``.
    """
    source_base = captured["source_base_outputs"]
    source_ft = captured["source_ft_outputs"]
    target_by_position = captured["target_base_outputs_by_position"]
    if set(target_by_position) != set(range(pairing.target_depth)):
        raise ValueError(
            "Captured target references do not exactly match the pairing's target positions: "
            f"expected={sorted(range(pairing.target_depth))}, found={sorted(target_by_position)}"
        )
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
        q64, mu_s64, mu_t64 = centered_rectangular_procrustes(s0_rows, t0_rows)

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
    source_rows: Tensor, target_rows: Tensor, *, eps: float = 1e-8
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the polar factor of the centered source/target cross-covariance.

    ``source_rows`` is ``[N, d_source]`` and ``target_rows`` is
    ``[N, d_target]``; the returned map is ``[d_source, d_target]``.
    For same-width maps and source-to-target extensions this also minimizes
    ``||X Q - Y||_F`` under the corresponding orthogonality constraint. For
    shrink maps it maximizes cross-covariance alignment, but generally does
    not minimize that least-squares objective because ``||X Q||`` varies with
    the selected source subspace.
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


@cost_phase_decorator("transformation")
def _procrustes_from_cross(cross: Tensor) -> Tensor:
    """Orthogonal Procrustes map from a cross-covariance matrix via SVD."""
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return u @ vh


def _check_rows(h: Tensor, e: Tensor, h_name: str, e_name: str) -> None:
    if h.ndim != 2 or e.ndim != 2:
        raise ValueError(f"{h_name} and {e_name} must be rank-2")
    if h.shape[0] != e.shape[0]:
        raise ValueError("activation and residual row counts must match")
