"""``anchors``: shared-anchor transfer of fine-tuning effects (universal functional rebasin).

Per target block ``j`` (source block ``i = pairing[j]``, the same relative-depth rule Ariadne uses)::

    E  = S_ft,i - S_base,i                    on the task calibration rows (uncentred), canonical token grid
    R  = (E / c_m,i) P_m,i^T                  encode: common anchor coordinates
    D  = scale * R (K_k,j + eps I)^{-1} P_k,j decode: target coordinates, scale = c_k,j (or c_m,i)
    tau = fit_direct_residual(H_k, D)         Ariadne's ridge (Eq. 9-10), the SAME function, unchanged

``D`` is resampled from the canonical grid to the target's native grid before the ridge; encode/decode and
the bilinear resampling are all linear row maps, so the order does not matter.

The frame (anchors, mu, c, P per block) comes from a generic pool, never from the task; the task calibration
only provides ``E`` and the ridge rows, exactly as in Ariadne (``paired_calibration``, same seed and budget).
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

from ....utils.cost_accounting import PhaseCostRecorder, recording
from ...discrete_layer_match import DiscreteLayerPairing
from .._ariadne.capture import (
    capture_paired_boundary_activations,
    capture_tokens,
    iter_capture_tokens,
    paired_calibration,
)
from .._ariadne.fit import fit_direct_residual
from .._ariadne.streaming import fit_direct_residual_sequential_streaming, prepare_direct_residual_streaming
from .canonical_stats import heldout_gamma_out, heldout_soft_phi, heldout_soft_phi_from_sigma
from .config import TEXT_ANCHOR_POOLS, AnchorsConfig, preserve_supported, validate_text_config
from .core import (
    CANONICAL_DECODES,
    CoverageAccumulator,
    canonical_factors_from_frames,
    canonical_transport_map,
    decode_rows,
    dedup_mask,
    encode_rows,
    gpa,
    gpa_transport_map,
    kernel,
    kernel_diagnostics,
    kernel_diagnostics_from_frames,
    kernel_spectrum,
    kernel_spectrum_from_frame,
    resolve_eps,
    resolve_eps_from_frame,
    round_trip_operator_primal,
    row_normalize,
    spectral_transport_map,
    transport_map_primal,
    whitened_transport_map,
)
from .state import AnchorState, RegisteredModel, to_canonical, to_rows
from .text_units import capture_paired_word_activations, paired_word_calibration
from .universe import fit_consensus, fit_universe

Tensor = torch.Tensor


@dataclass(frozen=True)
class AnchorsPrepared:
    """Same shape as ``AriadnePrepared``: ``task_vector`` at ``config.strength``, ``unit_task_vector`` at 1."""

    task_vector: dict[str, Tensor]
    unit_task_vector: dict[str, Tensor]
    timing: dict[str, dict[str, float]]
    diagnostics: list[dict[str, Any]]
    extra: dict[str, Any]
    config: AnchorsConfig = field(repr=False)


def _primal(p: Tensor) -> bool:
    """``T > d``: decode through the ``d x d`` Gram (core module docstring); otherwise the ``T x T`` kernel."""
    return int(p.shape[0]) > int(p.shape[1])


def filtered_frame(reg: RegisteredModel, block: int, keep: Tensor, k: float) -> tuple[Tensor, float]:
    """``P' = (S - mu_k) / c_k`` on the kept anchors, ``S = P c + mu`` (the stored frame), ``mu_k``/``c_k`` without
    the high-norm registration rows (``AnchorState.norm_filter``)."""
    flt = reg.extras[f"norm_k{k:g}"]
    s_rows = reg.p[block].to(torch.float64) * float(reg.c[block]) + reg.mu[block].to(torch.float64)
    c_f = float(flt["c"][block])
    return (s_rows[keep] - flt["mu"][block].to(torch.float64)) / c_f, c_f


def filtered_pair_inputs(
    source: RegisteredModel, source_block: int, target: RegisteredModel, target_block: int, k: float
) -> tuple[Tensor, float, Tensor, float, Tensor]:
    """``(mu_m, c_m, mu_k, c_k, keep)`` of the norm filter, i.e. exactly the centring / scale / row mask under which
    ``filtered_frame`` builds ``P' = (S - mu) / c`` (``keep``: ``[T]`` bool, anchors flagged in neither model).
    ``PairSigmaAccumulator(mu_m, c_m, mu_k, c_k)`` fed the raw rows ``S[keep]`` (or ``update(..., mask=keep[rows])``)
    then holds ``Sigma = P'^T P'`` of the frame the decoder uses (before the optional R2 dedup / R3 row norm)."""
    flt_m, flt_k = source.extras[f"norm_k{k:g}"], target.extras[f"norm_k{k:g}"]
    return (
        flt_m["mu"][source_block].to(torch.float64),
        float(flt_m["c"][source_block]),
        flt_k["mu"][target_block].to(torch.float64),
        float(flt_k["c"][target_block]),
        anchor_keep([(source, source_block), (target, target_block)], k),
    )


def decoder_scales(
    source: RegisteredModel, source_block: int, target: RegisteredModel, target_block: int, config: AnchorsConfig
) -> tuple[float, float]:
    """``(c_m, c_k)`` the decoder of ``pair_frames`` uses: the registration ``c``, or the high-norm-filtered ``c``
    (``AnchorState.norm_filter``) under ``anchor_filter``."""
    if config.anchor_filter is None:
        return float(source.c[source_block]), float(target.c[target_block])
    k = float(config.anchor_filter_k)
    return (
        float(source.extras[f"norm_k{k:g}"]["c"][source_block]),
        float(target.extras[f"norm_k{k:g}"]["c"][target_block]),
    )


def anchor_keep(regs_blocks: list[tuple[RegisteredModel, int]], k: float) -> Tensor:
    """Anchors flagged high-norm in none of the given (model, block) pairs."""
    flagged = None
    for reg, block in regs_blocks:
        f = reg.extras[f"norm_k{k:g}"]["anchor_flags"][block]
        flagged = f if flagged is None else (flagged | f)
    return ~flagged


def pair_frames(
    source: RegisteredModel, source_block: int, target: RegisteredModel, target_block: int, *, config: AnchorsConfig
) -> tuple[Tensor, float, Tensor, float]:
    """``(P_m, c_m, P_k, c_k)`` of one block pair, with the high-norm filter, then the R2 dedup (measured in the
    source) and the R3 row normalisation applied when configured."""
    if config.anchor_filter is None:
        p_m, c_m = source.p[source_block].to(torch.float64), source.c[source_block]
        p_k, c_k = target.p[target_block].to(torch.float64), target.c[target_block]
    else:
        k = float(config.anchor_filter_k)
        keep = anchor_keep([(source, source_block), (target, target_block)], k)
        p_m, c_m = filtered_frame(source, source_block, keep, k)
        p_k, c_k = filtered_frame(target, target_block, keep, k)
    if config.anchor_dedup_cos is not None:
        sel = dedup_mask(p_m, config.anchor_dedup_cos, max_count=config.anchor_dedup_max, seed=config.anchor_seed)
        p_m, p_k = p_m[sel], p_k[sel]
    if config.anchor_row_norm:
        p_m, p_k = row_normalize(p_m), row_normalize(p_k)
    return p_m, c_m, p_k, c_k


def _heldout_canonical_map(
    source: RegisteredModel, source_block: int, target: RegisteredModel, target_block: int,
    p_m: Tensor, c_m: float, p_k: Tensor, scale: float, config: AnchorsConfig,
):
    """Plan decoder D3 (``canonical_soft_weights='heldout'``): ``Phi`` with directions ``Y, Z, V`` from all rows of the pair
    frames and weights ``g = clip(Gamma_out, 0, 1)`` (zero below ``canonical_null_threshold``), ``Gamma_out`` from the
    id-split halves (``canonical_stats.heldout_gamma_out``). Info: ``canonical_soft_weights`` 'heldout' and the
    mean / min / max of ``Gamma_out`` and of the used weights."""
    ids = source.anchor_ids
    if ids is None or target.anchor_ids is None:
        raise ValueError("canonical_soft_weights='heldout' needs the row ids of the anchors (models registered through "
                         "AnchorState with anchor_sampling 'k_per_image' / 'k_per_document')")
    if config.anchor_filter is not None:
        ids = ids[anchor_keep([(source, source_block), (target, target_block)], float(config.anchor_filter_k))]
    if int(ids.shape[0]) != int(p_m.shape[0]):
        raise ValueError(f"row ids ({int(ids.shape[0])}) and anchor frame rows ({int(p_m.shape[0])}) differ")
    gamma_out, split_info = heldout_gamma_out(p_m, p_k, ids, config.anchor_seed)
    factors = canonical_factors_from_frames(p_m, p_k)
    phi, info = heldout_soft_phi(factors, gamma_out, c_m=c_m, scale=scale, null_threshold=config.canonical_null_threshold)
    info.update({f"canonical_{k}": v for k, v in split_info.items()})
    return phi, info


def transport_map(
    source: RegisteredModel,
    source_block: int,
    target: RegisteredModel,
    target_block: int,
    *,
    config: AnchorsConfig,
    gpa_maps: Mapping[int, tuple[Tensor, Tensor]] | None = None,
):
    """``(rows -> D, eps, spectral info)`` for one block pair: encode at the source block, decode at the target block.

    ``T <= d_k``: ``R = (E / c_m) P_m^T``, ``D = scale * R (K_k + eps I)^{-1} P_k`` (kernel form).
    ``T > d_k``: the same map as one ``[d_m, d_k]`` matrix ``Phi`` (``transport_map_primal``), built once.
    ``spectral_filter != 'tikhonov'``: ``Phi`` from ``spectral_transport_map`` (``eps`` is then None).
    ``kernel_decode`` zca / agreement: ``Phi`` from ``whitened_transport_map`` (``eps`` = the target's eps).
    """
    p_m, c_m, p_k, c_k = pair_frames(source, source_block, target, target_block, config=config)
    scale = c_k if config.rescale_target else c_m
    if config.frame_map == "consensus":
        if gpa_maps is None:
            raise ValueError("frame_map='consensus' needs the universe maps (fit_consensus)")
        a_m, b_k = gpa_maps[target_block]
        phi = gpa_transport_map(a_m, b_k, c_m, scale=scale)
        return (lambda e: e.to(torch.float64) @ phi.to(e.device)), None, {}
    if config.frame_map == "gpa":
        if gpa_maps is not None:
            a_m, a_k = gpa_maps[target_block]
            info = {}
        else:  # no universe given: the two-model GPA of this block pair, in memory
            _, (a_m, a_k), info = gpa([p_m, p_k], d_u=config.gpa_dim)
            info = {k: v for k, v in info.items() if k != "gpa_residual_per_model"}
        phi = gpa_transport_map(a_m, a_k, c_m, scale=scale, rank=config.gpa_rank)
        return (lambda e: e.to(torch.float64) @ phi.to(e.device)), None, info
    if config.spectral_filter != "tikhonov":
        phi, info = spectral_transport_map(
            p_m, c_m, p_k, scale=scale, mode=config.spectral_filter, rank=config.spectral_rank
        )
        return (lambda e: e.to(torch.float64) @ phi.to(e.device)), None, info
    if config.canonical_source == "sigma_all_tokens":
        sig = target.extras.get("canonical_sigma")
        if sig is None:
            raise ValueError("canonical_source='sigma_all_tokens' needs the streamed statistics (prepare computes them)")
        phi, info = heldout_soft_phi_from_sigma(
            sig.sigma(source_block, target_block, 0), sig.sigma(source_block, target_block, 1), c_m=c_m, scale=scale,
            null_threshold=config.canonical_null_threshold,
        )
        info["canonical_source"] = "sigma_all_tokens"
        return (lambda e: e.to(torch.float64) @ phi.to(e.device)), None, info
    if config.kernel_decode in CANONICAL_DECODES and config.canonical_soft_weights == "heldout":
        phi, info = _heldout_canonical_map(source, source_block, target, target_block, p_m, c_m, p_k, scale, config)
        return (lambda e: e.to(torch.float64) @ phi.to(e.device)), None, info
    if config.kernel_decode in CANONICAL_DECODES:
        phi, info = canonical_transport_map(
            p_m, c_m, p_k, scale=scale, mode=config.kernel_decode, rank=config.canonical_rank,
            null_threshold=0.0 if config.canonical_null_threshold is None else config.canonical_null_threshold,
        )
        return (lambda e: e.to(torch.float64) @ phi.to(e.device)), None, info
    if config.kernel_decode != "tikhonov":
        phi, info = whitened_transport_map(
            p_m, c_m, p_k, scale=scale, eps_rel=config.eps_rel, mode=config.kernel_decode, tau=config.agreement_tau
        )
        return (lambda e: e.to(torch.float64) @ phi.to(e.device)), info["kernel_eps_target"], info
    if _primal(p_k):
        eps = resolve_eps_from_frame(p_k, config.eps_rel)
        phi = transport_map_primal(p_m, c_m, p_k, scale=scale, eps=eps)
        return (lambda e: e.to(torch.float64) @ phi.to(e.device)), eps, {}
    k_k = kernel(p_k)
    eps = resolve_eps(k_k, config.eps_rel)
    return (lambda e: decode_rows(encode_rows(e, p_m, c_m), p_k, k_k, scale=scale, eps=eps)), eps, {}


def transport_rows(
    e: Tensor,
    source: RegisteredModel,
    source_block: int,
    target: RegisteredModel,
    target_block: int,
    *,
    config: AnchorsConfig,
) -> Tensor:
    """Encode ``e`` (``[n, d_m]`` canonical rows) at the source block and decode it at the target block."""
    fn, _, _ = transport_map(source, source_block, target, target_block, config=config)
    return fn(e)


def _source_coverage(acc: CoverageAccumulator, p_m: Tensor, eps_rel: float) -> dict[str, float]:
    if _primal(p_m):
        return acc.coverage_from_operator(round_trip_operator_primal(p_m, resolve_eps_from_frame(p_m, eps_rel)))
    k_m = kernel(p_m)
    return acc.coverage(p_m, k_m, resolve_eps(k_m, eps_rel))


def _pair_diagnostics(p_m: Tensor, p_k: Tensor) -> dict[str, float]:
    if _primal(p_m) or _primal(p_k):
        kd, ks = kernel_diagnostics_from_frames(p_m, p_k), kernel_spectrum_from_frame(p_k)
    else:
        kd, ks = kernel_diagnostics(kernel(p_m), kernel(p_k)), kernel_spectrum(kernel(p_k))
    return {
        **{f"anchors_kernel_{k}": v for k, v in kd.items()},
        **{f"anchors_target_kernel_{k}": v for k, v in ks.items()},
    }


def _frame_moments(reg: RegisteredModel, block: int, config: AnchorsConfig) -> tuple[Tensor, float]:
    """``(mu, c)`` the frame of ``block`` was centred and scaled with (the norm-filtered ones when configured)."""
    if config.anchor_filter is None:
        return reg.mu[block].to(torch.float64), float(reg.c[block])
    flt = reg.extras[f"norm_k{float(config.anchor_filter_k):g}"]
    return flt["mu"][block].to(torch.float64), float(flt["c"][block])


def _operator_keep(config: AnchorsConfig, source: RegisteredModel, target: RegisteredModel):
    """``keep(side, block, rows) -> bool mask`` of the anchor norm filter's rule (``||x - mu|| <= k median``), or None."""
    if not config.operator_norm_filter:
        return None
    k = float(config.anchor_filter_k)
    regs = {"source": source, "target": target}
    meds = {side: reg.extras[f"norm_k{k:g}"]["median"] for side, reg in regs.items()}

    def keep(side: str, block: int, rows: Tensor) -> Tensor:
        mu = regs[side].mu[block].to(device=rows.device, dtype=torch.float64)
        return (rows.to(torch.float64) - mu).norm(dim=-1) <= k * float(meds[side][block])

    return keep


def _pool_input_maps(source_model, target_model, pool_batches, pairing, source, target, *, config, gpa_maps, keep,
                     device) -> tuple[dict[int, tuple[Tensor, Tensor]], dict[int, tuple[Tensor, Tensor]]]:
    """``operator_input_map='pool_ridge'``: per target position j, the ridge ``u_k -> u_m`` (GPA coordinates, both
    frames' moments) on every canonical token of the anchor images kept by the norm filter in both models, moments
    streamed batch by batch. ``(maps, CLS maps)``: with ``operator_cls_split`` the first map is fitted on the patch
    rows and the second on the CLS rows; otherwise one map on all rows and no CLS maps."""
    from .operators import _Moments, common_rows, fit_from_moments

    if pool_batches is None:
        raise ValueError("operator_input_map='pool_ridge' needs the anchor pool batches")
    canon, rank = int(config.canonical_tokens), config.gpa_rank
    distinct = sorted({int(i) for i in pairing.pairing})
    split = bool(config.operator_cls_split)
    moms = {j: _Moments() for j in range(pairing.target_depth)}
    moms_cls = {j: _Moments() for j in range(pairing.target_depth)} if split else {}
    frames = {("s", i): _frame_moments(source, i, config) for i in distinct}
    frames.update({("t", j): _frame_moments(target, j, config) for j in range(pairing.target_depth)})
    for sb, tb in zip(*pool_batches, strict=True):
        s_out = capture_tokens(source_model, [sb], {str(i): (i, "boundary") for i in distinct}, device)
        t_out = capture_tokens(target_model, [tb], {str(j): (j, "boundary") for j in range(pairing.target_depth)}, device)
        for j in range(pairing.target_depth):
            i = int(pairing.pairing[j])
            sc = to_canonical(s_out[str(i)][0].double(), canon)
            tc = to_canonical(t_out[str(j)][0].double(), canon)
            m = torch.ones(sc.shape[0] * sc.shape[1], dtype=torch.bool, device=sc.device)
            if keep is not None:
                m = keep("source", i, sc).reshape(-1) & keep("target", j, tc).reshape(-1)
            (mu_m, c_m), (mu_k, c_k) = frames[("s", i)], frames[("t", j)]
            a_m, a_k = gpa_maps[j]
            uk, um = common_rows((tc - mu_k) / c_k, a_k, rank), common_rows((sc - mu_m) / c_m, a_m, rank)
            if split:
                cls = _cls_rows(sc)
                m = m | cls  # CLS rows never filtered under the split (the readout row)
                moms[j].update(uk[m & ~cls], um[m & ~cls])
                moms_cls[j].update(uk[m & cls], um[m & cls])
            else:
                moms[j].update(uk[m], um[m])
    fit = lambda mm: {j: (f.w, f.b) for j, f in ((j, fit_from_moments(mom)) for j, mom in mm.items())}
    return fit(moms), fit(moms_cls)


def _cls_rows(x: Tensor) -> Tensor:
    """``[B*T]`` bool, True on the CLS row (token 0) of every image of ``x [B, T, d]`` (rows flattened in order)."""
    flags = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
    flags[:, 0] = True
    return flags.reshape(-1)


def operator_desired_effects(
    source_base_model,
    source_ft_model,
    target_model,
    source_loader,
    target_loader,
    pairing: DiscreteLayerPairing,
    source: RegisteredModel,
    target: RegisteredModel,
    *,
    config: AnchorsConfig,
    gpa_maps: Mapping[int, tuple[Tensor, Tensor]],
    device,
    library_entry: Mapping[str, Any] | None = None,
    pool_batches: tuple[list[Any], list[Any]] | None = None,
) -> tuple[dict[str, Any], dict[int, list[Tensor]], dict[int, dict[str, Any]]]:
    """O1 (``library='operator'``): ``(captured for the ridge, {j: [D per batch]}, per-position diagnostics)``.

    One seeded draw of ``num_batches + operator_holdout_batches (+ num_batches if disjoint)`` batches (prefix
    consistent with the data library's draw): the first ``num_batches`` fit the source operators, the next
    ``operator_holdout_batches`` give their held-out R^2, and the target rows are the first ``num_batches`` ('same',
    exactly the data library's) or the last ``num_batches`` ('disjoint', never seen by the source).
    """
    from .operators import (
        common_rows, decode_operator, explained, fit_effect_operator, operator_nbytes, r_squared, save_library,
        to_common_space,
    )

    nb, ho = int(config.num_batches), int(config.operator_holdout_batches)
    extra = nb if config.target_calibration == "disjoint" else 0
    source_batches, target_batches, metadata = paired_calibration(
        source_loader, target_loader, num_batches=nb + ho + extra, seed=config.seed
    )
    if len(source_batches) != nb + ho + extra:
        raise ValueError(f"task calibration too small for library='operator': {len(source_batches)} batches")
    tgt = target_batches[:nb] if extra == 0 else target_batches[nb + ho :]
    distinct = sorted({int(i) for i in pairing.pairing})
    src_req = {str(i): (i, "boundary") for i in distinct}
    base_out = capture_tokens(source_base_model, source_batches[: nb + ho], src_req, device)
    ft_out = capture_tokens(source_ft_model, source_batches[: nb + ho], src_req, device)
    tgt_out = capture_tokens(target_model, tgt, {str(j): (j, "boundary") for j in range(pairing.target_depth)}, device)
    rank = config.gpa_rank
    keep = _operator_keep(config, source, target)
    canon = int(config.canonical_tokens)
    split = bool(config.operator_cls_split)
    src: dict[int, tuple[list[Tensor], list[Tensor], list[Tensor], float, list[Tensor]]] = {}
    ops: dict[int, Any] = {}
    for i in distinct:
        mu, c = _frame_moments(source, i, config)
        b, f = base_out[str(i)], ft_out[str(i)]
        hb = [(x.to(torch.float64) - mu) / c for x in b]
        eb = [(y.to(torch.float64) - x.to(torch.float64)) / c for x, y in zip(b, f, strict=True)]
        if keep is not None:
            masks = [keep("source", i, x).reshape(-1) for x in b]
        else:
            masks = [torch.ones(x.shape[0] * x.shape[1], dtype=torch.bool, device=x.device) for x in b]
        if split:
            masks = [m | _cls_rows(x) for x, m in zip(hb, masks, strict=True)]  # CLS rows never filtered
        h = [x.reshape(-1, x.shape[-1])[m] for x, m in zip(hb, masks, strict=True)]
        e = [y.reshape(-1, y.shape[-1])[m] for y, m in zip(eb, masks, strict=True)]
        cls = [_cls_rows(x)[m] for x, m in zip(hb, masks, strict=True)]
        src[i] = (hb, h, e, c, cls)
        if config.operator_space == "native":
            op = fit_effect_operator(h[:nb], e[:nb], rank=config.operator_rank)
            op.info["r2_train"] = r_squared(op, h[:nb], e[:nb])
            op.info["r2_heldout"] = r_squared(op, h[nb:], e[nb:])
            ops[i] = op
    pool_maps: dict[int, tuple[Tensor, Tensor]] = {}
    pool_maps_cls: dict[int, tuple[Tensor, Tensor]] = {}
    if config.operator_input_map == "pool_ridge":
        pool_maps, pool_maps_cls = _pool_input_maps(source_base_model, target_model, pool_batches, pairing, source, target,
                                     config=config, gpa_maps=gpa_maps, keep=keep, device=device)
    desired: dict[int, list[Tensor]] = {}
    diag: dict[int, dict[str, Any]] = {}
    library: dict[int, tuple[Tensor, Tensor]] = {}
    library_cls: dict[int, tuple[Tensor, Tensor]] = {}
    composed: dict[int, tuple[Tensor, Tensor]] = {}
    for j in range(pairing.target_depth):
        i = int(pairing.pairing[j])
        hb, h, e, c_m, cls = src[i]
        a_m, a_k = gpa_maps[j]
        uh = [common_rows(x, a_m, rank) for x in h]
        ue = [common_rows(y, a_m, rank) for y in e]
        if config.operator_space == "native":
            op = ops[i]
            w_u, b_u = to_common_space(op, a_m, rank)
        elif not split:
            op = fit_effect_operator(uh[:nb], ue[:nb], rank=config.operator_rank)
            op.info["r2_train"] = r_squared(op, uh[:nb], ue[:nb])
            op.info["r2_heldout"] = r_squared(op, uh[nb:], ue[nb:])
            w_u, b_u = op.w, op.b
        else:
            # patch operator (op, W_U) and CLS operator (op_c, W_c); the R^2 entries are those of the patch one
            pick = lambda rows, flags, want: [x[f == want] for x, f in zip(rows, flags, strict=True)]
            op = fit_effect_operator(pick(uh[:nb], cls[:nb], False), pick(ue[:nb], cls[:nb], False),
                                     rank=config.operator_rank)
            op_c = fit_effect_operator(pick(uh[:nb], cls[:nb], True), pick(ue[:nb], cls[:nb], True),
                                       rank=config.operator_rank)
            op.info["r2_train"] = r_squared(op, pick(uh[:nb], cls[:nb], False), pick(ue[:nb], cls[:nb], False))
            op.info["r2_heldout"] = r_squared(op, pick(uh[nb:], cls[nb:], False), pick(ue[nb:], cls[nb:], False))
            w_u, b_u = op.w, op.b
            library_cls[j] = (op_c.w, op_c.b)
        library[j] = (w_u, b_u)

        def source_pred(x: Tensor, flags: Tensor) -> Tensor:
            p = x @ w_u + b_u
            if split:
                p[flags] = x[flags] @ library_cls[j][0] + library_cls[j][1]
            return p

        # the operator in the common space, on the source's own held-out rows (no transfer of the inputs); the CLS
        # rows alone as well (the readout row)
        r2_src, cos_src = explained([source_pred(x, f) for x, f in zip(uh[nb:], cls[nb:], strict=True)], ue[nb:])
        r2_cls, _ = explained([source_pred(x, f)[f] for x, f in zip(uh[nb:], cls[nb:], strict=True)],
                              [y[f] for y, f in zip(ue[nb:], cls[nb:], strict=True)])
        mu_k, c_k = _frame_moments(target, j, config)
        hk = [(t.to(torch.float64) - mu_k) / c_k for t in tgt_out[str(j)]]
        input_map = pool_maps.get(j)
        input_map_cls = pool_maps_cls.get(j, input_map)
        if config.operator_input_map == "anchor_ridge":
            p_m, _, p_k, _ = pair_frames(source, i, target, j, config=config)
            fitted = fit_effect_operator([common_rows(p_k, a_k, rank)], [common_rows(p_m, a_m, rank)])
            input_map = input_map_cls = (fitted.w, fitted.b)
        input_diag: dict[str, Any] = {}
        if extra == 0:
            # the same task images through both models, on the canonical grid: how well the target's common
            # coordinates (after the input map) reproduce the source's (rows kept by the norm filter in both)
            uk_c, um_c, cls_c = [], [], []
            for t, x, sb in zip(tgt_out[str(j)], hb[:nb], base_out[str(i)][:nb], strict=True):
                tc, sc = to_canonical(t.double(), canon), to_canonical(sb.double(), canon)
                m = torch.ones(tc.shape[0] * tc.shape[1], dtype=torch.bool, device=tc.device)
                if keep is not None:
                    m = keep("target", j, tc).reshape(-1) & keep("source", i, sc).reshape(-1)
                    m = m | _cls_rows(tc) if split else m
                uk_c.append(common_rows((tc - mu_k) / c_k, a_k, rank)[m])
                um_c.append(common_rows(to_canonical(x, canon), a_m, rank)[m])
                cls_c.append(_cls_rows(tc)[m])
            if config.operator_input_map == "paired" and not split:
                fitted = fit_effect_operator(uk_c, um_c)
                input_map = input_map_cls = (fitted.w, fitted.b)
            elif config.operator_input_map == "paired":
                fitted = fit_effect_operator([u[~f] for u, f in zip(uk_c, cls_c)], [u[~f] for u, f in zip(um_c, cls_c)])
                fitted_c = fit_effect_operator([u[f] for u, f in zip(uk_c, cls_c)], [u[f] for u, f in zip(um_c, cls_c)])
                input_map, input_map_cls = (fitted.w, fitted.b), (fitted_c.w, fitted_c.b)
            carry = lambda u, mp: u if mp is None else u @ mp[0] + mp[1]
            pred = [torch.where(f.unsqueeze(-1), carry(u, input_map_cls), carry(u, input_map))
                    for u, f in zip(uk_c, cls_c, strict=True)]
            r2_in, cos_in = explained(pred, um_c)
            r2_in_cls, _ = explained([p[f] for p, f in zip(pred, cls_c)], [u[f] for u, f in zip(um_c, cls_c)])
            input_diag = {"anchors_op_input_r2": r2_in, "anchors_op_input_cos": cos_in,
                          "anchors_op_input_r2_cls": r2_in_cls}
        scale = c_k if config.rescale_target else c_m
        if library_entry is not None and not split:
            # the operator on the TARGET's common coordinates: input map and decode scale folded in
            # (D = (u W~ + b~) A_k), so operators of different sources towards this target can be fused (O4 F2 / F3)
            m_w, m_b = (None, None) if input_map is None else input_map
            w_t = w_u if m_w is None else m_w @ w_u
            b_t = b_u if m_b is None else m_b @ w_u + b_u
            composed[j] = (float(scale) * w_t, float(scale) * b_t)
        out = [decode_operator(x, w_u, b_u, a_k, rank, scale, input_map) for x in hk]
        if split:
            for d, x in zip(out, hk, strict=True):
                d[:, :1] = decode_operator(x[:, :1], *library_cls[j], a_k, rank, scale, input_map_cls)
        out = [d.float() for d in out]
        dropped = 0.0
        if keep is not None:
            masks = [keep("target", j, t) for t in tgt_out[str(j)]]
            if split:
                for m in masks:
                    m[:, 0] = True
            out = [d * m.unsqueeze(-1).to(d.dtype) for d, m in zip(out, masks, strict=True)]
            dropped = 1.0 - sum(float(m.sum()) for m in masks) / sum(m.numel() for m in masks)
        desired[j] = out
        diag[j] = {
            "anchors_source_block": i,
            "anchors_d_norm": float(sum(float((d.double() ** 2).sum()) for d in out) ** 0.5),
            "anchors_op_target_rows_dropped": dropped,
            "anchors_op_common_r2_heldout": r2_src,
            "anchors_op_common_cos_heldout": cos_src,
            "anchors_op_common_r2_heldout_cls": r2_cls,
            **input_diag,
            **{f"anchors_op_{k}": v for k, v in op.info.items()},
            "anchors_op_library_bytes": operator_nbytes(w_u, config.operator_rank) * (2 if split else 1),
        }
    if library_entry is not None:
        # One entry per (source, task), in the coordinates of the GPA frame it was encoded in; keyed by the target
        # grid point j (the map A_m of grid point j); c_m per entry so a decode can restore the source scale.
        from pathlib import Path

        path = Path(library_entry["dir"]) / str(library_entry["universe_version"])
        path = path / f"{library_entry['source_model_id']}__{library_entry['task']}.pt"
        save_library(path, library, {**dict(library_entry), "gpa_rank": rank, "operator_rank": config.operator_rank,
                                     "c_source": {j: float(src[int(pairing.pairing[j])][3]) for j in library},
                                     "cls_split": split})
        if composed:
            save_library(path.with_name(path.stem + "__composed.pt"), composed,
                         {**dict(library_entry), "gpa_rank": rank, "operator_rank": config.operator_rank,
                          "acts_on": "target common coordinates u_k = h_k A_k^T, input map and scale folded in",
                          "input_map": config.operator_input_map})
        if split:
            save_library(path.with_name(path.stem + "__cls.pt"), library_cls,
                         {**dict(library_entry), "gpa_rank": rank, "operator_rank": config.operator_rank, "rows": "cls"})
    if extra == 0:
        # Diagnostic only: the data library's D on the very same target rows (same calibration draw), and its cosine
        # with the operator's D per position (all rows, and the rows the norm filter keeps).
        data_d, _ = decode_desired_effects(
            {"source_base_outputs": {i: base_out[str(i)][:nb] for i in distinct},
             "source_ft_outputs": {i: ft_out[str(i)][:nb] for i in distinct},
             "target_base_outputs_by_position": {j: tgt_out[str(j)] for j in desired}},
            pairing, source, target, config=config, gpa_maps=gpa_maps,
        )
        for j in desired:
            dot = nd = no = 0.0
            for d_op, d_dat in zip(desired[j], data_d[j], strict=True):
                d_op, d_dat = d_op.double(), d_dat.double()
                dot += float((d_op * d_dat).sum())
                no += float((d_op**2).sum())
                nd += float((d_dat**2).sum())
            diag[j]["anchors_op_cos_data"] = dot / max((no * nd) ** 0.5, 1e-300)
            diag[j]["anchors_op_norm_over_data"] = (no / max(nd, 1e-300)) ** 0.5
    captured = {"target_batches": tgt, "target_base_outputs_by_position": {j: tgt_out[str(j)] for j in desired},
                "calibration": {**metadata, "target_calibration": config.target_calibration,
                                "operator_holdout_batches": ho}}
    return captured, desired, diag


def fused_operator_effects(
    source_loader,
    target_model,
    target_loader,
    pool_target_batches: list[Any] | None,
    pairing: DiscreteLayerPairing,
    source: RegisteredModel,
    target: RegisteredModel,
    *,
    config: AnchorsConfig,
    gpa_maps: Mapping[int, tuple[Tensor, Tensor]],
    universe_version: str,
    target_model_id: str,
    device,
) -> tuple[dict[str, Any], dict[int, list[Tensor]], dict[int, dict[str, Any]]]:
    """O4 F2/F3 (``operator_fusion_entries``): ``(captured for the ridge, {j: [D per batch]}, diagnostics)`` of the
    fused target-side operators, decoded once on the target rows of ``operator_decode_rows``."""
    from pathlib import Path

    from .operators import decode_operator, fuse_operators, load_library

    entries = []
    for path in config.operator_fusion_entries:
        ops, meta = load_library(Path(path))
        if meta.get("target_model_id") != target_model_id or str(meta.get("universe_version")) != str(universe_version):
            raise ValueError(f"{path}: entry for target {meta.get('target_model_id')!r} in frame "
                             f"{meta.get('universe_version')!r}, this run is {target_model_id!r} in {universe_version!r}")
        if "acts_on" not in meta:
            raise ValueError(f"{path}: not a target-side (__composed) library entry")
        if set(ops) != set(range(pairing.target_depth)):
            raise ValueError(f"{path}: grid points {sorted(ops)} != the target's {pairing.target_depth}")
        entries.append(ops)
    weights = list(config.operator_fusion_weights) or [1.0] * len(entries)
    nb = int(config.num_batches)
    if config.operator_decode_rows == "pool":
        if pool_target_batches is None:
            raise ValueError("operator_decode_rows='pool' needs the anchor pool batches")
        tgt, metadata = list(pool_target_batches), {"rows": "anchor pool"}
    else:
        _, target_batches, metadata = paired_calibration(source_loader, target_loader, num_batches=nb, seed=config.seed)
        tgt = target_batches[:nb]
    tgt_out = capture_tokens(target_model, tgt, {str(j): (j, "boundary") for j in range(pairing.target_depth)}, device)
    keep = _operator_keep(config, source, target)
    desired: dict[int, list[Tensor]] = {}
    diag: dict[int, dict[str, Any]] = {}
    for j in range(pairing.target_depth):
        w, b = fuse_operators([e[j] for e in entries], weights, config.operator_fusion)
        mu_k, c_k = _frame_moments(target, j, config)
        _, a_k = gpa_maps[j]
        out = [decode_operator((t.to(torch.float64) - mu_k) / c_k, w, b, a_k, config.gpa_rank, 1.0).float()
               for t in tgt_out[str(j)]]
        dropped = 0.0
        if keep is not None:
            masks = [keep("target", j, t) for t in tgt_out[str(j)]]
            out = [d * m.unsqueeze(-1).to(d.dtype) for d, m in zip(out, masks, strict=True)]
            dropped = 1.0 - sum(float(m.sum()) for m in masks) / sum(m.numel() for m in masks)
        desired[j] = out
        diag[j] = {"anchors_d_norm": float(sum(float((d.double() ** 2).sum()) for d in out) ** 0.5),
                   "anchors_fusion": config.operator_fusion, "anchors_fusion_entries": len(entries),
                   "anchors_fusion_w_norm": float(torch.linalg.norm(w)), "anchors_op_target_rows_dropped": dropped}
    captured = {"target_batches": tgt, "target_base_outputs_by_position": {j: tgt_out[str(j)] for j in desired},
                "calibration": {**metadata, "operator_decode_rows": config.operator_decode_rows,
                                "operator_fusion_entries": list(config.operator_fusion_entries)}}
    return captured, desired, diag


def decode_desired_effects(
    captured: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    source: RegisteredModel,
    target: RegisteredModel,
    *,
    config: AnchorsConfig,
    gpa_maps: Mapping[int, tuple[Tensor, Tensor]] | None = None,
    family_adapter=None,
) -> tuple[dict[int, list[Tensor]], dict[int, dict[str, Any]]]:
    """``{j: [D per batch]}`` on the target's native token grid, plus per-position diagnostics.

    Decoders (``family_adapter`` given): the captures are ``[1, N, d]`` banks of the same content rows in the three
    models (one combined row mask), so ``D`` is decoded row by row with no grid resampling.
    """
    base = captured["source_base_outputs"]
    ft = captured["source_ft_outputs"]
    target_ref = captured["target_base_outputs_by_position"]
    canonical = config.canonical_tokens
    text = family_adapter is not None
    desired: dict[int, list[Tensor]] = {}
    diag: dict[int, dict[str, Any]] = {}
    coverage: dict[int, dict[str, float]] = {}
    for j in range(pairing.target_depth):
        i = int(pairing.pairing[j])
        acc = CoverageAccumulator() if (config.coverage_diagnostics and i not in coverage) else None
        fn, eps, spectral = transport_map(source, i, target, j, config=config, gpa_maps=gpa_maps)
        out: list[Tensor] = []
        e_sq = d_sq = 0.0
        for b0, b1, t0 in zip(base[i], ft[i], target_ref[j], strict=True):
            e = to_rows(b1 - b0, canonical, text)
            batch, n_rows, d_m = e.shape
            rows = e.reshape(-1, d_m)
            if acc is not None:
                acc.update(rows)
            d = fn(rows)
            if text:
                if tuple(t0.shape[:2]) != (batch, n_rows):
                    raise ValueError(f"source rows {(batch, n_rows)} and target rows {tuple(t0.shape[:2])} differ")
                d = d.reshape(batch, n_rows, -1).float()
            else:
                d = to_canonical(d.reshape(batch, canonical, -1).float(), int(t0.shape[1]))
            e_sq += float((rows.double() ** 2).sum().item())
            d_sq += float((d.double() ** 2).sum().item())
            out.append(d)
        desired[j] = out
        p_m = source.p[i].to(torch.float64)
        p_k = target.p[j].to(torch.float64)
        if acc is not None:
            coverage[i] = _source_coverage(acc, p_m, config.eps_rel)
        diag[j] = {
            "anchors_source_block": i,
            "anchors_c_source": source.c[i],
            "anchors_c_target": target.c[j],
            **dict(zip(("anchors_c_source_used", "anchors_c_target_used"), decoder_scales(source, i, target, j, config), strict=True)),
            "anchors_eps": eps,
            **{f"anchors_{k}": v for k, v in spectral.items()},
            "anchors_e_norm": e_sq**0.5,
            "anchors_d_norm": d_sq**0.5,
            **_pair_diagnostics(p_m, p_k),
            **({f"anchors_{k}": v for k, v in coverage[i].items()} if i in coverage else {}),
        }
    return desired, diag


class SequentialDecoder:
    """``desired_fn`` for Ariadne's sequential fit: the anchor decode of one batch, plus the per-position diagnostics.

    Called as ``decoder(pos, i, b, f, out)`` with the source base/FT boundary of source block ``i`` (source grid)
    and the current target boundary ``out``; returns ``D_pos`` on the target grid, exactly as
    ``decode_desired_effects`` does for that batch. The sequential fit then regresses block ``pos`` on
    ``D_pos - U_pos`` (``U_pos`` = the change already produced by the fitted upstream blocks).
    """

    def __init__(
        self,
        source: RegisteredModel,
        target: RegisteredModel,
        config: AnchorsConfig,
        gpa_maps: Mapping[int, tuple[Tensor, Tensor]] | None = None,
        text: bool = False,
    ) -> None:
        self.source, self.target, self.config = source, target, config
        self.gpa_maps = gpa_maps
        # decoders: rows are the tokens of one shared token sequence, no canonical patch grid to resample
        self.text = text
        self._maps: dict[int, tuple[Any, Any, dict[str, float]]] = {}
        self._norms: dict[int, list[float]] = {}
        self._coverage_acc: dict[int, CoverageAccumulator] = {}
        self._source_of: dict[int, int] = {}

    def __call__(self, pos: int, i: int, b: Tensor, f: Tensor, out: Tensor) -> Tensor:
        cfg = self.config
        if pos not in self._maps:
            self._maps[pos] = transport_map(self.source, i, self.target, pos, config=cfg, gpa_maps=self.gpa_maps)
            self._norms[pos] = [0.0, 0.0]
            self._source_of[pos] = i
        fn = self._maps[pos][0]
        if self.text:
            if (f - b).shape[1] != out.shape[1]:
                raise ValueError("anchors sequential fit on decoders needs source and target on the same token sequence")
            e = f - b
        else:
            e = to_canonical(f - b, cfg.canonical_tokens)
        batch, n_tok, d_m = e.shape
        rows = e.reshape(-1, d_m)
        if cfg.coverage_diagnostics and self._owner(i) == pos:
            self._coverage_acc.setdefault(i, CoverageAccumulator()).update(rows)
        if self.text:
            d = fn(rows).reshape(batch, n_tok, -1).float()
        else:
            d = to_canonical(fn(rows).reshape(batch, cfg.canonical_tokens, -1).float(), int(out.shape[1]))
        self._norms[pos][0] += float((rows.double() ** 2).sum().item())
        self._norms[pos][1] += float((d.double() ** 2).sum().item())
        return d

    def _owner(self, i: int) -> int:
        """The first target position decoded from source block ``i`` collects its coverage (once per ``i``)."""
        return min(p for p, src in self._source_of.items() if src == i)

    def diagnostics(self) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        coverage = {
            i: _source_coverage(acc, self.source.p[i].to(torch.float64), self.config.eps_rel)
            for i, acc in self._coverage_acc.items()
        }
        for pos, (_, eps, spectral) in self._maps.items():
            i = self._source_of[pos]
            e_sq, d_sq = self._norms[pos]
            out[pos] = {
                "anchors_source_block": i,
                "anchors_c_source": self.source.c[i],
                "anchors_c_target": self.target.c[pos],
                **dict(zip(
                    ("anchors_c_source_used", "anchors_c_target_used"),
                    decoder_scales(self.source, i, self.target, pos, self.config), strict=True,
                )),
                "anchors_eps": eps,
                **{f"anchors_{k}": v for k, v in spectral.items()},
                "anchors_e_norm": e_sq**0.5,
                "anchors_d_norm": d_sq**0.5,
                **_pair_diagnostics(self.source.p[i].to(torch.float64), self.target.p[pos].to(torch.float64)),
                **({f"anchors_{k}": v for k, v in coverage[i].items()} if i in coverage else {}),
            }
        return out


@dataclass(frozen=True)
class AnchorsRebase:
    """Shared-anchor rebasin: register models once, encode experts in anchor coordinates, decode on any target.

    ``prepare`` is the standard per-task interface (same keyword set as ``AriadneRebase.prepare`` plus the
    anchor-pool loaders and model ids); ``open_state`` / ``AnchorState.register_model`` /
    ``transport_rows`` are the extended interface.
    """

    name: str = "anchors"

    @staticmethod
    def open_state(config: AnchorsConfig) -> AnchorState:
        return AnchorState(config, config.state_dir)

    def prepare(
        self,
        *,
        source_base_model: torch.nn.Module,
        source_ft_model: torch.nn.Module,
        target_model: torch.nn.Module,
        target_base_sd: Mapping[str, Tensor],
        source_loader: Any,
        target_loader: Any,
        pairing: DiscreteLayerPairing,
        config: AnchorsConfig,
        device: str,
        source_anchor_loader: Any,
        target_anchor_loader: Any,
        source_model_id: str,
        target_model_id: str,
        state: AnchorState | None = None,
        family_adapter: Any = None,
        recorder: PhaseCostRecorder | None = None,
        source_model_meta: dict[str, Any] | None = None,
        target_model_meta: dict[str, Any] | None = None,
        **_unused: Any,
    ) -> AnchorsPrepared:
        """Register both models if needed, encode/decode every target position, compile with the shared ridge.

        Extra keyword arguments of ``AriadneRebase.prepare`` (classifier, text features, ...) are accepted and
        ignored so both methods share a call site. With ``recorder=None`` the fit runs under its own
        ``PhaseCostRecorder`` and ``timing["cost_phases"]`` carries its summary.

        ``family_adapter`` (HF decoders sharing one tokenizer, AGENT_NEXT_PHASE.md L1): the anchor rows are tokens
        of a generic text pool, identical token ids in every model (checked by hash), the first token of every
        sequence (BOS / attention sink) and the padding excluded; the shared ridge fits ``mlp.down_proj``.
        ``config.text_units='word'``: the rows are words of the raw texts, each model reading a word at the last
        token of its own tokenization, so source and target tokenizers may differ (``text_units.py``); the
        loaders must be built with ``build_text_calibration_loader(word_units=True)``.
        """
        if family_adapter is not None:
            validate_text_config(config)
        elif config.anchor_pool in TEXT_ANCHOR_POOLS:
            raise ValueError(f"anchor_pool={config.anchor_pool!r} is a text pool: it needs a decoder family adapter")
        if config.synthesis == "ridge_preserve" and not preserve_supported(config):
            # e.g. dataclasses.replace() of a resolved config: the sequential / task-row paths would ignore D2 silently
            raise ValueError("synthesis='ridge_preserve' is not implemented for this configuration: set synthesis='ridge'")
        if state is None:
            state = self.open_state(config)
        kwargs = dict(
            source_base_model=source_base_model,
            source_ft_model=source_ft_model,
            target_model=target_model,
            target_base_sd=target_base_sd,
            source_loader=source_loader,
            target_loader=target_loader,
            pairing=pairing,
            config=config,
            device=device,
            source_anchor_loader=source_anchor_loader,
            target_anchor_loader=target_anchor_loader,
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            state=state,
            source_model_meta=source_model_meta,
            target_model_meta=target_model_meta,
            task_name=_unused.get("task_name"),
            family_adapter=family_adapter,
        )
        if recorder is not None:
            return _prepare_body(recorder, **kwargs)
        with recording(PhaseCostRecorder(device)) as own:
            prepared = _prepare_body(own, **kwargs)
        prepared.timing["cost_phases"] = own.summary()
        return prepared

    def apply(
        self,
        prepared: AnchorsPrepared,
        *,
        delta: Mapping[str, Tensor] | None = None,
        strength: float | None = None,
    ) -> dict[str, Tensor]:
        if delta is not None:
            raise ValueError("anchors fits its own task vector; apply() does not accept an input delta.")
        if strength is None:
            return prepared.task_vector
        s = float(strength)
        if s == 0.0:
            return {}
        return {key: s * v for key, v in prepared.unit_task_vector.items()}

    def transport(self, **kwargs) -> dict[str, Tensor]:
        raise NotImplementedError(
            "anchors does not transport a state-dict delta: call AnchorsRebase().prepare(...) with live models, "
            "task calibration loaders and anchor-pool loaders."
        )


def _register_pair(
    state: AnchorState,
    source_base_model,
    target_model,
    target_base_sd,
    source_anchor_loader,
    target_anchor_loader,
    source_model_id: str,
    target_model_id: str,
    *,
    device,
    source_model_meta=None,
    target_model_meta=None,
    family_adapter=None,
) -> tuple[RegisteredModel, RegisteredModel]:
    """Register source and target in the shared frame (idempotent; the target at its base weights)."""
    source = state.register_model(
        source_base_model,
        source_model_id,
        source_anchor_loader,
        device=device,
        model_meta=source_model_meta,
        family_adapter=family_adapter,
        frame_loaders=(target_anchor_loader,),  # word units: anchors among the words present in both tokenizations
    )
    # The target must be at its base weights when registered (the frame describes the base model).
    current = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    try:
        target_model.load_state_dict(dict(target_base_sd), strict=True)
        target = state.register_model(
            target_model,
            target_model_id,
            target_anchor_loader,
            device=device,
            model_meta=target_model_meta,
            family_adapter=family_adapter,
        )
    finally:
        target_model.load_state_dict(current, strict=True)
    return source, target


def _frame_from_rows(model_id: str, rows: list[Tensor], meta: dict[str, Any]) -> RegisteredModel:
    """``P = (S - mu) / c`` per block, ``mu`` and ``c = ||S - mu||_F / sqrt(n)`` from the same rows."""
    p, mu, c = [], [], []
    for s_rows in rows:
        x = s_rows.to(torch.float64)
        m = x.mean(0)
        scale = float(torch.linalg.norm(x - m).item()) / x.shape[0] ** 0.5
        p.append(((x - m) / scale).float())
        mu.append(m)
        c.append(scale)
    return RegisteredModel(model_id, meta, p, mu, c)


@torch.no_grad()
def task_map_frames(
    source_base_model,
    target_model,
    target_base_sd,
    source_loader,
    target_loader,
    pairing: DiscreteLayerPairing,
    *,
    config: AnchorsConfig,
    device,
    family_adapter=None,
) -> tuple[RegisteredModel, RegisteredModel]:
    """``map_rows='task'``: in-memory frames whose rows are the task's paired-calibration rows.

    Base-model boundaries of every block on ``paired_calibration(num_batches=map_num_batches or num_batches,
    seed=seed)`` (E's draw; the draw is prefix-consistent), on the canonical grid. The same (image, token) rows are
    kept in both models: ``map_rows_per_image`` random tokens per image, then ``map_num_rows`` random rows overall.
    Decoders: every content row of the draw (padding and first token dropped, as for E), then ``map_num_rows``.
    """
    n_batches = config.map_num_batches or config.num_batches
    text = family_adapter is not None
    if text and config.text_units == "word":  # each model's banks hold the words present in both, in one order
        source_batches, target_batches, _ = paired_word_calibration(
            source_loader, target_loader, num_batches=n_batches, seed=config.seed, family_adapter=family_adapter
        )
    else:
        text_kwargs = {"family_adapter": family_adapter, "exclude_first_token": True} if text else {}
        source_batches, target_batches, _ = paired_calibration(
            source_loader, target_loader, num_batches=n_batches, seed=config.seed, **text_kwargs
        )
    canonical = config.canonical_tokens
    src_requests = {str(i): (i, "boundary") for i in range(pairing.source_depth)}
    tgt_requests = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
    generator = torch.Generator().manual_seed(int(config.map_seed))
    src_rows: list[list[Tensor]] = [[] for _ in range(pairing.source_depth)]
    tgt_rows: list[list[Tensor]] = [[] for _ in range(pairing.target_depth)]
    current = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    try:
        target_model.load_state_dict(dict(target_base_sd), strict=True)
        gens = (
            iter_capture_tokens(source_base_model, source_batches, src_requests, device, family_adapter=family_adapter),
            iter_capture_tokens(target_model, target_batches, tgt_requests, device, family_adapter=family_adapter),
        )
        for src, tgt in zip(*gens, strict=True):
            n_img = int(src["0"].shape[0])
            if text:  # [1, N, d] content-row bank, the same rows in both models (one combined row mask)
                select = torch.arange(int(src["0"].shape[1]))
            elif config.map_rows_per_image is None:
                select = torch.arange(n_img * canonical)
            else:
                k = min(int(config.map_rows_per_image), canonical)
                pos = torch.stack([torch.randperm(canonical, generator=generator)[:k] for _ in range(n_img)])
                select = (torch.arange(n_img)[:, None] * canonical + pos).reshape(-1)
            for i in range(pairing.source_depth):
                x = to_rows(src[str(i)], canonical, text)
                src_rows[i].append(x.reshape(-1, x.shape[-1])[select])
            for j in range(pairing.target_depth):
                x = to_rows(tgt[str(j)], canonical, text)
                tgt_rows[j].append(x.reshape(-1, x.shape[-1])[select])
    finally:
        target_model.load_state_dict(current, strict=True)
    src_cat = [torch.cat(r, 0) for r in src_rows]
    tgt_cat = [torch.cat(r, 0) for r in tgt_rows]
    n_rows = int(src_cat[0].shape[0])
    if config.map_num_rows is not None and config.map_num_rows < n_rows:
        keep = torch.randperm(n_rows, generator=generator)[: int(config.map_num_rows)].sort().values
        src_cat = [r[keep] for r in src_cat]
        tgt_cat = [r[keep] for r in tgt_cat]
    meta = {"map_rows": "task", "map_num_batches": n_batches, "n_rows": int(src_cat[0].shape[0])}
    return _frame_from_rows("task_source", src_cat, meta), _frame_from_rows("task_target", tgt_cat, meta)


def generic_calibration_loaders(config: AnchorsConfig, source_anchor_loader, target_anchor_loader):
    """``calibration_source='generic'``: paired calibration loaders over generic pool images (labels unused).

    The frame's pool draw is the first ``anchor_batches + registration_batches`` batches of the permutation of the pool
    dataset seeded by ``pool_seed`` (registration, anchor images, ridge_preserve's G and refine's keep / holdout rows
    all lie inside it). The calibration images are ``num_batches * batch_size`` images taken, with ``config.seed``, from
    the rest of that permutation, so they are disjoint from all of them by construction (asserted). Each model keeps its
    own preprocessing (the loaders' datasets / collate functions). Returns (source_loader, target_loader, info)."""
    import hashlib

    from torch.utils.data import DataLoader, Subset

    from .._ariadne.capture import _dataset_identity, _stable_dataset_identity

    if not isinstance(source_anchor_loader, DataLoader) or not isinstance(target_anchor_loader, DataLoader):
        raise ValueError("calibration_source='generic' needs indexed anchor-pool DataLoaders")
    if _dataset_identity(source_anchor_loader.dataset) != _dataset_identity(target_anchor_loader.dataset):
        raise ValueError("Source/target anchor-pool image identities or ordering do not match")
    bs = source_anchor_loader.batch_size
    if not bs or bs != target_anchor_loader.batch_size:
        raise ValueError("calibration_source='generic' needs equal positive anchor-pool batch sizes")
    n = len(source_anchor_loader.dataset)
    n_pool = (int(config.anchor_batches) + int(config.registration_batches)) * bs
    n_cal = int(config.num_batches) * bs
    if n_pool + n_cal > n:
        raise ValueError(
            f"calibration_source='generic': the pool has {n} images, the frame uses {n_pool} and the calibration "
            f"needs {n_cal} more"
        )
    order = torch.randperm(n, generator=torch.Generator().manual_seed(int(config.pool_seed))).tolist()
    used, rest = order[:n_pool], order[n_pool:]
    pick = torch.randperm(len(rest), generator=torch.Generator().manual_seed(int(config.seed)))[:n_cal].tolist()
    indices = [rest[i] for i in pick]
    if set(indices) & set(used) or len(set(indices)) != len(indices):
        raise RuntimeError("generic calibration images overlap the frame's pool draw")
    kwargs = dict(batch_size=bs, shuffle=False, num_workers=0, drop_last=False)
    source = DataLoader(Subset(source_anchor_loader.dataset, indices), collate_fn=source_anchor_loader.collate_fn, **kwargs)
    target = DataLoader(Subset(target_anchor_loader.dataset, indices), collate_fn=target_anchor_loader.collate_fn, **kwargs)
    info = {
        "calibration_source": "generic",
        "pool": config.anchor_pool,
        "pool_seed": int(config.pool_seed),
        "calibration_seed": int(config.seed),
        "n_images": len(indices),
        "batch_size": bs,
        "pool_dataset_identity": _stable_dataset_identity(source_anchor_loader.dataset),
        "calibration_dataset_identity": _stable_dataset_identity(source.dataset),
        "calibration_indices_sha256": hashlib.sha256(repr(indices).encode()).hexdigest(),
        "frame_draw_indices_sha256": hashlib.sha256(repr(used).encode()).hexdigest(),
        "disjoint_from_frame_draw": True,
    }
    return source, target, info


def precomputed_calibration_loaders(config: AnchorsConfig, task_name: str | None, batch_size: int):
    """``calibration_source='precomputed'``: ``<calibration_images>/<task>.pt`` (stored preprocessed tensors, the first
    ``num_batches * batch_size`` in stored order) as one dataset shared by the source and the target loader."""
    from torch.utils.data import DataLoader

    from ....data.precomputed_images import PrecomputedImageDataset, load_precomputed_images, tensor_sha256

    if not task_name:
        raise ValueError("calibration_source='precomputed' needs the task name (per-task image files)")
    images, digest, path = load_precomputed_images(config.calibration_images, task_name)
    need = int(config.num_batches) * int(batch_size)
    if images.shape[0] < need:
        raise ValueError(f"{path}: {images.shape[0]} images but num_batches*batch_size = {need} are required")
    dataset = PrecomputedImageDataset(images[:need], tensor_sha256(images[:need]))
    kwargs = {"batch_size": int(batch_size), "shuffle": False, "num_workers": 0, "drop_last": False}
    info = {"calibration_source": "precomputed", "path": str(path), "file_sha256": digest,
            "n_stored": int(images.shape[0]), "n_used": need, "used_sha256": dataset.sha256,
            "shape": [need, *images.shape[1:]], "task": task_name}
    return DataLoader(dataset, **kwargs), DataLoader(dataset, **kwargs), info


def _prepare_body(
    recorder: PhaseCostRecorder,
    *,
    source_base_model,
    source_ft_model,
    target_model,
    target_base_sd,
    source_loader,
    target_loader,
    pairing: DiscreteLayerPairing,
    config: AnchorsConfig,
    device,
    source_anchor_loader,
    target_anchor_loader,
    source_model_id: str,
    target_model_id: str,
    state: AnchorState,
    source_model_meta=None,
    target_model_meta=None,
    task_name: str | None = None,
    family_adapter=None,
) -> AnchorsPrepared:
    """Registration + task capture + encode/decode (``alignment_calibration`` bracket), then the ridge
    (``correction_fit`` bracket); timing keys mirror Ariadne's so the run summary reads both alike."""
    generic_info = None
    if config.calibration_source == "precomputed":
        if family_adapter is not None or config.map_rows != "frame" or config.library != "data":
            raise ValueError("calibration_source='precomputed' is vision-only, map_rows='frame', library='data'")
        source_loader, target_loader, generic_info = precomputed_calibration_loaders(
            config, task_name, getattr(source_loader, "batch_size", None) or 32
        )
    elif config.calibration_source == "generic":
        if family_adapter is not None or config.map_rows != "frame" or config.library != "data":
            raise ValueError("calibration_source='generic' is vision-only, map_rows='frame', library='data'")
        source_loader, target_loader, generic_info = generic_calibration_loaders(
            config, source_anchor_loader, target_anchor_loader
        )
    alignment_mark = recorder.mark()
    started = time.perf_counter()
    stream_cal = None
    if config.calibration_streaming:
        if family_adapter is None:
            raise ValueError("calibration_streaming is implemented for decoders (family_adapter) only")
        from . import calibration_streaming as cs

        stream_cal = {"batches": cs.draw_streaming_calibration(source_loader, target_loader, config, family_adapter),
                      "task": None}
    if config.map_rows == "task" and stream_cal is not None:
        from . import calibration_streaming as cs

        map_s, map_t = cs.draw_map_batches(source_loader, target_loader, config, family_adapter)
        current = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
        try:
            target_model.load_state_dict(dict(target_base_sd), strict=True)
            stream_cal["task"] = cs.accumulate_task_frame(
                source_base_model, target_model, map_s, map_t, pairing, config=config, device=device,
                family_adapter=family_adapter,
            )
        finally:
            target_model.load_state_dict(current, strict=True)
        del map_s, map_t
        source, target = stream_cal["task"][:2]
    elif config.map_rows == "task":
        source, target = task_map_frames(
            source_base_model,
            target_model,
            target_base_sd,
            source_loader,
            target_loader,
            pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
    else:
        source, target = _register_pair(
            state,
            source_base_model,
            target_model,
            target_base_sd,
            source_anchor_loader,
            target_anchor_loader,
            source_model_id,
            target_model_id,
            device=device,
            source_model_meta=source_model_meta,
            target_model_meta=target_model_meta,
            family_adapter=family_adapter,
        )
    if stream_cal is not None and stream_cal["task"] is not None:  # statistics frames: depths = the blocks of the stats
        if (set(source.c) != set(int(i) for i in pairing.pairing)) or set(target.c) != set(range(pairing.target_depth)):
            raise ValueError("streamed task frame does not cover the pairing's blocks")
    elif source.depth != pairing.source_depth or target.depth != pairing.target_depth:
        raise ValueError(
            f"pairing ({pairing.source_depth}->{pairing.target_depth}) does not match the registered depths "
            f"({source.depth}->{target.depth})"
        )
    if config.canonical_source == "sigma_all_tokens":
        from .canonical_stream import streamed_sigma_for_pair

        s_b, t_b = _pool_draw(state, source_anchor_loader, target_anchor_loader)
        pairs = [(int(pairing.pairing[j]), j) for j in range(pairing.target_depth)]
        target.extras["canonical_sigma"] = streamed_sigma_for_pair(
            state, source, target, source_base_model, target_model, target_base_sd, s_b, t_b, pairs,
            config=config, device=device,
        )
    if config.anchor_filter is not None:
        k = float(config.anchor_filter_k)
        state.norm_filter(
            source_base_model, source_model_id, source_anchor_loader, k=k, device=device, family_adapter=family_adapter
        )
        current = {k_: v.detach().cpu().clone() for k_, v in target_model.state_dict().items()}
        try:
            target_model.load_state_dict(dict(target_base_sd), strict=True)
            state.norm_filter(
                target_model, target_model_id, target_anchor_loader, k=k, device=device, family_adapter=family_adapter
            )
        finally:
            target_model.load_state_dict(current, strict=True)
        if config.frame_map in ("gpa", "consensus"):  # models in the universe not live in this run: stored filters
            for extra in (*config.gpa_models, *config.gpa_add_models):
                if extra not in (source_model_id, target_model_id):
                    state.load_norm_filter(extra, k=k)
    gpa_maps = None
    universe_meta = None
    if config.frame_map in ("gpa", "consensus"):
        # gpa_add_models are added to the fixed G after the fit, even when one of them is the run's source/target.
        added = set(config.gpa_add_models)
        fit_models = [m for m in (source_model_id, target_model_id, *config.gpa_models) if m not in added]
        if len(set(fit_models)) < 2 and source_model_id != target_model_id:
            raise ValueError(f"frame_map={config.frame_map!r} needs at least two fitted models besides gpa_add_models")
        universe_fit = fit_universe if config.frame_map == "gpa" else partial(
            fit_consensus, k=config.consensus_k, construction=config.consensus, register=config.consensus_register,
            eps=config.consensus_eps)
        universe = universe_fit(
            state,
            fit_models,
            n_grid=target.depth,
            **({"d_u": config.gpa_dim} if config.frame_map == "gpa" else {}),
            add_models=config.gpa_add_models,
            filter_k=None if config.anchor_filter is None else float(config.anchor_filter_k),
            dedup=None
            if config.anchor_dedup_cos is None
            else (source_model_id, float(config.anchor_dedup_cos), config.anchor_dedup_max, int(config.anchor_seed)),
            row_norm=bool(config.anchor_row_norm),
        )
        gpa_maps = {j: universe.block_maps(j, source_model_id, target_model_id) for j in range(target.depth)}
        universe_meta = {
            "version": universe.version,
            "fit_models": universe.fit_models,
            "added": universe.added,
            "d_u": universe.d_u,
            "info": universe.info,
        }
    registration_seconds = time.perf_counter() - started
    native_seq = uses_native_depth_fit(config)
    streaming_seq = config.depth_fit == "sequential" and not native_seq
    if stream_cal is not None:
        # nothing stored per token: capture, decode and the ridge statistics are one sweep (correction_fit bracket)
        calibration = stream_cal["batches"][2]
    elif streaming_seq:
        prepared_stream = prepare_direct_residual_streaming(
            source_base_model,
            target_model,
            source_loader,
            target_loader,
            pairing,
            num_batches=config.num_batches,
            seed=config.seed,
            device=device,
            source_ft_model=source_ft_model,
            # decoders: padding and the first token of every sequence leave every bank, as in the independent fit
            **({} if family_adapter is None else {"family_adapter": family_adapter, "exclude_first_token": True}),
        )
        calibration = prepared_stream["calibration"]
    elif config.operator_fusion_entries:
        captured, desired, position_diag = fused_operator_effects(
            source_loader, target_model, target_loader,
            _pool_draw(state, source_anchor_loader, target_anchor_loader)[1]
            if config.operator_decode_rows == "pool" else None,
            pairing, source, target, config=config, gpa_maps=gpa_maps, universe_version=universe_meta["version"],
            target_model_id=target_model_id, device=device,
        )
        calibration = captured["calibration"]
    elif config.library == "operator":
        captured, desired, position_diag = operator_desired_effects(
            source_base_model,
            source_ft_model,
            target_model,
            source_loader,
            target_loader,
            pairing,
            source,
            target,
            config=config,
            gpa_maps=gpa_maps,
            device=device,
            library_entry=None
            if config.operator_library_dir is None
            else {"dir": config.operator_library_dir, "universe_version": universe_meta["version"],
                  "source_model_id": source_model_id, "target_model_id": target_model_id,
                  "task": task_name or "unnamed"},
            pool_batches=_pool_draw(state, source_anchor_loader, target_anchor_loader)
            if config.operator_input_map == "pool_ridge" else None,
        )
        calibration = captured["calibration"]
    elif family_adapter is not None and config.text_units == "word":
        # word units: each model on its own tokenization, rows = the words present in both (anchors/text_units.py)
        captured = capture_paired_word_activations(
            source_base_model,
            source_ft_model,
            target_model,
            source_loader,
            target_loader,
            pairing,
            num_batches=config.num_batches,
            seed=config.seed,
            device=device,
            family_adapter=family_adapter,
        )
        calibration = captured["calibration"]
        desired, position_diag = decode_desired_effects(
            captured, pairing, source, target, config=config, gpa_maps=gpa_maps, family_adapter=family_adapter
        )
    else:
        captured = capture_paired_boundary_activations(
            source_base_model,
            source_ft_model,
            target_model,
            source_loader,
            target_loader,
            pairing,
            num_batches=config.num_batches,
            seed=config.seed,
            device=device,
            # decoders: padding and the first token of every sequence (BOS / attention sink) leave every bank
            **({} if family_adapter is None else {"family_adapter": family_adapter, "exclude_first_token": True}),
        )
        calibration = captured["calibration"]
        desired, position_diag = decode_desired_effects(
            captured, pairing, source, target, config=config, gpa_maps=gpa_maps, family_adapter=family_adapter
        )
    if generic_info is not None:
        calibration = {**calibration, "split": "anchor_pool" if config.calibration_source == "generic"
                       else "precomputed", **generic_info}
    alignment_peak, alignment_host_peak = recorder.peaks_since(alignment_mark)
    alignment_timing = {
        "alignment_calibration_seconds": time.perf_counter() - started,
        "alignment_calibration_peak_memory_bytes": alignment_peak,
        "alignment_calibration_peak_host_rss_bytes": alignment_host_peak,
        "anchors_registration_seconds": registration_seconds,
    }
    fit_mark = recorder.mark()
    started = time.perf_counter()
    if stream_cal is not None:
        solve_hook = None
        if config.synthesis == "ridge_preserve":
            solve_hook = _synthesis_hook(
                state, target, target_model, target_base_sd, target_anchor_loader, config=config, device=device,
                family_adapter=family_adapter,
            )
        corrections, rows, position_diag = cs.fit_streaming_independent(
            target_model=target_model, target_base_sd=target_base_sd, source_base_model=source_base_model,
            source_ft_model=source_ft_model, source_batches=stream_cal["batches"][0],
            target_batches=stream_cal["batches"][1], pairing=pairing, source=source, target=target, config=config,
            device=device, family_adapter=family_adapter, gpa_maps=gpa_maps, solve_hook=solve_hook,
            task_stats=stream_cal["task"],
        )
    elif streaming_seq:
        decoder = SequentialDecoder(source, target, config, gpa_maps=gpa_maps, text=family_adapter is not None)
        corrections, rows = fit_direct_residual_sequential_streaming(
            target_model,
            target_base_sd,
            source_base_model,
            source_ft_model,
            prepared_stream,
            pairing,
            config=config.ridge_config(),
            device=device,
            desired_fn=decoder,
            **({} if family_adapter is None else {"family_adapter": family_adapter}),
        )
        position_diag = decoder.diagnostics()
    else:
        full_captured, full_desired = captured, desired
        if config.refine != "none":  # the depth fit sees only the batches the refinement does not hold out
            keep = len(captured["target_batches"]) - int(config.refine_holdout_batches)
            captured = {**captured, "target_batches": captured["target_batches"][:keep],
                        "target_base_outputs_by_position": {
                            j: v[:keep] for j, v in captured["target_base_outputs_by_position"].items()}}
            desired = {j: v[:keep] for j, v in desired.items()}
        solve_hook = None
        if config.synthesis == "ridge_preserve" or config.update_rank is not None:
            solve_hook = _synthesis_hook(
                state, target, target_model, target_base_sd, target_anchor_loader, config=config, device=device,
                family_adapter=family_adapter,
            )
        if config.depth_fit == "spread":
            from .synthesis import fit_spread, measure_blocks

            corrections, rows = fit_spread(
                target_model, target_base_sd, captured, desired,
                measure_blocks(config.measure_depths, pairing.target_depth),
                config=config.ridge_config(), device=device, solve_hook=solve_hook,
            )
        elif native_seq:
            from .synthesis import fit_tracking

            corrections, rows = fit_tracking(
                target_model, target_base_sd, captured, desired, config=config.ridge_config(), device=device,
                solve_hook=solve_hook, gain=config.gain_calibration, holdout=int(config.gain_holdout_batches),
                sequential_impl=config.sequential_impl,
            )
        elif config.gain_calibration != "none":
            corrections, rows = compile_with_gain(
                target_model, target_base_sd, captured, desired, pairing, config=config, device=device,
                solve_hook=solve_hook, family_adapter=family_adapter,
            )
        else:
            corrections, rows = compile_desired_effects(
                target_model, target_base_sd, captured, desired, pairing, config=config, device=device,
                solve_hook=solve_hook, family_adapter=family_adapter,
            )
    refine_info = None
    if config.refine != "none":
        corrections, rows, refine_info = _run_refine(
            target_model, target_base_sd, full_captured, full_desired, corrections, rows, state, target, pairing,
            target_anchor_loader, config=config, device=device,
        )
    fit_peak, fit_host_peak = recorder.peaks_since(fit_mark)
    fit_timing = {
        "correction_fit_seconds": time.perf_counter() - started,
        "correction_fit_peak_memory_bytes": fit_peak,
        "correction_fit_peak_host_rss_bytes": fit_host_peak,
    }
    for row in rows:
        row.update(position_diag.get(int(row.get("position", -1)), {}))
    corrections = restrict_to_depth(corrections, target.depth, config.compile_max_depth)
    strength = float(config.strength)
    scaled = {} if strength == 0.0 else {k: strength * v for k, v in corrections.items()}
    return AnchorsPrepared(
        task_vector=scaled,
        unit_task_vector=corrections,
        timing={"alignment_calibration": alignment_timing, "correction_fit": fit_timing},
        diagnostics=rows,
        extra={
            # Keys AriadneRunRecord.record_fit reads; the Ariadne-only analyses are not run here.
            "realization_by_position": None,
            "task_vector_stats": None,
            "alignment_diagnostics": None,
            "tv_scaling": None,
            "fidelity_holdout": None,
            "calibration": calibration,
            "anchors": {
                "config": config.to_dict(),
                "frame_pool": None if state.frame is None else dict(state.frame["pool"], indices=None),
                "source_model_id": source_model_id,
                "target_model_id": target_model_id,
                "gpa_universe": universe_meta,
                **({} if family_adapter is None else {"rows": _text_rows(config), "exclude_first_token": True}),
                **({} if refine_info is None else {"refine": refine_info}),
                **({} if generic_info is None else {"calibration": generic_info}),
            },
        },
        config=config,
    )


def _run_refine(
    target_model, target_base_sd, captured, desired, corrections, rows, state, target, pairing, target_anchor_loader,
    *, config: AnchorsConfig, device,
):
    """``refine='joint'`` (``anchors/refine.py``): jointly refine the depth fit's ``corrections`` on the calibration
    banks (``captured`` / ``desired``) and the generic anchor-pool rows; per-block diagnostics are merged into ``rows``
    (by position), the global ones are returned for ``extra['anchors']['refine']``."""
    from .refine import refine_blocks, refine_joint

    blocks = refine_blocks(config.refine_depths, sorted(desired), pairing.target_depth)
    generic = generic_hold = None
    norm = None
    if float(config.refine_rho) > 0.0:
        frame = state.config
        pool, _, _ = paired_calibration(
            target_anchor_loader, target_anchor_loader,
            num_batches=frame.anchor_batches + frame.registration_batches, seed=frame.pool_seed,
        )
        generic = pool[: min(frame.anchor_batches, int(config.refine_generic_batches))]
        # held-out generic batch: the first pool batch O2's G does not use (index anchor_batches); in-sample only if absent
        generic_hold = pool[frame.anchor_batches: frame.anchor_batches + 1] or None
        if config.anchor_filter is not None:
            flt = target.extras[f"norm_k{float(config.anchor_filter_k):g}"]
            norm = {"mu": target.mu, "k": float(flt["k"]), "median": flt["median"]}
    new, per_block, info = refine_joint(
        target_model, target_base_sd, captured, desired, corrections, generic, blocks=blocks,
        generic_holdout=generic_hold, warmup=float(config.refine_warmup),
        target=config.refine_target, rho=float(config.refine_rho), prox=float(config.refine_prox),
        steps=int(config.refine_steps), lr=float(config.refine_lr), batch=int(config.refine_batch),
        holdout=int(config.refine_holdout_batches), eval_every=int(config.refine_eval_every),
        amp=bool(config.refine_amp), use_checkpoint=bool(config.refine_checkpoint), seed=int(config.seed),
        device=device, norm_filter=norm,
    )
    stale = ("correction_norm", "bias_norm", "residual_norm_after", "residual_norm_before", "relative_residual_before",
             "generic_change", "generic_change_ridge", "update_rank_energy_kept")
    for row in rows:
        pos = int(row.get("position", -1))
        for key in [k for k in row if k in stale or k.startswith(("realized_over", "cosine_gain"))]:
            row[f"prefit_{key}"] = row.pop(key)  # statistics of the pre-refinement fit, kept under a prefix
        row.update(per_block.get(pos, {}))
        if pos in per_block:
            row["correction_norm"] = per_block[pos]["refine_dw_norm_after"]
            row["bias_norm"] = per_block[pos]["refine_bias_norm_after"]
            row["realized_over_desired_heldout"] = per_block[pos]["refine_ratio_after"]
            row["cosine_heldout"] = per_block[pos]["refine_cosine_after"]
    info["generic_holdout_in_sample"] = generic_hold is None and generic is not None
    return new, rows, info


def _text_rows(config: AnchorsConfig) -> str:
    return "words" if config.text_units == "word" else "tokens"


_BLOCK_KEY = re.compile(r"(?:^|\.)(?:resblocks|layers)\.(\d+)\.")


def restrict_to_depth(
    corrections: Mapping[str, Tensor], target_depth: int, max_depth: float | None
) -> dict[str, Tensor]:
    """Zero the update of every target block ``j`` with ``j / (L_k - 1) >= max_depth`` (``None``: unchanged)."""
    if max_depth is None:
        return dict(corrections)
    out: dict[str, Tensor] = {}
    for key, value in corrections.items():
        match = _BLOCK_KEY.search(key)
        if match is None:
            raise ValueError(f"compile_max_depth: cannot read the block index of {key!r}")
        j = int(match.group(1))
        keep = j / max(target_depth - 1, 1) < float(max_depth)
        out[key] = value if keep else torch.zeros_like(value)
    return out


def compile_desired_effects(
    target_model,
    target_base_sd: Mapping[str, Tensor],
    captured: Mapping[str, Any],
    desired: Mapping[int, list[Tensor]],
    pairing: DiscreteLayerPairing,
    *,
    config: AnchorsConfig,
    device,
    solve_hook=None,
    family_adapter=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Step 4b: Ariadne's ridge on the target base inputs, called unchanged with the decoded ``D`` (``solve_hook``:
    a synthesis variant, ``anchors/synthesis.py``; ``family_adapter``: decoders, ``mlp.c_proj`` = ``mlp.down_proj``)."""
    return fit_direct_residual(
        target_model, target_base_sd, captured, desired, pairing, config=config.ridge_config(), device=device,
        family_adapter=family_adapter, solve_hook=solve_hook,
    )


def uses_native_depth_fit(config: AnchorsConfig) -> bool:
    """Whether the depth fit runs in anchors' own closed-loop (``synthesis.fit_tracking``) instead of Ariadne's
    streaming sequential fit: ``depth_fit='tracking'`` (S2), or ``'sequential'`` with ``ridge_preserve`` (S1) or a gain
    calibration (S3), which Ariadne's streaming fit cannot host. Plain ``'sequential'`` is unchanged."""
    return config.depth_fit == "tracking" or (
        config.depth_fit == "sequential"
        and (config.synthesis == "ridge_preserve" or config.gain_calibration != "none")
    )


def compile_with_gain(
    target_model,
    target_base_sd: Mapping[str, Tensor],
    captured: Mapping[str, Any],
    desired: Mapping[int, list[Tensor]],
    pairing: DiscreteLayerPairing,
    *,
    config: AnchorsConfig,
    device,
    solve_hook=None,
    family_adapter=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """S3 on the independent fit: the ridge on the first ``num_batches - gain_holdout_batches`` calibration batches
    (all of them when the holdout is 0), then the per-block gains ``g_j`` (``synthesis.fit_tracking`` with the fitted
    corrections fixed) on the held-out batches, in depth order with the scaled upstream mounted. The returned
    corrections are the scaled ones; the rows carry the gain and the realized / desired diagnostics of the pass."""
    from .synthesis import fit_tracking

    hold = int(config.gain_holdout_batches)
    n = len(captured["target_batches"])
    if hold:
        keep = n - hold
        fit_captured = {
            **captured,
            "target_batches": captured["target_batches"][:keep],
            "target_base_outputs_by_position": {
                j: v[:keep] for j, v in captured["target_base_outputs_by_position"].items()
            },
        }
        fit_desired = {j: v[:keep] for j, v in desired.items()}
    else:
        fit_captured, fit_desired = captured, desired
    fitted, rows = compile_desired_effects(
        target_model, target_base_sd, fit_captured, fit_desired, pairing, config=config, device=device,
        solve_hook=solve_hook, family_adapter=family_adapter,
    )
    scaled, gain_rows = fit_tracking(
        target_model, target_base_sd, captured, desired, config=config.ridge_config(), device=device,
        fixed=fitted, gain=config.gain_calibration, holdout=hold,
    )
    by_pos = {int(r["position"]): r for r in gain_rows}
    for row in rows:
        extra = by_pos[int(row["position"])]
        row.update({k: v for k, v in extra.items()
                    if k.startswith(("gain", "realized_over", "cosine_gain")) or k == "upstream_over_desired"})
    return scaled, rows


def _pool_draw(state, source_anchor_loader, target_anchor_loader) -> tuple[list[Any], list[Any]]:
    """The frame's anchor images (first anchor_batches of the pool draw), as paired source / target batches."""
    frame = state.config
    s_b, t_b, _ = paired_calibration(
        source_anchor_loader, target_anchor_loader,
        num_batches=frame.anchor_batches + frame.registration_batches, seed=frame.pool_seed,
    )
    return s_b[: frame.anchor_batches], t_b[: frame.anchor_batches]


def _synthesis_hook(
    state, target, target_model, target_base_sd, target_anchor_loader, *, config, device, family_adapter=None
):
    """D2 / D5 solve hook. For D2, the G statistics of the target base on the frame's anchor data (the first
    anchor_batches of the pool draw). Vision: all blocks in one sweep before the fit, the target loaded at its base
    weights for the capture and restored. Decoders (``family_adapter``): the content tokens of the anchor sequences
    (padding and the first token of every sequence dropped, as in the frame; with ``text_units='word'`` still all
    content tokens of the target's own tokenization, not only the word-final ones), one block at a time inside the
    solve (``LazyGenericInputs``), while the independent fit holds the target at its base weights."""
    from .synthesis import LazyGenericInputs, capture_generic_inputs, make_synthesis_hook

    if config.synthesis != "ridge_preserve":
        return make_synthesis_hook(rank=config.update_rank)
    frame = state.config
    kwargs = {} if family_adapter is None else {"family_adapter": family_adapter, "exclude_first_token": True}
    batches, _, _ = paired_calibration(
        target_anchor_loader, target_anchor_loader,
        num_batches=frame.anchor_batches + frame.registration_batches, seed=frame.pool_seed, **kwargs,
    )
    norm = None
    if config.anchor_filter is not None:
        flt = target.extras[f"norm_k{float(config.anchor_filter_k):g}"]
        norm = {"mu": target.mu, "k": float(flt["k"]), "median": flt["median"]}
    if family_adapter is not None:
        if config.depth_fit != "independent":
            raise ValueError("ridge_preserve on decoders needs depth_fit='independent'")
        from .._ariadne.layouts import _layout_for

        shim = _layout_for(family_adapter)
        keys = {j: shim.component_key(j, "mlp.c_proj", prefixed=True) for j in range(target.depth)}
        weights = {j: target_base_sd[keys[j]] for j in range(target.depth)}

        def at_base(j: int) -> None:
            live = target_model.get_parameter(keys[j])
            if not torch.equal(live.detach().to("cpu"), weights[j].detach().to("cpu", live.dtype)):
                raise RuntimeError(f"ridge_preserve: target block {j} is not at its base weights during the solve")

        generic = LazyGenericInputs(
            target_model, batches[: frame.anchor_batches], component="mlp.c_proj",
            tokens_per_block=config.preserve_tokens, seed=config.seed, device=device, norm_filter=norm,
            family_adapter=family_adapter, before_capture=at_base,
        )
        return make_synthesis_hook(generic=generic, rho_g=float(config.rho_g), base_weights=weights,
                                   rank=config.update_rank)
    if config.preserve_inputs == "updated":
        # S1/S2: G re-captured per block on the model with the upstream blocks already updated (the target is mounted
        # block by block by fit_tracking, so LazyGenericInputs sees the current weights when the solve asks for block j)
        weights = {j: target_base_sd[f"visual.transformer.resblocks.{j}.mlp.c_proj.weight"] for j in range(target.depth)}
        if config.sequential_impl == "incremental":
            from .synthesis import IncrementalGenericInputs

            # third running stream (generic images, current model); fit_tracking starts / advances it
            generic = IncrementalGenericInputs(
                batches[: frame.anchor_batches], tokens_per_block=config.preserve_tokens, seed=config.seed,
                norm_filter=norm,
            )
            hook = make_synthesis_hook(generic=generic, rho_g=float(config.rho_g), base_weights=weights,
                                       rank=config.update_rank)
            hook.generic_provider = generic
            return hook
        generic = LazyGenericInputs(
            target_model, batches[: frame.anchor_batches], component="mlp.c_proj",
            tokens_per_block=config.preserve_tokens, seed=config.seed, device=device, norm_filter=norm,
        )
        return make_synthesis_hook(generic=generic, rho_g=float(config.rho_g), base_weights=weights, rank=config.update_rank)
    current = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    try:
        target_model.load_state_dict(dict(target_base_sd), strict=True)
        generic = capture_generic_inputs(
            target_model, batches[: frame.anchor_batches], target.depth, component="mlp.c_proj",
            tokens_per_block=config.preserve_tokens, seed=config.seed, device=device, norm_filter=norm,
        )
    finally:
        target_model.load_state_dict(current, strict=True)
    weights = {j: target_base_sd[f"visual.transformer.resblocks.{j}.mlp.c_proj.weight"] for j in range(target.depth)}
    return make_synthesis_hook(generic=generic, rho_g=float(config.rho_g), base_weights=weights, rank=config.update_rank)
