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
from .config import AnchorsConfig
from .core import (
    CoverageAccumulator,
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
from .state import AnchorState, RegisteredModel, to_canonical
from .universe import fit_universe

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
                     device) -> dict[int, tuple[Tensor, Tensor]]:
    """``operator_input_map='pool_ridge'``: per target position j, the ridge ``u_k -> u_m`` (GPA coordinates, both
    frames' moments) on every canonical token of the anchor images kept by the norm filter in both models, moments
    streamed batch by batch."""
    from .operators import _Moments, common_rows, fit_from_moments

    if pool_batches is None:
        raise ValueError("operator_input_map='pool_ridge' needs the anchor pool batches")
    canon, rank = int(config.canonical_tokens), config.gpa_rank
    distinct = sorted({int(i) for i in pairing.pairing})
    moms = {j: _Moments() for j in range(pairing.target_depth)}
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
            moms[j].update(common_rows((tc - mu_k) / c_k, a_k, rank)[m], common_rows((sc - mu_m) / c_m, a_m, rank)[m])
    out = {}
    for j, mom in moms.items():
        fitted = fit_from_moments(mom)
        out[j] = (fitted.w, fitted.b)
    return out


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
    src: dict[int, tuple[list[Tensor], list[Tensor], list[Tensor], float]] = {}
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
        h = [x.reshape(-1, x.shape[-1])[m] for x, m in zip(hb, masks, strict=True)]
        e = [y.reshape(-1, y.shape[-1])[m] for y, m in zip(eb, masks, strict=True)]
        src[i] = (hb, h, e, c)
        if config.operator_space == "native":
            op = fit_effect_operator(h[:nb], e[:nb], rank=config.operator_rank)
            op.info["r2_train"] = r_squared(op, h[:nb], e[:nb])
            op.info["r2_heldout"] = r_squared(op, h[nb:], e[nb:])
            ops[i] = op
    pool_maps: dict[int, tuple[Tensor, Tensor]] = {}
    if config.operator_input_map == "pool_ridge":
        pool_maps = _pool_input_maps(source_base_model, target_model, pool_batches, pairing, source, target,
                                     config=config, gpa_maps=gpa_maps, keep=keep, device=device)
    desired: dict[int, list[Tensor]] = {}
    diag: dict[int, dict[str, Any]] = {}
    library: dict[int, tuple[Tensor, Tensor]] = {}
    for j in range(pairing.target_depth):
        i = int(pairing.pairing[j])
        hb, h, e, c_m = src[i]
        a_m, a_k = gpa_maps[j]
        uh = [common_rows(x, a_m, rank) for x in h]
        ue = [common_rows(y, a_m, rank) for y in e]
        if config.operator_space == "native":
            op = ops[i]
            w_u, b_u = to_common_space(op, a_m, rank)
        else:
            op = fit_effect_operator(uh[:nb], ue[:nb], rank=config.operator_rank)
            op.info["r2_train"] = r_squared(op, uh[:nb], ue[:nb])
            op.info["r2_heldout"] = r_squared(op, uh[nb:], ue[nb:])
            w_u, b_u = op.w, op.b
        library[j] = (w_u, b_u)
        # the operator in the common space, on the source's own held-out rows (no transfer of the inputs)
        r2_src, cos_src = explained([x @ w_u + b_u for x in uh[nb:]], ue[nb:])
        mu_k, c_k = _frame_moments(target, j, config)
        hk = [(t.to(torch.float64) - mu_k) / c_k for t in tgt_out[str(j)]]
        input_map = pool_maps.get(j)
        if config.operator_input_map == "anchor_ridge":
            p_m, _, p_k, _ = pair_frames(source, i, target, j, config=config)
            fitted = fit_effect_operator([common_rows(p_k, a_k, rank)], [common_rows(p_m, a_m, rank)])
            input_map = (fitted.w, fitted.b)
        input_diag: dict[str, Any] = {}
        if extra == 0:
            # the same task images through both models, on the canonical grid: how well the target's common
            # coordinates (after the input map) reproduce the source's (rows kept by the norm filter in both)
            uk_c, um_c = [], []
            for t, x, sb in zip(tgt_out[str(j)], hb[:nb], base_out[str(i)][:nb], strict=True):
                tc, sc = to_canonical(t.double(), canon), to_canonical(sb.double(), canon)
                m = torch.ones(tc.shape[0] * tc.shape[1], dtype=torch.bool, device=tc.device)
                if keep is not None:
                    m = keep("target", j, tc).reshape(-1) & keep("source", i, sc).reshape(-1)
                uk_c.append(common_rows((tc - mu_k) / c_k, a_k, rank)[m])
                um_c.append(common_rows(to_canonical(x, canon), a_m, rank)[m])
            if config.operator_input_map == "paired":
                fitted = fit_effect_operator(uk_c, um_c)
                input_map = (fitted.w, fitted.b)
            pred = uk_c if input_map is None else [u @ input_map[0] + input_map[1] for u in uk_c]
            r2_in, cos_in = explained(pred, um_c)
            input_diag = {"anchors_op_input_r2": r2_in, "anchors_op_input_cos": cos_in}
        scale = c_k if config.rescale_target else c_m
        out = [decode_operator(x, w_u, b_u, a_k, rank, scale, input_map).float() for x in hk]
        dropped = 0.0
        if keep is not None:
            masks = [keep("target", j, t) for t in tgt_out[str(j)]]
            out = [d * m.unsqueeze(-1).to(d.dtype) for d, m in zip(out, masks, strict=True)]
            dropped = 1.0 - sum(float(m.sum()) for m in masks) / sum(m.numel() for m in masks)
        desired[j] = out
        diag[j] = {
            "anchors_source_block": i,
            "anchors_d_norm": float(sum(float((d.double() ** 2).sum()) for d in out) ** 0.5),
            "anchors_op_target_rows_dropped": dropped,
            "anchors_op_common_r2_heldout": r2_src,
            "anchors_op_common_cos_heldout": cos_src,
            **input_diag,
            **{f"anchors_op_{k}": v for k, v in op.info.items()},
            "anchors_op_library_bytes": operator_nbytes(w_u, config.operator_rank),
        }
    if library_entry is not None:
        # One entry per (source, task), in the coordinates of the GPA frame it was encoded in; keyed by the target
        # grid point j (the map A_m of grid point j); c_m per entry so a decode can restore the source scale.
        from pathlib import Path

        path = Path(library_entry["dir"]) / str(library_entry["universe_version"])
        path = path / f"{library_entry['source_model_id']}__{library_entry['task']}.pt"
        save_library(path, library, {**dict(library_entry), "gpa_rank": rank, "operator_rank": config.operator_rank,
                                     "c_source": {j: float(src[int(pairing.pairing[j])][3]) for j in library}})
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


def decode_desired_effects(
    captured: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    source: RegisteredModel,
    target: RegisteredModel,
    *,
    config: AnchorsConfig,
    gpa_maps: Mapping[int, tuple[Tensor, Tensor]] | None = None,
) -> tuple[dict[int, list[Tensor]], dict[int, dict[str, Any]]]:
    """``{j: [D per batch]}`` on the target's native token grid, plus per-position diagnostics."""
    base = captured["source_base_outputs"]
    ft = captured["source_ft_outputs"]
    target_ref = captured["target_base_outputs_by_position"]
    canonical = config.canonical_tokens
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
            e = to_canonical(b1 - b0, canonical)
            batch, _, d_m = e.shape
            rows = e.reshape(-1, d_m)
            if acc is not None:
                acc.update(rows)
            d = fn(rows)
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
    ) -> None:
        self.source, self.target, self.config = source, target, config
        self.gpa_maps = gpa_maps
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
        e = to_canonical(f - b, cfg.canonical_tokens)
        batch, _, d_m = e.shape
        rows = e.reshape(-1, d_m)
        if cfg.coverage_diagnostics and self._owner(i) == pos:
            self._coverage_acc.setdefault(i, CoverageAccumulator()).update(rows)
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
        """
        if family_adapter is not None:
            raise NotImplementedError("anchors is vision-only for now (text units: milestone M8-M9)")
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
) -> tuple[RegisteredModel, RegisteredModel]:
    """Register source and target in the shared frame (idempotent; the target at its base weights)."""
    source = state.register_model(
        source_base_model, source_model_id, source_anchor_loader, device=device, model_meta=source_model_meta
    )
    # The target must be at its base weights when registered (the frame describes the base model).
    current = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    try:
        target_model.load_state_dict(dict(target_base_sd), strict=True)
        target = state.register_model(
            target_model, target_model_id, target_anchor_loader, device=device, model_meta=target_model_meta
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
) -> tuple[RegisteredModel, RegisteredModel]:
    """``map_rows='task'``: in-memory frames whose rows are the task's paired-calibration rows.

    Base-model boundaries of every block on ``paired_calibration(num_batches=map_num_batches or num_batches,
    seed=seed)`` (E's draw; the draw is prefix-consistent), on the canonical grid. The same (image, token) rows are
    kept in both models: ``map_rows_per_image`` random tokens per image, then ``map_num_rows`` random rows overall.
    """
    n_batches = config.map_num_batches or config.num_batches
    source_batches, target_batches, _ = paired_calibration(
        source_loader, target_loader, num_batches=n_batches, seed=config.seed
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
            iter_capture_tokens(source_base_model, source_batches, src_requests, device),
            iter_capture_tokens(target_model, target_batches, tgt_requests, device),
        )
        for src, tgt in zip(*gens, strict=True):
            n_img = int(src["0"].shape[0])
            if config.map_rows_per_image is None:
                select = torch.arange(n_img * canonical)
            else:
                k = min(int(config.map_rows_per_image), canonical)
                pos = torch.stack([torch.randperm(canonical, generator=generator)[:k] for _ in range(n_img)])
                select = (torch.arange(n_img)[:, None] * canonical + pos).reshape(-1)
            for i in range(pairing.source_depth):
                x = to_canonical(src[str(i)], canonical)
                src_rows[i].append(x.reshape(-1, x.shape[-1])[select])
            for j in range(pairing.target_depth):
                x = to_canonical(tgt[str(j)], canonical)
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
) -> AnchorsPrepared:
    """Registration + task capture + encode/decode (``alignment_calibration`` bracket), then the ridge
    (``correction_fit`` bracket); timing keys mirror Ariadne's so the run summary reads both alike."""
    alignment_mark = recorder.mark()
    started = time.perf_counter()
    if config.map_rows == "task":
        source, target = task_map_frames(
            source_base_model,
            target_model,
            target_base_sd,
            source_loader,
            target_loader,
            pairing,
            config=config,
            device=device,
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
        )
    if source.depth != pairing.source_depth or target.depth != pairing.target_depth:
        raise ValueError(
            f"pairing ({pairing.source_depth}->{pairing.target_depth}) does not match the registered depths "
            f"({source.depth}->{target.depth})"
        )
    if config.anchor_filter is not None:
        k = float(config.anchor_filter_k)
        state.norm_filter(source_base_model, source_model_id, source_anchor_loader, k=k, device=device)
        current = {k_: v.detach().cpu().clone() for k_, v in target_model.state_dict().items()}
        try:
            target_model.load_state_dict(dict(target_base_sd), strict=True)
            state.norm_filter(target_model, target_model_id, target_anchor_loader, k=k, device=device)
        finally:
            target_model.load_state_dict(current, strict=True)
        if config.frame_map == "gpa":  # models in the universe that are not live in this run: stored filters
            for extra in (*config.gpa_models, *config.gpa_add_models):
                if extra not in (source_model_id, target_model_id):
                    state.load_norm_filter(extra, k=k)
    gpa_maps = None
    universe_meta = None
    if config.frame_map == "gpa":
        # gpa_add_models are added to the fixed G after the fit, even when one of them is the run's source/target.
        added = set(config.gpa_add_models)
        fit_models = [m for m in (source_model_id, target_model_id, *config.gpa_models) if m not in added]
        if len(set(fit_models)) < 2 and source_model_id != target_model_id:
            raise ValueError("frame_map='gpa' needs at least two fitted models besides gpa_add_models")
        universe = fit_universe(
            state,
            fit_models,
            n_grid=target.depth,
            d_u=config.gpa_dim,
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
    if config.depth_fit == "sequential":
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
        )
        calibration = prepared_stream["calibration"]
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
                  "source_model_id": source_model_id, "task": task_name or "unnamed"},
            pool_batches=_pool_draw(state, source_anchor_loader, target_anchor_loader)
            if config.operator_input_map == "pool_ridge" else None,
        )
        calibration = captured["calibration"]
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
        )
        calibration = captured["calibration"]
        desired, position_diag = decode_desired_effects(
            captured, pairing, source, target, config=config, gpa_maps=gpa_maps
        )
    alignment_peak, alignment_host_peak = recorder.peaks_since(alignment_mark)
    alignment_timing = {
        "alignment_calibration_seconds": time.perf_counter() - started,
        "alignment_calibration_peak_memory_bytes": alignment_peak,
        "alignment_calibration_peak_host_rss_bytes": alignment_host_peak,
        "anchors_registration_seconds": registration_seconds,
    }
    fit_mark = recorder.mark()
    started = time.perf_counter()
    if config.depth_fit == "sequential":
        decoder = SequentialDecoder(source, target, config, gpa_maps=gpa_maps)
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
        )
        position_diag = decoder.diagnostics()
    else:
        solve_hook = None
        if config.synthesis == "ridge_preserve" or config.update_rank is not None:
            solve_hook = _synthesis_hook(
                state, target, target_model, target_base_sd, target_anchor_loader, config=config, device=device
            )
        if config.depth_fit == "spread":
            from .synthesis import fit_spread, measure_blocks

            corrections, rows = fit_spread(
                target_model, target_base_sd, captured, desired,
                measure_blocks(config.measure_depths, pairing.target_depth),
                config=config.ridge_config(), device=device, solve_hook=solve_hook,
            )
        else:
            corrections, rows = compile_desired_effects(
                target_model, target_base_sd, captured, desired, pairing, config=config, device=device,
                solve_hook=solve_hook,
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
            },
        },
        config=config,
    )


_BLOCK_KEY = re.compile(r"(?:^|\.)resblocks\.(\d+)\.")


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
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Step 4b: Ariadne's ridge on the target base inputs, called unchanged with the decoded ``D`` (``solve_hook``:
    a synthesis variant, ``anchors/synthesis.py``)."""
    return fit_direct_residual(
        target_model, target_base_sd, captured, desired, pairing, config=config.ridge_config(), device=device,
        solve_hook=solve_hook,
    )


def _pool_draw(state, source_anchor_loader, target_anchor_loader) -> tuple[list[Any], list[Any]]:
    """The frame's anchor images (first anchor_batches of the pool draw), as paired source / target batches."""
    frame = state.config
    s_b, t_b, _ = paired_calibration(
        source_anchor_loader, target_anchor_loader,
        num_batches=frame.anchor_batches + frame.registration_batches, seed=frame.pool_seed,
    )
    return s_b[: frame.anchor_batches], t_b[: frame.anchor_batches]


def _synthesis_hook(state, target, target_model, target_base_sd, target_anchor_loader, *, config, device):
    """D2 / D5 solve hook. For D2, the G statistics of the target base on the frame's anchor images (the first
    anchor_batches of the pool draw); the target is loaded at its base weights for the capture and restored."""
    from .synthesis import capture_generic_inputs, make_synthesis_hook

    if config.synthesis != "ridge_preserve":
        return make_synthesis_hook(rank=config.update_rank)
    frame = state.config
    batches, _, _ = paired_calibration(
        target_anchor_loader, target_anchor_loader,
        num_batches=frame.anchor_batches + frame.registration_batches, seed=frame.pool_seed,
    )
    norm = None
    if config.anchor_filter is not None:
        flt = target.extras[f"norm_k{float(config.anchor_filter_k):g}"]
        norm = {"mu": target.mu, "k": float(flt["k"]), "median": flt["median"]}
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
