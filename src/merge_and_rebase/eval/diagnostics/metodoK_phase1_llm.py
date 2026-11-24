"""Metodo K, Phase 1 (forward only) for decoder LLM pairs: how many anchor rows (N documents x k units) give reliable
canonical correlations between two models, block by block?

The statistics, strata, halves, ICC, coverage, cost and writers are those of the vision runner
(``metodoK_phase1``, imported as ``v1``, NOT modified). This module adds what a text pool needs:

* **Ragged rows.** Documents have different numbers of units ``n_i`` (units = rows of the anchors: content tokens, or
  word units, see below). Cell ``(N, k)`` uses the documents with pool position ``p < N`` and, per document, the units
  of rank ``< k`` in the document's own seeded permutation of its ``n_i`` units (``unit_ranks``; depends on
  ``(seed, p)`` and ``n_i`` only; ``k = all`` is the last level: every unit). Nested in ``k`` and in ``N``; strata =
  (document stratum) x (unit level) x (half), exactly as in vision, so one forward pass of the largest cell gives every
  cell. Halves: ``v1.half_labels`` (``split_ids`` by document inside each document stratum, never by unit). A batch
  is padded to its longest document (``to_padded``); padded slots are never rows.
  The nominal ``T = N k`` of a cell is an upper bound: the realised numbers are reported per cell (``T_realised`` =
  units of the cell before the norm filter; ``rows`` = kept rows after it; ``mean_units_per_doc``).
* **Rows.** ``rows = "token"`` (P -> Q: Qwen2.5-0.5B -> 1.5B share one tokenizer): every content token except the first
  of each document (attention sink), the same position in both models. ``rows = "word"`` (L -> Q: Llama-3.2-1B ->
  Qwen2.5-1.5B): word units (maximal ``\\S+`` runs of the raw text), represented by the LAST token of the word; only
  the units present in both tokenizations (``anchors/text_units.pair_unit_batches``); word order = token order, so the
  banks of the two models are row-aligned. Exactly the rows of ``text_units='word'`` anchors states.
* **Frame.** ``mu``, ``c`` and the ``anchor_filter='norm'`` k = 3 flags come from a fixed REGISTRATION set (default 320
  documents), disjoint from the pool, computed here with the procedure of ``AnchorState.register_model`` /
  ``norm_filter`` (three passes: moments, median of the centred norms, moments without the rows above ``k m``). The
  old anchors states (pre-K, 192 registration documents, no stored norm filter) are not read.
* **Nulls (pluggable, ``NullScheme``).** Primary ``N1`` (random unit): the source row ``(i, u)`` is paired with a
  random unit of a DIFFERENT document ``pi_r(i)`` (derangement inside the (batch, half) group, as vision; units of the
  partner drawn by an independent permutation per (replicate, document); source rows beyond the partner's unit count
  are dropped uniformly at random and the null ``Sigma_mk`` is rescaled by ``n_rows / n_null_rows``). Secondary ``N2``
  (position kept): the old vision null, partner row ``(pi_r(i), same slot)``; valid only after per-position centring,
  kept as a control. A new null = a class with ``prepare`` / ``rows`` registered in ``NULL_SCHEMES``.
* **r*.** One threshold pooled over indices and replicates (``null_quantile`` of all null ``Gamma_out`` values,
  ``interpolation='higher'``); ``above = Gamma_out > threshold``; ``r_star = r_star_prefix`` = largest ``r`` such that
  >= ``prefix_frac`` (0.95) of the indices ``< r`` are above; ``r_star_excess = n_above - 0.05 (q - n_above)``;
  ``r_star_contig`` = leading contiguous run (the old rule, kept for transparency). Coverage uses ``r_star``.
* **rho_icc / n_eff.** As vision (nominal ``k``) plus ``n_eff_*_real`` with the size-weighted mean cluster size
  ``sum n_i^2 / sum n_i`` of the kept rows (the design effect of ragged clusters).
* **Selection.** ``select_cell_abs`` / ``select_global_abs``: cheapest cell whose mean ``|dGamma_out|`` over the top
  ``r_ref`` directions (``r_ref`` = ``r_star`` of the largest cell of the block) against every larger cell is
  ``< tol_g`` (absolute, 0.02); optional relative ``r*`` criterion (``tol_r``, off by default).

Coverage. ``E = H_instruct - H_base`` of the SOURCE model on the IFEval VAL prompts (selection slice of
``ifeval_k_v1``, never the test slice; prompt text, same units as the anchors, base tokenizer for both models).

Run: ``python -m merge_and_rebase.eval.diagnostics.metodoK_phase1_llm --config configs/examples/metodoK_phase1/P_Q.json``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from ...rebase.methods.anchors.core import (
    CanonicalFactors,
    MomentAccumulator,
    _svd_canonical,
    canonical_correlations,
    canonical_correlations_out,
    canonical_factors_from_sigma,
)
from . import metodoK_phase1 as v1

Tensor = torch.Tensor
STATUS_DEGENERATE = v1.STATUS_DEGENERATE
ALL = "all"


# ----------------------------------------------------------------------------------------------------------------
# pure pieces: ragged rows, ranks, padding
# ----------------------------------------------------------------------------------------------------------------


def resolve_ks(ks: Sequence[int | str], max_units: int) -> tuple[int, ...]:
    """``ks`` with ``"all"`` (the last level: every unit of a document) replaced by ``max_units`` (>= any n_i)."""
    out = tuple(int(max_units) if (isinstance(k, str) and k.lower() == ALL) else int(k) for k in ks)
    v1.Grid((1,), out)  # validates: positive, strictly increasing
    return out


def unit_ranks(positions: Tensor, counts: Tensor, n_slots: int, seed: int) -> Tensor:
    """``[B, n_slots]`` rank of every unit slot in its document's seeded permutation of its ``counts[b]`` units (rank 0 =
    first drawn). Padded slots get rank 0 (they are never valid). Depends on ``(seed, position, count)`` only."""
    out = torch.zeros(len(positions), int(n_slots), dtype=torch.long)
    for n, (p, c) in enumerate(zip(positions.tolist(), counts.tolist(), strict=True)):
        if c == 0:
            continue
        gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(p) + 7919)
        perm = torch.randperm(int(c), generator=gen)
        out[n, perm] = torch.arange(int(c))
    return out


def to_padded(bank: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """``[1, N, d]`` bank of the rows of ``mask`` ``[B, T]`` (row-major, as ``iter_capture_tokens`` returns them) ->
    ``([B, S, d]`` padded per-document rows in unit order, ``counts [B])``."""
    mask = mask.bool().cpu()
    counts = mask.sum(1)
    if bank.shape[0] != 1 or int(bank.shape[1]) != int(counts.sum()):
        raise ValueError(f"bank {tuple(bank.shape)} does not hold the {int(counts.sum())} rows of the mask")
    n_slots = max(int(counts.max()), 1)
    slot = mask.long().cumsum(1) - 1
    rows, cols = mask.nonzero(as_tuple=True)
    out = torch.zeros(mask.shape[0], n_slots, bank.shape[-1], dtype=bank.dtype, device=bank.device)
    out[rows.to(bank.device), slot[rows, cols].to(bank.device)] = bank[0]
    return out, counts


def slot_valid(counts: Tensor, n_slots: int) -> Tensor:
    """``[B, S]`` bool: slot ``< counts[b]``."""
    return torch.arange(int(n_slots))[None, :] < counts[:, None]


# ----------------------------------------------------------------------------------------------------------------
# pluggable nulls
# ----------------------------------------------------------------------------------------------------------------


class NullScheme:
    """Pairs every source row ``(i, slot)`` of a batch with a target row of another document.

    ``prepare`` is called once per batch and half assignment (``half [B]``, global pool ``positions [B]``, unit
    ``counts [B]``); ``rows(ctx, r, pk, valid)`` returns, for replicate ``r``, ``(pk_r [B, S, d_k], valid_r [B, S])``:
    the target row paired with source row ``(i, s)`` is ``pk_r[i, s]`` and it is usable iff ``valid_r[i, s]``."""

    name = "null"

    def prepare(self, half: Tensor, positions: Tensor, counts: Tensor, batch_index: int, n_perm: int, seed: int) -> Any:
        raise NotImplementedError

    def rows(self, ctx: Any, r: int, pk: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class PositionKeptNull(NullScheme):
    """N2: the vision null: partner row = (derangement partner document, SAME slot)."""

    name = "N2"

    def prepare(self, half, positions, counts, batch_index, n_perm, seed):
        return v1.null_partners(half, n_perm, seed, batch_index)

    def rows(self, ctx, r, pk, valid):
        part = ctx[r].to(pk.device)
        pc = part.clamp(min=0)
        return pk[pc], valid[pc] & (part >= 0)[:, None]


class RandomUnitNull(NullScheme):
    """N1: partner = (derangement partner document j, a RANDOM unit of j). Per (replicate, document) an independent
    permutation ``rho`` of the source units and ``tau`` of the partner units: source slot ``s`` is paired with partner
    unit ``tau[rho[s]]`` when ``rho[s] < n_j`` (so the dropped source rows are a uniformly random subset)."""

    name = "N1"

    def prepare(self, half, positions, counts, batch_index, n_perm, seed):
        return {"partners": v1.null_partners(half, n_perm, seed, batch_index), "positions": positions.tolist(),
                "counts": counts.tolist(), "seed": int(seed)}

    def rows(self, ctx, r, pk, valid):
        part = ctx["partners"][r]
        b, s = int(valid.shape[0]), int(valid.shape[1])
        idx = torch.zeros(b, s, dtype=torch.long)
        ok = torch.zeros(b, s, dtype=torch.bool)
        counts = ctx["counts"]
        for i in range(b):
            j = int(part[i])
            ni = counts[i]
            if j < 0 or ni == 0 or counts[j] == 0:
                continue
            nj = counts[j]
            gen = torch.Generator().manual_seed(ctx["seed"] * 1_000_003 + r * 10_007 + ctx["positions"][i] * 131 + 15_485_863)
            rho = torch.randperm(ni, generator=gen)
            tau = torch.randperm(nj, generator=gen)
            m = rho < nj
            idx[i, :ni][m] = tau[rho[m]]
            ok[i, :ni] = m
        pc = part.clamp(min=0).to(pk.device)
        idx, ok = idx.to(pk.device), ok.to(pk.device)
        return pk[pc[:, None], idx], valid[pc[:, None], idx] & ok


NULL_SCHEMES: dict[str, type[NullScheme]] = {"N1": RandomUnitNull, "N2": PositionKeptNull}


def make_nulls(names: Sequence[str]) -> list[NullScheme]:
    unknown = [n for n in names if n not in NULL_SCHEMES]
    if unknown or not names:
        raise ValueError(f"nulls must be a non-empty subset of {sorted(NULL_SCHEMES)}, got {list(names)}")
    return [NULL_SCHEMES[n]() for n in names]


# ----------------------------------------------------------------------------------------------------------------
# streaming statistics per stratum (ragged rows, pluggable nulls)
# ----------------------------------------------------------------------------------------------------------------


class RaggedStratifiedSigma:
    """``v1.StratifiedSigma`` for ragged documents with pluggable nulls (same tensors, same strata). ``update`` takes
    rows already centred and scaled ``pm [B, S, d_m]``, ``pk [B, S, d_k]``, ``valid [B, S]`` (real slot AND kept by the
    norm filter), ``real [B, S]`` (real slot), ``level [B, S]``, ``half [B]`` and the batch description the nulls need."""

    def __init__(self, d_m: int, d_k: int, grid: v1.Grid, n_perm: int, nulls: Sequence[NullScheme], *,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64, seed: int = 0,
                 compute_device: str | torch.device | None = None, compute_dtype: torch.dtype | None = None) -> None:
        """Storage (``device``, ``dtype``: float64, host RAM by default) and compute (``compute_device`` /
        ``compute_dtype``: where and in which precision the per-batch products are formed, default = storage) are
        separate, as in ``v1.StratifiedSigma``: float32 products on a GPU added to float64 CPU storage keep the
        accumulation exact (the rows are pre-centred)."""
        self.d_m, self.d_k, self.grid, self.n_perm, self.seed = int(d_m), int(d_k), grid, int(n_perm), int(seed)
        self.device, self.dtype = torch.device(device), dtype
        self.cdev = torch.device(compute_device) if compute_device is not None else self.device
        self.cdtype = compute_dtype or dtype
        self.nulls = list(nulls) if n_perm else []
        n_a, n_l = len(grid.n_images), len(grid.ks)
        z = lambda *s: torch.zeros(*s, dtype=dtype, device=self.device)  # noqa: E731
        self.mm, self.kk, self.mk = z(n_a, n_l, 2, d_m, d_m), z(n_a, n_l, 2, d_k, d_k), z(n_a, n_l, 2, d_m, d_k)
        self.n = torch.zeros(n_a, n_l, 2, dtype=torch.long)
        self.units = torch.zeros(n_a, n_l, 2, dtype=torch.long)  # real units (before the norm filter)
        self.null_mk = {s.name: z(self.n_perm, n_a, n_l, 2, d_m, d_k) for s in self.nulls}
        self.null_n = {s.name: torch.zeros(self.n_perm, n_a, n_l, 2, dtype=torch.long) for s in self.nulls}
        self.icc_b, self.icc_s = z(n_a, n_l, d_m, d_m), z(n_a, n_l, d_m)
        self.icc_stats = torch.zeros(n_a, n_l, 3, dtype=torch.float64)

    def _add(self, target: Tensor, x: Tensor, y: Tensor) -> None:
        target += (x.T @ y).to(self.device, self.dtype)

    @torch.no_grad()
    def update(self, a: int, pm: Tensor, pk: Tensor, valid: Tensor, real: Tensor, level: Tensor, half: Tensor,
               positions: Tensor, counts: Tensor, batch_index: int) -> None:
        pm_s = pm.to(self.device, self.dtype)  # storage precision (ICC sums)
        pm, pk = pm.to(self.cdev, self.cdtype), pk.to(self.cdev, self.cdtype)
        valid, real, level, half = valid.to(self.cdev), real.to(self.cdev), level.to(self.cdev), half.to(self.cdev)
        n_l = len(self.grid.ks)
        halves = [(h, (half == h).nonzero().flatten()) for h in (0, 1)]
        for h, ih in halves:
            if ih.numel() == 0:
                continue
            pmh, pkh, vh, lh, rh = pm[ih], pk[ih], valid[ih], level[ih], real[ih]
            for lv in range(n_l):
                m = vh & (lh == lv)
                x, y = pmh[m], pkh[m]
                self._add(self.mm[a, lv, h], x, x)
                self._add(self.kk[a, lv, h], y, y)
                self._add(self.mk[a, lv, h], x, y)
                self.n[a, lv, h] += int(x.shape[0])
                self.units[a, lv, h] += int((rh & (lh == lv)).sum())
        for scheme in self.nulls:
            ctx = scheme.prepare(half.cpu(), positions, counts, batch_index, self.n_perm, self.seed)
            for r in range(self.n_perm):
                pk_r, v_r = scheme.rows(ctx, r, pk, valid)
                for h, ih in halves:
                    if ih.numel() == 0:
                        continue
                    pmh, pkr, vh, lh, vr = pm[ih], pk_r[ih], valid[ih], level[ih], v_r[ih]
                    for lv in range(n_l):
                        mn = vh & (lh == lv) & vr
                        self._add(self.null_mk[scheme.name][r, a, lv, h], pmh[mn], pkr[mn])
                        self.null_n[scheme.name][r, a, lv, h] += int(mn.sum())
        valid_s, level_s = valid.to(self.device), level.to(self.device)
        for lv in range(n_l):  # ICC sums at the cumulative level <= lv, all documents of the batch
            m = valid_s & (level_s <= lv)
            n_i = m.sum(1).to(self.dtype)
            s = torch.einsum("bt,btd->bd", m.to(self.dtype), pm_s)
            ok = n_i > 0
            sn = s[ok] / n_i[ok].sqrt()[:, None]
            self.icc_b[a, lv] += sn.T @ sn
            self.icc_s[a, lv] += s[ok].sum(0)
            self.icc_stats[a, lv] += torch.tensor(
                [float(ok.sum()), float(n_i.sum()), float((n_i[ok] ** 2).sum())], dtype=torch.float64)

    def cell(self, ia: int, il: int) -> tuple[v1.CellSums, dict[str, tuple[Tensor, Tensor]], Tensor]:
        """``(CellSums without null, {null name: (null_mk [R,2,dm,dk], null_n [R,2])}, real units per half [2])``."""
        sl = (slice(0, ia + 1), slice(0, il + 1))
        sums = v1.CellSums(
            n_images=int(self.grid.n_images[ia]), k=int(self.grid.ks[il]),
            mm=self.mm[sl].sum((0, 1)), kk=self.kk[sl].sum((0, 1)), mk=self.mk[sl].sum((0, 1)), n=self.n[sl].sum((0, 1)),
            null_mk=None, null_n=None,
            icc_b=self.icc_b[:ia + 1, il].sum(0), icc_s=self.icc_s[:ia + 1, il].sum(0),
            icc_stats=self.icc_stats[:ia + 1, il].sum(0))
        nulls = {name: (self.null_mk[name][:, sl[0], sl[1]].sum((1, 2)), self.null_n[name][:, sl[0], sl[1]].sum((1, 2)))
                 for name in self.null_mk}
        return sums, nulls, self.units[sl].sum((0, 1))


# ----------------------------------------------------------------------------------------------------------------
# per-cell analysis (r* rules, several nulls)
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class LlmAnalysisOptions:
    n_null: int = 19
    null_quantile: float = 0.95
    quantile_interpolation: str = "higher"
    prefix_frac: float = 0.95
    icc_top: int = 10
    linalg_device: str = "cpu"


def rstar_rules(gamma_out: Tensor, null: Tensor, opts: LlmAnalysisOptions) -> dict[str, Any]:
    """The r* rules against ONE pooled threshold: ``null [R, q]`` (all replicates and indices pooled)."""
    q = int(gamma_out.numel())
    thr = torch.quantile(null.reshape(-1).to(torch.float64), float(opts.null_quantile),
                         interpolation=opts.quantile_interpolation)
    above = gamma_out > thr
    n_above = int(above.sum())
    contig = int(torch.cumprod(above.to(torch.long), 0).sum())
    frac = torch.cumsum(above.to(torch.float64), 0) / torch.arange(1, q + 1, dtype=torch.float64)
    ok = (frac >= float(opts.prefix_frac)).nonzero().flatten()
    prefix = int(ok[-1]) + 1 if ok.numel() else 0
    excess = n_above - (1.0 - float(opts.null_quantile)) * (q - n_above)
    return {"threshold": float(thr), "above": above, "n_above": n_above, "r_star_contig": contig,
            "r_star_prefix": prefix, "r_star_excess": float(excess)}


def _null_gamma_out(fac: CanonicalFactors, null_a: Tensor, sig_b: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    """Null ``Gamma_out`` of ``R`` replicates: directions from half A's null ``Sigma_mk`` [R, d_m, d_k] in the true
    half-A whitening ``fac``, evaluated on half B's null ``Sigma_mk`` (same semantics as ``v1._null_gamma_out``)."""
    out = []
    smm_b, skk_b, mk_b = sig_b
    for r in range(null_a.shape[0]):
        m_null = (fac.v_m.T @ null_a[r] @ fac.v_k) / fac.s_m[:, None] / fac.s_k[None, :]
        f_null = CanonicalFactors(fac.v_m, fac.s_m, fac.v_k, fac.s_k, m_null, fac.n_rows)
        out.append(canonical_correlations_out(f_null, (smm_b, skk_b, mk_b[r])))
    return torch.stack(out)


def analyze_cell_llm(
    sums: v1.CellSums, nulls: Mapping[str, tuple[Tensor, Tensor]], opts: LlmAnalysisOptions,
    gram_e: Mapping[str, Tensor] | None = None, units_real: Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Scalars (JSON) and tensors (``.pt``) of one cell. The FIRST null of ``nulls`` is primary (keys without a prefix);
    the others write ``<name>_...`` keys. A degenerate cell returns status and counts only."""
    dev = opts.linalg_device
    s = sums.to(dev)
    d_m, d_k = int(s.mm.shape[-1]), int(s.kk.shape[-1])
    n_half = [int(x) for x in s.n]
    n_tot = sum(n_half)
    info: dict[str, Any] = {
        "N": sums.n_images, "k": sums.k, "T_nominal": sums.n_images * sums.k, "rows": n_tot, "rows_half": n_half,
        "d_m": d_m, "d_k": d_k,
        "status": v1.classify_rows(min(n_half), d_m, d_k) if min(n_half) > 0 else STATUS_DEGENERATE,
    }
    if units_real is not None:
        info["T_realised"] = int(units_real.sum())
    if info["status"] == STATUS_DEGENERATE:
        info["reason"] = f"rows of a half {min(n_half)} <= d_m + d_k = {d_m + d_k}"
        return info, {}
    f64 = torch.float64
    mm, kk, mk = s.mm.to(f64), s.kk.to(f64), s.mk.to(f64)
    try:
        fac_all = canonical_factors_from_sigma(mm.sum(0), kk.sum(0), mk.sum(0), n_tot)
        fac = [canonical_factors_from_sigma(mm[h], kk[h], mk[h], n_half[h]) for h in (0, 1)]
        out_ab = canonical_correlations_out(fac[0], (mm[1], kk[1], mk[1]))
        out_ba = canonical_correlations_out(fac[1], (mm[0], kk[0], mk[0]))
        nulls_raw = {}
        for name, (nm, nn_) in nulls.items():
            scale = s.n.to(f64)[None, :] / nn_.to(dev, f64).clamp(min=1.0)  # [R, 2]
            nmk = nm.to(dev, f64) * scale[:, :, None, None]
            nab = _null_gamma_out(fac[0], nmk[:, 0], (mm[1], kk[1], nmk[:, 1]))
            nba = _null_gamma_out(fac[1], nmk[:, 1], (mm[0], kk[0], nmk[:, 0]))
            nulls_raw[name] = (nab, nba)
    except ValueError as exc:
        info["status"], info["reason"] = STATUS_DEGENERATE, f"{exc}"[:200]
        return info, {}
    gamma_in = canonical_correlations(fac_all)
    q = min(out_ab.numel(), out_ba.numel())
    gamma_out = 0.5 * (out_ab[:q] + out_ba[:q])
    tensors: dict[str, Tensor] = {
        "gamma_in": gamma_in.cpu(), "gamma_out": gamma_out.cpu(), "gamma_out_ab": out_ab.cpu(), "gamma_out_ba": out_ba.cpu(),
    }
    info.update({
        "q": int(q), "rank_m": int(fac_all.s_m.numel()), "rank_k": int(fac_all.s_k.numel()),
        "gamma_in_top1": float(gamma_in[0]), "gamma_in_mean": float(gamma_in.mean()),
        "gamma_out_top1": float(gamma_out[0]), "gamma_out_mean": float(gamma_out.mean()),
        "gamma_out_ab_mean": float(out_ab.mean()), "gamma_out_ba_mean": float(out_ba.mean()),
    })
    r_star = 0
    for n_idx, (name, (nab, nba)) in enumerate(nulls_raw.items()):
        null_mean = 0.5 * (nab[:, :q] + nba[:, :q])  # [R, q]
        rules = rstar_rules(gamma_out, null_mean, opts)
        pre = "" if n_idx == 0 else f"{name}_"
        info.update({
            f"{pre}r_star": rules["r_star_prefix"], f"{pre}r_star_prefix": rules["r_star_prefix"],
            f"{pre}r_star_contig": rules["r_star_contig"], f"{pre}r_star_excess": rules["r_star_excess"],
            f"{pre}n_above": rules["n_above"], f"{pre}threshold": rules["threshold"],
            f"{pre}threshold_top1": rules["threshold"], f"{pre}n_null": int(null_mean.shape[0]),
        })
        suffix = "" if n_idx == 0 else f"_{name}"
        tensors.update({f"null_gamma_out{suffix}": null_mean.cpu(), f"above{suffix}": rules["above"].cpu()})
        if n_idx == 0:
            r_star = rules["r_star_prefix"]
            info["null_primary"] = name
    y, _, _ = _svd_canonical(fac_all.m)
    n_top = min(int(opts.icc_top), y.shape[1])
    w = fac_all.v_m @ (y[:, :n_top] / fac_all.s_m[:, None])
    rho_dir, rho_tr = v1.icc_directions(s, w)
    rho_top = float(rho_dir.mean()) if not bool(torch.isnan(rho_dir).any()) else float("nan")
    rho_tr = float(rho_tr)
    n_img, n_rows, sum_n2 = (float(x) for x in s.icc_stats)
    n_w = sum_n2 / n_rows if n_rows > 0 else float("nan")  # size-weighted mean cluster size of the kept rows

    def n_eff(rho: float) -> float:
        k = sums.k
        return float(sums.n_images * k / (1.0 + (k - 1) * max(rho, 0.0))) if not math.isnan(rho) else float(sums.n_images)

    def n_eff_real(rho: float) -> float:
        return float(n_rows / (1.0 + (n_w - 1.0) * max(rho, 0.0))) if not math.isnan(rho) else float(n_img)

    tensors["rho_icc_top"] = rho_dir.cpu()
    info.update({
        "rho_icc_top1": float(rho_dir[0]), "rho_icc_top_mean": rho_top, "rho_icc_trace": rho_tr,
        "n_eff_top": n_eff(rho_top), "n_eff_trace": n_eff(rho_tr),
        "n_eff_top_real": n_eff_real(rho_top), "n_eff_trace_real": n_eff_real(rho_tr),
        "mean_units_per_doc": n_rows / n_img if n_img > 0 else float("nan"),
    })
    if gram_e:
        info["coverage"] = {task: v1.task_coverage(g.to(dev), fac_all.v_m, y[:, :r_star]) for task, g in gram_e.items()}
        info["coverage_isotropic_reference"] = r_star / int(fac_all.s_m.numel())
    return info, tensors


# ----------------------------------------------------------------------------------------------------------------
# selection (absolute |dGamma_out| over the top r_ref directions)
# ----------------------------------------------------------------------------------------------------------------


def _stable_abs(c: Mapping[str, Any], o: Mapping[str, Any], r_ref: int, tol_g: float, tol_r: float | None) -> bool:
    if tol_r is not None and abs(int(o["r_star"]) - int(c["r_star"])) > tol_r * max(int(o["r_star"]), 1):
        return False
    g_c, g_o = c["gamma_out_vec"], o["gamma_out_vec"]
    n = min(int(r_ref), len(g_c), len(g_o))
    return n == 0 or float((g_c[:n] - g_o[:n]).abs().mean()) < tol_g


def reference_cell(cells: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The largest valid cell (nominal T, ties: more documents) of a block."""
    ok = [c for c in cells if c["status"] != STATUS_DEGENERATE]
    return max(ok, key=lambda c: (c["T_nominal"], c["N"])) if ok else None


def select_cell_abs(cells: Sequence[Mapping[str, Any]], *, tol_g: float = 0.02, tol_r: float | None = None) -> dict[str, Any]:
    ok = [c for c in cells if c["status"] != STATUS_DEGENERATE]
    if not ok:
        return {"cell": None, "reason": "every cell is degenerate"}
    ref = reference_cell(ok)
    r_ref = int(ref["r_star"])
    for c in sorted(ok, key=lambda c: (c["T_nominal"], c["N"])):
        larger = [o for o in ok if o["T_nominal"] > c["T_nominal"]]
        if all(_stable_abs(c, o, r_ref, tol_g, tol_r) for o in larger):
            return {"cell": {"N": c["N"], "k": c["k"], "T_nominal": c["T_nominal"]}, "trivial": not larger,
                    "r_ref": r_ref, "tol_gamma": tol_g, "tol_r": tol_r}
    return {"cell": None, "reason": "no cell is stable against all larger cells", "r_ref": r_ref}


def select_global_abs(per_block: Sequence[Sequence[Mapping[str, Any]]], *, tol_g: float = 0.02,
                      tol_r: float | None = None) -> dict[str, Any]:
    keys = {(c["N"], c["k"]) for c in per_block[0]} if per_block else set()
    good = []
    for key in keys:
        valid = True
        for cells in per_block:
            by = {(c["N"], c["k"]): c for c in cells}
            c = by.get(key)
            ref = reference_cell(cells)
            if c is None or c["status"] == STATUS_DEGENERATE or ref is None:
                valid = False
                break
            larger = [o for o in cells if o["status"] != STATUS_DEGENERATE and o["T_nominal"] > c["T_nominal"]]
            if not all(_stable_abs(c, o, int(ref["r_star"]), tol_g, tol_r) for o in larger):
                valid = False
                break
        if valid:
            good.append(key)
    if not good:
        return {"cell": None, "reason": "no cell is valid in every block"}
    n, k = min(good, key=lambda nk: (nk[0] * nk[1], nk[0]))
    return {"cell": {"N": n, "k": k, "T_nominal": n * k}, "tol_gamma": tol_g, "tol_r": tol_r}


# ----------------------------------------------------------------------------------------------------------------
# cost estimate
# ----------------------------------------------------------------------------------------------------------------


def estimate_cost_llm(d_m: int, d_k: int, grid: v1.Grid, n_perm: int, n_nulls: int, n_blocks: int, *,
                      mean_units: float, params_m: float, params_k: float, ram_budget_gb: float | None = None,
                      blocks_per_pass: int | None = None) -> dict[str, Any]:
    """``v1.estimate_cost`` with ``n_nulls`` null schemes (the null term is linear in ``n_perm * n_nulls``) and the
    flops of the forward passes (``2 * params`` per token, both models, every pass runs the whole network)."""
    cost = v1.estimate_cost(d_m, d_k, grid, n_perm * n_nulls, n_blocks, ram_budget_gb=ram_budget_gb,
                            blocks_per_pass=blocks_per_pass, rows_per_image=max(int(round(mean_units)), 1))
    tokens = grid.n_max * int(grid.n_tokens)  # padded length bound: every document padded to max_length
    cost["forward_flops_per_pass"] = 2.0 * (params_m + params_k) * tokens
    cost["forward_flops_total"] = cost["forward_flops_per_pass"] * cost["n_passes"]
    cost["n_nulls"] = n_nulls
    return cost


# ----------------------------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class LlmModelSpec:
    model: str  # HF id of the base model
    arch: str = "auto"
    instruct: str | None = None  # source only: the IFEval expert (Instruct) used for the task coverage


@dataclass
class LlmPhase1Config:
    source: LlmModelSpec
    target: LlmModelSpec
    out_dir: str
    pair: str = "pair"
    rows: str = "token"  # "token" (same tokenizer) | "word" (word units, any tokenizers)
    pool_split: str = "train"  # WikiText-103 raw split of the anchor documents (the validation split holds only 898
    pool_min_chars: int = 600  # paragraphs >= 600 chars, fewer than N_max = 4096)
    pool_seed: int = 0
    max_length: int = 128  # tokens per document (BOS included), as the anchors pool of the existing LLM runs
    registration_split: str = "validation"
    registration_docs: int = 320
    registration_seed: int = 0
    n_images: tuple[int, ...] = (256, 1024, 4096)  # documents (name kept for the vision grid)
    ks: tuple[Any, ...] = (1, 16, ALL)
    seed: int = 0
    n_null: int = 19
    null_quantile: float = 0.95
    prefix_frac: float = 0.95
    nulls: tuple[str, ...] = ("N1",)
    anchor_filter: str | None = "norm"
    anchor_filter_k: float = 3.0
    tasks: tuple[str, ...] = ("ifeval",)
    ifeval_split: str = "ifeval_k_v1"
    ifeval_part: str = "selection"  # VAL slice; "test" is refused
    ifeval_n: int | None = None
    ifeval_max_length: int = 512
    ifeval_batch_size: int = 4
    batch_size: int = 16
    device: str = "cuda"
    dtype: str | None = "bf16"
    acc_device: str = "cpu"  # storage of the float64 accumulators and of the captured banks (host RAM)
    compute_device: str | None = None  # None = acc_device; "cuda" forms the batch products on the GPU
    compute_dtype: str | None = None  # None = float64; "float32" with a GPU (products added to float64 storage)
    linalg_device: str = "cpu"
    blocks: tuple[int, ...] | None = None
    blocks_per_pass: int | None = None
    ram_budget_gb: float = 24.0
    icc_top: int = 10
    tol_g: float = 0.02
    offline: bool = True
    overwrite: bool = False


def parse_config(payload: Mapping[str, Any]) -> LlmPhase1Config:
    p = dict(payload)
    for side in ("source", "target"):
        spec = dict(p[side])
        unknown = sorted(set(spec) - {f.name for f in fields(LlmModelSpec)})
        if unknown:
            raise ValueError(f"unknown {side} keys: {unknown}")
        p[side] = LlmModelSpec(**spec)
    unknown = sorted(set(p) - {f.name for f in fields(LlmPhase1Config)})
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    for key in ("tasks", "n_images", "ks", "blocks", "nulls"):
        if p.get(key) is not None:
            p[key] = tuple(p[key]) if key in ("tasks", "nulls", "ks") else tuple(int(x) for x in p[key])
    cfg = LlmPhase1Config(**p)
    resolve_ks(cfg.ks, cfg.max_length)
    v1.Grid(cfg.n_images, (1,))
    if cfg.rows not in ("token", "word"):
        raise ValueError("rows must be 'token' or 'word'")
    if cfg.anchor_filter not in {None, "norm"}:
        raise ValueError("anchor_filter must be null or 'norm'")
    if cfg.ifeval_part != "selection":
        raise ValueError("coverage uses the VAL (selection) prompts of the IFEval split only, never the test slice")
    if cfg.tasks != ("ifeval",):
        raise ValueError("LLM Phase 1: tasks = ['ifeval'] only (GSM8K dropped)")
    if cfg.source.instruct is None:
        raise ValueError("source.instruct (the IFEval expert) is required")
    if any(n % cfg.batch_size for n in cfg.n_images):
        raise ValueError(f"batch_size {cfg.batch_size} must divide every N in {cfg.n_images}")
    if cfg.compute_dtype not in (None, "float32", "float64"):
        raise ValueError("compute_dtype must be null, 'float32' or 'float64'")
    if cfg.n_null < 0 or not 0.0 < cfg.null_quantile < 1.0 or not 0.0 < cfg.prefix_frac <= 1.0:
        raise ValueError("n_null >= 0, 0 < null_quantile < 1, 0 < prefix_frac <= 1 required")
    make_nulls(cfg.nulls)
    return cfg


# ----------------------------------------------------------------------------------------------------------------
# text pool, paired unit batches
# ----------------------------------------------------------------------------------------------------------------


def _texts_sha(texts: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(texts)).encode()).hexdigest()


def pool_documents(cfg: LlmPhase1Config) -> tuple[list[str], list[str]]:
    """``(registration texts, pool texts)``. Registration: ``registration_docs`` paragraphs of ``registration_split``
    (the recipe of ``anchors_pool.load_pool_texts``). Pool: ``N_max`` paragraphs of ``pool_split`` (same recipe, then
    one seeded permutation = the pool order, position = document id). Disjoint by text, checked."""
    from ...eval.llm_rebase.anchors_pool import POOL_DATASETS, load_pool_texts

    path, name, _, column = POOL_DATASETS["wikitext103"]
    base = {"hf_path": path, "hf_name": name, "text_column": column, "min_chars": cfg.pool_min_chars}
    reg = load_pool_texts({**base, "split": cfg.registration_split, "n_sequences": cfg.registration_docs,
                           "seed": cfg.registration_seed})
    pool = load_pool_texts({**base, "split": cfg.pool_split, "n_sequences": max(cfg.n_images), "seed": cfg.pool_seed})
    perm = torch.randperm(len(pool), generator=torch.Generator().manual_seed(int(cfg.seed))).tolist()
    pool = [pool[i] for i in perm]
    if set(reg) & set(pool):
        raise ValueError("the pool and the registration documents overlap")
    return reg, pool


@dataclass
class PairBatch:
    """One batch of documents under both tokenizers, with the row mask of each model (same rows, same order)."""

    batch_m: dict[str, Any]
    batch_k: dict[str, Any]
    mask_m: Tensor
    mask_k: Tensor
    counts: Tensor  # [B] rows (units) per document
    stats: dict[str, int]


def pair_batch(raw_m: Mapping[str, Any], raw_k: Mapping[str, Any], family_adapter, rows: str) -> PairBatch:
    from ...rebase.methods._ariadne.capture import ROW_MASK_KEY, _attach_row_masks
    from ...rebase.methods.anchors.text_units import pair_unit_batches

    if rows == "word":
        wm, wk, st = pair_unit_batches(raw_m, raw_k, family_adapter)
    else:
        wm, wk, total, content, _ = _attach_row_masks(raw_m, raw_k, family_adapter, True)
        mask = wm[ROW_MASK_KEY].bool()
        ids_m = family_adapter.extract_calibration_batch(raw_m)["input_ids"][mask]
        ids_k = family_adapter.extract_calibration_batch(raw_k)["input_ids"][mask]
        if not torch.equal(ids_m, ids_k):
            raise ValueError("rows='token' needs identical token ids in the two models (same tokenizer)")
        st = {"source": int(mask.sum()), "target": int(mask.sum()), "common": int(mask.sum())}
    mm, mk = wm[ROW_MASK_KEY].bool().cpu(), wk[ROW_MASK_KEY].bool().cpu()
    if not torch.equal(mm.sum(1), mk.sum(1)):
        raise RuntimeError("source / target unit counts differ per document")
    return PairBatch(wm, wk, mm, mk, mm.sum(1), st)


class TextPairSource:
    """The modality seam of the LLM pair: documents in pool order, both tokenizations, paired unit rows."""

    def __init__(self, texts: Sequence[str], tok_m, tok_k, family_adapter, rows: str, batch_size: int, max_length: int):
        from ...data.llm_calibration import build_text_calibration_loader

        kw: dict[str, Any] = {"texts": list(texts), "batch_size": int(batch_size), "max_length": int(max_length)}
        if rows == "word":
            kw["word_units"] = True
        self.rows, self.family, self.batch_size, self.n_docs = rows, family_adapter, int(batch_size), len(texts)
        self.loader_m = build_text_calibration_loader(tokenizer=tok_m, **kw)
        self.loader_k = build_text_calibration_loader(tokenizer=tok_k, **kw)
        self.texts_sha256 = _texts_sha(texts)

    def paired(self) -> list[PairBatch]:
        out = [pair_batch(a, b, self.family, self.rows) for a, b in zip(self.loader_m, self.loader_k, strict=True)]
        return out

    def own(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Registration batches (rows of the state's registration pass): token mode = the shared rows; word mode = each
        model's OWN units (``own_unit_batch``)."""
        if self.rows == "token":
            pairs = self.paired()
            return [p.batch_m for p in pairs], [p.batch_k for p in pairs]
        from ...rebase.methods.anchors.text_units import own_unit_batch

        return ([own_unit_batch(b, self.family)[0] for b in self.loader_m],
                [own_unit_batch(b, self.family)[0] for b in self.loader_k])


# ----------------------------------------------------------------------------------------------------------------
# registration frame (mu, c, norm filter) of one model, from the registration batches
# ----------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def registration_stats(model, family_adapter, batches: Sequence[Mapping[str, Any]], blocks: Sequence[int], device: str,
                       k: float | None) -> dict[int, dict[str, Any]]:
    """Per block: ``mu``, ``c`` (all rows), and with ``k``: ``median`` of the centred norms and ``mu_f``, ``c_f`` of the
    rows within ``k * median`` (the three passes of ``AnchorState.register_model`` / ``norm_filter``)."""
    from ...rebase.methods._ariadne.capture import iter_capture_tokens

    req = {str(b): (b, "boundary") for b in blocks}
    mom = {b: MomentAccumulator() for b in blocks}
    for vals in iter_capture_tokens(model, batches, req, device, family_adapter=family_adapter):
        for b in blocks:
            mom[b].update(vals[str(b)])
    out: dict[int, dict[str, Any]] = {}
    for b in blocks:
        mu, c = mom[b].finalize()
        out[b] = {"mu": mu, "c": c, "rows": mom[b].n}
    if k is None:
        return out
    norms: dict[int, list[Tensor]] = {b: [] for b in blocks}
    for vals in iter_capture_tokens(model, batches, req, device, family_adapter=family_adapter):
        for b in blocks:
            x = vals[str(b)].double().reshape(-1, vals[str(b)].shape[-1])
            norms[b].append((x - out[b]["mu"]).norm(dim=-1))
    for b in blocks:
        out[b]["median"] = float(torch.cat(norms[b]).median())
    filt = {b: MomentAccumulator() for b in blocks}
    for vals in iter_capture_tokens(model, batches, req, device, family_adapter=family_adapter):
        for b in blocks:
            x = vals[str(b)].double().reshape(-1, vals[str(b)].shape[-1])
            filt[b].update(x[(x - out[b]["mu"]).norm(dim=-1) <= float(k) * out[b]["median"]])
    for b in blocks:
        out[b]["mu_f"], out[b]["c_f"] = filt[b].finalize()
        out[b]["rows_kept"] = filt[b].n
    return out


def block_filters(reg_m, reg_k, pairing: Sequence[int], blocks: Sequence[int], k: float | None,
                  device: str) -> dict[int, v1.BlockFilter]:
    f64 = torch.float64
    out = {}
    for j in blocks:
        i = int(pairing[j])
        rm, rk = reg_m[i], reg_k[j]
        if k is None:
            out[j] = v1.BlockFilter(j, i, rm["mu"].to(device, f64), rm["c"], rk["mu"].to(device, f64), rk["c"])
        else:
            out[j] = v1.BlockFilter(j, i, rm["mu_f"].to(device, f64), rm["c_f"], rk["mu_f"].to(device, f64), rk["c_f"],
                                    rm["mu"].to(device, f64), rm["median"], rk["mu"].to(device, f64), rk["median"], float(k))
    return out


# ----------------------------------------------------------------------------------------------------------------
# forward passes
# ----------------------------------------------------------------------------------------------------------------


def accumulate_batch(accs: Mapping[int, RaggedStratifiedSigma], filters: Mapping[int, v1.BlockFilter], blocks: Sequence[int],
                     vals_m: Mapping[str, Tensor], vals_k: Mapping[str, Tensor], pb: PairBatch, positions: Tensor,
                     grid: v1.Grid, halves: Tensor, seed: int, acc_device: str, batch_index: int) -> None:
    """Add one batch (captured banks of the two models) to the accumulators of ``blocks`` (pure, no model)."""
    a = int(v1.image_stratum(positions[:1], grid))
    if int(v1.image_stratum(positions[-1:], grid)) != a:
        raise RuntimeError("a batch straddles two document strata")
    half = halves[positions]
    n_slots = max(int(pb.counts.max()), 1)
    if n_slots > grid.n_tokens:
        raise ValueError(f"a document has {n_slots} units > the last k ({grid.n_tokens}); raise max_length / ks[-1]")
    level = v1.token_levels(unit_ranks(positions, pb.counts, n_slots, seed), grid.ks)
    real = slot_valid(pb.counts, n_slots)
    for j in blocks:
        f = filters[j]
        t_m, _ = to_padded(vals_m[str(f.i)].to(acc_device), pb.mask_m)
        t_k, _ = to_padded(vals_k[str(j)].to(acc_device), pb.mask_k)
        pm, pk, valid = f.rows(t_m, t_k)
        valid = valid & real.to(valid.device)
        accs[j].update(a, pm, pk, valid, real, level, half, positions, pb.counts, batch_index)


def forward_pass(pairs: Sequence[PairBatch], model_m, fam_m, model_k, fam_k, filters, blocks: Sequence[int],
                 grid: v1.Grid, halves: Tensor, cfg: LlmPhase1Config, nulls: Sequence[NullScheme], pair: str,
                 ) -> dict[int, RaggedStratifiedSigma]:
    from ...rebase.methods._ariadne.capture import iter_capture_tokens

    m_req = {str(i): (i, "boundary") for i in sorted({filters[j].i for j in blocks})}
    k_req = {str(j): (j, "boundary") for j in blocks}
    d_m = {j: int(filters[j].mu_m.numel()) for j in blocks}
    d_k = {j: int(filters[j].mu_k.numel()) for j in blocks}
    cdtype = None if cfg.compute_dtype is None else getattr(torch, cfg.compute_dtype)
    accs = {j: RaggedStratifiedSigma(d_m[j], d_k[j], grid, cfg.n_null, nulls, device=cfg.acc_device, seed=cfg.seed,
                                     compute_device=cfg.compute_device, compute_dtype=cdtype) for j in blocks}
    bs = cfg.batch_size
    for b, pb in enumerate(pairs):
        pos = torch.arange(b * bs, b * bs + int(pb.counts.shape[0]))
        vals_m = next(iter_capture_tokens(model_m, [pb.batch_m], m_req, cfg.device, family_adapter=fam_m,
                                          store_device=cfg.acc_device))
        vals_k = next(iter_capture_tokens(model_k, [pb.batch_k], k_req, cfg.device, family_adapter=fam_k,
                                          store_device=cfg.acc_device))
        accumulate_batch(accs, filters, blocks, vals_m, vals_k, pb, pos, grid, halves, cfg.seed, cfg.acc_device, b)
        print(f"[metodoK_phase1_llm] {pair}: blocks {list(blocks)} batch {b + 1}/{len(pairs)}", flush=True)
    return accs


# ----------------------------------------------------------------------------------------------------------------
# task coverage: E = H_instruct - H_base of the source model on IFEval VAL prompts
# ----------------------------------------------------------------------------------------------------------------


def ifeval_val_prompts(split_name: str, part: str, limit: int | None) -> tuple[list[str], list[int]]:
    import datasets

    from ..llm_rebase.harness_split import load_harness_split

    if part != "selection":
        raise ValueError("only the selection (VAL) slice is allowed")
    idx = [int(i) for i in load_harness_split(split_name)[part]]
    if limit is not None:
        idx = idx[: int(limit)]
    ds = datasets.load_dataset("google/IFEval", split="train")
    return [str(ds[i]["prompt"]) for i in idx], idx


@torch.no_grad()
def ifeval_gram_pass(cfg: LlmPhase1Config, model_base, tok_base, fam, source_blocks: Sequence[int],
                     ) -> tuple[dict[int, Tensor], dict[str, Any]]:
    """``G_E = sum_rows dH^T dH`` per source block (``dH = H_instruct - H_base`` at the units of the IFEval VAL prompts)."""
    from transformers import AutoTokenizer

    from ...data.llm_calibration import build_text_calibration_loader
    from ...models.text_lm import TextBuildConfig, TextLM
    from ...rebase.methods._ariadne.capture import ROW_MASK_KEY, _first_content_position, iter_capture_tokens
    from ...rebase.methods.anchors.text_units import own_unit_batch

    prompts, idx = ifeval_val_prompts(cfg.ifeval_split, cfg.ifeval_part, cfg.ifeval_n)
    tok_it = AutoTokenizer.from_pretrained(cfg.source.instruct)
    a = tok_base(prompts, truncation=True, max_length=cfg.ifeval_max_length)["input_ids"]
    b = tok_it(prompts, truncation=True, max_length=cfg.ifeval_max_length)["input_ids"]
    if a != b:
        raise ValueError("the base and Instruct tokenizers differ on the IFEval prompts")
    instruct = TextLM.build(TextBuildConfig(cfg.source.instruct, cfg.source.arch, cfg.device, cfg.dtype)).model
    req = {str(i): (i, "boundary") for i in source_blocks}
    loader = build_text_calibration_loader(tokenizer=tok_base, texts=prompts, batch_size=cfg.ifeval_batch_size,
                                           max_length=cfg.ifeval_max_length, word_units=cfg.rows == "word")
    gram: dict[int, Tensor] = {}
    n_rows = 0
    for batch in loader:
        if cfg.rows == "word":
            wrapped, _ = own_unit_batch(batch, fam)
        else:
            content = fam.content_mask(batch).bool().cpu()
            wrapped = dict(fam.extract_calibration_batch(batch))
            wrapped[ROW_MASK_KEY] = content & ~_first_content_position(content)
        v_base = next(iter_capture_tokens(model_base, [wrapped], req, cfg.device, family_adapter=fam,
                                          store_device=cfg.acc_device))
        v_ft = next(iter_capture_tokens(instruct, [wrapped], req, cfg.device, family_adapter=fam,
                                        store_device=cfg.acc_device))
        for i in source_blocks:
            e = (v_ft[str(i)] - v_base[str(i)]).reshape(-1, v_ft[str(i)].shape[-1]).to(torch.float64)
            gram[i] = gram.get(i, 0) + e.T @ e
        n_rows += int(e.shape[0])
    del instruct
    meta = {"task": "ifeval", "expert": cfg.source.instruct, "split": f"{cfg.ifeval_split}:{cfg.ifeval_part} (VAL)",
            "n_prompts": len(prompts), "n_rows": n_rows, "rows": cfg.rows,
            "prompt_indices_sha256": hashlib.sha256(repr(idx).encode()).hexdigest(),
            "text": "raw IFEval prompt (no chat template), base tokenizer for both models"}
    return {i: g.cpu() for i, g in gram.items()}, meta


# ----------------------------------------------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------------------------------------------

CSV_COLUMNS = (
    "target_block", "source_block", "N", "k", "T_nominal", "T_realised", "status", "rows", "rows_half_min",
    "mean_units_per_doc", "rank_m", "rank_k", "q", "gamma_in_top1", "gamma_in_mean", "gamma_out_top1", "gamma_out_mean",
    "gamma_out_ab_mean", "gamma_out_ba_mean", "threshold", "r_star", "r_star_prefix", "r_star_excess", "r_star_contig",
    "n_above", "rho_icc_top1", "rho_icc_top_mean", "rho_icc_trace", "n_eff_top", "n_eff_trace", "n_eff_top_real",
    "n_eff_trace_real", "cov_isotropic",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _f(x: Any, nd: int = 3) -> str:
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def csv_rows_llm(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = list(summary["setting"]["tasks"])
    rows = []
    for blk in summary["blocks"]:
        for c in blk["cells"]:
            row = {"target_block": blk["target_block"], "source_block": blk["source_block"],
                   "rows_half_min": min(c["rows_half"]), "cov_isotropic": c.get("coverage_isotropic_reference")}
            row.update({k: c.get(k) for k in CSV_COLUMNS if k not in row})
            for key in c:  # secondary nulls: <name>_r_star_prefix, ...
                if key.endswith("_r_star_prefix"):
                    name = key[: -len("r_star_prefix")]
                    row.update({f"{name}{k}": c.get(f"{name}{k}") for k in ("r_star_prefix", "r_star_excess",
                                                                             "r_star_contig", "n_above", "threshold")})
            for t in tasks:
                cv = (c.get("coverage") or {}).get(t, {})
                row[f"cov_total_{t}"] = cv.get("cov_total")
                row[f"cov_span_{t}"] = cv.get("cov_span")
            rows.append(row)
    return rows


def render_md_llm(summary: Mapping[str, Any]) -> str:
    st = summary["setting"]
    lines = [f"# Metodo K, Phase 1 (LLM): {st['pair']}", "", "## Setting", ""]
    for k in ("rows", "source", "target", "pool_registration", "units", "grid", "frame", "filter", "seeds", "halves", "null",
              "nulls", "rstar_rule", "selection_rule", "icc", "coverage", "code", "cost_estimate", "measured", "seconds"):
        if k in st:
            lines.append(f"- **{k}**: {json.dumps(st[k], default=str)}")
    lines += ["", "## Selection (cheapest cell whose mean |dGamma_out| over the top r_ref directions is below the tolerance "
              "against every larger cell; r_ref = r* of the largest cell)", "",
              f"- global (every block): {json.dumps(summary['selection_global'])}", "",
              "| target block | source block | r_ref | cheapest cell | r* at the largest cell | q |", "|---|---|---|---|---|---|"]
    for blk in summary["blocks"]:
        sel = blk["selection"]
        big = blk["cells"][-1]
        label = "N={} k={}".format(sel["cell"]["N"], sel["cell"]["k"]) if sel.get("cell") else sel.get("reason")
        lines.append(f"| {blk['target_block']} | {blk['source_block']} | {_f(sel.get('r_ref'))} | {label} | "
                     f"{_f(big.get('r_star'))} | {_f(big.get('q'))} |")
    prof = summary["r_star_profile_largest_cell"]
    lines += ["", f"## r* by depth at the largest cell {json.dumps(prof['cell'])}", "",
              "| target block | source block | status | r* (prefix) | r* contig | r* excess | q | rank_m | d_m + d_k |",
              "|---|---|---|---|---|---|---|---|---|"]
    for e in prof["blocks"]:
        lines.append(f"| {e['target_block']} | {e['source_block']} | {e['status']} | {_f(e.get('r_star'))} | "
                     f"{_f(e.get('r_star_contig'))} | {_f(e.get('r_star_excess'), 1)} | {_f(e.get('q'))} | "
                     f"{_f(e.get('rank_m'))} | {_f(e.get('d_sum'))} |")
    lines += ["", "## Realised rows per cell (block 0)", "",
              "| N | k | T nominal | T realised (units) | rows kept | mean kept units / doc |", "|---|---|---|---|---|---|"]
    for c in summary["blocks"][0]["cells"]:
        lines.append(f"| {c['N']} | {c['k']} | {c['T_nominal']} | {_f(c.get('T_realised'))} | {c['rows']} | "
                     f"{_f(c.get('mean_units_per_doc'), 1)} |")
    lines += ["", "## Per block and cell", "",
              "| block | N | k | status | rows | Gamma_in top1 | Gamma_out top1 | thr (pooled) | r* prefix | r* excess | "
              "r* contig | n_above | q | rho_icc (top mean) | rho_icc (trace) | n_eff (trace, real) |", "|" + "---|" * 16]
    for blk in summary["blocks"]:
        for c in blk["cells"]:
            lines.append(
                f"| {blk['target_block']} | {c['N']} | {c['k']} | {c['status']} | {c['rows']} | {_f(c.get('gamma_in_top1'))} | "
                f"{_f(c.get('gamma_out_top1'))} | {_f(c.get('threshold'))} | {_f(c.get('r_star_prefix'))} | "
                f"{_f(c.get('r_star_excess'), 1)} | {_f(c.get('r_star_contig'))} | {_f(c.get('n_above'))} | {_f(c.get('q'))} | "
                f"{_f(c.get('rho_icc_top_mean'))} | {_f(c.get('rho_icc_trace'))} | {_f(c.get('n_eff_trace_real'), 1)} |")
    tasks = list(st["tasks"])
    lines += ["", "## Task coverage at r* (fraction of the energy of E; isotropic reference r*/rank)", "",
              "| block | N | k | r* | iso | " + " | ".join(f"{t} total / span" for t in tasks) + " |",
              "|" + "---|" * (5 + len(tasks))]
    for blk in summary["blocks"]:
        for c in blk["cells"]:
            if c["status"] == STATUS_DEGENERATE or not c.get("coverage"):
                continue
            cv = [f"{_f(c['coverage'][t]['cov_total'])} / {_f(c['coverage'][t]['cov_span'])}" for t in tasks]
            lines.append(f"| {blk['target_block']} | {c['N']} | {c['k']} | {c.get('r_star')} | "
                         f"{_f(c.get('coverage_isotropic_reference'))} | " + " | ".join(cv) + " |")
    lines += ["", "Degenerate cells (rows of a half <= d_m + d_k) report no Gamma. `borderline`: rows of a half < 2 (d_m + d_k)."]
    return "\n".join(lines) + "\n"


def write_outputs(summary: Mapping[str, Any], tensors: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = _clean(dict(summary))
    (out_dir / "summary.json").write_text(json.dumps(clean, indent=1, default=str))
    rows = csv_rows_llm(summary)
    cols = list(CSV_COLUMNS) + sorted({k for r in rows for k in r if k not in CSV_COLUMNS})
    with open(out_dir / "blocks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "summary.md").write_text(render_md_llm(clean))
    torch.save(dict(tensors), out_dir / "tensors.pt")


# ----------------------------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------------------------


def _build(spec: LlmModelSpec, cfg: LlmPhase1Config):
    from ...models.text_lm import TextBuildConfig, TextLM
    from ...rebase.model_families.registry import infer_family

    lm = TextLM.build(TextBuildConfig(spec.model, spec.arch, cfg.device, cfg.dtype))
    fam = infer_family(lm.model)
    if fam is None:
        raise ValueError(f"no family adapter for {spec.model}")
    return lm.model, lm.tokenizer, fam


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    from ...rebase.discrete_layer_match import DiscreteLayerPairing
    from ...run_logging import _code_fingerprint

    cfg = parse_config(payload)
    if cfg.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    out_dir = Path(cfg.out_dir)
    if (out_dir / "summary.json").exists() and not cfg.overwrite:
        raise FileExistsError(f"{out_dir / 'summary.json'} exists (set overwrite to replace)")
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    ks = resolve_ks(cfg.ks, cfg.max_length)
    grid = v1.Grid(cfg.n_images, ks)
    nulls = make_nulls(cfg.nulls)

    reg_texts, pool_texts = pool_documents(cfg)
    model_m, tok_m, fam_m = _build(cfg.source, cfg)
    model_k, tok_k, fam_k = _build(cfg.target, cfg)
    depth_m, depth_k = fam_m.block_count(model_m), fam_k.block_count(model_k)
    pairing = DiscreteLayerPairing.compute(depth_m, depth_k)
    blocks = list(cfg.blocks) if cfg.blocks is not None else list(range(depth_k))
    src_blocks = sorted({int(pairing.pairing[j]) for j in blocks})
    k_filt = float(cfg.anchor_filter_k) if cfg.anchor_filter == "norm" else None

    reg_src = TextPairSource(reg_texts, tok_m, tok_k, fam_k, cfg.rows, cfg.batch_size, cfg.max_length)
    reg_bm, reg_bk = reg_src.own()
    reg_m = registration_stats(model_m, fam_m, reg_bm, src_blocks, cfg.device, k_filt)
    reg_k = registration_stats(model_k, fam_k, reg_bk, blocks, cfg.device, k_filt)
    filters = block_filters(reg_m, reg_k, pairing.pairing, blocks, k_filt, cfg.acc_device)
    del reg_bm, reg_bk

    pool_src = TextPairSource(pool_texts, tok_m, tok_k, fam_k, cfg.rows, cfg.batch_size, cfg.max_length)
    pairs = pool_src.paired()
    counts_all = torch.cat([p.counts for p in pairs])
    unit_stats = {k: sum(p.stats[k] for p in pairs) for k in ("source", "target", "common")}
    if int(counts_all.max()) > grid.n_tokens:
        raise ValueError(f"a document has {int(counts_all.max())} units > the last k {grid.n_tokens}")
    d_m, d_k = int(filters[blocks[0]].mu_m.numel()), int(filters[blocks[0]].mu_k.numel())
    params = lambda m: float(sum(p.numel() for p in m.parameters()))  # noqa: E731
    cost = estimate_cost_llm(d_m, d_k, grid, cfg.n_null, len(nulls), len(blocks),
                             mean_units=float(counts_all.float().mean()), params_m=params(model_m),
                             params_k=params(model_k), ram_budget_gb=cfg.ram_budget_gb, blocks_per_pass=cfg.blocks_per_pass)
    print(f"[metodoK_phase1_llm] cost estimate: {json.dumps(cost)}", flush=True)

    grams, task_meta = ifeval_gram_pass(cfg, model_m, tok_m, fam_m, src_blocks)
    halves = v1.half_labels(grid, cfg.seed)
    opts = LlmAnalysisOptions(n_null=cfg.n_null, null_quantile=cfg.null_quantile, prefix_frac=cfg.prefix_frac,
                              icc_top=cfg.icc_top, linalg_device=cfg.linalg_device)
    block_res: dict[int, dict[str, Any]] = {}
    tensors: dict[str, Any] = {}
    bpp = cost["blocks_per_pass"]
    analysis_seconds = 0.0
    for s0 in range(0, len(blocks), bpp):
        group = blocks[s0: s0 + bpp]
        accs = forward_pass(pairs, model_m, fam_m, model_k, fam_k, filters, group, grid, halves, cfg, nulls, cfg.pair)
        t0 = time.perf_counter()
        for j in group:
            i = filters[j].i
            cells = []
            for ia, il in grid.cells():
                sums, nul, units = accs[j].cell(ia, il)
                info, t = analyze_cell_llm(sums, nul, opts, {"ifeval": grams[i]}, units)
                if il == len(grid.ks) - 1:
                    info["k_label"] = ALL
                if t:
                    info["gamma_out_vec"] = t["gamma_out"]
                    tensors[f"b{j}_N{info['N']}_k{info['k']}"] = t
                cells.append(info)
            sel = (select_cell_abs(cells, tol_g=cfg.tol_g) if cfg.n_null
                   else {"cell": None, "reason": "n_null = 0: no r*"})
            block_res[j] = {"target_block": j, "source_block": i, "cells": cells, "selection": sel}
            print(f"[metodoK_phase1_llm] block {j}: " + ", ".join(
                f"(N={c['N']},k={c['k']}) {c['status']}" + (f" r*={c['r_star']}" if "r_star" in c else "") for c in cells),
                flush=True)
        analysis_seconds += time.perf_counter() - t0
        del accs
    sel_global = (select_global_abs([block_res[j]["cells"] for j in blocks], tol_g=cfg.tol_g) if cfg.n_null
                  else {"cell": None, "reason": "n_null = 0"})
    big = grid.cells()[-1]
    profile = [{"target_block": j, "source_block": block_res[j]["source_block"], "d_sum": d_m + d_k, **{
        k: block_res[j]["cells"][-1].get(k)
        for k in ("status", "r_star", "r_star_contig", "r_star_excess", "n_above", "q", "rank_m", "rank_k")}}
        for j in blocks]
    for blk in block_res.values():
        for c in blk["cells"]:
            c.pop("gamma_out_vec", None)
    root = Path(__file__).resolve().parents[2]
    code = {**_code_fingerprint(), "module_sha256": _sha256(Path(__file__)),
            "metodoK_phase1_sha256": _sha256(Path(v1.__file__)),
            "canonical_stats_sha256": _sha256(root / "rebase/methods/anchors/canonical_stats.py"),
            "core_sha256": _sha256(root / "rebase/methods/anchors/core.py"),
            "text_units_sha256": _sha256(root / "rebase/methods/anchors/text_units.py")}
    t_real = {f"N={n}": {f"k={k}": block_res[blocks[0]]["cells"][a * len(ks) + level].get("T_realised")
                         for level, k in enumerate(ks)} for a, n in enumerate(grid.n_images)}
    setting = {
        "pair": cfg.pair, "rows": cfg.rows,
        "source": {"model": cfg.source.model, "expert_ifeval": cfg.source.instruct, "d": d_m, "depth": depth_m},
        "target": {"model": cfg.target.model, "d": d_k, "depth": depth_k},
        "pool_registration": {
            "dataset": f"Salesforce/wikitext wikitext-103-raw-v1 paragraphs >= {cfg.pool_min_chars} chars",
            "pool": {"split": cfg.pool_split, "n_docs": max(cfg.n_images), "seed": cfg.pool_seed, "order_seed": cfg.seed,
                     "texts_sha256": _texts_sha(pool_texts)},
            "registration": {"split": cfg.registration_split, "n_docs": cfg.registration_docs,
                             "seed": cfg.registration_seed, "texts_sha256": _texts_sha(reg_texts)},
            "disjoint": "checked by text; different WikiText-103 splits (disjoint articles)",
            "max_length_tokens": cfg.max_length, "batch_size": cfg.batch_size},
        "units": {"unit": "last token of each word present in both tokenizations" if cfg.rows == "word"
                  else "content token (first token of each document excluded)",
                  "pool_unit_counts": unit_stats, "units_per_doc": {
                      "min": int(counts_all.min()), "mean": float(counts_all.float().mean()), "max": int(counts_all.max())},
                  "realised_T_by_cell": t_real},
        "grid": {"N": list(grid.n_images), "k": list(grid.ks), "k_last": "all units of a document (max_length bound)",
                 "T_nominal": {f"N={n}": {f"k={k}": n * k for k in grid.ks} for n in grid.n_images}},
        "frame": {"block_pairing": "DiscreteLayerPairing", "pairing": list(pairing.pairing),
                  "capture": "decoder layer output (boundary)", "registration_rows": {
                      "source": {i: reg_m[i]["rows"] for i in src_blocks[:1]}, "target": {j: reg_k[j]["rows"] for j in blocks[:1]}}},
        "filter": {"anchor_filter": cfg.anchor_filter, "k": cfg.anchor_filter_k},
        "seeds": {"seed": cfg.seed, "pool_seed": cfg.pool_seed},
        "halves": "split_ids by document position inside each document stratum; identical across cells",
        "null": {"permutations": cfg.n_null, "quantile": cfg.null_quantile, "interpolation": "higher",
                 "schemes": list(cfg.nulls), "primary": cfg.nulls[0]},
        "nulls": {"N1": "derangement of documents in (batch, half) groups + independent random permutation of partner units per (replicate, document)",
                  "N2": "derangement, same slot (old vision null; control only)"},
        "rstar_rule": {"threshold": "pooled null_quantile of all null Gamma_out values", "r_star": "r_star_prefix",
                       "prefix_frac": cfg.prefix_frac, "excess": "n_above - 0.05 (q - n_above)", "contig": "kept"},
        "selection_rule": {"tol_gamma_abs": cfg.tol_g, "r_ref": "r_star of the largest cell of the block"},
        "icc": "one-way unbalanced ANOVA (Searle), see metodoK_phase1; n_eff_*_real uses sum n_i^2 / sum n_i",
        "coverage": {"tasks": {"ifeval": task_meta}, "directions": "V_m Y_r* of the in-sample fit of the cell"},
        "tasks": list(cfg.tasks), "blocks_run": blocks, "largest_cell": {"N": grid.n_images[big[0]], "k": grid.ks[big[1]]},
        "code": code, "cost_estimate": cost, "config": dict(payload), "seconds": time.perf_counter() - started,
        "measured": {"peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                     "analysis_seconds": analysis_seconds},
    }
    summary = {"setting": setting, "blocks": [block_res[j] for j in blocks], "selection_global": sel_global,
               "r_star_profile_largest_cell": {"cell": {"N": grid.n_images[big[0]], "k": grid.ks[big[1]]}, "blocks": profile}}
    write_outputs(summary, tensors, out_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", help="overrides out_dir of the config")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.config).read_text())
    if args.out_dir:
        payload["out_dir"] = args.out_dir
    run(payload)


if __name__ == "__main__":
    main()
