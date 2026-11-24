"""Metodo K, Phase 1 (forward only): how many anchor rows (N images x k tokens) give reliable canonical correlations?

Question. For a (source, target) model pair and every target block ``j`` (source block ``i = pairing[j]``,
``DiscreteLayerPairing``, as ``update_kernel_cka``), with anchor rows ``P_m = (S_m - mu_m) / c_m``, ``P_k`` (registration
``mu``, ``c`` and the ``anchor_filter='norm'`` row filter of the anchors state, exactly the frame of the decoder) the
canonical correlations ``Gamma`` are the singular values of ``M = U_m^T U_k``. With few effective rows the in-sample
``Gamma`` is inflated. This module measures, per block and per grid cell ``(N, k)``:

* ``Gamma`` in sample (all rows of the cell) and ``Gamma_out`` (directions from half A, correlation on half B, and the
  swap; reported value = mean of the two, ``core.canonical_correlations_out``);
* a permutation null of ``Gamma_out`` per rank index; ``r*`` = leading directions (contiguous from the top) above it;
* ``rho_icc`` (intra-image correlation) and ``n_eff = N k / (1 + (k - 1) rho_icc)``;
* task coverage of the source experts' update ``E = dH_m`` in the ``r*`` directions;
* cell status ``ok`` / ``borderline`` / ``degenerate`` (a degenerate cell reports no ``Gamma`` at all).

Run: ``python -m merge_and_rebase.eval.diagnostics.metodoK_phase1 --config configs/examples/metodoK_phase1/A_B.json``

Design
------
**One data pass per block group, nested cells from strata.** The pool images are the Tiny-ImageNet ``valid`` images that
are NOT in the registration set of the state (``frame.json`` pool indices after the anchor batches), in the order of one
seeded permutation (``seed``); image position ``p`` is the id. Cell ``(N, k)`` uses the images ``p < N`` and, per image,
the tokens of rank ``< k`` in that image's own seeded permutation of the ``n_tok`` canonical positions (``rank 0`` = a
random token, not always CLS; nested: ``k = 1 < 16 < 197``; the permutation of image ``p`` depends on ``(seed, p)`` only,
never on the batch). Statistics are accumulated per STRATUM = (image stratum ``a``: ``N_{a-1} <= p < N_a``) x (token
level ``l``: ``k_{l-1} <= rank < k_l``) x (half); a cell is the sum of the strata ``a <= a_cell``, ``l <= l_cell``, so
one forward pass of the largest cell yields all cells. Halves: ``canonical_stats.split_ids`` applied to the image
positions of every image stratum separately (never to tokens): balanced in every stratum and identical across cells.
Rows flagged high-norm (``anchor_filter``, in the source OR the target block, the rule of ``anchors.method.anchor_keep``
applied per row) are dropped; the number of kept rows of each cell is reported (``T = N k`` is nominal).

**Null (permuted images), pluggable.** A null plan (``NullRandomPosition`` / ``NullPositionKept``, any callable with the
same signature, see "NULL PLAN" above ``NullPositionKept``) decides which target rows replace the true ones in the null
``Sigma_mk``; ``Sigma_mm``, ``Sigma_kk`` are the true ones. PRIMARY (N1): image derangement ``pi_r`` AND an independent
random position of the partner image, ``pk_null[i, t] = pk[pi_r(i), sigma_{r,i}(t)]`` with ``sigma_{r,i}`` seeded by
``(seed, r, image position)``: the shared per-position mean structure is destroyed too, so r* counts every real
direction (positional or content). The old position-kept null is only valid on rows centred per position and is kept as
the SECONDARY null ``r*_content`` (beyond position): true and null rows are centred by the per-position means of the
registration images (one extra forward pass over them). ``pi_r`` acts within groups of images that are in the same batch,
stratum and half (a global permutation would need every pool image resident), independent per half. Rows whose
partner target row is high-norm are skipped, so the null ``Sigma_mk`` of a half is rescaled by ``n_rows / n_null_rows``
(min / max of ``n_null / n`` per cell are reported). ``Gamma_out`` under the null: directions from the null ``Sigma_mk`` of
half A (true half-A whitening), correlation on the null ``Sigma_mk`` of half B, the swap, mean.

**r*.** One pooled threshold: the 95% quantile (``interpolation='higher'``) of all ``R x q`` null values over the rank
indices (the null ``Gamma_out`` is out of sample and signed, hence exchangeable over the indices; the per-index spread is
reported). ``r*_prefix`` (primary, handed to Phase 2 and used for the coverage; ``info['r_star']``): the largest ``r``
such that >= 95% of the indices ``< r`` are above the threshold. ``r*_excess = n_above - 0.05 (q - n_above)``;
``r*_contig`` (contiguous run from the top) is a transparency column. The same three for ``r*_content``.

**Selection.** ``r_ref`` = ``r*_prefix`` at the largest cell; the cheapest cell after which the mean ``|dGamma_out|`` over the
top ``r_ref`` directions is < 0.02 ABSOLUTE in every larger cell (independent of the power of the null test); ``r*_prefix``
per cell is a profile. ``--compare-seeds`` gives the split-to-split ``|dGamma_out|``, the noise floor of that 0.02.

**rho_icc.** One-way ANOVA with unequal group sizes (images = groups, tokens = observations) for a scalar variate ``x``
(image ``i`` has ``n_i`` kept rows, ``N`` images, ``n = sum n_i``)::

    SSB = sum_i S_i^2 / n_i - (sum_i S_i)^2 / n,      S_i = sum of x over the rows of image i
    SST = sum x^2 - (sum x)^2 / n,                    SSW = SST - SSB
    MSB = SSB / (N - 1),  MSW = SSW / (n - N),  n0 = (n - sum n_i^2 / n) / (N - 1)
    rho = (MSB - MSW) / (MSB + (n0 - 1) MSW)          (Searle; equals the balanced ICC(1) when all n_i = k)

For a direction ``w`` (the canonical variate of ``P_m``, ``w_i = V_m S_m^-1 y_i``, in-sample fit of the cell): ``sum
S_i^2 / n_i = w^T B w`` with ``B = sum_i s_i s_i^T / n_i`` and ``s_i`` the sum of the rows of image i; ``sum x^2 = w^T
Sigma_mm w``, ``sum x = w^T s``. Only ``B`` (``d_m^2``) and ``s`` are accumulated per (image stratum, cumulative token
level) because ``B`` is not additive over token levels of the same image. Trace version: the same formulas with
``SSB = tr B - |s|^2 / n``, ``SST = tr Sigma_mm - |s|^2 / n`` (variance-weighted average over the coordinates of
``P_m``). ``rho`` is NaN for ``k = 1`` (no within-image variation). ``n_eff`` uses ``max(rho, 0)`` and the NOMINAL ``k``.

**Task coverage.** For each source expert, ``G_E = sum_rows dH^T dH`` (``dH = H_ft - H_base`` at source block ``i``, task
TRAIN images, seeded random subset in random order, all canonical tokens). With ``W = V_m Y_{r*}`` (orthonormal columns;
``P_m = U_m S_m V_m^T``, ``M = Y Gamma Z^T``): coverage = ``tr(W^T G_E W) / tr G_E`` and, relative to the span of ``P_m``,
``tr(W^T G_E W) / tr(V_m^T G_E V_m)``. Isotropic reference: ``r* / rank(P_m)``.

**Memory / time** (``estimate_cost``; float64 accumulators on ``acc_device``): per block ``8 B * (S (d_m^2 + d_k^2 +
d_m d_k) + R S d_m d_k + n_l n_a (d_m^2 + d_m))`` with ``S = n_a n_l 2`` strata; the blocks are processed in groups
(``blocks_per_pass`` or automatically from ``ram_budget_gb``), one forward pass of the largest cell per group.
A -> B (768 x 1024, 3 x 3 strata, R = 19): 2.54 GB per block; with ``ram_budget_gb = 24``: 8 blocks per pass, 3 passes,
peak accumulators 20.8 GB. A -> C (768 x 768): 1.91 GB per block, 11 blocks per pass, 2 passes, 21.3 GB. Flops of the
accumulation at the full grid: 0.67 PFLOP (A -> B) and 0.25 PFLOP (A -> C), 93 % of it the null; in float64 on a
consumer GPU (~0.6 TFLOP/s) about 19 min and 7 min, on 4 CPU threads hours. The forward passes are cheap (a few minutes
per pass, loader bound). ``acc_device`` puts the accumulators on the GPU (then they count as GPU memory, not RSS).

LLM pairs are not implemented: see ``VisionPairSource`` for the seam (pool, forward capture, rows per image).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from ...rebase.methods.anchors.canonical_stats import split_ids
from ...rebase.methods.anchors.core import (
    CanonicalFactors,
    _svd_canonical,
    canonical_correlations,
    canonical_correlations_out,
    canonical_factors_from_sigma,
)
from .update_kernel_cka import ModelSpec, image_derangement

Tensor = torch.Tensor

STATUS_OK, STATUS_BORDERLINE, STATUS_DEGENERATE = "ok", "borderline", "degenerate"


# ----------------------------------------------------------------------------------------------------------------
# grid, strata, nested token subsets, halves, null partners (pure, no model)
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """Cells ``(N, k)``: ``n_images`` and ``ks`` ascending; the last ``k`` is the number of tokens of an image."""

    n_images: tuple[int, ...]
    ks: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, v in (("n_images", self.n_images), ("ks", self.ks)):
            if not v or any(int(x) < 1 for x in v) or list(v) != sorted(set(v)):
                raise ValueError(f"{name} must be positive, strictly increasing, got {v}")

    @property
    def n_max(self) -> int:
        return int(self.n_images[-1])

    @property
    def n_tokens(self) -> int:
        return int(self.ks[-1])

    def cells(self) -> list[tuple[int, int]]:
        """``(a, l)`` index pairs, row-major (images outer)."""
        return [(a, level) for a in range(len(self.n_images)) for level in range(len(self.ks))]


def image_stratum(positions: Tensor, grid: Grid) -> Tensor:
    """Image stratum of pool positions: ``N_{a-1} <= p < N_a``."""
    return torch.bucketize(positions, torch.as_tensor(grid.n_images, dtype=torch.long), right=True)


def half_labels(grid: Grid, seed: int) -> Tensor:
    """``[N_max]`` half (0 / 1) of every pool position: ``split_ids`` inside each image stratum (balanced in every
    stratum, hence nested across cells; by image, never by token)."""
    labels = torch.zeros(grid.n_max, dtype=torch.long)
    lo = 0
    for a, hi in enumerate(grid.n_images):
        if hi - lo < 2:
            raise ValueError("every image stratum needs at least 2 images to be split in halves")
        _, h1 = split_ids(torch.arange(lo, hi), int(seed) * 7919 + a)
        labels[h1] = 1
        lo = hi
    return labels


def token_ranks(positions: Tensor, n_tokens: int, seed: int) -> Tensor:
    """``[len(positions), n_tokens]`` rank of every token position in its image's seeded permutation (rank 0 = the
    first token drawn). Depends on ``(seed, image position)`` only, never on the batch."""
    out = torch.empty(len(positions), int(n_tokens), dtype=torch.long)
    for n, p in enumerate(positions.tolist()):
        perm = torch.randperm(int(n_tokens), generator=torch.Generator().manual_seed(int(seed) * 1_000_003 + int(p) + 7919))
        out[n, perm] = torch.arange(int(n_tokens))
    return out


def token_levels(ranks: Tensor, ks: Sequence[int]) -> Tensor:
    """Token level ``l`` with ``k_{l-1} <= rank < k_l`` (``k_{-1} = 0``): level-<=l tokens are the nested subset ``k_l``."""
    return torch.bucketize(ranks, torch.as_tensor(list(ks), dtype=torch.long), right=True)


def null_partners(half: Tensor, n_perm: int, seed: int, batch_index: int) -> Tensor:
    """``[R, B]`` partner image (index in the batch) of every image under each of ``R`` permutations: a derangement
    inside each half group of the batch, drawn independently per half and permutation; ``-1`` if the group of the image
    has a single member (no partner: its rows are skipped)."""
    b = int(half.shape[0])
    out = torch.full((int(n_perm), b), -1, dtype=torch.long)
    for r in range(int(n_perm)):
        gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + r * 10_007 + int(batch_index) + 104_729)
        for h in (0, 1):
            idx = (half == h).nonzero().flatten()
            if idx.numel() < 2:
                continue
            out[r, idx] = idx[image_derangement(int(idx.numel()), gen)]
    return out


def position_permutation(r: int, image_position: int, n_tokens: int, seed: int) -> Tensor:
    """Permutation ``sigma_{r, i}`` of the ``n_tokens`` token positions of one image, seeded by ``(seed, r, image
    position)`` only (independent per replicate and per image, never dependent on the batch)."""
    gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(r) * 10_007 + int(image_position) * 97 + 31_337)
    return torch.randperm(int(n_tokens), generator=gen)


# A NULL PLAN is the pluggable part of the null (the LLM module supplies its own): a callable object with ``n_perm``
# and ``__call__(pk [B, T, d_k], valid [B, T], ih [nh], r) -> (pk_null [nh, T, d_k], valid_null [nh, T])``. For the
# images ``ih`` of one half group of the batch and replicate ``r`` it returns the target rows that replace the true
# ones in the null pairing (source row ``(i, t)`` is paired with ``pk_null[i, t]``) and whether they are kept rows.
# ``StratifiedSigma.update(..., null=plan)``; a plain ``[R, B]`` partner tensor is accepted (position-kept null).


class NullPositionKept:
    """SECONDARY null (``r*_content``, beyond position): image derangement ``pi_r``, token position kept:
    ``pk_null[i, t] = pk[pi(i), t]``. Valid only on rows centred per position (``centre_by_position``): otherwise
    the shared per-position mean term survives the permutation and the test is "beyond position"."""

    def __init__(self, half: Tensor, n_perm: int, seed: int, batch_index: int) -> None:
        self.n_perm = int(n_perm)
        self.partners = null_partners(half, n_perm, seed, batch_index)

    def __call__(self, pk: Tensor, valid: Tensor, ih: Tensor, r: int) -> tuple[Tensor, Tensor]:
        part = self.partners[r].to(pk.device)[ih]
        ok = part >= 0
        pc = part.clamp(min=0)
        return pk[pc], valid[pc] & ok[:, None]


class NullRandomPosition:
    """PRIMARY null (N1): image derangement ``pi_r`` AND an independent random position of the partner image for
    every source row: ``pk_null[i, t] = pk[pi(i), sigma_{r,i}(t)]`` with ``sigma_{r,i}`` a permutation of the token
    positions seeded by ``(seed, r, image position)``. Destroys the shared per-position (positional) structure as
    well as the content pairing, so r* counts every real direction, positional or not. ``positions`` = pool position
    of each image of the batch."""

    def __init__(self, half: Tensor, positions: Tensor, n_tokens: int, n_perm: int, seed: int, batch_index: int) -> None:
        self.n_perm = int(n_perm)
        self.partners = null_partners(half, n_perm, seed, batch_index)
        self.sigma = torch.stack([
            torch.stack([position_permutation(r, int(p), n_tokens, seed) for p in positions.tolist()])
            for r in range(self.n_perm)
        ]) if self.n_perm else torch.zeros(0, int(half.shape[0]), n_tokens, dtype=torch.long)

    def __call__(self, pk: Tensor, valid: Tensor, ih: Tensor, r: int) -> tuple[Tensor, Tensor]:
        part = self.partners[r].to(pk.device)[ih]
        ok = part >= 0
        pc = part.clamp(min=0)
        sig = self.sigma[r].to(pk.device)[ih]  # [nh, T]
        return pk[pc[:, None], sig], valid[pc[:, None], sig] & ok[:, None]


def centre_by_position(rows: Tensor, position_mean: Tensor) -> Tensor:
    """Rows ``[B, T, d]`` minus the per-position mean ``[T, d]`` (registration images)."""
    return rows - position_mean.to(rows.device, rows.dtype)[None]


# ----------------------------------------------------------------------------------------------------------------
# streaming statistics per stratum
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class CellSums:
    """Statistics of one cell (sum of strata). Halves on dim 0; ``null_*`` are ``None`` without a null."""

    n_images: int
    k: int
    mm: Tensor  # [2, d_m, d_m]
    kk: Tensor  # [2, d_k, d_k]
    mk: Tensor  # [2, d_m, d_k]
    n: Tensor  # [2] rows per half
    null_mk: Tensor | None  # [R, 2, d_m, d_k]
    null_n: Tensor | None  # [R, 2]
    icc_b: Tensor  # [d_m, d_m]  sum_i s_i s_i^T / n_i  (empty without ICC)
    icc_s: Tensor  # [d_m]       sum of all rows
    icc_stats: Tensor  # [3]      N images with rows, n rows, sum n_i^2

    def to(self, device: str | torch.device) -> CellSums:
        kw = {f.name: getattr(self, f.name) for f in fields(self)}
        return CellSums(**{k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in kw.items()})


class StratifiedSigma:
    """``Sigma_mm``, ``Sigma_kk``, ``Sigma_mk`` (and null ``Sigma_mk``, ICC sums) of one block pair per stratum
    (image stratum ``a`` x token level ``l`` x half), float64 on ``device`` (storage). ``update`` takes rows already
    centred and scaled (``P = (S - mu) / c``, the frame of ``canonical_stats``), so the sums are those of
    ``PairSigmaAccumulator`` (tested). Sigma are NOT divided by the number of rows.

    ``compute_device`` / ``compute_dtype`` (default: storage ones) choose where and in which precision the per-batch
    products are formed; they are added to the float64 storage, so ``compute_dtype=float32`` on a GPU with CPU storage
    keeps the accumulation exact (rows are pre-centred; precision of the products ~1e-7 relative per batch).
    ``icc=False`` skips the ICC sums (secondary accumulators)."""

    def __init__(
        self, d_m: int, d_k: int, grid: Grid, n_perm: int, *, device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64, compute_device: str | torch.device | None = None,
        compute_dtype: torch.dtype | None = None, icc: bool = True,
    ) -> None:
        self.d_m, self.d_k, self.grid, self.n_perm = int(d_m), int(d_k), grid, int(n_perm)
        self.device, self.dtype = torch.device(device), dtype
        self.cdev = torch.device(compute_device) if compute_device is not None else self.device
        self.cdtype = compute_dtype or dtype
        self.icc = bool(icc)
        n_a, n_l = len(grid.n_images), len(grid.ks)
        z = lambda *s: torch.zeros(*s, dtype=dtype, device=self.device)  # noqa: E731
        self.mm, self.kk, self.mk = z(n_a, n_l, 2, d_m, d_m), z(n_a, n_l, 2, d_k, d_k), z(n_a, n_l, 2, d_m, d_k)
        self.n = torch.zeros(n_a, n_l, 2, dtype=torch.long)
        self.null_mk = z(self.n_perm, n_a, n_l, 2, d_m, d_k) if self.n_perm else None
        self.null_n = torch.zeros(self.n_perm, n_a, n_l, 2, dtype=torch.long)
        di = d_m if self.icc else 0
        self.icc_b, self.icc_s = z(n_a, n_l, di, di), z(n_a, n_l, di)
        self.icc_stats = torch.zeros(n_a, n_l, 3, dtype=torch.float64)

    def _add(self, store: Tensor, idx: tuple, x: Tensor, y: Tensor) -> None:
        store[idx] += (x.T @ y).to(self.device, self.dtype)

    @torch.no_grad()
    def update(
        self, a: int, pm: Tensor, pk: Tensor, valid: Tensor, level: Tensor, half: Tensor, null=None
    ) -> None:
        """One batch of images of image stratum ``a``: ``pm [B, T, d_m]``, ``pk [B, T, d_k]`` (the same rows in the two
        models), ``valid [B, T]`` bool (kept rows), ``level [B, T]`` token levels, ``half [B]``, ``null`` a null plan
        (see above) or a ``[R, B]`` partner tensor (position-kept null); required when ``n_perm > 0``."""
        if self.n_perm and null is None:
            raise ValueError("n_perm > 0 needs a null plan")
        if isinstance(null, Tensor):
            null = _LegacyPartners(null)
        pm, pk = pm.to(self.cdev, self.cdtype), pk.to(self.cdev, self.cdtype)
        valid, level, half = valid.to(self.cdev), level.to(self.cdev), half.to(self.cdev)
        n_l = len(self.grid.ks)
        for h in (0, 1):
            ih = (half == h).nonzero().flatten()
            if ih.numel() == 0:
                continue
            pmh, pkh, vh, lh = pm[ih], pk[ih], valid[ih], level[ih]
            pk_part = [null(pk, valid, ih, r) for r in range(self.n_perm)]
            for lv in range(n_l):
                m = vh & (lh == lv)
                x, y = pmh[m], pkh[m]
                self._add(self.mm, (a, lv, h), x, x)
                self._add(self.kk, (a, lv, h), y, y)
                self._add(self.mk, (a, lv, h), x, y)
                self.n[a, lv, h] += int(x.shape[0])
                for r, (pkp, vp) in enumerate(pk_part):
                    mn = m & vp
                    self._add(self.null_mk, (r, a, lv, h), pmh[mn], pkp[mn])
                    self.null_n[r, a, lv, h] += int(mn.sum())
        if not self.icc:
            return
        # ICC sums at the cumulative level <= lv, all images of the batch (both halves)
        for lv in range(n_l):
            m = valid & (level <= lv)
            n_i = m.sum(1).to(self.cdtype)
            s = torch.einsum("bt,btd->bd", m.to(self.cdtype), pm)
            ok = n_i > 0
            sn = s[ok] / n_i[ok].sqrt()[:, None]
            self.icc_b[a, lv] += (sn.T @ sn).to(self.device, self.dtype)
            self.icc_s[a, lv] += s[ok].sum(0).to(self.device, self.dtype)
            self.icc_stats[a, lv] += torch.tensor(
                [float(ok.sum()), float(n_i.sum()), float((n_i[ok] ** 2).sum())], dtype=torch.float64
            )

    def cell(self, ia: int, il: int) -> CellSums:
        """Sum of the strata ``a <= ia``, ``l <= il`` (token level ``l <= il`` = the nested tokens ``k_il``)."""
        sl = (slice(0, ia + 1), slice(0, il + 1))
        return CellSums(
            n_images=int(self.grid.n_images[ia]), k=int(self.grid.ks[il]),
            mm=self.mm[sl].sum((0, 1)), kk=self.kk[sl].sum((0, 1)), mk=self.mk[sl].sum((0, 1)),
            n=self.n[sl].sum((0, 1)),
            null_mk=None if self.null_mk is None else self.null_mk[:, sl[0], sl[1]].sum((1, 2)),
            null_n=None if self.null_mk is None else self.null_n[:, sl[0], sl[1]].sum((1, 2)),
            icc_b=self.icc_b[:ia + 1, il].sum(0), icc_s=self.icc_s[:ia + 1, il].sum(0),
            icc_stats=self.icc_stats[:ia + 1, il].sum(0),
        )


class _LegacyPartners(NullPositionKept):
    def __init__(self, partners: Tensor) -> None:
        self.n_perm = int(partners.shape[0])
        self.partners = partners


# ----------------------------------------------------------------------------------------------------------------
# per-cell analysis
# ----------------------------------------------------------------------------------------------------------------


def classify_rows(rows_half_min: int, d_m: int, d_k: int) -> str:
    """``degenerate``: rows of a half <= d_m + d_k (Gamma saturated by geometry); ``borderline``: < 2 (d_m + d_k)."""
    d = int(d_m) + int(d_k)
    if int(rows_half_min) <= d:
        return STATUS_DEGENERATE
    if int(rows_half_min) < 2 * d:
        return STATUS_BORDERLINE
    return STATUS_OK


def icc_anova(ssb: Tensor, sst: Tensor, n_img: float, n_tot: float, sum_n2: float) -> Tensor:
    """Unbalanced one-way ANOVA ICC (formulas in the module docstring) from ``SSB`` and ``SST`` (any shape); NaN where
    a degree of freedom is missing (``k = 1``)."""
    df_b, df_w = n_img - 1.0, n_tot - n_img
    if df_b <= 0 or df_w <= 0:
        return torch.full_like(ssb, float("nan"))
    msb, msw = ssb / df_b, (sst - ssb) / df_w
    n0 = (n_tot - sum_n2 / n_tot) / df_b
    return (msb - msw) / (msb + (n0 - 1.0) * msw)


def icc_directions(sums: CellSums, w: Tensor) -> tuple[Tensor, Tensor]:
    """``(rho per column of w [d_m, q], rho of the trace)`` from the cell's ICC sums and ``Sigma_mm`` (both halves)."""
    n_img, n_tot, sum_n2 = (float(x) for x in sums.icc_stats)
    smm = sums.mm.sum(0)
    s = sums.icc_s
    ssb = (w * (sums.icc_b @ w)).sum(0) - (w.T @ s) ** 2 / n_tot
    sst = (w * (smm @ w)).sum(0) - (w.T @ s) ** 2 / n_tot
    rho = icc_anova(ssb, sst, n_img, n_tot, sum_n2)
    ssb_tr = torch.trace(sums.icc_b) - (s @ s) / n_tot
    sst_tr = torch.trace(smm) - (s @ s) / n_tot
    return rho, icc_anova(ssb_tr, sst_tr, n_img, n_tot, sum_n2)


def task_coverage(gram_e: Tensor, v_m: Tensor, y_r: Tensor) -> dict[str, float]:
    """Energy of ``E`` (``gram_e = E^T E``) in the directions ``W = V_m Y_r`` (orthonormal columns): ``cov_total`` =
    ``tr(W^T G W) / tr G``, ``cov_span`` = relative to ``V_m`` (``tr(V_m^T G V_m)``), ``span_fraction`` = the latter
    over ``tr G``."""
    g = gram_e.to(v_m.dtype)
    tr_g = float(torch.trace(g))
    span = float((v_m * (g @ v_m)).sum())
    num = 0.0
    if y_r.shape[1]:
        w = v_m @ y_r
        num = float((w * (g @ w)).sum())
    return {
        "cov_total": num / tr_g, "cov_span": num / span if span > 0 else float("nan"), "span_fraction": span / tr_g,
    }


def _null_gamma_out(
    fac: CanonicalFactors, null_a: Tensor, sig_b: tuple[Tensor, Tensor, Tensor]
) -> Tensor:
    """Null ``Gamma_out`` of ``R`` permutations for one direction of estimation (directions from half A's null
    ``Sigma_mk`` [R, d_m, d_k] in the true half-A whitening ``fac``; evaluated on half B's null ``Sigma_mk``)."""
    out = []
    smm_b, skk_b, mk_b = sig_b
    for r in range(null_a.shape[0]):
        m_null = (fac.v_m.T @ null_a[r] @ fac.v_k) / fac.s_m[:, None] / fac.s_k[None, :]
        f_null = CanonicalFactors(fac.v_m, fac.s_m, fac.v_k, fac.s_k, m_null, fac.n_rows)
        out.append(canonical_correlations_out(f_null, (smm_b, skk_b, mk_b[r])))
    return torch.stack(out)


@dataclass
class AnalysisOptions:
    n_null: int = 19
    null_quantile: float = 0.95
    quantile_interpolation: str = "higher"
    icc_top: int = 10
    linalg_device: str = "cpu"
    icc: bool = True  # False for the secondary (content) accumulators, which carry no ICC sums
    prefix_fraction: float = 0.95  # r*_prefix: the largest r with >= this fraction of the indices < r above threshold


def r_star_rules(gamma_out: Tensor, null_mean: Tensor, opts: AnalysisOptions) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """The three counts of directions above the null, from ``Gamma_out`` ``[q]`` and the null ``[R, q]``.

    ``info['threshold']`` is the pooled scalar (``null_quantile`` quantile of all ``R x q`` null values); the threshold
    actually applied is ``thr_i = max(pooled, quantile of the null values in the window of i)``, windows [0,10), [10,50),
    [50,q) (tensor ``threshold`` [q], ``info['threshold_windows']``): the leading null directions overfit more.
    The per-index quantiles are kept (``threshold_per_index``, min/max in ``info``).

    * ``r_star_prefix`` (primary): the largest ``r`` such that at least ``prefix_fraction`` of the indices ``< r`` are
      above the threshold (0 if none);
    * ``r_star_excess`` = ``n_above - (1 - quantile) (q - n_above)`` (expected-false-positive corrected count);
    * ``r_star_contig`` (transparency): the contiguous run from the top; ``n_above``: the total count."""
    q = int(gamma_out.numel())
    flat = null_mean.reshape(-1).to(torch.float64)
    quant = lambda x: torch.quantile(x, float(opts.null_quantile), interpolation=opts.quantile_interpolation)  # noqa: E731
    thr_pool = quant(flat) if flat.numel() <= 16_000_000 else torch.sort(flat).values[int(math.ceil(opts.null_quantile * (flat.numel() - 1)))]
    # the null is NOT homogeneous at the top indices (the leading null directions overfit more): per-window thresholds
    # (indices [0,10), [10,50), [50,q)), each at least the pooled one
    thr = torch.full((q,), float(thr_pool), dtype=torch.float64)
    windows = []
    for lo, hi in ((0, 10), (10, 50), (50, q)):
        hi = min(hi, q)
        if lo >= hi:
            continue
        w = float(max(thr_pool, quant(null_mean[:, lo:hi].reshape(-1).to(torch.float64))))
        thr[lo:hi] = w
        windows.append({"indices": [lo, hi], "threshold": w})
    per_index = torch.quantile(null_mean, float(opts.null_quantile), dim=0, interpolation=opts.quantile_interpolation)
    above = gamma_out > thr
    n_above = int(above.sum())
    run = torch.cumsum(above.to(torch.float64), 0) / torch.arange(1, q + 1, dtype=torch.float64)
    ok = (run >= float(opts.prefix_fraction)).nonzero().flatten()
    prefix = int(ok.max()) + 1 if ok.numel() else 0
    # trim the trailing failures the 95 % tolerance lets in: the prefix ends on an index above the threshold
    prefix = int(above[:prefix].nonzero().max()) + 1 if prefix and bool(above[:prefix].any()) else 0
    info = {
        "r_star_prefix": prefix, "r_star_excess": float(n_above - (1.0 - opts.null_quantile) * (q - n_above)),
        "r_star_contig": int(torch.cumprod(above.to(torch.long), 0).sum()), "n_above": n_above,
        "threshold": float(thr_pool), "threshold_windows": windows, "threshold_per_index_min": float(per_index.min()),
        "threshold_per_index_max": float(per_index.max()), "n_null": int(null_mean.shape[0]),
    }
    return info, {"null_gamma_out": null_mean.cpu(), "threshold": thr.cpu(), "threshold_per_index": per_index.cpu(),
                  "above": above.cpu()}


def analyze_cell(
    sums: CellSums, opts: AnalysisOptions, gram_e: Mapping[str, Tensor] | None = None
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Scalars (JSON) and tensors (``.pt``) of one cell. A degenerate cell returns the status and the row counts only.
    ``info['r_star']`` is ``r_star_prefix`` (the count handed to Phase 2; also used for the coverage)."""
    dev = opts.linalg_device
    s = sums.to(dev)
    d_m, d_k = int(s.mm.shape[-1]), int(s.kk.shape[-1])
    n_half = [int(x) for x in s.n]
    n_tot = sum(n_half)
    info: dict[str, Any] = {
        "N": sums.n_images, "k": sums.k, "T_nominal": sums.n_images * sums.k, "rows": n_tot, "rows_half": n_half,
        "d_m": d_m, "d_k": d_k,
        "status": classify_rows(min(n_half), d_m, d_k) if min(n_half) > 0 else STATUS_DEGENERATE,
    }
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
        if s.null_mk is not None:
            frac = s.null_n.to(f64) / s.n.to(f64)[None, :]  # [R, 2] fraction of rows kept in the null
            scale = 1.0 / frac.clamp(min=1e-12)
            nmk = s.null_mk.to(f64) * scale[:, :, None, None]
            nab = _null_gamma_out(fac[0], nmk[:, 0], (mm[1], kk[1], nmk[:, 1]))
            nba = _null_gamma_out(fac[1], nmk[:, 1], (mm[0], kk[0], nmk[:, 0]))
    except ValueError as exc:  # rank-deficient frame or half: never a result
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
    if s.null_mk is not None:
        null_mean = 0.5 * (nab[:, :q] + nba[:, :q])  # [R, q]
        rinfo, rt = r_star_rules(gamma_out, null_mean, opts)
        r_star = rinfo["r_star_prefix"]
        info.update(rinfo)
        info["r_star"] = r_star
        info["threshold_top1"] = rinfo["threshold"]  # kept name: the pooled threshold
        info["null_row_fraction_min"], info["null_row_fraction_max"] = float(frac.min()), float(frac.max())
        tensors.update(rt)
    y, _, _ = _svd_canonical(fac_all.m)
    if opts.icc:
        # ICC of the top canonical variates of P_m and of the trace; n_eff
        n_top = min(int(opts.icc_top), y.shape[1])
        w = fac_all.v_m @ (y[:, :n_top] / fac_all.s_m[:, None])
        rho_dir, rho_tr = icc_directions(s, w)
        rho_top = float(rho_dir.mean()) if not bool(torch.isnan(rho_dir).any()) else float("nan")
        rho_tr = float(rho_tr)

        def n_eff(rho: float) -> float:
            k = sums.k
            return float(sums.n_images * k / (1.0 + (k - 1) * max(rho, 0.0))) if not math.isnan(rho) else float(sums.n_images)

        tensors["rho_icc_top"] = rho_dir.cpu()
        info.update({
            "rho_icc_top1": float(rho_dir[0]), "rho_icc_top_mean": rho_top, "rho_icc_trace": rho_tr,
            "n_eff_top": n_eff(rho_top), "n_eff_trace": n_eff(rho_tr),
        })
    if gram_e:
        cov: dict[str, Any] = {}
        for task, g in gram_e.items():
            cov[task] = task_coverage(g.to(dev), fac_all.v_m, y[:, :r_star])
        info["coverage"] = cov
        info["coverage_isotropic_reference"] = r_star / int(fac_all.s_m.numel())
    return info, tensors


# ----------------------------------------------------------------------------------------------------------------
# selection helper
# ----------------------------------------------------------------------------------------------------------------


def _gamma_shift(c: Mapping[str, Any], o: Mapping[str, Any], r_ref: int) -> float:
    """Mean ``|Gamma_out,c - Gamma_out,o|`` over the top ``r_ref`` directions (ABSOLUTE; no dependence on the power of
    the null test)."""
    g_c, g_o = c["gamma_out_vec"], o["gamma_out_vec"]
    n = min(int(r_ref), len(g_c), len(g_o))
    return float((g_c[:n] - g_o[:n]).abs().mean())


def _largest(ok: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return max(ok, key=lambda c: (c["T_nominal"], c["N"]))


def select_cell(
    cells: Sequence[Mapping[str, Any]], *, tol_g: float = 0.02, tol_r: float | None = None
) -> dict[str, Any]:
    """Cheapest cell (nominal ``T = N k``, ties: fewer images) such that, in EVERY larger cell (larger ``T``), the mean
    ``|dGamma_out|`` over the top ``r_ref`` directions is ``< tol_g`` (absolute). ``r_ref`` = ``r_star`` (the prefix
    count) of the LARGEST non-degenerate cell: the selection never depends on the power of the null test at the cell
    being judged. ``r_star`` per cell is a profile (``r_star_prefix`` column), not a criterion (``tol_r`` is accepted
    and ignored, for callers of the earlier version). Degenerate cells are neither candidates nor comparators.
    Needs ``r_star`` and ``gamma_out_vec`` in every non-degenerate cell. Returns ``cell``, ``r_ref``, ``trivial`` (no
    larger valid cell: the largest cell itself) and ``compared`` (the larger cells judged for the chosen cell)."""
    ok = [c for c in cells if c["status"] != STATUS_DEGENERATE]
    if not ok:
        return {"cell": None, "reason": "every cell is degenerate"}
    big = _largest(ok)
    r_ref = int(big["r_star"])
    if r_ref == 0:
        return {"cell": None, "r_ref": 0, "reason": "r_ref = 0 at the largest cell: no direction above the null"}
    for c in sorted(ok, key=lambda c: (c["T_nominal"], c["N"])):
        larger = [o for o in ok if o["T_nominal"] > c["T_nominal"]]
        if all(_gamma_shift(c, o, r_ref) < tol_g for o in larger):
            return {"cell": {"N": c["N"], "k": c["k"], "T_nominal": c["T_nominal"]}, "r_ref": r_ref,
                    "trivial": not larger, "compared": [{"N": o["N"], "k": o["k"]} for o in larger], "tol_gamma": tol_g}
    return {"cell": None, "r_ref": r_ref, "reason": "no cell is stable against all larger cells"}


def select_global(
    per_block: Sequence[Sequence[Mapping[str, Any]]], *, tol_g: float = 0.02, tol_r: float | None = None
) -> dict[str, Any]:
    """Cheapest cell that is non-degenerate in EVERY block and stable (``select_cell`` rule, ``r_ref`` of each block)
    against every larger valid cell of each block. Blocks with ``r_ref = 0`` have nothing to stabilise: they impose no
    constraint and are listed in ``blocks_without_directions``. ``trivial`` / ``compared`` as in ``select_cell``
    (``compared``: the larger cells judged, per block)."""
    keys = {(c["N"], c["k"]) for c in per_block[0]} if per_block else set()
    r_refs, constrained = [], []
    for cells in per_block:
        okc = [c for c in cells if c["status"] != STATUS_DEGENERATE]
        r_refs.append(int(_largest(okc)["r_star"]) if okc else 0)
        constrained.append(bool(okc) and r_refs[-1] > 0)
    good = []
    for key in keys:
        valid = True
        for cells, r_ref, con in zip(per_block, r_refs, constrained, strict=True):
            by = {(c["N"], c["k"]): c for c in cells}
            c = by.get(key)
            if c is None or c["status"] == STATUS_DEGENERATE:
                valid = False
                break
            larger = [o for o in cells if o["status"] != STATUS_DEGENERATE and o["T_nominal"] > c["T_nominal"]]
            if con and not all(_gamma_shift(c, o, r_ref) < tol_g for o in larger):
                valid = False
                break
        if valid:
            good.append(key)
    none_blocks = [i for i, con in enumerate(constrained) if not con]
    if not good:
        return {"cell": None, "reason": "no cell is valid in every block", "blocks_without_directions": none_blocks}
    n, k = min(good, key=lambda nk: (nk[0] * nk[1], nk[0]))
    compared = [
        [{"N": o["N"], "k": o["k"]} for o in cells if o["status"] != STATUS_DEGENERATE and o["T_nominal"] > n * k]
        for cells in per_block
    ]
    return {"cell": {"N": n, "k": k, "T_nominal": n * k}, "r_ref_per_block": r_refs, "tol_gamma": tol_g,
            "trivial": not any(compared), "compared": compared, "blocks_without_directions": none_blocks}


# ----------------------------------------------------------------------------------------------------------------
# cost estimate
# ----------------------------------------------------------------------------------------------------------------


def device_free_gb(device: str | torch.device) -> float | None:
    """Free memory (GB) of a CUDA device, ``None`` for the CPU (host RAM is bounded by ``ram_budget_gb``)."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    return float(torch.cuda.mem_get_info(dev)[0]) / 1e9


def estimate_cost(d_m: int, d_k: int, grid: Grid, n_perm: int, n_blocks: int, *, ram_budget_gb: float | None = None,
                  blocks_per_pass: int | None = None, rows_per_image: int | None = None, content_null: bool = False,
                  acc_device: str = "cpu", free_gb: float | None = None, gpu_margin_gb: float = 4.0) -> dict[str, Any]:
    """Bytes of the float64 accumulators of ONE block, the analysis transient, the blocks per pass and the flops.

    ``content_null`` adds the secondary accumulators (centred rows, position-kept null; no ICC). On a CUDA
    ``acc_device`` the budget is the FREE device memory (``free_gb``; default: queried) minus ``gpu_margin_gb`` for the
    models and activations, never ``ram_budget_gb``: ``blocks_per_pass`` is reduced to fit (``reduced_from`` records
    it) and a ``ValueError`` refuses a single block that does not fit."""
    n_a, n_l = len(grid.n_images), len(grid.ks)
    n_s = n_a * n_l * 2
    base = n_s * (d_m * d_m + d_k * d_k + d_m * d_k)
    null = n_perm * n_s * d_m * d_k
    icc = n_a * n_l * (d_m * d_m + d_m)
    doubles = base + null + icc + ((base + null) if content_null else 0)
    per_block_gb = 8.0 * doubles / 1e9
    transient_gb = 8.0 * (8 * (d_m * d_m + d_k * d_k + d_m * d_k) + 2 * (n_perm + 1) * d_m * d_k) / 1e9
    on_gpu = torch.device(acc_device).type == "cuda"
    if on_gpu:
        free = device_free_gb(acc_device) if free_gb is None else float(free_gb)
        budget = free - float(gpu_margin_gb)
        fit = int(budget // per_block_gb)
        if fit < 1:
            raise ValueError(
                f"one block needs {per_block_gb:.2f} GB of accumulators but only {free:.1f} GB are free on {acc_device} "
                f"(margin {gpu_margin_gb} GB): use acc_device='cpu' (host RAM) or reduce n_null / the grid"
            )
    else:
        budget = 24.0 if ram_budget_gb is None else float(ram_budget_gb)
        fit = max(1, int((budget - transient_gb - 2.0) // per_block_gb))
    requested = blocks_per_pass
    blocks_per_pass = fit if requested is None else (min(int(requested), fit) if on_gpu else int(requested))
    blocks_per_pass = min(int(blocks_per_pass), int(n_blocks))
    rows = grid.n_max * (rows_per_image or grid.n_tokens)
    flops_block = (2.0 * rows * (d_m * d_m + d_k * d_k + d_m * d_k) + n_perm * 2.0 * rows * d_m * d_k) * (2 if content_null else 1)
    out = {
        "per_block_accumulator_gb": per_block_gb, "analysis_transient_gb": transient_gb,
        "blocks_per_pass": blocks_per_pass, "n_passes": math.ceil(n_blocks / blocks_per_pass),
        "peak_accumulator_gb": per_block_gb * blocks_per_pass + transient_gb,
        "flops_per_block": flops_block, "flops_total": flops_block * n_blocks, "acc_device": str(acc_device),
        "content_null": bool(content_null),
        "note": "accumulators live on acc_device (host RAM when cpu, GPU memory otherwise); models excluded",
    }
    if on_gpu:
        out.update({"free_gb": free, "gpu_margin_gb": float(gpu_margin_gb)})
        if requested is not None and int(requested) > fit:
            out["reduced_from"] = int(requested)
    return out


# ----------------------------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Phase1Config:
    source: ModelSpec
    target: ModelSpec
    state_dir: str
    out_dir: str
    pair: str = "pair"
    suite: str = "vision8"
    tasks: tuple[str, ...] = ("EuroSAT", "GTSRB", "DTD")  # source experts for the coverage
    n_images: tuple[int, ...] = (256, 1024, 4096)
    ks: tuple[int, ...] = (1, 16, 197)
    seed: int = 0  # pool permutation, token subsets, halves, null permutations
    n_null: int = 19
    null_quantile: float = 0.95
    anchor_filter: str | None = "norm"
    anchor_filter_k: float = 3.0
    task_images: int = 1024
    task_seed: int = 0
    val_size: int = 1000
    batch_size: int = 64
    num_workers: int = 0
    device: str = "cuda"  # model forward
    acc_device: str = "cpu"  # where the float64 accumulators live (GPU memory is NOT enough for A -> B)
    compute_device: str | None = None  # where the per-batch products are formed (None = acc_device)
    compute_dtype: str = "float64"  # 'float32' only after checking the precision on real data
    linalg_device: str = "cpu"
    content_null: bool = True  # secondary null r*_content (rows centred per position by the registration images)
    cudnn_allow_tf32: bool = False  # torch.backends.cudnn.allow_tf32 of the forward passes
    matmul_allow_tf32: bool = False  # torch.backends.cuda.matmul.allow_tf32 of the forward passes
    frame_check_images: int = 64  # startup check of the recomputed P against the state's anchor rows (0 = skip)
    gpu_margin_gb: float = 4.0
    blocks: tuple[int, ...] | None = None
    blocks_per_pass: int | None = None
    ram_budget_gb: float = 24.0
    icc_top: int = 10
    overwrite: bool = False


def parse_config(payload: Mapping[str, Any]) -> Phase1Config:
    p = dict(payload)
    for side in ("source", "target"):
        spec = dict(p[side])
        unknown = sorted(set(spec) - {f.name for f in fields(ModelSpec)})
        if unknown:
            raise ValueError(f"unknown {side} keys: {unknown}")
        p[side] = ModelSpec(**spec)
    allowed = {f.name for f in fields(Phase1Config)}
    unknown = sorted(set(p) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}; allowed: {sorted(allowed)}")
    for key in ("tasks", "n_images", "ks", "blocks"):
        if p.get(key) is not None:
            p[key] = tuple(p[key]) if key == "tasks" else tuple(int(x) for x in p[key])
    cfg = Phase1Config(**p)
    Grid(cfg.n_images, cfg.ks)  # validates
    if cfg.anchor_filter not in {None, "norm"}:
        raise ValueError("anchor_filter must be null or 'norm'")
    if any(n % cfg.batch_size for n in cfg.n_images):
        raise ValueError(f"batch_size {cfg.batch_size} must divide every N in {cfg.n_images} (a batch stays in one stratum)")
    if cfg.compute_dtype not in {"float32", "float64"}:
        raise ValueError("compute_dtype must be float32|float64")
    if cfg.n_null < 0 or not 0.0 < cfg.null_quantile < 1.0:
        raise ValueError("n_null >= 0 and 0 < null_quantile < 1 required")
    missing = [t for t in cfg.tasks if t not in cfg.source.experts]
    if missing:
        raise ValueError(f"source experts missing for tasks {missing}")
    return cfg


# ----------------------------------------------------------------------------------------------------------------
# vision pipeline (needs models, data)
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class BlockFilter:
    """Per target block ``j`` (source block ``i``): the frame of the decoder (``filtered_pair_inputs`` and the row rule
    of ``AnchorState.norm_filter``), tensors float64 on the accumulation device."""

    j: int
    i: int
    mu_m: Tensor  # filtered mu (centring of P)
    c_m: float
    mu_k: Tensor
    c_k: float
    reg_mu_m: Tensor | None = None  # registration mu and median of the norm flag
    med_m: float = 0.0
    reg_mu_k: Tensor | None = None
    med_k: float = 0.0
    k_filter: float = 3.0

    def rows(self, t_m: Tensor, t_k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(P_m, P_k, valid)`` of canonical tokens ``[B, T, d]`` (``valid``: not high-norm in either model)."""
        t_m, t_k = t_m.to(self.mu_m.dtype), t_k.to(self.mu_k.dtype)
        valid = torch.ones(t_m.shape[:2], dtype=torch.bool, device=t_m.device)
        if self.reg_mu_m is not None:
            valid &= (t_m - self.reg_mu_m).norm(dim=-1) <= self.k_filter * self.med_m
            valid &= (t_k - self.reg_mu_k).norm(dim=-1) <= self.k_filter * self.med_k
        return (t_m - self.mu_m) / self.c_m, (t_k - self.mu_k) / self.c_k, valid


def open_state(cfg: Phase1Config, depth_m: int, depth_k: int):
    """``(frame_json, acfg, reg_m, reg_k)``: the anchors state of the pair (read only), norm filters loaded."""
    from ...rebase.methods.anchors import AnchorState, parse_anchors_config
    from .update_kernel_cka import model_id

    frame_json = json.loads((Path(cfg.state_dir) / "frame.json").read_text())
    if frame_json["config"]["anchor_pool"] != "tiny_imagenet":
        raise NotImplementedError("Phase 1 vision: the Tiny-ImageNet pool only")
    params: dict[str, Any] = {**frame_json["config"], "state_dir": str(cfg.state_dir)}
    if cfg.anchor_filter is not None:
        params.update({"anchor_filter": cfg.anchor_filter, "anchor_filter_k": float(cfg.anchor_filter_k)})
    acfg = parse_anchors_config(params)
    state = AnchorState(acfg, acfg.state_dir)
    reg_m, reg_k = state.model(model_id(cfg.source)), state.model(model_id(cfg.target))
    if reg_m.depth != depth_m or reg_k.depth != depth_k:
        raise ValueError(f"registered depths {reg_m.depth}/{reg_k.depth} != model depths {depth_m}/{depth_k}")
    if cfg.anchor_filter is not None:
        for mid in (model_id(cfg.source), model_id(cfg.target)):
            state.load_norm_filter(mid, k=float(cfg.anchor_filter_k))
    return frame_json, acfg, reg_m, reg_k


def load_registration(cfg: Phase1Config, depth_m: int, depth_k: int, blocks: Sequence[int], device: str):
    """``({j: BlockFilter}, frame_json, registration indices, pairing)`` from the anchors state (read only)."""
    from ...rebase.discrete_layer_match import DiscreteLayerPairing

    frame_json, _, reg_m, reg_k = open_state(cfg, depth_m, depth_k)
    k = float(cfg.anchor_filter_k)
    pairing = DiscreteLayerPairing.compute(depth_m, depth_k)
    f64 = torch.float64
    out: dict[int, BlockFilter] = {}
    for j in blocks:
        i = int(pairing.pairing[j])
        if cfg.anchor_filter is None:
            out[j] = BlockFilter(j, i, reg_m.mu[i].to(device, f64), float(reg_m.c[i]), reg_k.mu[j].to(device, f64),
                                 float(reg_k.c[j]))
            continue
        fm, fk = reg_m.extras[f"norm_k{k:g}"], reg_k.extras[f"norm_k{k:g}"]
        out[j] = BlockFilter(
            j, i, fm["mu"][i].to(device, f64), float(fm["c"][i]), fk["mu"][j].to(device, f64), float(fk["c"][j]),
            reg_m.mu[i].to(device, f64), float(fm["median"][i]), reg_k.mu[j].to(device, f64), float(fk["median"][j]), k,
        )
    sizes = frame_json["pool"]["batch_sizes"]
    n_anchor_images = sum(sizes[: int(frame_json["config"]["anchor_batches"])])
    reg_indices = [int(x) for x in frame_json["pool"]["indices"][n_anchor_images:]]
    return out, frame_json, reg_indices, pairing


def gather_rows(p: Tensor, images: Tensor, positions: Tensor) -> Tensor:
    """Rows ``p[image, position]`` of ``p [B, T, d]``."""
    return p[images.to(p.device), positions.to(p.device)]


def relative_errors(run: Tensor, ref: Tensor) -> dict[str, float]:
    """Relative Frobenius error of ``run`` against ``ref`` (rows ``[n, d]``) and the median / max of the per-row
    relative errors."""
    d = run.to(torch.float64) - ref.to(torch.float64)
    row = d.norm(dim=1) / ref.to(torch.float64).norm(dim=1).clamp(min=1e-300)
    return {"rel_fro": float(d.norm() / ref.to(torch.float64).norm()), "row_median": float(row.median()),
            "row_max": float(row.max()), "n_rows": int(ref.shape[0])}


class VisionPairSource:
    """The modality seam: pool images in the Phase-1 order, forward capture of canonical tokens, task images.

    A decoder (LLM) source would provide the same three things for documents: ``order`` of the pool documents
    (disjoint from the registration documents), per-document units (tokens / words) captured at the paired blocks
    with the document id and unit rank, and task prompts. ``StratifiedSigma`` / ``analyze_cell`` need nothing else."""

    def __init__(self, cfg: Phase1Config, clf_m, clf_k, reg_indices: Sequence[int], canonical_tokens: int, grid: Grid):
        from ...data.vision_loaders import HFVisionDataset, load_hf_splits
        from ...rebase.methods._ariadne.capture import _stable_dataset_identity

        self.cfg, self.clf_m, self.clf_k, self.grid, self.canonical = cfg, clf_m, clf_k, grid, int(canonical_tokens)
        hf = load_hf_splits("zh-plus/tiny-imagenet", config=None, requested_splits=("valid",))["valid"]
        self.ds_m = HFVisionDataset(hf, transform=clf_m.preprocess)
        self.ds_k = HFVisionDataset(hf, transform=clf_k.preprocess)
        self.identity = _stable_dataset_identity(self.ds_m)
        avail = torch.ones(len(self.ds_m), dtype=torch.bool)
        avail[torch.as_tensor(list(reg_indices), dtype=torch.long)] = False
        pool = avail.nonzero().flatten()
        if grid.n_max > pool.numel():
            raise ValueError(f"pool too small: N_max={grid.n_max} > {int(pool.numel())} images outside the registration set")
        perm = torch.randperm(int(pool.numel()), generator=torch.Generator().manual_seed(int(cfg.seed)))
        self.order = pool[perm[: grid.n_max]]
        self.n_registration = len(reg_indices)

    def loaders(self, indices: Sequence[int] | None = None):
        from torch.utils.data import DataLoader, Subset

        idx = self.order.tolist() if indices is None else [int(i) for i in indices]
        kw = {"batch_size": self.cfg.batch_size, "shuffle": False, "num_workers": self.cfg.num_workers}
        return DataLoader(Subset(self.ds_m, idx), **kw), DataLoader(Subset(self.ds_k, idx), **kw)


FRAME_CHECK_TOL = 1e-3


def frame_check(
    src: VisionPairSource, cfg: Phase1Config, filters: Mapping[int, BlockFilter], frame_json: Mapping[str, Any], acfg,
    reg_m, reg_k, blocks: Sequence[int],
) -> dict[str, Any]:
    """Recompute ``P`` for the state's anchor rows on the first ``frame_check_images`` images of the state's pool draw
    under THIS run's forward settings and compare it with ``pair_frames`` of the state (the frame the decoder uses).
    Relative Frobenius / per-row errors per block and model. Detects a change of forward numerics (TF32, precision,
    device) between the state's registration and this run."""
    from ...rebase.methods._ariadne.capture import iter_capture_tokens
    from ...rebase.methods.anchors.method import anchor_keep, pair_frames
    from ...rebase.methods.anchors.state import to_canonical

    n_chk = min(int(cfg.frame_check_images), 512)
    idx = [int(i) for i in frame_json["pool"]["indices"][:n_chk]]
    rows = torch.as_tensor(frame_json["anchor_rows"], dtype=torch.long)
    m_req = {str(i): (i, "boundary") for i in sorted({filters[j].i for j in blocks})}
    k_req = {str(j): (j, "boundary") for j in blocks}
    dl_m, dl_k = src.loaders(idx)
    runs: dict[int, list[tuple[Tensor, Tensor]]] = {j: [] for j in blocks}
    for bm, bk in zip(dl_m, dl_k, strict=True):
        vm = next(iter_capture_tokens(src.clf_m.model, [bm], m_req, cfg.device, store_device=cfg.device))
        vk = next(iter_capture_tokens(src.clf_k.model, [bk], k_req, cfg.device, store_device=cfg.device))
        for j in blocks:
            f = filters[j]
            pm, pk, _ = f.rows(to_canonical(vm[str(f.i)], src.canonical).to(f.mu_m.device),
                               to_canonical(vk[str(j)], src.canonical).to(f.mu_m.device))
            runs[j].append((pm.cpu(), pk.cpu()))
    out: dict[str, Any] = {}
    for j in blocks:
        f = filters[j]
        pm_all = torch.cat([x[0] for x in runs[j]])
        pk_all = torch.cat([x[1] for x in runs[j]])
        p_m, _, p_k, _ = pair_frames(reg_m, f.i, reg_k, j, config=acfg)
        keep = anchor_keep([(reg_m, f.i), (reg_k, j)], float(cfg.anchor_filter_k)) if cfg.anchor_filter is not None \
            else torch.ones(rows.shape[0], dtype=torch.bool)
        kept = rows[keep.cpu()]
        sel = (kept[:, 0] < n_chk).nonzero().flatten()
        ref_m, ref_k = p_m.cpu()[sel], p_k.cpu()[sel]
        run_m = gather_rows(pm_all, kept[sel, 0], kept[sel, 1])
        run_k = gather_rows(pk_all, kept[sel, 0], kept[sel, 1])
        out[str(j)] = {"source_block": f.i, "source": relative_errors(run_m, ref_m), "target": relative_errors(run_k, ref_k)}
    worst = max(max(e["source"]["rel_fro"], e["target"]["rel_fro"]) for e in out.values())
    res = {"n_images": n_chk, "blocks": out, "max_rel_fro": worst, "warning": worst > FRAME_CHECK_TOL}
    if res["warning"]:
        import logging

        logging.getLogger(__name__).warning(
            "frame check: the recomputed P differs from the state's by rel_fro %.2e > %.0e in %s; the forward numerics "
            "(TF32 flags, device, precision) of this run differ from the state's registration",
            worst, FRAME_CHECK_TOL, [j for j, e in out.items() if max(e["source"]["rel_fro"], e["target"]["rel_fro"]) > FRAME_CHECK_TOL])
    return res


def registration_position_means(
    src: VisionPairSource, cfg: Phase1Config, filters: Mapping[int, BlockFilter], blocks: Sequence[int],
    reg_indices: Sequence[int],
) -> dict[int, tuple[Tensor, Tensor]]:
    """Per-position mean ``[T, d]`` of the rows ``P`` of the registration images (kept rows only) per block and model:
    the centring of the secondary (content) null. One cheap forward pass over the 320 registration images."""
    from ...rebase.methods._ariadne.capture import iter_capture_tokens
    from ...rebase.methods.anchors.state import to_canonical

    dev = cfg.device
    cdev = cfg.compute_device or cfg.acc_device
    m_req = {str(i): (i, "boundary") for i in sorted({filters[j].i for j in blocks})}
    k_req = {str(j): (j, "boundary") for j in blocks}
    dl_m, dl_k = src.loaders(reg_indices)
    sums: dict[int, list[Tensor]] = {}
    for bm, bk in zip(dl_m, dl_k, strict=True):
        vm = next(iter_capture_tokens(src.clf_m.model, [bm], m_req, dev, store_device=dev))
        vk = next(iter_capture_tokens(src.clf_k.model, [bk], k_req, dev, store_device=dev))
        for j in blocks:
            f = filters[j]
            pm, pk, valid = f.rows(to_canonical(vm[str(f.i)], src.canonical).to(cdev), to_canonical(vk[str(j)], src.canonical).to(cdev))
            w = valid.to(pm.dtype)
            cur = [torch.einsum("bt,btd->td", w, pm), torch.einsum("bt,btd->td", w, pk), w.sum(0)]
            sums[j] = cur if j not in sums else [a + b for a, b in zip(sums[j], cur, strict=True)]
    return {j: (v[0] / v[2].clamp(min=1)[:, None], v[1] / v[2].clamp(min=1)[:, None]) for j, v in sums.items()}


def _model_forward_pass(
    src: VisionPairSource, filters: Mapping[int, BlockFilter], blocks: Sequence[int], grid: Grid, halves: Tensor,
    cfg: Phase1Config, pos_means: Mapping[int, tuple[Tensor, Tensor]] | None = None,
) -> tuple[dict[int, StratifiedSigma], dict[int, StratifiedSigma] | None]:
    """One forward pass of the largest cell over the pool, accumulating the strata of ``blocks``: the primary
    accumulators (raw ``P`` rows, null N1 = derangement + random position) and, with ``pos_means``, the secondary
    ones (rows centred per position, position-kept null; no ICC)."""
    from ...rebase.methods._ariadne.capture import iter_capture_tokens
    from ...rebase.methods.anchors.state import to_canonical

    device = cfg.device
    cdev = cfg.compute_device or cfg.acc_device
    cdt = torch.float32 if cfg.compute_dtype == "float32" else torch.float64
    m_req = {str(i): (i, "boundary") for i in sorted({filters[j].i for j in blocks})}
    k_req = {str(j): (j, "boundary") for j in blocks}
    dl_m, dl_k = src.loaders()
    d_m = {j: int(filters[j].mu_m.numel()) for j in blocks}
    d_k = {j: int(filters[j].mu_k.numel()) for j in blocks}
    mk = lambda j, icc: StratifiedSigma(  # noqa: E731
        d_m[j], d_k[j], grid, cfg.n_null, device=cfg.acc_device, compute_device=cdev, compute_dtype=cdt, icc=icc)
    acc = {j: mk(j, True) for j in blocks}
    acc_c = {j: mk(j, False) for j in blocks} if pos_means is not None else None
    bs = cfg.batch_size
    for b, (bm, bk) in enumerate(zip(dl_m, dl_k, strict=True)):
        if not torch.equal(torch.as_tensor(bm[1]), torch.as_tensor(bk[1])):
            raise RuntimeError("the two models' loaders do not yield the same images (labels differ)")
        pos = torch.arange(b * bs, b * bs + int(bm[0].shape[0]))
        a = int(image_stratum(pos[:1], grid))
        if int(image_stratum(pos[-1:], grid)) != a:
            raise RuntimeError("a batch straddles two image strata")
        half = halves[pos]
        level = token_levels(token_ranks(pos, grid.n_tokens, cfg.seed), grid.ks)
        plan = NullRandomPosition(half, pos, grid.n_tokens, cfg.n_null, cfg.seed, b) if cfg.n_null else None
        plan_c = NullPositionKept(half, cfg.n_null, cfg.seed, b) if (cfg.n_null and pos_means is not None) else None
        vals_m = next(iter_capture_tokens(src.clf_m.model, [bm], m_req, device, store_device=device))
        vals_k = next(iter_capture_tokens(src.clf_k.model, [bk], k_req, device, store_device=device))
        for j in blocks:
            f = filters[j]
            t_m = to_canonical(vals_m[str(f.i)], src.canonical).to(cdev)
            t_k = to_canonical(vals_k[str(j)], src.canonical).to(cdev)
            pm, pk, valid = f.rows(t_m, t_k)
            acc[j].update(a, pm, pk, valid, level, half, plan)
            if acc_c is not None:
                acc_c[j].update(a, centre_by_position(pm, pos_means[j][0]), centre_by_position(pk, pos_means[j][1]),
                                valid, level, half, plan_c)
        print(f"[metodoK_phase1] {cfg.pair}: blocks {list(blocks)} batch {b + 1}/{grid.n_max // bs}", flush=True)
    return acc, acc_c


def task_gram_pass(
    src: VisionPairSource, task: str, source_blocks: Sequence[int], cfg: Phase1Config, device: str, acc_device: str,
) -> tuple[dict[int, Tensor], dict[str, Any]]:
    """``G_E = sum_rows dH^T dH`` per source block for the source expert of ``task`` on seeded random TRAIN images
    (all canonical tokens, random order: indices are NOT sorted by class)."""
    import copy

    from torch.utils.data import DataLoader, Subset

    from ...data.vision_loaders import build_vision_loaders, load_hf_splits
    from ...rebase.methods._ariadne.capture import iter_capture_tokens
    from ...rebase.methods.anchors.state import to_canonical
    from ..datasets.vision8_14_20 import SUITES
    from .update_kernel_cka import expert_update

    clf = src.clf_m
    hf_path, hf_config, split_map = SUITES[cfg.suite].resolver(task)
    hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
    loaders = build_vision_loaders(
        hf_ds=hf_ds, hf_path=hf_path, preprocess=clf.preprocess, ft_epochs=1, split_map=split_map,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=False, seed=cfg.task_seed,
        val_size=cfg.val_size,
    )
    ds = loaders.train.dataset  # the TRAIN split (minus the val slice); never the test split
    n = min(int(cfg.task_images), len(ds))
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(int(cfg.task_seed)))[:n].tolist()
    base_sd = {k: v.detach().cpu().clone() for k, v in clf.model.state_dict().items()}
    upd, info = expert_update(base_sd, cfg.source.experts[task])
    work = copy.deepcopy(clf.model)
    work.load_state_dict({**base_sd, **upd}, strict=True)
    req = {str(i): (i, "boundary") for i in source_blocks}
    gram: dict[int, Tensor] = {}
    n_rows = 0
    loader = DataLoader(Subset(ds, idx), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    for batch in loader:
        batch = [batch[0], batch[1]]
        v_base = next(iter_capture_tokens(clf.model, [batch], req, device, store_device=device))
        v_ft = next(iter_capture_tokens(work, [batch], req, device, store_device=device))
        for i in source_blocks:
            e = to_canonical(v_ft[str(i)] - v_base[str(i)], src.canonical).reshape(-1, v_ft[str(i)].shape[-1])
            e = e.to(acc_device, torch.float64)
            gram[i] = gram.get(i, 0) + e.T @ e
        n_rows += int(e.shape[0])
    meta = {"task": task, "split": "train (minus val slice)", "n_images": n, "n_rows": n_rows, "expert": info,
            "image_index_order": "seeded random, unsorted", "index_sha256": hashlib.sha256(repr(idx).encode()).hexdigest()}
    return {i: g.cpu() for i, g in gram.items()}, meta


# ----------------------------------------------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------------------------------------------


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


CSV_COLUMNS = (
    "target_block", "source_block", "N", "k", "T_nominal", "status", "rows", "rows_half_min", "rank_m", "rank_k", "q",
    "gamma_in_top1", "gamma_in_mean", "gamma_out_top1", "gamma_out_mean", "gamma_out_ab_mean", "gamma_out_ba_mean",
    "threshold", "r_star_prefix", "r_star_excess", "r_star_contig", "n_above", "null_row_fraction_min",
    "null_row_fraction_max", "rho_icc_top1", "rho_icc_top_mean", "rho_icc_trace", "n_eff_top", "n_eff_trace",
    "cov_isotropic", "r_star_content_prefix", "r_star_content_excess", "r_star_content_contig", "n_above_content",
    "threshold_content",
)
CONTENT_KEYS = {"r_star_prefix": "r_star_content_prefix", "r_star_excess": "r_star_content_excess",
                "r_star_contig": "r_star_content_contig", "n_above": "n_above_content", "threshold": "threshold_content"}


def csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = list(summary["setting"]["tasks"])
    rows = []
    for blk in summary["blocks"]:
        for c in blk["cells"]:
            row = {"target_block": blk["target_block"], "source_block": blk["source_block"],
                   "rows_half_min": min(c["rows_half"]), "cov_isotropic": c.get("coverage_isotropic_reference")}
            row.update({k: c.get(k) for k in CSV_COLUMNS if k not in row and k not in CONTENT_KEYS.values()})
            row.update({v: (c.get("content") or {}).get(k) for k, v in CONTENT_KEYS.items()})
            for t in tasks:
                cv = (c.get("coverage") or {}).get(t, {})
                row[f"cov_total_{t}"] = cv.get("cov_total")
                row[f"cov_span_{t}"] = cv.get("cov_span")
            rows.append(row)
    return rows


def _f(x: Any, nd: int = 3) -> str:
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(summary: Mapping[str, Any]) -> str:
    st = summary["setting"]
    lines = [f"# Metodo K, Phase 1: {st['pair']}", "", "## Setting", ""]
    for k in ("source", "target", "pool", "registration", "grid", "frame", "filter", "seeds", "forward", "halves", "null",
              "r_star", "icc", "coverage", "code", "cost_estimate", "measured", "seconds"):
        if k in st:
            lines.append(f"- **{k}**: {json.dumps(st[k], default=str)}")
    fc = st.get("frame_check")
    if fc:
        lines += ["", f"## Frame check{(' (WARNING: max rel_fro {:.1e} > 1e-3)'.format(fc['max_rel_fro'])) if fc.get('warning') else ''} "
                  f"(recomputed P vs the state's anchor rows, {fc['n_images']} images, forward flags "
                  f"{json.dumps(st.get('forward'))})", "", "| target block | source block | rel. Fro. source | rel. Fro. target "
                  "| median row target | max row target | rows |", "|---|---|---|---|---|---|---|"]
        for j, e in fc["blocks"].items():
            lines.append(f"| {j} | {e['source_block']} | {e['source']['rel_fro']:.2e} | {e['target']['rel_fro']:.2e} | "
                         f"{e['target']['row_median']:.2e} | {e['target']['row_max']:.2e} | {e['target']['n_rows']} |")
    sg = summary["selection_global"]
    lines += ["", "## Selection (cheapest cell with mean |dGamma_out| < 0.02 absolute over the top r_ref directions in every "
              "larger cell; r_ref = r*_prefix at the largest cell)", "",
              f"- global (every block): cell {json.dumps(sg.get('cell'))}, trivial: {sg.get('trivial')}, "
              f"r_ref per block: {sg.get('r_ref_per_block')}, blocks without directions: {sg.get('blocks_without_directions')}, "
              f"reason: {sg.get('reason')}",
              f"- larger cells compared (per block): {json.dumps(sg.get('compared'))}", ""]
    lines += ["| target block | source block | r_ref | cheapest cell | trivial | larger cells compared |", "|---|---|---|---|---|---|"]
    for blk in summary["blocks"]:
        sl = blk["selection"]
        sel = sl.get("cell")
        label = "N={} k={}".format(sel["N"], sel["k"]) if sel else sl.get("reason")
        cmp_ = ", ".join(f"({o['N']},{o['k']})" for o in sl.get("compared", []))
        lines.append(f"| {blk['target_block']} | {blk['source_block']} | {_f(sl.get('r_ref'))} | {label} | "
                     f"{sl.get('trivial', '-')} | {cmp_} |")
    lines += ["", "## r* profile per cell (profile, not a criterion): prefix / excess / contiguous; content null (beyond position)",
              "", "| block | N | k | status | r*_prefix | r*_excess | r*_contig | n_above | q | r*_content_prefix | "
              "null rows kept (min-max) |", "|" + "---|" * 11]
    for blk in summary["blocks"]:
        for c in blk["cells"]:
            ct = c.get("content") or {}
            lines.append(f"| {blk['target_block']} | {c['N']} | {c['k']} | {c['status']} | {_f(c.get('r_star_prefix'))} | "
                         f"{_f(c.get('r_star_excess'), 1)} | {_f(c.get('r_star_contig'))} | {_f(c.get('n_above'))} | "
                         f"{_f(c.get('q'))} | {_f(ct.get('r_star_prefix'))} | "
+ ("-" if c.get("null_row_fraction_min") is None else
                            f"{c['null_row_fraction_min']:.3f}-{c['null_row_fraction_max']:.3f}") + " |")
    lines += ["", "## Per block and cell", "",
              "| block | N | k | status | rows | Gamma_in top1 | Gamma_out top1 | thr (pooled) | r*_prefix | rho_icc (top mean) "
              "| rho_icc (trace) | n_eff (trace) |", "|" + "---|" * 12]
    for blk in summary["blocks"]:
        for c in blk["cells"]:
            lines.append(
                f"| {blk['target_block']} | {c['N']} | {c['k']} | {c['status']} | {c['rows']} | {_f(c.get('gamma_in_top1'))} | "
                f"{_f(c.get('gamma_out_top1'))} | {_f(c.get('threshold'))} | {_f(c.get('r_star_prefix'))} | "
                f"{_f(c.get('rho_icc_top_mean'))} | {_f(c.get('rho_icc_trace'))} | {_f(c.get('n_eff_trace'), 1)} |")
    lines += ["", "## Task coverage at r*_prefix (fraction of the energy of E; isotropic reference r*/rank)", ""]
    tasks = list(st["tasks"])
    lines += ["| block | N | k | r* | iso | " + " | ".join(f"{t} total / span" for t in tasks) + " |", "|" + "---|" * (5 + len(tasks))]
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
    (out_dir / "summary.json").write_text(json.dumps(_clean(dict(summary)), indent=1, default=str))
    rows = csv_rows(summary)
    cols = list(CSV_COLUMNS) + sorted({k for r in rows for k in r if k not in CSV_COLUMNS})
    with open(out_dir / "blocks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "summary.md").write_text(render_md(_clean(dict(summary))))
    torch.save(dict(tensors), out_dir / "tensors.pt")


# ----------------------------------------------------------------------------------------------------------------
# seed comparison (noise floor of the 0.02 criterion)
# ----------------------------------------------------------------------------------------------------------------


def compare_seeds(dir0: str | Path, dir1: str | Path) -> dict[str, Any]:
    """Split-to-split noise: for two runs that differ only in ``seed`` (pool order, halves, null), per block and cell
    the mean ``|Gamma_out,seed0 - Gamma_out,seed1|`` over the top ``r_ref`` directions, ``r_ref`` = the smaller of the two
    ``r*_prefix`` at the largest cell of the block (0 -> no entry). This is the floor against which the 0.02 of the
    selection is judged: a ``dGamma`` between cells below it is not distinguishable from resampling."""
    out = []
    sums = [json.loads((Path(d) / "summary.json").read_text()) for d in (dir0, dir1)]
    ten = [torch.load(Path(d) / "tensors.pt", weights_only=True) for d in (dir0, dir1)]
    if sums[0]["setting"]["null"] != sums[1]["setting"]["null"]:
        raise ValueError("the runs differ in the null (type or n_null)")
    for k in ("grid", "source", "target", "filter"):
        if sums[0]["setting"].get(k) != sums[1]["setting"].get(k):
            raise ValueError(f"the runs differ in {k}")
    if sums[0]["setting"]["seeds"]["seed"] == sums[1]["setting"]["seeds"]["seed"]:
        raise ValueError("both runs have the same seed")
    for b0, b1 in zip(sums[0]["blocks"], sums[1]["blocks"], strict=True):
        j = b0["target_block"]
        big = [(c0, c1) for c0, c1 in zip(b0["cells"], b1["cells"], strict=True)
               if c0["status"] != STATUS_DEGENERATE and c1["status"] != STATUS_DEGENERATE]
        if not big:
            continue
        c0, c1 = big[-1]
        r_ref = min(int(c0.get("r_star_prefix") or 0), int(c1.get("r_star_prefix") or 0))
        for c0, _ in big:
            key = f"b{j}_N{c0['N']}_k{c0['k']}"
            g0, g1 = ten[0][key]["gamma_out"], ten[1][key]["gamma_out"]
            n = min(r_ref, len(g0), len(g1))
            out.append({"target_block": j, "N": c0["N"], "k": c0["k"], "r_ref": r_ref,
                        "mean_abs_dgamma_out": float((g0[:n] - g1[:n]).abs().mean()) if n else None})
    vals = [e["mean_abs_dgamma_out"] for e in out if e["mean_abs_dgamma_out"] is not None]
    return {"seeds": [sums[0]["setting"]["seeds"]["seed"], sums[1]["setting"]["seeds"]["seed"]], "entries": out,
            "median": float(torch.tensor(vals).median()) if vals else None,
            "max": max(vals) if vals else None}


def render_compare(res: Mapping[str, Any]) -> str:
    lines = [f"# Split-to-split |dGamma_out| (seeds {res['seeds'][0]} vs {res['seeds'][1]})", "",
             f"median {_f(res['median'], 4)}, max {_f(res['max'], 4)}", "",
             "| block | N | k | r_ref | mean abs dGamma_out |", "|---|---|---|---|---|"]
    lines += [f"| {e['target_block']} | {e['N']} | {e['k']} | {e['r_ref']} | {_f(e['mean_abs_dgamma_out'], 4)} |"
              for e in res["entries"]]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------------------------

CONTENT_INFO_KEYS = ("status", "r_star_prefix", "r_star_excess", "r_star_contig", "n_above", "threshold", "gamma_out_top1",
                     "gamma_out_mean", "null_row_fraction_min", "null_row_fraction_max")


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    from ...run_logging import _code_fingerprint
    from .update_kernel_cka import _build_model

    cfg = parse_config(payload)
    out_dir = Path(cfg.out_dir)
    if (out_dir / "summary.json").exists() and not cfg.overwrite:
        raise FileExistsError(f"{out_dir / 'summary.json'} exists (set overwrite to replace)")
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.matmul_allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.cudnn_allow_tf32)
    forward = {"cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
               "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "model_dtype": "fp32",
               "device": cfg.device, "acc_device": cfg.acc_device, "compute_device": cfg.compute_device or cfg.acc_device,
               "compute_dtype": cfg.compute_dtype, "torch": torch.__version__}
    device = cfg.device
    cdev = cfg.compute_device or cfg.acc_device
    grid = Grid(cfg.n_images, cfg.ks)
    clf_m, clf_k = _build_model(cfg.source, device), _build_model(cfg.target, device)
    depth_m, depth_k = len(clf_m.model.visual.transformer.resblocks), len(clf_k.model.visual.transformer.resblocks)
    blocks = list(cfg.blocks) if cfg.blocks is not None else list(range(depth_k))
    filters, frame_json, reg_indices, pairing = load_registration(cfg, depth_m, depth_k, blocks, cdev)
    canonical = int(frame_json["config"]["canonical_tokens"])
    if canonical != grid.n_tokens:
        raise ValueError(f"the last k ({grid.n_tokens}) must be the number of canonical tokens ({canonical})")
    src = VisionPairSource(cfg, clf_m, clf_k, reg_indices, canonical, grid)
    if src.identity != frame_json["pool"]["dataset_identity"]:
        raise RuntimeError(f"pool dataset {src.identity} differs from the frame's {frame_json['pool']['dataset_identity']}")
    d_m, d_k = int(next(iter(filters.values())).mu_m.numel()), int(next(iter(filters.values())).mu_k.numel())
    cost = estimate_cost(d_m, d_k, grid, cfg.n_null, len(blocks), ram_budget_gb=cfg.ram_budget_gb,
                         blocks_per_pass=cfg.blocks_per_pass, content_null=cfg.content_null, acc_device=cfg.acc_device,
                         gpu_margin_gb=cfg.gpu_margin_gb)
    print(f"[metodoK_phase1] cost estimate: {json.dumps(cost)}", flush=True)
    halves = half_labels(grid, cfg.seed)
    opts = AnalysisOptions(n_null=cfg.n_null, null_quantile=cfg.null_quantile, icc_top=cfg.icc_top,
                           linalg_device=cfg.linalg_device)
    opts_c = AnalysisOptions(n_null=cfg.n_null, null_quantile=cfg.null_quantile, linalg_device=cfg.linalg_device, icc=False)

    fcheck = None
    if cfg.frame_check_images > 0 and cfg.anchor_filter is not None:
        _, acfg, reg_m, reg_k = open_state(cfg, depth_m, depth_k)
        fcheck = frame_check(src, cfg, filters, frame_json, acfg, reg_m, reg_k, blocks)
        print(f"[metodoK_phase1] frame check WARNING={fcheck['warning']} (rel. Fro. source/target): " + ", ".join(
            f"b{j}: {e['source']['rel_fro']:.1e}/{e['target']['rel_fro']:.1e}" for j, e in fcheck["blocks"].items()), flush=True)
    pos_means = registration_position_means(src, cfg, filters, blocks, reg_indices) if cfg.content_null and cfg.n_null else None

    # task gram matrices of the source experts (E = dH_m), per source block
    src_blocks = sorted({filters[j].i for j in blocks})
    grams: dict[str, dict[int, Tensor]] = {}
    task_meta: dict[str, Any] = {}
    for task in cfg.tasks:
        grams[task], task_meta[task] = task_gram_pass(src, task, src_blocks, cfg, device, cdev)

    block_res: dict[int, dict[str, Any]] = {}
    tensors: dict[str, Any] = {}
    cells_idx = grid.cells()
    bpp = cost["blocks_per_pass"]
    for s0 in range(0, len(blocks), bpp):
        group = blocks[s0 : s0 + bpp]
        acc, acc_c = _model_forward_pass(src, filters, group, grid, halves, cfg, pos_means)
        for j in group:
            i = filters[j].i
            cells, tens = [], {}
            for ia, il in cells_idx:
                info, t = analyze_cell(acc[j].cell(ia, il), opts, {task: grams[task][i] for task in cfg.tasks})
                key = f"b{j}_N{info['N']}_k{info['k']}"
                if t:
                    info["gamma_out_vec"] = t["gamma_out"]  # tensor, dropped before the JSON
                    tens[key] = t
                if acc_c is not None and info["status"] != STATUS_DEGENERATE and "r_star" in info:
                    info_c, t_c = analyze_cell(acc_c[j].cell(ia, il), opts_c, None)
                    info["content"] = {k: info_c.get(k) for k in CONTENT_INFO_KEYS}
                    if t_c:
                        tens[key + "_content"] = {k: t_c[k] for k in ("gamma_out", "null_gamma_out", "threshold", "above") if k in t_c}
                cells.append(info)
            sel = select_cell(cells) if cfg.n_null else {"cell": None, "reason": "n_null = 0: no r*"}
            block_res[j] = {"target_block": j, "source_block": i, "cells": cells, "selection": sel}
            tensors.update(tens)
            print(f"[metodoK_phase1] block {j}: " + ", ".join(
                f"(N={c['N']},k={c['k']}) {c['status']}" + (f" r*={c['r_star']}" if "r_star" in c else "") for c in cells), flush=True)
        del acc, acc_c
    sel_global = select_global([block_res[j]["cells"] for j in blocks]) if cfg.n_null else {"cell": None, "reason": "n_null = 0"}
    big = grid.cells()[-1]
    profile = [{"target_block": j, "source_block": block_res[j]["source_block"], **{
        k: block_res[j]["cells"][-1].get(k) for k in ("status", "r_star", "r_star_excess", "r_star_contig", "n_above", "q", "rank_m", "rank_k")}}
        for j in blocks]
    for blk in block_res.values():
        for c in blk["cells"]:
            c.pop("gamma_out_vec", None)
    root = Path(__file__).resolve().parents[2]
    code = {**_code_fingerprint(), "module_sha256": _sha256(Path(__file__)),
            "canonical_stats_sha256": _sha256(root / "rebase/methods/anchors/canonical_stats.py"),
            "core_sha256": _sha256(root / "rebase/methods/anchors/core.py")}
    large = max(grid.cells(), key=lambda c: (c[0], c[1]))
    setting = {
        "pair": cfg.pair,
        "source": {"model": cfg.source.model, "pretrained": cfg.source.pretrained, "experts": {t: cfg.source.experts[t] for t in cfg.tasks}},
        "target": {"model": cfg.target.model, "pretrained": cfg.target.pretrained},
        "pool": {"dataset": "zh-plus/tiny-imagenet valid", "identity": src.identity,
                 "images": "positions of one seeded permutation (seed) of the pool minus the registration images",
                 "n_registration_excluded": src.n_registration, "state_dir": cfg.state_dir,
                 "order_sha256": hashlib.sha256(repr(src.order.tolist()).encode()).hexdigest()},
        "registration": "mu, c (and the high-norm filter flags) of the state: fixed, disjoint from the Phase-1 images",
        "grid": {"N": list(grid.n_images), "k": list(grid.ks),
                 "T_nominal": {f"N={n}": {f"k={k}": n * k for k in grid.ks} for n in grid.n_images},
                 "token_unit": f"canonical tokens ({canonical}, CLS included), nested per-image seeded subsets"},
        "frame": {"block_pairing": "DiscreteLayerPairing", "pairing": list(pairing.pairing),
                  "canonical_tokens": canonical, "capture": "block boundary (output of resblock)"},
        "filter": {"anchor_filter": cfg.anchor_filter, "k": cfg.anchor_filter_k},
        "seeds": {"seed": cfg.seed, "task_seed": cfg.task_seed, "scope": "pool order, token subsets, halves, null"},
        "forward": forward,
        "halves": "split_ids by image position inside each image stratum; identical across cells",
        "null": {"permutations": cfg.n_null, "quantile": cfg.null_quantile, "interpolation": "higher",
                 "primary": "N1: derangement of images within (batch, stratum, half) groups AND an independent random "
                            "position of the partner image per (replicate, image); positional structure is destroyed",
                 "secondary": ("content (r*_content, beyond position): derangement, position kept, rows centred per position "
                               f"by the {src.n_registration} registration images (true and null rows)" if cfg.content_null
                               else "off")},
        "r_star": "one pooled threshold (95% quantile of all R x q null values over the indices); r*_prefix (primary): largest "
                  "r with >= 95% of the indices < r above it; r*_excess = n_above - 0.05 (q - n_above); r*_contig = contiguous run",
        "icc": "one-way unbalanced ANOVA (Searle), top canonical variates and trace, see module docstring",
        "coverage": {"tasks": task_meta, "directions": "V_m Y_r*(prefix) of the in-sample fit of the cell"},
        "tasks": list(cfg.tasks), "blocks_run": blocks, "largest_cell": {"N": grid.n_images[large[0]], "k": grid.ks[large[1]]},
        "code": code, "cost_estimate": cost, "config": dict(payload), "seconds": time.perf_counter() - started,
        "measured": {"peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6},
    }
    if fcheck is not None:
        setting["frame_check"] = fcheck
    summary = {"setting": setting, "blocks": [block_res[j] for j in blocks], "selection_global": sel_global,
               "r_star_profile_largest_cell": {"cell": {"N": grid.n_images[big[0]], "k": grid.ks[big[1]]}, "blocks": profile}}
    write_outputs(summary, tensors, out_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config")
    parser.add_argument("--out-dir", help="overrides out_dir of the config")
    parser.add_argument("--seed", type=int, help="overrides seed of the config (pool order, halves, null)")
    parser.add_argument("--compare-seeds", nargs=2, metavar=("DIR0", "DIR1"),
                        help="split-to-split |dGamma_out| of two finished runs that differ in the seed")
    parser.add_argument("--out", help="file for --compare-seeds (markdown); default: print")
    args = parser.parse_args(argv)
    if args.compare_seeds:
        res = compare_seeds(*args.compare_seeds)
        text = render_compare(res)
        if args.out:
            Path(args.out).write_text(text)
            Path(args.out).with_suffix(".json").write_text(json.dumps(_clean(res), indent=1))
        else:
            print(text)
        return
    if not args.config:
        parser.error("--config or --compare-seeds is required")
    payload = json.loads(Path(args.config).read_text())
    if args.out_dir:
        payload["out_dir"] = args.out_dir
    if args.seed is not None:
        payload["seed"] = args.seed
    run(payload)


if __name__ == "__main__":
    main()
