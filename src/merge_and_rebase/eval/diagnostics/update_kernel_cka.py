"""Forward-only diagnostic: do two models change their representation in the same way under the same task expert?

For one task ``t`` and two models ``m`` (source) and ``k`` (target), at every block pair ``(i, j)`` (``i =
pairing[j]``, the relative-depth rule of the anchors code, ``DiscreteLayerPairing``), the update of the residual
stream ``dH_m = H_m^ft - H_m^base`` (block output, same task images, expert minus base) is expressed in the
coordinates of the SHARED ANCHORS of the PAIRWISE frame (``pair_frames``; not the gpa / consensus frame)::

    P_m  = pair_frames(...)  (centred, scaled, ``anchor_filter='norm'`` exactly as the decoder), divided by c_m
    D_m  = dH_m P_m^T        [rows x T]      column a = the same anchor (image, canonical position) in every model

Measures (rows = tokens of the task images, anchors = the T columns)

1. ``m1``  CKA of the update kernels on the anchors, ``K_m = D_m^T D_m`` vs ``K_k = D_k^T D_k`` (T x T), rows of the
   CKA = anchors. Uses ALL native tokens of each model; also CLS-only and patch-only. Three Gram variants: row-centred
   (primary), ``uncentered`` ("decoder-relevant (uncentred)") and ``position_centered`` (mean over images removed per
   token position). With anchor-shared frames m1 is dominated by frame agreement, so it is read through its EXCESS over
   comparators: model k with the other task's expert on the same images, the within-model cross-task kernels, and the
   base-representation kernel (all with delete-one-image-cluster jackknife intervals on the same deleted anchors).
2. ``m2``  CKA over task rows, CKA(D_m, D_k) with rows = matched tokens (CLS per image; patches matched by position,
   k bilinearly resampled to the reference grid when the grids differ, an APPROXIMATION; optional mean-patch row).
3. ``m3``  direct agreement on matched rows: cosine (uncentred and centred) and the mean per-anchor-column Pearson r.

m2/m3 intervals: delete-one jackknife over G image chunks (t_{G-1}). Also logged per block and side: the common-mode
fraction ``rho = n h_bar^T Q h_bar / tr(Q S)`` and the cosine of the anchor-centred ``P h_bar`` of m and k.

Sanity checks (not comparators): null (anchors of D_k permuted for m1/m3; images mismatched with positions matched
for m2, z-score and empirical p), ``base_reference`` (CKA of ``P P^T``). Controls (same code path): ``other_task``
(expert-only: k with the expert of t' on the task-t images), ``other_task_other_images`` (the images of t'),
``random_update`` (k: base + a random direction per tensor with the norm of the real tau_k), ``transported``.

Streaming: nothing of size rows x T is ever stored. With ``x_i = P_m h_i`` (h_i a row of dH_m), every quantity of the
debiased linear CKA (Song et al. 2012 / Nguyen et al. 2021, the estimator of ``analysis/cka.py``) is a sum over rows of
d-space statistics (``SelfStats``, ``CrossStats``, ``NativeSide``); the T-space matrices are formed once per block pair.

    python -m merge_and_rebase.eval.diagnostics.update_kernel_cka --config <cfg.json>
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

Tensor = torch.Tensor

ROW_TYPES = ("cls", "patch", "mean")
CONTROLS = ("null", "other_task", "other_task_other_images", "random_update", "base_reference", "transported")
M1_VARIANTS = ("centered", "uncentered", "position_centered")


# ----------------------------------------------------------------------------------------------------------------
# Gram-form CKA over T anchors (measure 1, base reference). O(T^2) per evaluation (no T x T matmul).
# ----------------------------------------------------------------------------------------------------------------


def _zero_diag(mat: Tensor) -> Tensor:
    out = mat.clone()
    out.fill_diagonal_(0.0)
    return out


def _hsic_unbiased_zd(k: Tensor, l_: Tensor) -> Tensor:
    """Song et al. 2012 unbiased HSIC on zero-diagonal Grams. Same formula as ``analysis.cka._hsic_unbiased``
    (``trace(KL) = sum(K * L)`` and ``1^T K L 1 = (K1).(L1)`` for symmetric K, L: no T x T matmul)."""
    n = k.shape[0]
    ks, ls = k.sum(1), l_.sum(1)
    term1 = (k * l_).sum()
    term2 = ks.sum() * ls.sum() / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * (ks @ ls)
    return (term1 + term2 - term3) / (n * (n - 3))


def _center_gram(k: Tensor) -> Tensor:
    return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()


class GramPrep:
    """A Gram matrix prepared for CKA against many partners: zero-diagonal (debiased) or doubly centred (biased)
    form, and its self term. Permuting the anchors permutes rows and columns of the prepared matrix."""

    def __init__(self, gram: Tensor, *, debiased: bool = True) -> None:
        self.debiased = debiased
        g = gram.to(torch.float64)
        if g.shape[0] < (4 if debiased else 2):
            raise ValueError(f"CKA over anchors needs more anchors, got {g.shape[0]}")
        self.m = _zero_diag(g) if debiased else _center_gram(g)
        self.self_term = float(self._cross(self.m, self.m))

    def _cross(self, a: Tensor, b: Tensor) -> Tensor:
        return _hsic_unbiased_zd(a, b) if self.debiased else (a * b).sum()

    def cka_with(self, other: GramPrep, *, perm: Tensor | None = None) -> float:
        """CKA(self, other) where the anchors of ``other`` are permuted by ``perm`` (``other.m[perm][:, perm]``)."""
        if self.self_term <= 0.0 or other.self_term <= 0.0:
            return float("nan")
        o = other.m if perm is None else other.m[perm][:, perm]
        return float(self._cross(self.m, o)) / (self.self_term * other.self_term) ** 0.5


def gram_cka(k: Tensor, l_: Tensor, *, debiased: bool = True) -> float:
    """Linear CKA from the two Gram matrices (anchors x anchors). ``debiased=True`` equals
    ``analysis.cka.linear_cka(X, Y)`` for ``K = X X^T``, ``L = Y Y^T`` (checked in the tests); no explicit centring
    is needed for the debiased estimator (the n-1/n-2 terms centre implicitly), the biased one centres the Gram."""
    return GramPrep(k, debiased=debiased).cka_with(GramPrep(l_, debiased=debiased))


def gram_cka_bootstrap(k: Tensor, l_: Tensor, *, n_boot: int, seed: int, debiased: bool = True) -> dict[str, float]:
    """``analysis.cka.cka_bootstrap`` for Gram inputs: same generator protocol (``randint`` over rows, here the
    anchors), the resampled Gram is ``K[idx][:, idx]`` (= the Gram of the resampled features), same quantiles."""
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    k, l_ = k.to(torch.float64), l_.to(torch.float64)
    n = k.shape[0]
    value = gram_cka(k, l_, debiased=debiased)
    gen = torch.Generator().manual_seed(int(seed))
    samples = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=gen).to(k.device)
        samples[b] = gram_cka(k[idx][:, idx], l_[idx][:, idx], debiased=debiased)
    return {"value": value, **_quantile_ci(samples)}


def _quantile_ci(samples: Tensor) -> dict[str, float]:
    finite = samples[torch.isfinite(samples)]
    if finite.numel() == 0:
        return {"lo": float("nan"), "hi": float("nan")}
    return {
        "lo": float(torch.quantile(finite, 0.025).item()),
        "hi": float(torch.quantile(finite, 0.975).item()),
    }


# ----------------------------------------------------------------------------------------------------------------
# d-space streaming statistics (measures 2 and 3)
# ----------------------------------------------------------------------------------------------------------------


class _Acc:
    """Sums over rows: every field is a float64 tensor, so chunks add and bootstrap multiplicities scale."""

    names: tuple[str, ...] = ()

    def zeros_like(self):
        out = copy.deepcopy(self)
        for f in self.names:
            getattr(out, f).zero_()
        return out

    @classmethod
    def lincomb(cls, objs: Sequence[Any], weights: Sequence[float] | None = None):
        out = objs[0].zeros_like()
        for idx, o in enumerate(objs):
            w = 1.0 if weights is None else float(weights[idx])
            if w == 0.0:
                continue
            for f in cls.names:
                getattr(out, f).add_(getattr(o, f), alpha=w)
        return out


def _f64(x: Tensor) -> Tensor:
    return x.detach().to(device="cpu", dtype=torch.float64)


class SelfStats(_Acc):
    """One side's statistics of its rows ``h`` (d-dim) with ``a_i = h_i^T Q h_i = |P h_i|^2`` (``Q = P^T P``):
    ``n``, ``s = sum h``, ``S = sum h h^T``, ``a1 = sum a``, ``a2 = sum a^2``, ``ua = sum a h``."""

    names = ("n", "s", "S", "a1", "a2", "ua")

    def __init__(self, d: int) -> None:
        self.n = torch.zeros((), dtype=torch.float64)
        self.s = torch.zeros(d, dtype=torch.float64)
        self.S = torch.zeros(d, d, dtype=torch.float64)
        self.a1 = torch.zeros((), dtype=torch.float64)
        self.a2 = torch.zeros((), dtype=torch.float64)
        self.ua = torch.zeros(d, dtype=torch.float64)

    def update(self, h: Tensor, a: Tensor) -> None:
        self.n += h.shape[0]
        self.s += _f64(h.sum(0))
        self.S += _f64(h.T @ h)
        self.a1 += _f64(a.sum())
        self.a2 += _f64((a.double() * a.double()).sum())
        self.ua += _f64(a.to(h.dtype) @ h)


class CrossStats(_Acc):
    """Pairing-dependent sums of row pairs ``(h_i, g_i)`` (row i of m with row i of k): ``M = sum h g^T``,
    ``uhb = sum b_i h_i``, ``vga = sum a_i g_i``, ``ab = sum a_i b_i`` (``a``, ``b``: the rows' squared anchor norms)."""

    names = ("M", "uhb", "vga", "ab")

    def __init__(self, dm: int, dk: int) -> None:
        self.M = torch.zeros(dm, dk, dtype=torch.float64)
        self.uhb = torch.zeros(dm, dtype=torch.float64)
        self.vga = torch.zeros(dk, dtype=torch.float64)
        self.ab = torch.zeros((), dtype=torch.float64)

    def update(self, h: Tensor, g: Tensor, a: Tensor, b: Tensor) -> None:
        self.M += _f64(h.T @ g)
        self.uhb += _f64(b.to(h.dtype) @ h)
        self.vga += _f64(a.to(g.dtype) @ g)
        self.ab += _f64((a.double() * b.double()).sum())


def row_sq_norms(h: Tensor, q: Tensor) -> Tensor:
    """``a_i = h_i^T Q h_i`` for the rows of ``h`` (``= |P h_i|^2``)."""
    return ((h @ q) * h).sum(1)


def _hsic_debiased_stats(
    n: float, sx: Tensor, a1x: Tensor, sy: Tensor, a1y: Tensor, qx: Tensor, qy: Tensor,
    m: Tensor, uxb: Tensor, vya: Tensor, ab: Tensor,
) -> float:
    """Unbiased HSIC of the row Grams ``K = X X^T``, ``L = Y Y^T`` (``x_i = P_x h_i``, ``y_i = P_y g_i``) from the
    d-space sums. With ``K~``, ``L~`` the zero-diagonal Grams::

        sum K~ o L~      = ||P_x M P_y^T||_F^2 - sum a b = sum((Qx M) o (M Qy)) - ab
        1^T K~ 1         = s_x^T Qx s_x - a1,                    (K~ 1)_i = x_i . s_x - a_i
        (K~1).(L~1)      = qh^T M qg - qh . uxb - qg . vya + ab,    qh = Qx s_x,  qg = Qy s_y
        HSIC = (term1 + (1^T K~ 1)(1^T L~ 1) / ((n-1)(n-2)) - 2 / (n-2) (K~1).(L~1)) / (n (n-3)).
    """
    qh, qg = qx @ sx, qy @ sy
    term1 = ((qx @ m) * (m @ qy)).sum() - ab
    ex, ey = sx @ qh - a1x, sy @ qg - a1y
    term2 = ex * ey / ((n - 1) * (n - 2))
    term3 = qh @ m @ qg - qh @ uxb - qg @ vya + ab
    return float((term1 + term2 - (2.0 / (n - 2)) * term3) / (n * (n - 3)))


def self_hsic(sx: SelfStats, q: Tensor, dev: torch.device | str = "cpu") -> float:
    """HSIC(K, K) of one side (the CKA denominator terms)."""
    s, a1, ua = sx.s.to(dev), sx.a1.to(dev), sx.ua.to(dev)
    return _hsic_debiased_stats(float(sx.n), s, a1, s, a1, q, q, sx.S.to(dev), ua, ua, sx.a2.to(dev))


def _check_rows(n: float) -> None:
    if n < 4:
        raise ValueError(f"debiased CKA over rows needs n >= 4, got n={n}")


def cka_rows_from_stats(
    sx: SelfStats, sy: SelfStats, cr: CrossStats, qx: Tensor, qy: Tensor, *, debiased: bool = True,
    hxx: float | None = None, hyy: float | None = None,
) -> float:
    """Linear CKA(D_x, D_y) with rows = matched tokens (measure 2), from streamed statistics. ``hxx``/``hyy``
    (the self terms) may be passed when they are reused (permutation null: the self terms do not change)."""
    dev = qx.device
    n = float(sx.n)
    if debiased:
        _check_rows(n)
        hxy = _hsic_debiased_stats(
            n, sx.s.to(dev), sx.a1.to(dev), sy.s.to(dev), sy.a1.to(dev), qx, qy,
            cr.M.to(dev), cr.uhb.to(dev), cr.vga.to(dev), cr.ab.to(dev),
        )
        hxx = self_hsic(sx, qx, dev) if hxx is None else hxx
        hyy = self_hsic(sy, qy, dev) if hyy is None else hyy
        if hxx <= 0.0 or hyy <= 0.0:
            return float("nan")
        return hxy / (hxx * hyy) ** 0.5
    sxv, syv = sx.s.to(dev), sy.s.to(dev)
    mc = cr.M.to(dev) - torch.outer(sxv, syv) / n
    sxc = sx.S.to(dev) - torch.outer(sxv, sxv) / n
    syc = sy.S.to(dev) - torch.outer(syv, syv) / n
    num = ((qx @ mc) * (mc @ qy)).sum()
    nx = ((qx @ sxc) * (sxc @ qx)).sum().sqrt()
    ny = ((qy @ syc) * (syc @ qy)).sum().sqrt()
    if float(nx) <= 0.0 or float(ny) <= 0.0:
        return float("nan")
    return float(num / (nx * ny))


def agreement_terms(
    cxy_diag: Tensor, cxx_diag: Tensor, cyy_diag: Tensor, sx: Tensor, sy: Tensor, n: float
) -> tuple[float, float, float]:
    """``(cosine, mean per-anchor-column Pearson r, centred cosine)`` from the per-anchor diagonals of ``D_x^T D_y``,
    ``D_x^T D_x``, ``D_y^T D_y`` and the column sums ``sx = 1^T D_x``, ``sy = 1^T D_y`` over the ``n`` matched rows.
    The cosine is the uncentred cosine of the whole matrices (it contains the common-mode / mean shift); the centred
    cosine is the cosine of the matrices with the column means over the rows removed; columns with zero variance
    are left out of the mean r."""
    den = (cxx_diag.sum() * cyy_diag.sum()).sqrt()
    cos = float(cxy_diag.sum() / den) if float(den) > 0 else float("nan")
    cov = cxy_diag - sx * sy / n
    vx = cxx_diag - sx * sx / n
    vy = cyy_diag - sy * sy / n
    den_c = (vx.sum() * vy.sum()).sqrt()
    cos_c = float(cov.sum() / den_c) if float(den_c) > 0 else float("nan")
    ok = (vx > 1e-12 * vx.max().clamp_min(1e-300)) & (vy > 1e-12 * vy.max().clamp_min(1e-300))
    if not bool(ok.any()):
        return cos, float("nan"), cos_c
    r = cov[ok] / (vx[ok] * vy[ok]).sqrt()
    return cos, float(r.mean()), cos_c


def agreement_from_stats(
    sx: SelfStats, sy: SelfStats, cr: CrossStats, p_x: Tensor, p_y: Tensor
) -> tuple[float, float, float]:
    """Measure 3 from streamed statistics (``p_x``, ``p_y``: ``[T, d]`` frames): (cosine, mean r, centred cosine)."""
    dev = p_x.device
    cxy = ((p_x @ cr.M.to(dev)) * p_y).sum(1)
    cxx = ((p_x @ sx.S.to(dev)) * p_x).sum(1)
    cyy = ((p_y @ sy.S.to(dev)) * p_y).sum(1)
    return agreement_terms(cxy, cxx, cyy, p_x @ sx.s.to(dev), p_y @ sy.s.to(dev), float(sx.n))


def match_patch_grid(tokens: Tensor, n_tokens: int) -> Tensor:
    """Resample ``[B, N, d]`` tokens (CLS first, square patch grid) to ``n_tokens`` tokens: bilinear,
    ``align_corners=False``, CLS kept; the identity (the same tensor) when the grids are equal. This is
    ``to_canonical`` of the anchors state with a different target grid. An approximation when the grids differ."""
    from ...rebase.methods._shared import _interp_2d_tokens

    return _interp_2d_tokens(tokens, int(n_tokens))


def image_derangement(n: int, generator: torch.Generator) -> Tensor:
    """A random permutation of ``n`` images without fixed points (one cycle: random order, random shift)."""
    if n < 2:
        return torch.arange(n)
    order = torch.randperm(n, generator=generator)
    shift = int(torch.randint(1, n, (1,), generator=generator))
    sigma = torch.empty(n, dtype=torch.long)
    sigma[order] = order.roll(-shift)
    return sigma


def class_derangement(labels: Sequence[int] | Tensor, generator: torch.Generator, *, max_swaps: int = 64) -> Tensor:
    """A random permutation ``sigma`` of the ``n`` images of a batch such that image ``i`` is paired with an image
    ``sigma[i]`` of a DIFFERENT class (so also without fixed points), positions kept by the caller. Start from a random
    permutation and repair every same-label pair by swapping its target with the target of a random partner position
    that makes neither pair same-label (the number of bad pairs never grows). Exists iff no class has more than ``n / 2``
    images in the batch; when the repair fails (or there are fewer than 2 classes) the cyclic shift of the class-sorted
    order (``s`` in ``[max class size, n - max class size]``, the minimum number of same-class pairs when infeasible) or,
    with one class, ``image_derangement`` is used. Seeded through ``generator`` only."""
    lab = torch.as_tensor(labels).long().flatten()
    n = int(lab.numel())
    classes, inv = torch.unique(lab, return_inverse=True)
    if n < 2 or classes.numel() < 2:
        return image_derangement(n, generator)
    sigma = torch.randperm(n, generator=generator)
    for _ in range(max_swaps * n):
        bad = (lab[sigma] == lab).nonzero().flatten()
        if bad.numel() == 0:
            return sigma
        i = int(bad[int(torch.randint(0, bad.numel(), (1,), generator=generator))])
        ok = ((lab[sigma] != lab[i]) & (lab != lab[sigma[i]])).nonzero().flatten()
        ok = ok[ok != i]
        if ok.numel() == 0:
            break
        k = int(ok[int(torch.randint(0, ok.numel(), (1,), generator=generator))])
        sigma[i], sigma[k] = sigma[k].clone(), sigma[i].clone()
    counts = torch.bincount(inv)
    max_l = int(counts.max())
    hi = max(max_l, n - max_l)
    shift = max_l + int(torch.randint(0, hi - max_l + 1, (1,), generator=generator))
    crank = torch.randperm(classes.numel(), generator=generator)
    key = crank[inv] * n + torch.randperm(n, generator=generator)
    order = torch.argsort(key)
    out = torch.empty(n, dtype=torch.long)
    out[order] = order.roll(-shift)
    return out


# ----------------------------------------------------------------------------------------------------------------
# sampling design: seeded shuffle + class-stratified jackknife chunks
# ----------------------------------------------------------------------------------------------------------------


def chunk_image_counts(batch_sizes: Sequence[int], n_chunks: int) -> list[int]:
    """Images per jackknife chunk (``chunk_of_batch`` over the batches of the given sizes)."""
    n_b = len(batch_sizes)
    counts = [0] * max(1, min(int(n_chunks), n_b))
    for b, n in enumerate(batch_sizes):
        counts[chunk_of_batch(b, n_b, n_chunks)] += int(n)
    return counts


def stratified_chunk_order(
    labels: Sequence[int], batch_sizes: Sequence[int], n_chunks: int, seed: int
) -> tuple[list[int], list[int]]:
    """Order of the images (``order[new_position] = old_position``) such that every jackknife chunk (the consecutive
    batches ``chunk_of_batch`` assigns to it) holds the class proportions of the whole sample (every class count within
    one image of ``n_class * chunk_size / n`` for equal chunks, two images otherwise) and the images inside a chunk (so inside the batches) are in random order.
    Also returns the chunk id of every new position. Construction: the images are sorted by (random class order, random
    image order); the i-th of them goes to the chunk of the i-th entry of the chunk-size-proportional interleaving of
    the chunk ids (round robin for equal chunks); each chunk is then shuffled. Seeded (``seed``) and a pure function of
    the labels, so two models that load the same images get the same order."""
    n = len(labels)
    if sum(int(b) for b in batch_sizes) != n:
        raise ValueError(f"batch sizes sum to {sum(batch_sizes)}, expected {n}")
    sizes = chunk_image_counts(batch_sizes, n_chunks)
    gen = torch.Generator().manual_seed(int(seed))
    lab = torch.as_tensor(list(labels)).long()
    classes, inv = torch.unique(lab, return_inverse=True)
    crank = torch.randperm(classes.numel(), generator=gen)
    key = crank[inv] * n + torch.randperm(n, generator=gen)
    by_class = torch.argsort(key)  # old positions, class-sorted, random inside a class
    # interleaving of the chunk ids proportional to the sizes: the k-th slot of chunk c sits at (k + 1/2) / size_c
    pos = torch.cat([(torch.arange(s, dtype=torch.float64) + 0.5) / s for s in sizes])
    cid = torch.cat([torch.full((s,), c, dtype=torch.long) for c, s in enumerate(sizes)])
    slot = cid[torch.argsort(pos, stable=True)]  # ties (equal sizes) keep the chunk order: round robin
    order: list[int] = []
    chunk_ids: list[int] = []
    for c in range(len(sizes)):
        members = by_class[slot == c]
        members = members[torch.randperm(members.numel(), generator=gen)]
        order += [int(i) for i in members]
        chunk_ids += [c] * int(members.numel())
    return order, chunk_ids


def reorder_batches(
    batches: Sequence[tuple[Tensor, Tensor]], idx: Sequence[int], n_chunks: int, seed: int, *, stratified: bool
) -> tuple[list[tuple[Tensor, Tensor]], list[int], list[int]]:
    """The loaded ``batches`` (images of the positions ``idx``) in the sampling design's order, same batch sizes:
    ``stratified``: seeded shuffle with class-stratified chunks (``stratified_chunk_order``), images, labels and ``idx``
    permuted together; otherwise unchanged. Returns ``(batches, idx, chunk_id of every image in the new order)``."""
    sizes = [int(x.shape[0]) for x, _ in batches]
    if not stratified:
        return list(batches), list(idx), [chunk_of_batch(b, len(sizes), n_chunks) for b, n in enumerate(sizes) for _ in range(n)]
    labels0 = torch.cat([y for _, y in batches]).tolist()
    order, chunk_ids = stratified_chunk_order(labels0, sizes, n_chunks, seed)
    xs, ys = torch.cat([x for x, _ in batches])[order], torch.cat([y for _, y in batches])[order]
    offs = [sum(sizes[:i]) for i in range(len(sizes))]
    return [(xs[o:o + n], ys[o:o + n]) for o, n in zip(offs, sizes, strict=True)], [idx[i] for i in order], chunk_ids


def chunk_class_balance(labels: Sequence[int], chunk_ids: Sequence[int]) -> dict[str, Any]:
    """How far the chunks are from the class proportions of the sample: the largest ``|count_{chunk, class} - n_class
    * size_chunk / n|`` (images), the mean number of classes per chunk and the number of classes."""
    lab = torch.as_tensor(list(labels)).long()
    cid = torch.as_tensor(list(chunk_ids)).long()
    classes, inv = torch.unique(lab, return_inverse=True)
    g = int(cid.max()) + 1
    table = torch.zeros(g, classes.numel(), dtype=torch.float64)
    table.index_put_((cid, inv), torch.ones(len(lab), dtype=torch.float64), accumulate=True)
    expected = table.sum(1, keepdim=True) * table.sum(0, keepdim=True) / float(len(lab))
    return {
        "n_classes": int(classes.numel()), "n_chunks": g,
        "max_abs_dev_from_proportional": float((table - expected).abs().max()),
        "mean_classes_per_chunk": float((table > 0).sum(1).double().mean()),
    }


# ----------------------------------------------------------------------------------------------------------------
# per-block-pair accumulators and their finalisation
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class PairFrame:
    """The frame of one block pair: ``P`` already divided by the decoder scale ``c`` (CPU float64).
    ``anchor_images``: pool-image id of every row of ``P`` (same order); the clusters of the m1 jackknife."""

    target_block: int
    source_block: int
    p_m: Tensor
    p_k: Tensor
    c_m: float
    c_k: float
    anchor_images: Tensor | None = None

    @property
    def n_anchors(self) -> int:
        return int(self.p_m.shape[0])

    def q_m(self, dev) -> Tensor:
        p = self.p_m.to(dev)
        return p.T @ p

    def q_k(self, dev) -> Tensor:
        p = self.p_k.to(dev)
        return p.T @ p


class NativeSide(_Acc):
    """Native-token statistics of one side (one model / expert / base), all images, all native tokens, for m1:
    ``n`` images, ``s_pos`` ``[N, d]`` per-position sums over images, ``S_cls = sum h h^T`` over the CLS rows and
    ``S_patch`` over the patch rows. Three Gram variants of a row group (``cls``, ``patch``, ``all``):

    * ``uncentered``: ``S`` (the decoder-relevant ``D^T D``);
    * ``centered``: ``S - s s^T / n_rows`` (rows centred over the group's rows);
    * ``position_centered``: ``S - sum_p s_p s_p^T / n`` (the mean over images removed per token position; the
      CLS position is its own position).
    """

    names = ("n", "s_pos", "S_cls", "S_patch")

    def __init__(self, n_tok: int, d: int) -> None:
        self.n = torch.zeros((), dtype=torch.float64)
        self.s_pos = torch.zeros(n_tok, d, dtype=torch.float64)
        self.S_cls = torch.zeros(d, d, dtype=torch.float64)
        self.S_patch = torch.zeros(d, d, dtype=torch.float64)

    def update(self, h: Tensor) -> None:
        if h.ndim != 3 or h.shape[1] < 2:
            raise ValueError(f"NativeSide.update needs [B, 1 + patches, d] tokens, got {tuple(h.shape)}")
        d = h.shape[-1]
        self.n += h.shape[0]
        self.s_pos += _f64(h.sum(0))
        c, p = h[:, 0], h[:, 1:].reshape(-1, d)
        self.S_cls += _f64(c.T @ c)
        self.S_patch += _f64(p.T @ p)

    def _positions(self, group: str) -> slice:
        return {"cls": slice(0, 1), "patch": slice(1, None), "all": slice(None)}[group]

    def n_rows(self, group: str) -> float:
        return float(self.n) * self.s_pos[self._positions(group)].shape[0]

    def moments(self, group: str) -> tuple[float, Tensor, Tensor]:
        """``(n_rows, s = sum of the rows, S = sum h h^T)`` of the group."""
        s = self.s_pos[self._positions(group)].sum(0)
        big = {"cls": self.S_cls, "patch": self.S_patch, "all": self.S_cls + self.S_patch}[group]
        return self.n_rows(group), s, big

    def gram_matrix(self, group: str, variant: str) -> Tensor:
        n_rows, s, big = self.moments(group)
        if variant == "uncentered":
            return big
        if variant == "centered":
            return big - torch.outer(s, s) / n_rows
        if variant == "position_centered":
            sp = self.s_pos[self._positions(group)]
            return big - sp.T @ sp / float(self.n)
        raise ValueError(f"unknown m1 variant {variant!r}")


class PairAcc:
    """Streamed row statistics (m2, m3) of one block pair, one variant and one chunk of images."""

    def __init__(self, dm: int, dk: int, row_types: Sequence[str]) -> None:
        self.row_types = tuple(row_types)
        self.sm = {rt: SelfStats(dm) for rt in self.row_types}
        self.sk = {rt: SelfStats(dk) for rt in self.row_types}
        self.cross = {rt: CrossStats(dm, dk) for rt in self.row_types}

    def update(
        self, rows_m: Mapping[str, Tensor], rows_k: Mapping[str, Tensor], q_m: Tensor, q_k: Tensor,
        *, patches_per_image: int, perms: Sequence[Tensor] = (), perm_cross: Mapping[str, Sequence[CrossStats]] | None = None,
    ) -> None:
        for rt in self.row_types:
            h, g = rows_m[rt], rows_k[rt]
            a, b = row_sq_norms(h, q_m), row_sq_norms(g, q_k)
            self.sm[rt].update(h, a)
            self.sk[rt].update(g, b)
            self.cross[rt].update(h, g, a, b)
            per = patches_per_image if rt == "patch" else 1
            for p, sigma in enumerate(perms):
                idx = (sigma[:, None] * per + torch.arange(per)[None]).reshape(-1).to(g.device)
                perm_cross[rt][p].update(h, g[idx], a, b[idx])

    # groups ---------------------------------------------------------------------------------------------------
    def _rts(self, group: str) -> tuple[str, ...]:
        return ("cls", "patch") if group == "all" else (group,)

    def bundle2(self, group: str) -> tuple[SelfStats, SelfStats, CrossStats]:
        rts = self._rts(group)
        return (
            SelfStats.lincomb([self.sm[r] for r in rts]),
            SelfStats.lincomb([self.sk[r] for r in rts]),
            CrossStats.lincomb([self.cross[r] for r in rts]),
        )


def combine_chunks(chunks: Sequence[PairAcc], weights: Sequence[float]) -> PairAcc:
    """Weighted sum of per-chunk accumulators (jackknife: weight 0 deletes a chunk; all ones = the total)."""
    out = copy.copy(chunks[0])
    out.sm = {rt: SelfStats.lincomb([c.sm[rt] for c in chunks], weights) for rt in out.row_types}
    out.sk = {rt: SelfStats.lincomb([c.sk[rt] for c in chunks], weights) for rt in out.row_types}
    out.cross = {rt: CrossStats.lincomb([c.cross[rt] for c in chunks], weights) for rt in out.row_types}
    return out


# --- direct comparison of the encodings (PRIMARY): D_x vs D_y on the same images, anchor-space statistics -----------


DIRECT_VARIANTS = (
    "uncentered", "position_centered", "centered", "row_centered", "mean_anchor_corr", "mean_anchor_corr_position_centered",
)
DIRECT_PRIMARY = "uncentered"  # the RAW cosine of D_m and D_k (what the decoder receives); the other variants are secondary
DIRECT_GROUPS = ("cls", "patch", "all")
CHUNKED_PAIRS = ("true", "other_task", "other_task_rev", "within_m", "within_k", "random_update")
ACC_DTYPE = torch.float64  # per-batch encodings are cast to float64 BEFORE the sums (x.sum, (x*x).sum, (x*y).sum)


def _pos_slice(group: str) -> slice:
    return {"cls": slice(0, 1), "patch": slice(1, None), "all": slice(None)}[group]


class SideStats(_Acc):
    """Anchor-space statistics of one side's encoding ``x = rows P^T`` ``[B, N, T]`` (N matched positions, CLS first):
    ``n`` images, ``s = sum_i x`` and ``q = sum_i x^2`` per (position, anchor), ``r = sum_i xbar^2`` per position
    (``xbar`` the mean over anchors of a row)."""

    names = ("n", "s", "q", "r")

    def __init__(self, n_pos: int, t: int) -> None:
        self.n = torch.zeros((), dtype=torch.float64)
        self.s = torch.zeros(n_pos, t, dtype=torch.float64)
        self.q = torch.zeros(n_pos, t, dtype=torch.float64)
        self.r = torch.zeros(n_pos, dtype=torch.float64)

    def update(self, x: Tensor) -> None:
        x = x.to(ACC_DTYPE)
        xb = x.mean(2)
        self.n += x.shape[0]
        self.s += _f64(x.sum(0))
        self.q += _f64((x * x).sum(0))
        self.r += _f64((xb * xb).sum(0))


class PairCross(_Acc):
    """``c = sum_i x y`` per (position, anchor) and ``rc = sum_i xbar ybar`` per position of two encodings."""

    names = ("c", "rc")

    def __init__(self, n_pos: int, t: int) -> None:
        self.c = torch.zeros(n_pos, t, dtype=torch.float64)
        self.rc = torch.zeros(n_pos, dtype=torch.float64)

    def update(self, x: Tensor, y: Tensor) -> None:
        x, y = x.to(ACC_DTYPE), y.to(ACC_DTYPE)
        self.c += _f64((x * y).sum(0))
        self.rc += _f64((x.mean(2) * y.mean(2)).sum(0))


class DirectAcc:
    """Direct-comparison statistics of one block, one pass and one chunk of images."""

    def __init__(self, n_pos: int, t: int) -> None:
        self.n_pos, self.t = n_pos, t
        self.sides: dict[str, SideStats] = {}
        self.pairs: dict[str, PairCross] = {}

    def update(self, enc: Mapping[str, Tensor], side_tags: Sequence[str], pairs: Sequence[tuple[str, str, str]]) -> None:
        for tag in side_tags:
            self.sides.setdefault(tag, SideStats(self.n_pos, self.t)).update(enc[tag])
        for name, tx, ty in pairs:
            self.pairs.setdefault(name, PairCross(self.n_pos, self.t)).update(enc[tx], enc[ty])


class PermCross:
    """Image-mismatched cross statistics (rows of ``y`` of another image of the same batch, same position), one per
    permutation; the marginals of ``y`` are unchanged by the permutation."""

    def __init__(self, n_perm: int, n_pos: int, t: int) -> None:
        self.c = torch.zeros(n_perm, n_pos, t, dtype=torch.float64)
        self.rc = torch.zeros(n_perm, n_pos, dtype=torch.float64)

    def update(self, x: Tensor, y: Tensor, sigmas: Sequence[Tensor]) -> None:
        x, y = x.to(ACC_DTYPE), y.to(ACC_DTYPE)
        for p, sigma in enumerate(sigmas):
            ys = y[sigma.to(y.device)]
            self.c[p] += _f64((x * ys).sum(0))
            self.rc[p] += _f64((x.mean(2) * ys.mean(2)).sum(0))


@dataclass
class DirectBlock:
    """All direct statistics of one block: per tag and per pair a list over image chunks (a value-only control has one
    chunk), the pair definitions ``name -> (tag_x, tag_y)`` and the image-mismatched statistics of the pair ``true``."""

    sides: dict[str, list[SideStats]]
    pairs: dict[str, list[PairCross]]
    pair_tags: dict[str, tuple[str, str]]
    perm: PermCross | None = None


# --- mean-shift agreement: where the dataset lands / how it is displaced, in anchor coordinates -----------------------

MS_GROUPS = ("cls", "patch", "all")
MS_KINDS = ("displacement", "base_position", "ft_position")
MS_PAIRS = (
    ("other_task", "m_t", "k_o"), ("other_task_rev", "m_o", "k_t"), ("random_update", "m_t", "k_rand"),
)  # comparators of the true pair (m_t, k_t); ``other_task_mean`` is the mean of the first two


class MeanAcc(_Acc):
    """Sums over images (and positions) of the ENCODED rows ``x = rows P^T`` of one side, one chunk of images: ``n`` images,
    ``cls`` (``[T]``, the CLS row) and ``patch`` (``[T]``, sum over images and matched patch positions). ``all`` is
    ``cls + patch``. Fed with the update ``dH`` (displacement) and with the base activations ``H_base`` (position); the
    fine-tuned position is their sum (the encoding is linear)."""

    names = ("n", "cls", "patch")

    def __init__(self, t: int) -> None:
        self.n = torch.zeros((), dtype=torch.float64)
        self.cls = torch.zeros(t, dtype=torch.float64)
        self.patch = torch.zeros(t, dtype=torch.float64)

    def update(self, x: Tensor) -> None:
        if x.ndim != 3 or x.shape[1] < 2:
            raise ValueError(f"MeanAcc.update needs [B, 1 + patches, T] encodings, got {tuple(x.shape)}")
        x = x.to(ACC_DTYPE)
        self.n += x.shape[0]
        self.cls += _f64(x[:, 0].sum(0))
        self.patch += _f64(x[:, 1:].sum((0, 1)))


def _centred_cos(a: Tensor, b: Tensor) -> float:
    """Cosine of the anchor-centred (mean over the T anchors removed) vectors, as ``common_mode``."""
    a, b = a - a.mean(), b - b.mean()
    den = float(a.norm() * b.norm())
    return float(a @ b) / den if den > 0 else float("nan")


def mean_vectors(means: Mapping[str, Sequence[MeanAcc]]) -> dict[str, dict[str, Tensor]]:
    """``{tag: {'cls': [G, T], 'patch': [G, T], 'n': [G]}}`` per-chunk sums of a block's ``MeanAcc`` lists."""
    return {
        tag: {"cls": torch.stack([a.cls for a in accs]), "patch": torch.stack([a.patch for a in accs]),
              "n": torch.stack([a.n for a in accs])}
        for tag, accs in means.items()
    }


def ms_side_chunks(parts: Mapping[str, Mapping[str, Tensor]], kind: str, tag: str, group: str) -> Tensor:
    """Per-chunk encoded vectors ``[G, T]`` (float64) of the side ``tag``: the update (``displacement``), the base
    activations of its model (``base_position``) or base + update (``ft_position``), for the row group ``cls`` /
    ``patch`` / ``all``."""
    def grp(d: Mapping[str, Tensor]) -> Tensor:
        return {"cls": d["cls"], "patch": d["patch"], "all": d["cls"] + d["patch"]}[group].to(torch.float64)

    if kind == "displacement":
        return grp(parts[tag])
    base = grp(parts[f"base_{tag[0]}"])
    if kind == "base_position":
        return base
    if kind == "ft_position":
        return base + grp(parts[tag])
    raise ValueError(f"unknown mean-shift kind {kind!r}")


def ms_side_vector(parts: Mapping[str, Mapping[str, Tensor]], kind: str, tag: str, group: str, w: Tensor | None = None) -> Tensor:
    """Sum over the chunks (weights ``w`` ``[G]``, all ones by default; weight 0 deletes a chunk) of ``ms_side_chunks``."""
    x = ms_side_chunks(parts, kind, tag, group)
    return x.sum(0) if w is None else w.to(torch.float64) @ x


def mean_shift_block(parts: Mapping[str, Mapping[str, Tensor]], *, ci: bool = True) -> dict[str, Any]:
    """Mean-shift agreement of one block, from the per-chunk vectors ``mean_vectors``. For every ``kind`` (update /
    base position / fine-tuned position) and row group the cosine of the anchor-centred T-vectors of model m (``m_t``) and
    model k (``k_t``), with a delete-one-chunk jackknife interval (``t_{G-1}``); for ``displacement`` and ``ft_position``
    also the comparators ``other_task`` (m_t, k_o), ``other_task_rev`` (m_o, k_t), their mean and ``random_update`` (draw
    0) and the excesses true - comparator on the same deleted chunks (``base_position`` has no comparator: the base is
    the same). Keys follow the direct measure (``excess_other_task_mean``, ``true_minus_random_draw0``)."""
    g = int(parts["m_t"]["cls"].shape[0])
    do_ci = ci and g >= 2
    ws: list[Tensor | None] = [None]
    if do_ci:
        ws += [torch.ones(g, dtype=torch.float64).index_fill_(0, torch.tensor([d]), 0.0) for d in range(g)]

    def reps(kind: str, group: str, tx: str, ty: str) -> Tensor:
        return torch.tensor(
            [_centred_cos(ms_side_vector(parts, kind, tx, group, w), ms_side_vector(parts, kind, ty, group, w)) for w in ws],
            dtype=torch.float64,
        )

    out: dict[str, Any] = {"n_chunks": g, "images_per_chunk": [float(x) for x in parts["m_t"]["n"]]}
    for kind in MS_KINDS:
        out[kind] = {}
        for group in MS_GROUPS:
            true = reps(kind, group, "m_t", "k_t")
            e = _ci_entry(true)
            if kind != "base_position":
                comps: dict[str, Tensor] = {}
                for name, tx, ty in MS_PAIRS:
                    if tx in parts and ty in parts:
                        comps[name] = reps(kind, group, tx, ty)
                for name, rep in comps.items():
                    e[name] = _ci_entry(rep)
                    e["true_minus_random_draw0" if name == "random_update" else f"excess_{name}"] = _ci_entry(true - rep)
                if "other_task" in comps and "other_task_rev" in comps:
                    mean = (comps["other_task"] + comps["other_task_rev"]) / 2.0
                    e["other_task_mean"] = _ci_entry(mean)
                    e["excess_other_task_mean"] = _ci_entry(true - mean)
            out[kind][group] = e
    return out


def _ratio(num: Tensor, a: Tensor, b: Tensor) -> float:
    den = float(a) * float(b)
    return float(num) / den**0.5 if den > 0 else float("nan")


def _mean_r(cov: Tensor, vx: Tensor, vy: Tensor) -> float:
    ok = (vx > 1e-12 * vx.max().clamp_min(1e-300)) & (vy > 1e-12 * vy.max().clamp_min(1e-300))
    if not bool(ok.any()):
        return float("nan")
    return float((cov[ok] / (vx[ok] * vy[ok]).sqrt()).mean())


def direct_values(sx: SideStats, sy: SideStats, pc: PairCross, group: str) -> dict[str, float]:
    """The direct agreement of ``D_x`` and ``D_y`` (rows matched by image and position, columns = the same anchors) for
    a row group, from anchor-space sums. The update enters ``D`` linearly, so a shared frame alone does not make it high.

    * ``position_centered`` (PRIMARY): global Pearson correlation after removing, per anchor column, the mean over
      images at each token position (CLS its own position);
    * ``uncentered``: cosine of the whole matrices (the decoder-relevant number, contains the mean shift);
    * ``centered``: Pearson correlation after removing the column mean over all rows of the group;
    * ``row_centered``: cosine after removing each row's mean over the anchors (common mode across anchors);
    * ``mean_anchor_corr`` / ``mean_anchor_corr_position_centered``: the per-anchor Pearson r (column-centred over all
      rows / per position), averaged over anchors with non-zero variance."""
    sl = _pos_slice(group)
    n = float(sx.n)
    t = sx.s.shape[1]
    s_x, s_y, c_xx, c_yy, c_xy = sx.s[sl], sy.s[sl], sx.q[sl], sy.q[sl], pc.c[sl]
    n_rows = n * s_x.shape[0]
    out = {"uncentered": _ratio(c_xy.sum(), c_xx.sum(), c_yy.sum())}
    sxa, sya = s_x.sum(0), s_y.sum(0)
    cov, vx, vy = c_xy.sum(0) - sxa * sya / n_rows, c_xx.sum(0) - sxa**2 / n_rows, c_yy.sum(0) - sya**2 / n_rows
    out["centered"] = _ratio(cov.sum(), vx.sum(), vy.sum())
    out["mean_anchor_corr"] = _mean_r(cov, vx, vy)
    covp, vxp, vyp = c_xy - s_x * s_y / n, c_xx - s_x**2 / n, c_yy - s_y**2 / n
    out["position_centered"] = _ratio(covp.sum(), vxp.sum(), vyp.sum())
    out["mean_anchor_corr_position_centered"] = _mean_r(covp.sum(0), vxp.sum(0), vyp.sum(0))
    out["row_centered"] = _ratio(
        c_xy.sum() - t * pc.rc[sl].sum(), c_xx.sum() - t * sx.r[sl].sum(), c_yy.sum() - t * sy.r[sl].sum()
    )
    return {k: out[k] for k in DIRECT_VARIANTS}


def _comb(items: Sequence[Any], drop: int | None, g: int):
    """Sum of the chunk statistics, deleting chunk ``drop`` when the list has the ``g`` main chunks."""
    w = [1.0] * len(items)
    if drop is not None and len(items) == g:
        w[drop] = 0.0
    return items[0].__class__.lincomb(items, w)


def _is_extra_draw(name: str) -> bool:
    return name.startswith("random_update_d") and name[len("random_update_d"):].isdigit()


def _draw_index(name: str) -> int:
    return int(name[len("random_update_d"):])


def _draw_summary(values: Sequence[float], true: float, antithetic: bool = False) -> dict[str, Any]:
    """The random-update draws as a NULL DISTRIBUTION for the cosine ``true``. ``values``: per-draw cosines on all images
    (draw order). Independent draws: every draw is a group, and

    * ``null_mean`` / ``null_sd`` (ddof = 1; ``None`` with fewer than 2 groups) / ``null_max_abs`` over the draws;
    * ``z_random = (true - null_mean) / null_sd`` (``None`` without a spread);
    * ``p_random = (1 + #{cos >= true}) / (1 + N)`` and the two-sided ``p_random_two_sided``, centred on the null:
      ``(1 + #{|cos - null_mean| >= |true - null_mean|}) / (1 + N)``;
    * z and p are ``None`` when ``true`` or any draw is not finite;
    * ``mean`` / ``min`` / ``max`` and the per-draw ``values``.

    Antithetic draws ``(+g, -g)`` in pairs are mirrored, not independent: z and p are ``None`` (``z_p_note``); the
    summary keeps the mean over all draws, ``null_mean`` / ``null_sd`` over the ``+g`` draws (``n_groups = N / 2``) and
    ``even`` / ``odd`` per pair, ``(c+ + c-)/2`` and ``(c+ - c-)/2``."""
    vals = [float(x) for x in values]
    n = len(vals)
    if antithetic and n % 2:
        raise ValueError("antithetic draws come in pairs")
    groups = vals[0::2] if antithetic else vals
    k = len(groups)
    mean = sum(groups) / k
    sd = (sum((x - mean) ** 2 for x in groups) / (k - 1)) ** 0.5 if k > 1 else None
    finite = math.isfinite(true) and all(math.isfinite(x) for x in vals)
    out: dict[str, Any] = {
        "n": n, "n_groups": k, "antithetic": bool(antithetic), "values": vals, "mean": sum(vals) / n, "min": min(vals),
        "max": max(vals), "null_mean": mean, "null_sd": sd, "null_max_abs": max(abs(x) for x in groups), "sd_ddof": 1,
        "z_random": None, "p_random": None, "p_random_two_sided": None,
    }
    if antithetic:
        out["z_p_note"] = "antithetic draws are mirrored (not independent): no z / p"
        out["even"] = [(vals[i] + vals[i + 1]) / 2.0 for i in range(0, n, 2)]
        out["odd"] = [(vals[i] - vals[i + 1]) / 2.0 for i in range(0, n, 2)]
    elif finite:
        out["z_random"] = (true - mean) / sd if sd else None
        out["p_random"] = (1 + sum(1 for x in groups if x >= true)) / (1 + k)
        out["p_random_two_sided"] = (1 + sum(1 for x in groups if abs(x - mean) >= abs(true - mean))) / (1 + k)
    return out


def direct_block(db: DirectBlock, opts: FinalizeOptions) -> tuple[dict[str, Any], dict[str, Any]]:
    """The primary measure of one block: per variant and group the value, its delete-one-chunk jackknife interval
    (``t_{G-1}``), the comparators on the same images (``other_task``: model k with the other expert; ``within_model_
    cross_task``: D_m(t) vs D_m(t') and D_k(t) vs D_k(t') averaged; ``random_update``) and the paired excesses
    ``true - comparator`` (same deleted chunks), plus the image-mismatched null (sanity). Returns ``(entries,
    controls)``; ``controls[name][variant][group]`` are the values of every non-main pair (value-only controls
    included)."""
    g_chunks = len(db.pairs["true"])

    def values(drop: int | None) -> dict[str, dict[str, dict[str, float]]]:
        res: dict[str, dict[str, dict[str, float]]] = {}
        for name, (tx, ty) in db.pair_tags.items():
            if _is_extra_draw(name):  # null draws >= 1 (one chunk): only the all-images value, computed below in ``total``
                continue
            sx, sy, pc = _comb(db.sides[tx], drop, g_chunks), _comb(db.sides[ty], drop, g_chunks), _comb(db.pairs[name], drop, g_chunks)
            res[name] = {g: direct_values(sx, sy, pc, g) for g in DIRECT_GROUPS}
        return res

    total = values(None)
    do_ci = opts.ci and g_chunks >= 2
    loo = [values(d) for d in range(g_chunks)] if do_ci else []
    chunked = [n for n in CHUNKED_PAIRS if n in db.pair_tags and len(db.pairs[n]) == g_chunks]
    # random-update NULL draws: draw 0 is ``random_update`` (chunked: value, jackknife CI, m1 / m2), the others
    # ``random_update_d<i>`` (one chunk each, only the all-images cosine); z and p use the cosines over all images
    draw_names = [n for n in ("random_update", *sorted((n for n in db.pair_tags if _is_extra_draw(n)), key=_draw_index))
                  if n in db.pair_tags]
    for n in draw_names:
        if n not in total:
            tx, ty = db.pair_tags[n]
            sx, sy, pc = _comb(db.sides[tx], None, g_chunks), _comb(db.sides[ty], None, g_chunks), _comb(db.pairs[n], None, g_chunks)
            total[n] = {g: direct_values(sx, sy, pc, g) for g in DIRECT_GROUPS}

    def replicates(fn, g, v) -> tuple[float, list[float]]:
        return fn(total, g, v), [fn(r, g, v) for r in loo]

    def ent(theta_loo: tuple[float, list[float]]) -> dict[str, Any]:
        e: dict[str, Any] = {"value": theta_loo[0]}
        if do_ci:
            e.update(jackknife_ci(theta_loo[0], theta_loo[1]))
        return e

    def get(name):
        return lambda r, g, v: r[name][g][v]

    def within(r, g, v):
        return (r["within_m"][g][v] + r["within_k"][g][v]) / 2.0

    def other_mean(r, g, v):
        return (r["other_task"][g][v] + r["other_task_rev"][g][v]) / 2.0

    def diff(fa, fb):
        return lambda r, g, v: fa(r, g, v) - fb(r, g, v)

    entries: dict[str, Any] = {v: {} for v in DIRECT_VARIANTS}
    has_within = "within_m" in chunked and "within_k" in chunked
    for v in DIRECT_VARIANTS:
        for g in DIRECT_GROUPS:
            e = ent(replicates(get("true"), g, v))
            for name in ("other_task", "other_task_rev", "random_update"):
                if name in chunked:
                    fn = get(name)  # random_update: draw 0 alone (the chunked draw)
                    e[name] = ent(replicates(fn, g, v))
                    # excess_* for other tasks; for random_update: true - draw 0, reference only (not a chance-level test; see random_update_draws z/p)
                    e["true_minus_random_draw0" if name == "random_update" else f"excess_{name}"] = ent(
                        replicates(diff(get("true"), fn), g, v))
                    if name == "random_update":
                        e["random_update_draws"] = _draw_summary([total[n][g][v] for n in draw_names], total["true"][g][v], opts.random_antithetic)
            if "other_task" in chunked and "other_task_rev" in chunked:  # mean of (m_t,k_o) and the mirror (m_o,k_t)
                e["other_task_mean"] = ent(replicates(other_mean, g, v))
                e["excess_other_task_mean"] = ent(replicates(diff(get("true"), other_mean), g, v))
            if has_within:
                e["within_model_cross_task"] = {
                    **ent(replicates(within, g, v)), "m": total["within_m"][g][v], "k": total["within_k"][g][v],
                }
                e["excess_within_model_cross_task"] = ent(replicates(diff(get("true"), within), g, v))
            entries[v][g] = e
    if opts.null and db.perm is not None:  # sanity: images mismatched, positions matched
        tx, ty = db.pair_tags["true"]
        sx, sy = _comb(db.sides[tx], None, g_chunks), _comb(db.sides[ty], None, g_chunks)
        samples: dict[str, dict[str, list[float]]] = {v: {g: [] for g in DIRECT_GROUPS} for v in DIRECT_VARIANTS}
        for p in range(db.perm.c.shape[0]):
            pc = PairCross.__new__(PairCross)
            pc.c, pc.rc = db.perm.c[p], db.perm.rc[p]
            for g in DIRECT_GROUPS:
                for v, x in direct_values(sx, sy, pc, g).items():
                    samples[v][g].append(x)
        for v in DIRECT_VARIANTS:
            for g in DIRECT_GROUPS:
                entries[v][g]["image_null"] = _null_summary(samples[v][g], entries[v][g]["value"])
    controls: dict[str, Any] = {}
    for name in db.pair_tags:
        if name in ("true", "within_m", "within_k") or _is_extra_draw(name):
            continue
        controls[name] = {v: {g: total[name][g][v] for g in DIRECT_GROUPS} for v in DIRECT_VARIANTS}
    return entries, controls


def chunk_of_batch(b: int, n_batches: int, n_chunks: int) -> int:
    """Chunk index of batch ``b``: ``min(n_chunks, n_batches)`` non-empty consecutive chunks (the number of chunks
    is exactly the requested one whenever ``n_batches >= n_chunks``; chunk sizes differ by at most one batch)."""
    g = max(1, min(int(n_chunks), int(n_batches)))
    return (int(b) * g) // int(n_batches)


def _quant(samples: Sequence[float], q: float) -> float:
    t = torch.tensor([s for s in samples if math.isfinite(s)], dtype=torch.float64)
    return float(torch.quantile(t, q)) if t.numel() else float("nan")


def _null_summary(samples: Sequence[float], value: float | None = None) -> dict[str, Any]:
    """Summary of a permutation null: q95, mean, std, and (given the observed ``value``) the z-score
    ``(value - mean) / std`` and the empirical one-sided p ``(1 + #{null >= value}) / (1 + n)``."""
    fin = [s for s in samples if math.isfinite(s)]
    t = torch.tensor(fin, dtype=torch.float64)
    out: dict[str, Any] = {
        "q95": _quant(samples, 0.95),
        "mean": float(t.mean()) if t.numel() else float("nan"),
        "std": float(t.std()) if t.numel() > 1 else float("nan"),
        "n_perm": len(samples),
    }
    if value is not None and math.isfinite(value):
        std = out["std"]
        out["z"] = (value - out["mean"]) / std if math.isfinite(std) and std > 0 else float("nan")
        out["p_empirical"] = (1 + sum(1 for s in fin if s >= value)) / (1 + len(fin))
    return out


def jackknife_ci(theta: float, loo: Sequence[float]) -> dict[str, float]:
    """Delete-one-group jackknife interval around ``theta``: ``se^2 = (G-1)/G sum (loo_g - mean(loo))^2`` and
    ``theta +- t_{G-1, 0.975} se``. Non-finite leave-one-out values are dropped (G counts the finite ones)."""
    from scipy.stats import t as student_t

    vals = torch.tensor([x for x in loo if math.isfinite(x)], dtype=torch.float64)
    g = int(vals.numel())
    if g < 2 or not math.isfinite(theta):
        return {"lo": float("nan"), "hi": float("nan"), "se": float("nan"), "n_groups": g}
    se = math.sqrt((g - 1) / g * float(((vals - vals.mean()) ** 2).sum()))
    q = float(student_t.ppf(0.975, g - 1))
    return {"lo": theta - q * se, "hi": theta + q * se, "se": se, "n_groups": g}


@dataclass
class FinalizeOptions:
    groups_rows: tuple[str, ...] = ("cls", "patch", "all")
    debiased: bool = True
    ci: bool = False
    null: bool = False
    n_perm: int = 200
    seed: int = 0
    device: str = "cpu"
    random_antithetic: bool = False  # the random_update draws come in antithetic pairs (+g, -g)
    center_rows_m1: bool = True  # primary m1 = row-centred; the uncentred m1 is always also reported

    @property
    def primary_variant(self) -> str:
        return "centered" if self.center_rows_m1 else "uncentered"


def _gram(p: Tensor, s: Tensor) -> Tensor:
    return p @ s.to(device=p.device, dtype=p.dtype) @ p.T


# --- m1: masked (delete-one-cluster) debiased CKA over the anchors -------------------------------------------------


def cluster_weights(anchor_images: Tensor | None, n_anchors: int, device, *, deletions: bool = True) -> Tensor:
    """``[1 + G, T]`` 0/1 anchor masks: row 0 keeps everything, row ``1 + g`` deletes all anchors of image cluster
    ``g`` (anchors of one image stay together). ``anchor_images=None``: every anchor is its own cluster."""
    full = torch.ones(1, n_anchors, dtype=torch.float64)
    if not deletions:
        return full.to(device)
    if anchor_images is None:
        codes = torch.arange(n_anchors)
    else:
        if int(anchor_images.numel()) != n_anchors:
            raise ValueError(f"anchor_images has {int(anchor_images.numel())} entries, expected {n_anchors}")
        _, codes = torch.unique(anchor_images.cpu(), return_inverse=True)
    n_cl = int(codes.max()) + 1
    onehot = torch.zeros(n_cl, n_anchors, dtype=torch.float64)
    onehot[codes, torch.arange(n_anchors)] = 1.0
    return torch.cat([full, 1.0 - onehot], dim=0).to(device)


def _masked_hsic(a: Tensor, b: Tensor, w: Tensor) -> Tensor:
    """Unbiased HSIC (Song et al. 2012) of zero-diagonal symmetric Grams ``a``, ``b`` restricted to the anchors of every
    0/1 mask row of ``w`` (``[R, T]``): identical to ``_hsic_unbiased_zd(a[idx][:, idx], b[idx][:, idx])`` per row."""
    n = w.sum(1)
    term1 = ((w @ (a * b)) * w).sum(1)
    aw, bw = w @ a, w @ b
    term2 = (aw * w).sum(1) * (bw * w).sum(1) / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * (aw * bw * w).sum(1)
    return (term1 + term2 - term3) / (n * (n - 3))


class KernelSet:
    """Gram matrices over the anchors that enter several CKA terms; ``cka(a, b)`` returns the CKA of every mask
    row (the same deleted clusters for every pair of terms, so differences of terms are paired)."""

    def __init__(self, w: Tensor, debiased: bool) -> None:
        self.w, self.debiased = w, debiased
        self.g: dict[str, Tensor] = {}
        self.self_: dict[str, Tensor] = {}

    def add(self, name: str, gram: Tensor) -> None:
        g = gram.to(torch.float64)
        if self.debiased:
            g = _zero_diag(g)
            self.self_[name] = _masked_hsic(g, g, self.w)
        self.g[name] = g

    def cka(self, a: str, b: str) -> Tensor:
        if self.debiased:
            num = _masked_hsic(self.g[a], self.g[b], self.w)
            den = (self.self_[a] * self.self_[b]).clamp_min(0.0).sqrt()
            out = num / den
            out[den <= 0] = float("nan")
            return out
        vals = []
        for row in self.w:
            idx = row.nonzero().flatten()
            vals.append(gram_cka(self.g[a][idx][:, idx], self.g[b][idx][:, idx], debiased=False))
        return torch.tensor(vals, dtype=torch.float64)


def _ci_entry(rep: Tensor) -> dict[str, Any]:
    theta = float(rep[0])
    entry: dict[str, Any] = {"value": theta}
    if rep.numel() > 1:
        entry.update(jackknife_ci(theta, rep[1:].tolist()))
    return entry


def m1_block(
    frame: PairFrame, sides: Mapping[str, NativeSide], opts: FinalizeOptions, *,
    control_tags: Mapping[str, str] | None = None, perms: Sequence[Tensor] = (),
) -> dict[str, Any]:
    """m1 of one block pair for the three Gram variants and the row groups ``all, cls, patch``.

    ``sides``: ``m_t``, ``k_t`` (required), ``m_o``, ``k_o`` (expert of the other task on the SAME task images),
    ``base_m``, ``base_k`` (base activations), and one tag per control. Per (variant, group) the entry holds the
    value and the delete-one-image-cluster jackknife interval (``t_{G-1}``), the reference terms (other task =
    model k with the other expert; within-model cross-task = CKA(K_m(t), K_m(t')) and CKA(K_k(t), K_k(t')), mean
    of both; base representation) and the excesses ``m1 - reference``, all on the same deleted clusters."""
    dev = torch.device(opts.device)
    p_m, p_k = frame.p_m.to(dev), frame.p_k.to(dev)
    w = cluster_weights(frame.anchor_images, frame.n_anchors, dev, deletions=opts.ci)
    ctl = dict(control_tags or {})
    out: dict[str, Any] = {
        "variants": {v: {} for v in M1_VARIANTS}, "controls": {n: {v: {} for v in M1_VARIANTS} for n in ctl},
        "n_clusters": int(w.shape[0] - 1),
    }
    for variant in M1_VARIANTS:
        for g in ("all", "cls", "patch"):
            def gk(tag: str, p: Tensor, _g=g, _v=variant) -> Tensor:
                return _gram(p, sides[tag].gram_matrix(_g, _v).to(dev))

            ks = KernelSet(w, opts.debiased)
            k_mt, k_kt = gk("m_t", p_m), gk("k_t", p_k)
            ks.add("m_t", k_mt)
            ks.add("k_t", k_kt)
            terms: dict[str, tuple[str, str]] = {"true": ("m_t", "k_t")}
            if "k_o" in sides:
                ks.add("k_o", gk("k_o", p_k))
                terms["other_task"] = ("m_t", "k_o")
                if "m_o" in sides:
                    ks.add("m_o", gk("m_o", p_m))
                    terms["within_m"], terms["within_k"] = ("m_t", "m_o"), ("k_t", "k_o")
            if "base_m" in sides and "base_k" in sides:
                ks.add("b_m", gk("base_m", p_m))
                ks.add("b_k", gk("base_k", p_k))
                terms["base_repr"] = ("b_m", "b_k")
            reps = {name: ks.cka(*ab) for name, ab in terms.items()}
            entry = _ci_entry(reps["true"])
            entry["n_anchors"] = frame.n_anchors
            if "other_task" in reps:
                entry["other_task"] = _ci_entry(reps["other_task"])
                entry["excess_other_task"] = _ci_entry(reps["true"] - reps["other_task"])
            if "within_m" in reps:
                within = (reps["within_m"] + reps["within_k"]) / 2.0
                entry["within_model_cross_task"] = {
                    **_ci_entry(within), "m": float(reps["within_m"][0]), "k": float(reps["within_k"][0]),
                }
                entry["excess_within_model_cross_task"] = _ci_entry(reps["true"] - within)
            if "base_repr" in reps:
                entry["base_repr"] = _ci_entry(reps["base_repr"])
                entry["excess_base_repr"] = _ci_entry(reps["true"] - reps["base_repr"])
            if opts.null and variant == opts.primary_variant and perms:
                px, py = GramPrep(k_mt, debiased=opts.debiased), GramPrep(k_kt, debiased=opts.debiased)
                entry["null"] = _null_summary([px.cka_with(py, perm=pi) for pi in perms], entry["value"])
            for name, tag in ctl.items():
                out["controls"][name][variant][g] = gram_cka(k_mt, gk(tag, p_k), debiased=opts.debiased)
            out["variants"][variant][g] = entry
            del ks, k_mt, k_kt
    return out


def common_mode(frame: PairFrame, sides: Mapping[str, NativeSide], device: str = "cpu") -> dict[str, float]:
    """Common-mode (mean-shift) diagnostics of one block pair, all native tokens.

    ``rho_m = n h_bar^T Q h_bar / tr(Q S)`` with ``Q = P^T P``, ``h_bar`` the mean update row, ``S = sum h h^T``
    (the fraction of the uncentred update energy in anchor space that is the task-independent-or-not mean shift),
    per side; ``cos_mean_*``: cosine between the anchor-centred ``P_m h_bar_m`` and ``P_k h_bar_k`` for the true
    task and (model k with the other expert on the same images) for the other task."""
    dev = torch.device(device)
    p_m, p_k = frame.p_m.to(dev), frame.p_k.to(dev)

    def rho_and_vec(tag: str, p: Tensor) -> tuple[float, Tensor]:
        n_rows, s, big = sides[tag].moments("all")
        s, big = s.to(dev), big.to(dev)
        ps = p @ s
        rho = float((ps @ ps) / n_rows / ((p @ big @ p.T).trace()))
        v = ps / n_rows
        return rho, v - v.mean()

    def cos(a: Tensor, b: Tensor) -> float:
        den = float(a.norm() * b.norm())
        return float(a @ b) / den if den > 0 else float("nan")

    rho_m, v_m = rho_and_vec("m_t", p_m)
    rho_k, v_k = rho_and_vec("k_t", p_k)
    out = {"rho_m": rho_m, "rho_k": rho_k, "cos_mean_true_task": cos(v_m, v_k)}
    if "k_o" in sides:
        rho_ko, v_ko = rho_and_vec("k_o", p_k)
        out["rho_k_other_task"] = rho_ko
        out["cos_mean_other_task"] = cos(v_m, v_ko)
    return out


def groups_rt(group: str) -> tuple[str, ...]:
    return ("cls", "patch") if group == "all" else (group,)


def finalize_pair(
    chunks: Sequence[PairAcc],
    frame: PairFrame,
    opts: FinalizeOptions,
    perm_cross: Mapping[str, Sequence[CrossStats]] | None = None,
    native: Mapping[str, NativeSide] | None = None,
    control_tags: Mapping[str, str] | None = None,
    direct: DirectBlock | None = None,
) -> dict[str, Any]:
    """The measures of one block pair for one variant, the PRIMARY first.

    * ``direct`` (needs ``direct``): the direct comparison of the encodings, see ``direct_block``.
    * ``m1``/``m1_variants`` (needs ``native``): CKA of the Gram kernels on the anchors, SECONDARY, see ``m1_block``
      (``m1`` is the primary Gram variant, row-centred by default).
    * ``m2``: CKA over matched rows, SECONDARY (delete-one-chunk jackknife with ``opts.ci``; image-mismatched null).
    The d-space streamed statistics also give the anchor-permutation null of the direct uncentred / centred values."""
    dev = torch.device(opts.device)
    p_m, p_k = frame.p_m.to(dev), frame.p_k.to(dev)
    q_m, q_k = p_m.T @ p_m, p_k.T @ p_k
    groups2 = tuple(opts.groups_rows)
    out: dict[str, Any] = {}
    gen_cpu = torch.Generator().manual_seed(int(opts.seed))
    perms_t = [torch.randperm(frame.n_anchors, generator=gen_cpu).to(dev) for _ in range(opts.n_perm)] if opts.null else []
    direct_controls: dict[str, Any] = {}
    if direct is not None:
        out["direct"], direct_controls = direct_block(direct, opts)
        out["direct_primary_variant"] = DIRECT_PRIMARY
        out["direct_controls"] = direct_controls
    if native is not None:
        m1 = m1_block(frame, native, opts, control_tags=control_tags, perms=perms_t)
        out["m1_variants"] = m1["variants"]
        out["m1"] = m1["variants"][opts.primary_variant]
        out["m1_controls"] = m1["controls"]
        out["m1_n_clusters"] = m1["n_clusters"]
        out["m1_primary_variant"] = opts.primary_variant
        if "m_t" in native and "k_t" in native:
            out["common_mode"] = common_mode(frame, native, opts.device)
    out["m2"] = {}
    if not chunks:  # m2 off (config m2: false): no d-space row statistics, hence no m2 and no anchor-permutation null
        if direct is not None:
            side = direct.sides[direct.pair_tags["true"][0]][0]
            out["n_rows"] = {g: float(sum(x.n for x in direct.sides[direct.pair_tags["true"][0]])) * side.s[_pos_slice(g)].shape[0]
                             for g in DIRECT_GROUPS}
        return out
    total = combine_chunks(chunks, [1.0] * len(chunks))

    # --- m2: matched rows, delete-one-chunk jackknife --------------------------------------------------------------
    loo: dict[str, list[float]] = {g: [] for g in groups2}
    if opts.ci and len(chunks) >= 2:
        for drop in range(len(chunks)):
            w = [1.0] * len(chunks)
            w[drop] = 0.0
            bx = combine_chunks(chunks, w)
            for g in groups2:
                bsx, bsy, bcr = bx.bundle2(g)
                loo[g].append(
                    cka_rows_from_stats(bsx, bsy, bcr, q_m, q_k, debiased=opts.debiased) if float(bsx.n) >= 4 else float("nan")
                )
    for g in groups2:
        sx, sy, cr = total.bundle2(g)
        e2: dict[str, Any] = {"value": cka_rows_from_stats(sx, sy, cr, q_m, q_k, debiased=opts.debiased)}
        if loo[g]:
            e2.update(jackknife_ci(e2["value"], loo[g]))
        if opts.null:
            if perm_cross:  # m2: images mismatched, positions matched (self terms unchanged, only the pairing changes)
                hxx, hyy = self_hsic(sx, q_m, dev), self_hsic(sy, q_k, dev)
                rts = groups_rt(g)
                samples = []
                for p in range(len(perm_cross[rts[0]])):
                    cr_p = CrossStats.lincomb([perm_cross[r][p] for r in rts])
                    samples.append(cka_rows_from_stats(sx, sy, cr_p, q_m, q_k, debiased=opts.debiased, hxx=hxx, hyy=hyy))
                e2["null"] = _null_summary(samples, e2["value"])
            if direct is not None and g in DIRECT_GROUPS:  # anchor-permutation null (columns of D_k permuted), sanity
                cxy_full = p_m @ cr.M.to(dev) @ p_k.T
                cxx = ((p_m @ sx.S.to(dev)) * p_m).sum(1)
                cyy = ((p_k @ sy.S.to(dev)) * p_k).sum(1)
                sxv, syv, n = p_m @ sx.s.to(dev), p_k @ sy.s.to(dev), float(sx.n)
                ar = torch.arange(frame.n_anchors, device=dev)
                cs, rs, ccs = [], [], []
                for pi in perms_t:
                    c_, r_, cc_ = agreement_terms(cxy_full[ar, pi], cxx, cyy[pi], sxv, syv[pi], n)
                    cs.append(c_)
                    rs.append(r_)
                    ccs.append(cc_)
                dv = out["direct"]
                dv["uncentered"][g]["anchor_null"] = _null_summary(cs, dv["uncentered"][g]["value"])
                dv["centered"][g]["anchor_null"] = _null_summary(ccs, dv["centered"][g]["value"])
                dv["mean_anchor_corr"][g]["anchor_null"] = _null_summary(rs, dv["mean_anchor_corr"][g]["value"])
                del cxy_full
        out["m2"][g] = e2
    out["n_rows"] = {g: float(total.bundle2(g)[0].n) for g in groups2}
    return out


def base_reference(frame: PairFrame, *, debiased: bool, ci: bool, device: str) -> dict[str, float]:
    """Sanity reference: CKA of the anchor-frame kernels ``P_m P_m^T`` vs ``P_k P_k^T`` (no update, scale-free); the
    jackknife interval deletes image clusters of anchors. Not a comparator of m1 (see ``m1_block`` references)."""
    dev = torch.device(device)
    p_m, p_k = frame.p_m.to(dev), frame.p_k.to(dev)
    w = cluster_weights(frame.anchor_images, frame.n_anchors, dev, deletions=ci)
    ks = KernelSet(w, debiased)
    ks.add("a", p_m @ p_m.T)
    ks.add("b", p_k @ p_k.T)
    return _ci_entry(ks.cka("a", "b"))


# ----------------------------------------------------------------------------------------------------------------
# controls on the weights
# ----------------------------------------------------------------------------------------------------------------


def random_update_state(
    base: Mapping[str, Tensor], tuned: Mapping[str, Tensor], *, seed: int, sign: float = 1.0
) -> tuple[dict[str, Tensor], dict[str, dict[str, float]]]:
    """``base + r`` per parameter tensor with ``r`` a Gaussian direction rescaled to ``||tau||_F`` of the real
    ``tau = tuned - base`` of that tensor (tensors with ``tau = 0`` stay at base). Seeded; keys in sorted order.
    Returns the updated entries (cast to the model dtype) and ``{key: {tau_norm, random_norm, cast_norm,
    rel_norm_error}}``: ``random_norm`` is the norm of the float64 ``r``, ``cast_norm`` the norm of
    ``out[key] - base[key]`` AFTER the cast to the model dtype, and ``rel_norm_error = |cast_norm - tau_norm| /
    tau_norm`` is the error of the update actually applied.

    ``sign`` (+1 / -1) multiplies ``r`` after the rescaling: the same ``seed`` with ``sign=-1`` gives exactly ``-r``
    (the antithetic partner of a draw, same per-tensor norm); ``sign=1`` is the original behaviour."""
    gen = torch.Generator().manual_seed(int(seed))
    out: dict[str, Tensor] = {}
    norms: dict[str, dict[str, float]] = {}
    for key in sorted(tuned):
        if key not in base or not torch.is_floating_point(base[key]):
            continue
        tau = tuned[key].to(torch.float64) - base[key].to(torch.float64)
        tn = float(tau.norm())
        if tn == 0.0:
            continue
        r = torch.randn(tau.shape, generator=gen, dtype=torch.float64)
        r = r * (tn / float(r.norm()))
        if sign != 1.0:
            r = r * float(sign)
        out[key] = (base[key].to(torch.float64) + r).to(base[key].dtype)
        cast = float((out[key].to(torch.float64) - base[key].to(torch.float64)).norm())
        norms[key] = {"tau_norm": tn, "random_norm": float(r.norm()), "cast_norm": cast, "rel_norm_error": abs(cast - tn) / tn}
    return out, norms


RANDOM_SEED_STRIDE = 1_000_003


def random_draw_plan(control_seed: int, n_draws: int, antithetic: bool) -> list[dict[str, Any]]:
    """The random-update draws ``[{draw, seed, sign}]``. Gaussian group ``q`` has the seed ``control_seed + q *
    RANDOM_SEED_STRIDE`` (``q = 0`` is ``control_seed`` itself: ``n_draws=1`` without antithetic is the original
    single draw). With ``antithetic`` the draws come in pairs ``(+g_q, -g_q)`` (same Gaussian, opposite sign), so
    ``q = draw // 2``; otherwise every draw is its own group, ``q = draw``. Raises ``ValueError`` for ``n_draws < 1``
    or an odd ``n_draws`` with ``antithetic``."""
    if int(n_draws) < 1:
        raise ValueError(f"random_draws must be >= 1, got {n_draws}")
    if antithetic and int(n_draws) % 2:
        raise ValueError(f"random_draws must be even with random_antithetic (pairs +g, -g), got {n_draws}")
    plan = []
    for i in range(int(n_draws)):
        q = i // 2 if antithetic else i
        plan.append({"draw": i, "seed": int(control_seed) + q * RANDOM_SEED_STRIDE, "sign": -1.0 if antithetic and i % 2 else 1.0})
    return plan


def random_draw_names(i: int) -> tuple[str, str]:
    """``(pair name, side tag)`` of draw ``i``. Draw 0 keeps the names of the single-draw control (``random_update``,
    ``k_rand``): it also feeds m1 / m2; the draws ``i >= 1`` are direct-measure only."""
    return ("random_update", "k_rand") if i == 0 else (f"random_update_d{i}", f"k_rand{i}")


def expert_update(base_sd: Mapping[str, Tensor], ckpt_path: str) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """The expert's tensors renamed to the model's keyspace (same loader as the pipelines: ``load_ckpt`` then the
    conservative ``align_to_base_keys``). Raises if any ``visual.transformer.resblocks`` tensor is not covered."""
    from ...io.ckpt import align_to_base_keys, load_ckpt

    aligned = align_to_base_keys(load_ckpt(str(ckpt_path)), base_sd)
    blocks = [k for k in base_sd if k.startswith("visual.transformer.resblocks.") and torch.is_floating_point(base_sd[k])]
    missing = [k for k in blocks if k not in aligned]
    if missing:
        raise RuntimeError(f"{ckpt_path}: {len(missing)}/{len(blocks)} resblock tensors not found, e.g. {missing[:3]}")
    tau_sq = {k: float((aligned[k].to(torch.float64) - base_sd[k].to(torch.float64)).pow(2).sum()) for k in blocks}
    total = sum(tau_sq.values()) ** 0.5
    if total == 0.0:
        raise RuntimeError(f"{ckpt_path}: expert equals base on every resblock tensor")
    return dict(aligned), {"path": str(ckpt_path), "n_resblock_tensors": len(blocks), "tau_norm_resblocks": total}


# ----------------------------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class ModelSpec:
    model: str
    pretrained: str
    experts: dict[str, str] = field(default_factory=dict)
    base_checkpoint: str | None = None


@dataclass
class UpdateKernelCkaConfig:
    source: ModelSpec
    target: ModelSpec
    task: str
    state_dir: str
    out_dir: str
    suite: str = "vision8"
    other_task: str | None = None
    anchor_filter: str | None = "norm"
    anchor_filter_k: float = 3.0
    split: str = "test"  # evaluation images: 'test' (seeded random subset of the test split) | 'val' (train slice)
    num_images: int = 1000
    val_size: int = 1000
    seed: int = 89
    batch_size: int = 32
    num_workers: int = 0
    device: str = "cuda"
    compute_dtype: str = "float32"
    controls: tuple[str, ...] = ("null", "other_task", "random_update", "base_reference")
    permutations: int = 200
    image_permutations: int = 20
    ci: bool = True  # delete-one jackknife intervals (m1: image clusters of anchors; m2/m3: image chunks)
    jackknife_chunks: int = 16  # G, the number of image chunks of the m2/m3 jackknife
    control_seed: int = 0
    random_draws: int = 19  # random_update null distribution: number of random draws
    random_antithetic: bool = False  # True: draws in pairs (+g, -g) (random_draws must be even); False: independent draws
    debiased: bool = True
    grid_match: str = "auto"
    grid_reference: str = "canonical"
    include_mean_rows: bool = False
    m2: bool = False  # secondary row-CKA (m2) + its d-space statistics and permutations; ~20 GB on an L/14 target when on
    m1_center_rows: bool = True
    blocks: tuple[int, ...] | None = None
    transported_run_dir: str | None = None
    transported_alpha: float = 1.0
    overwrite: bool = False
    # image order and null pairing. 'legacy': the selected images sorted by dataset index (batches and jackknife chunks
    # are in class order for class-sorted datasets), image-null partners = any other image of the batch (the original
    # behaviour). 'stratified': seeded shuffle, class-STRATIFIED jackknife chunks, image-null partners of another class.
    sampling: str = "legacy"
    # cudnn TF32 for the forward passes. The anchors states were registered with cudnn TF32 ON (PyTorch default) and
    # matmul TF32 off; True reproduces the state frames to ~1e-6 (frame check of metodoK_phase1, 2026-11-06), False
    # (legacy) leaves a 1e-3..2e-2 relative gap on ViT-L/14. Matmul TF32 is always off.
    cudnn_allow_tf32: bool = False
    mean_shift: bool = True  # accumulate / report / save the per-chunk mean-shift T-vectors (mean_vectors.pt)


def parse_config(payload: Mapping[str, Any]) -> UpdateKernelCkaConfig:
    p = dict(payload)
    for side in ("source", "target"):
        spec = dict(p[side])
        unknown = sorted(set(spec) - {f.name for f in fields(ModelSpec)})
        if unknown:
            raise ValueError(f"unknown {side} keys: {unknown}")
        p[side] = ModelSpec(**spec)
    allowed = {f.name for f in fields(UpdateKernelCkaConfig)}
    unknown = sorted(set(p) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}; allowed: {sorted(allowed)}")
    if "controls" in p:
        p["controls"] = tuple(p["controls"])
    if p.get("blocks") is not None:
        p["blocks"] = tuple(int(b) for b in p["blocks"])
    cfg = UpdateKernelCkaConfig(**p)
    bad = sorted(set(cfg.controls) - set(CONTROLS))
    if bad:
        raise ValueError(f"unknown controls {bad}; allowed {list(CONTROLS)}")
    needs_other = {"other_task", "other_task_other_images"} & set(cfg.controls)
    if needs_other and not cfg.other_task:
        raise ValueError(f"controls {sorted(needs_other)} need 'other_task'")
    if needs_other and cfg.other_task == cfg.task:
        raise ValueError("other_task must differ from task")
    if "transported" in cfg.controls and not cfg.transported_run_dir:
        raise ValueError("control 'transported' needs 'transported_run_dir'")
    random_draw_plan(cfg.control_seed, cfg.random_draws, cfg.random_antithetic)  # validates (ValueError)
    if cfg.grid_match not in {"auto", "direct", "resample"}:
        raise ValueError("grid_match must be auto|direct|resample")
    if cfg.grid_reference not in {"m", "k", "canonical"}:
        raise ValueError("grid_reference must be m|k|canonical")
    if cfg.split not in {"test", "val"}:
        raise ValueError("split must be test|val")
    if cfg.sampling not in {"legacy", "stratified"}:
        raise ValueError("sampling must be legacy|stratified")
    if cfg.anchor_filter not in {None, "norm"}:
        raise ValueError("anchor_filter must be null or 'norm'")
    if cfg.compute_dtype not in {"float32", "float64"}:
        raise ValueError("compute_dtype must be float32|float64")
    if cfg.task not in cfg.source.experts or cfg.task not in cfg.target.experts:
        raise ValueError(f"experts of both models must include task {cfg.task!r}")
    if needs_other and cfg.other_task not in cfg.target.experts:
        raise ValueError("other_task: the target model needs an expert for it")
    if "other_task" in cfg.controls and cfg.other_task not in cfg.source.experts:
        raise ValueError("other_task: the source model needs an expert for it (within-model cross-task reference)")
    return cfg


# ----------------------------------------------------------------------------------------------------------------
# pipeline (needs models, data, GPU)
# ----------------------------------------------------------------------------------------------------------------


def model_id(spec: ModelSpec) -> str:
    return f"{spec.model}__{spec.pretrained}"


def load_anchor_frames(cfg: UpdateKernelCkaConfig, depth_m: int, depth_k: int, blocks: Sequence[int]):
    """``{j: PairFrame}`` for the target blocks ``blocks`` from ``pair_frames`` (the decoder's own PAIRWISE frame
    function, filter included; not the gpa / consensus frame). Read-only on the state. ``PairFrame.anchor_images``
    are the pool-image ids of the kept rows (``frame.json`` ``anchor_rows`` and the norm-filter keep mask). Also
    returns the frame meta and the pairing."""
    from ...rebase.discrete_layer_match import DiscreteLayerPairing
    from ...rebase.methods.anchors import AnchorState, parse_anchors_config
    from ...rebase.methods.anchors.method import anchor_keep, pair_frames

    frame_json = json.loads((Path(cfg.state_dir) / "frame.json").read_text())
    params: dict[str, Any] = {**frame_json["config"], "state_dir": str(cfg.state_dir)}
    if cfg.anchor_filter is not None:
        params.update({"anchor_filter": cfg.anchor_filter, "anchor_filter_k": float(cfg.anchor_filter_k)})
    acfg = parse_anchors_config(params)
    if acfg.anchor_dedup_cos is not None:
        raise NotImplementedError("anchor_dedup_cos is set: the anchor-image clusters of the kept rows are not tracked")
    state = AnchorState(acfg, acfg.state_dir)
    reg_m, reg_k = state.model(model_id(cfg.source)), state.model(model_id(cfg.target))
    if reg_m.depth != depth_m or reg_k.depth != depth_k:
        raise ValueError(f"registered depths {reg_m.depth}/{reg_k.depth} != model depths {depth_m}/{depth_k}")
    if cfg.anchor_filter is not None:
        for mid in (model_id(cfg.source), model_id(cfg.target)):
            state.load_norm_filter(mid, k=float(cfg.anchor_filter_k))
    images = torch.tensor([int(r[0]) for r in frame_json["anchor_rows"]], dtype=torch.long)
    if bool((images[1:] < images[:-1]).any()):
        raise RuntimeError("frame.json anchor_rows are not sorted by image: the row order of P is not recoverable")
    pairing = DiscreteLayerPairing.compute(depth_m, depth_k)
    frames: dict[int, PairFrame] = {}
    for j in blocks:
        i = int(pairing.pairing[j])
        p_m, c_m, p_k, c_k = pair_frames(reg_m, i, reg_k, j, config=acfg)
        if cfg.anchor_filter is None:
            imgs = images
        else:
            imgs = images[anchor_keep([(reg_m, i), (reg_k, j)], float(cfg.anchor_filter_k)).cpu()]
        if int(imgs.numel()) != int(p_m.shape[0]):
            raise RuntimeError(f"block {j}: {int(imgs.numel())} anchor image ids for {int(p_m.shape[0])} frame rows")
        frames[j] = PairFrame(j, i, (p_m / float(c_m)).cpu(), (p_k / float(c_k)).cpu(), float(c_m), float(c_k), imgs)
    return frames, frame_json, pairing


def _absolute_indices(ds, idx: Sequence[int]) -> tuple[list[int], str]:
    """Indices of ``idx`` (positions in the torch dataset) in the original HF split, when the split is a view."""
    table = getattr(getattr(ds, "split", None), "_indices", None)
    if table is None:
        return [int(i) for i in idx], "natural order of the HF split"
    rows = table.column(0).to_pylist()
    return [int(rows[i]) for i in idx], "HF split _indices (positions in the parent split)"


def _eval_batches(cfg: UpdateKernelCkaConfig, task: str, clf) -> tuple[list[tuple[Tensor, Tensor]], dict[str, Any]]:
    """The evaluation images of ``task``. ``split='test'`` (default): a seeded random subset of the original TEST
    split (disjoint from every expert's training data); ``split='val'``: the first images of the train slice
    ``val_size`` (the experts may have been trained on them). Records the labels and the image indices (sha256)."""
    from torch.utils.data import DataLoader, Subset

    from ...data.vision_loaders import build_vision_loaders, load_hf_splits
    from ..datasets.vision8_14_20 import SUITES

    hf_path, hf_config, split_map = SUITES[cfg.suite].resolver(task)
    hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
    loaders = build_vision_loaders(
        hf_ds=hf_ds, hf_path=hf_path, preprocess=clf.preprocess, ft_epochs=1, split_map=split_map,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=False, seed=cfg.seed, val_size=cfg.val_size,
    )
    ds = (loaders.test if cfg.split == "test" else loaders.val).dataset
    n_total = len(ds)
    n = min(int(cfg.num_images), n_total)
    if cfg.split == "test":
        gen = torch.Generator().manual_seed(int(cfg.seed))
        idx = sorted(torch.randperm(n_total, generator=gen)[:n].tolist())
    else:
        idx = list(range(n))
    loader = DataLoader(Subset(ds, idx), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    batches = [(batch[0], batch[1]) for batch in loader]
    sizes = [int(x.shape[0]) for x, _ in batches]
    batches, idx, chunk_ids = reorder_batches(
        batches, idx, cfg.jackknife_chunks, int(cfg.seed) * 1000003 + 17, stratified=cfg.sampling == "stratified"
    )
    labels = torch.cat([y for _, y in batches]).tolist()
    abs_idx, idx_source = _absolute_indices(ds, idx)
    identity = image_identity(ds, idx)
    meta = {
        **identity, "preprocess": repr(clf.preprocess),
        "preprocess_sha256": hashlib.sha256(_ADDR_RE.sub(" at 0x?", repr(clf.preprocess)).encode()).hexdigest(),
        "task": task, "split": cfg.split, "n_images": len(labels), "n_batches": len(batches),
        "labels_sha256": hashlib.sha256(repr(labels).encode()).hexdigest(),
        "image_indices_sha256": hashlib.sha256(repr(abs_idx).encode()).hexdigest(), "index_source": idx_source,
        "n_split_total": n_total, "sampling": cfg.sampling, "n_chunks": len(set(chunk_ids)),
        "chunk_class_balance": chunk_class_balance(labels, chunk_ids),
        # full record of the selected images (in the order used); ``run`` writes it to ``selected_images.json``
        "selection": {
            "task": task, "split": cfg.split, "sampling": cfg.sampling, "seed": int(cfg.seed), "n_split_total": n_total,
            "dataset_positions": [int(i) for i in idx], "hf_indices": abs_idx, "labels": [int(y) for y in labels],
            "chunk_ids": [int(c) for c in chunk_ids], "batch_sizes": sizes, "index_source": idx_source,
        },
    }
    return batches, meta


def _row_digest(value: Any) -> bytes:
    """sha256 of the raw content of one image cell: the encoded bytes (HF ``Image`` with ``decode=False``), else the
    decoded PIL pixels with mode and size, else an array's bytes."""
    if isinstance(value, Mapping) and value.get("bytes") is not None:
        return hashlib.sha256(bytes(value["bytes"])).digest()
    if hasattr(value, "tobytes") and hasattr(value, "size") and hasattr(value, "mode"):
        return hashlib.sha256(repr((value.mode, value.size)).encode() + value.tobytes()).digest()
    if isinstance(value, Tensor):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).digest()
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).digest()


ID_COLUMNS = ("id", "image_id", "file_name", "filename", "path", "image_path", "name")


def image_identity(ds, idx: Sequence[int]) -> dict[str, Any]:
    """Identity of the selected rows: sha256 over the per-row sha256 of the RAW image content (same rows give the same
    hash whatever the preprocess), and, when the HF dataset has a stable id column, the sha256 of those ids."""
    hasher, ids, id_col = hashlib.sha256(), [], None
    image_key = getattr(ds, "image_key", "image")
    cols = getattr(getattr(ds, "split", None), "column_names", None) or ()
    id_col = next((c for c in ID_COLUMNS if c in cols), None)
    for i in idx:
        ex = ds.split[int(i)]
        hasher.update(_row_digest(ex[image_key]))
        if id_col is not None:
            ids.append(ex[id_col])
    out: dict[str, Any] = {"raw_image_bytes_sha256": hasher.hexdigest(), "id_column": id_col}
    out["row_ids_sha256"] = hashlib.sha256(repr(ids).encode()).hexdigest() if id_col is not None else None
    return out


def check_equal_chunks(batch_sizes: Sequence[int], n_chunks: int) -> dict[str, Any]:
    """Images per jackknife chunk (``chunk_of_batch``); warns when the chunks are unequal."""
    import warnings

    n_b = len(batch_sizes)
    counts = [0] * max(1, min(int(n_chunks), n_b))
    for b, n in enumerate(batch_sizes):
        counts[chunk_of_batch(b, n_b, n_chunks)] += int(n)
    equal = len(set(counts)) == 1
    if not equal:
        msg = f"unequal jackknife chunks: images per chunk {counts} (use num_images divisible by batch_size * chunks)"
        warnings.warn(msg, stacklevel=2)
        print(f"[update_kernel_cka] WARNING {msg}", flush=True)
    return {"images_per_chunk": counts, "equal": equal}


def estimate_accumulator_gb(
    frames: Mapping[int, PairFrame], *, n_ref: int, n_native_m: int, n_native_k: int, controls: Sequence[str], m2: bool,
    n_chunks: int, n_perm_images: int, n_row_types: int = 2, random_draws: int = 1,
) -> dict[str, Any]:
    """Bytes (GB, 1e9) of the float64 accumulators that stay resident until the finalisation (all blocks stream together
    per batch, so none can be freed earlier), computed from the shapes of the classes above:

    * ``direct``: ``SideStats`` ``(2 n_ref T + n_ref + 1)`` and ``PairCross`` ``(n_ref T + n_ref)`` doubles per chunk, per
      side tag / pair (main pass: m_t, k_t (+ m_o, k_o with other_task); pairs true (+ other_task, other_task_rev,
      within_m, within_k); random_update one more side and pair per chunk for draw 0 plus one chunk per further random draw (``random_draws``; the draws ``>= 1`` are direct-only,
      single-chunk: no native (m1) or row (m2) statistics, only draw 0 has them); other_task_other_images / transported one
      chunk) and ``PermCross`` ``n_perm (n_ref T + n_ref)``;
    * ``native`` (m1): ``NativeSide`` ``1 + N d + 2 d^2`` per tag (m_t, k_t, base_m, base_k, + the others);
    * ``m2`` (only with m2: true): ``PairAcc`` per chunk of the pair true, one chunk for each control, and the image-
      permutation ``CrossStats``.
    Excludes the models, the frames P and the GPU transients."""
    ctrl = set(controls)
    run_null = "null" in ctrl and n_perm_images > 0
    comp = {"direct": 0.0, "native": 0.0, "m2": 0.0, "frames": 0.0}
    selfs = lambda d: 3 + 2 * d + d * d  # noqa: E731
    cross = lambda a, b: 1 + a * b + a + b  # noqa: E731
    for fr in frames.values():
        t, dm, dk = fr.n_anchors, int(fr.p_m.shape[1]), int(fr.p_k.shape[1])
        side, pair = 2 * n_ref * t + n_ref + 1, n_ref * t + n_ref
        n_sides = 2 + (2 if "other_task" in ctrl else 0)
        n_pairs = 1 + (4 if "other_task" in ctrl else 0)
        d = (n_sides * side + n_pairs * pair) * n_chunks
        if "random_update" in ctrl:
            d += (side + pair) * (n_chunks + max(1, int(random_draws)) - 1)  # draw 0 chunked, draws >= 1 one chunk
        d += sum(side + pair for c in ("other_task_other_images", "transported") if c in ctrl)
        if run_null:
            d += n_perm_images * pair
        comp["direct"] += 8.0 * d
        nat = lambda n_tok, dd: 1 + n_tok * dd + 2 * dd * dd  # noqa: E731
        n = nat(n_native_m, dm) * (1 + (1 if "other_task" in ctrl else 0) + 1)  # m_t, [m_o], base_m
        n += nat(n_native_k, dk) * (1 + (1 if "other_task" in ctrl else 0) + 1)  # k_t, [k_o], base_k
        n += sum(nat(n_native_k, dk) for c in ("other_task_other_images", "random_update", "transported") if c in ctrl)
        comp["native"] += 8.0 * n
        if m2:
            acc = n_row_types * (selfs(dm) + selfs(dk) + cross(dm, dk))
            n_acc = n_chunks + sum(1 for c in ("other_task", "other_task_other_images", "random_update", "transported") if c in ctrl)
            r = acc * n_acc + (n_perm_images * n_row_types * cross(dm, dk) if run_null else 0)
            comp["m2"] += 8.0 * r
        comp["frames"] += 8.0 * t * (dm + dk)
    comp_gb = {k: v / 1e9 for k, v in comp.items()}
    return {"total_gb": sum(comp_gb.values()), "components_gb": comp_gb,
            "n_blocks": len(frames), "n_chunks": n_chunks, "m2": m2, "random_draws": int(random_draws)}


def _build_model(spec: ModelSpec, device: str):
    from ...models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier

    # fp32 always: the OpenAI ``clip`` loader returns fp16 weights on CUDA, which would round the expert's tau
    clf = OpenClipClassifier.build(
        OpenClipBuildConfig(model_name=spec.model, pretrained=spec.pretrained, device=device, dtype="fp32")
    )
    if spec.base_checkpoint:
        from ...io.ckpt import align_to_base_keys, load_ckpt

        sd = align_to_base_keys(load_ckpt(spec.base_checkpoint), clf.model.state_dict())
        clf.model.load_state_dict(sd, strict=False)
    return clf


def _find_transported(run_dir: str, task: str) -> Path:
    tv_dir = Path(run_dir) / "transported_tvs"
    found = sorted(tv_dir.glob(f"{task}_*_transported_native.pt"))
    if len(found) != 1:
        raise FileNotFoundError(f"{tv_dir}: expected one {task}_*_transported_native.pt, found {len(found)}")
    return found[0]


def _rows(delta: Tensor, n_ref: int, mean_rows: bool) -> dict[str, Tensor]:
    """CLS / matched-patch / mean-patch rows of ``delta [B, N, d]`` (patch grid resampled to ``n_ref`` tokens)."""
    matched = match_patch_grid(delta, n_ref)
    d = matched.shape[-1]
    out = {"cls": matched[:, 0], "patch": matched[:, 1:].reshape(-1, d)}
    if mean_rows:
        out["mean"] = matched[:, 1:].mean(1)
    return out


SLOT = {
    "m_t": "m_work", "m_o": "m_work2", "k_t": "k_work", "k_o": "k_work2", "k_oi": "k_work2", "k_rand": "k_work2",
    "k_tr": "k_work2",
}


def _slot(tag: str) -> str:
    """The work-model slot of a side tag (``k_rand<i>``, the extra random draws, share ``k_rand``'s slot)."""
    if tag in SLOT:
        return SLOT[tag]
    if tag.startswith("k_rand"):
        return SLOT["k_rand"]
    raise KeyError(tag)


@dataclass
class Collector:
    """Everything the passes accumulate, per block ``j``."""

    native: dict[int, dict[str, NativeSide]]
    sides: dict[int, dict[str, list[SideStats]]]
    pairs: dict[int, dict[str, list[PairCross]]]
    pair_tags: dict[str, tuple[str, str]] = field(default_factory=dict)
    perm: dict[int, PermCross] = field(default_factory=dict)
    rows: dict[str, list[dict[int, PairAcc]]] = field(default_factory=dict)
    perm_cross: dict[int, dict[str, list[CrossStats]]] | None = None
    means: dict[int, dict[str, list[MeanAcc]]] = field(default_factory=dict)  # per block, tag, chunk (mean-shift sums)
    native_means: dict[int, dict[str, list[MeanAcc]]] = field(default_factory=dict)  # same sums in the native d-space
    null_pairs: list[int] = field(default_factory=lambda: [0, 0])  # [same-label image-null pairs, all image-null pairs]

    @classmethod
    def empty(cls, blocks: Sequence[int]) -> Collector:
        return cls({j: {} for j in blocks}, {j: {} for j in blocks}, {j: {} for j in blocks})

    def direct_block(self, j: int) -> DirectBlock:
        return DirectBlock(self.sides[j], self.pairs[j], dict(self.pair_tags), self.perm.get(j))


def _stream_pass(
    *, name: str, ctx: dict[str, Any], col: Collector, states: Mapping[str, Mapping[str, Tensor]],
    batches: Mapping[str, list] | None = None, frames: Mapping[int, PairFrame], n_ref: int, row_types: Sequence[str],
    native_tags: Sequence[str] = (), side_tags: Sequence[str] = (), pairs: Sequence[tuple[str, str, str]] = (),
    row_pairs: Sequence[tuple[str, str, str, bool]] = (), collect_base: bool = False, chunk_fn, n_perm_images: int = 0,
    seed: int, collect_means: bool = False, null_across_classes: bool = False,
) -> None:
    """One pass over the task images with the sides ``states`` (tag -> weights) live at once; the SAME images in every
    side (rows correspond), only the expert (state) differs between sides. Accumulates, into ``col``: the native
    statistics (m1) of ``native_tags`` (and of the base activations with ``collect_base``), the anchor-space statistics
    of ``side_tags`` and of ``pairs`` ``(name, tag_x, tag_y)`` (the primary direct measure; ``n_perm_images`` image-
    mismatched permutations for the pair ``true``) and the row statistics of ``row_pairs`` ``(name, tag_x, tag_y,
    chunked)`` (m2, only the main one is chunked). ``collect_means``: per chunk, the sums of the encoded update of every
    tag of ``side_tags`` (``MeanAcc``; with ``collect_base`` also of the encoded base activations, ``base_m`` /
    ``base_k``) into ``col.means``. ``null_across_classes``: the image-mismatched permutations pair every image with an
    image of another class of the same batch (``class_derangement``; the same-label pairs are counted in
    ``col.null_pairs``); False: ``image_derangement`` (any other image)."""
    from ...rebase.methods._ariadne.capture import iter_capture_tokens

    device, dtype = ctx["device"], ctx["dtype"]
    blocks = sorted(frames)
    m_req = {str(i): (i, "boundary") for i in sorted({frames[j].source_block for j in blocks})}
    k_req = {str(j): (j, "boundary") for j in blocks}
    m_batches = ctx["m_batches"]
    bt = {tag: (batches or {}).get(tag) or (m_batches if tag[0] == "m" else ctx["k_batches"]) for tag in states}
    gens: dict[Any, Any] = {}
    for tag, state in states.items():
        work = ctx[_slot(tag)]
        work.load_state_dict(dict(state), strict=True)
        req = m_req if tag[0] == "m" else k_req
        base = ctx["m_base"] if tag[0] == "m" else ctx["k_base"]
        gens[("w", tag)] = iter_capture_tokens(work, bt[tag], req, device, store_device=device)
        gens[("b", tag[0], id(bt[tag]))] = iter_capture_tokens(base, bt[tag], req, device, store_device=device)
    if collect_base and not ({"m_t", "k_t"} <= set(states)):
        raise ValueError("collect_base needs m_t and k_t")
    pt = {j: (frames[j].p_m.to(device, dtype).T.contiguous(), frames[j].p_k.to(device, dtype).T.contiguous()) for j in blocks}
    qs = {j: (frames[j].q_m(device).to(dtype), frames[j].q_k(device).to(dtype)) for j in blocks}
    need = set(side_tags) | {t for _, tx, ty in pairs for t in (tx, ty)}
    dchunks: list[dict[int, DirectAcc]] = []
    rchunks: dict[str, list[dict[int, PairAcc]]] = {rp[0]: [] for rp in row_pairs}
    pair_names = [pr[0] for pr in pairs]
    do_perm = n_perm_images > 0 and "true" in pair_names
    if do_perm:
        col.perm.update({j: PermCross(n_perm_images, n_ref, frames[j].n_anchors) for j in blocks})
        if row_pairs and col.perm_cross is None:
            col.perm_cross = {
                j: {rt: [CrossStats(frames[j].p_m.shape[1], frames[j].p_k.shape[1]) for _ in range(n_perm_images)] for rt in row_types}
                for j in blocks
            }
    keys = list(gens)
    for b, vals in enumerate(zip(*gens.values(), strict=True)):
        v = dict(zip(keys, vals, strict=True))
        c = chunk_fn(b)
        if pairs or side_tags:
            while len(dchunks) <= c:
                dchunks.append({j: DirectAcc(n_ref, frames[j].n_anchors) for j in blocks})
        for rp in row_pairs:
            while len(rchunks[rp[0]]) <= (c if rp[3] else 0):
                rchunks[rp[0]].append({
                    j: PairAcc(frames[j].p_m.shape[1], frames[j].p_k.shape[1], row_types) for j in blocks
                })
        bsz = int(m_batches[b][0].shape[0])
        gen = torch.Generator().manual_seed(int(seed) * 100003 + b)
        if do_perm and null_across_classes:
            lab_b = m_batches[b][1]
            sigmas = [class_derangement(lab_b, gen) for _ in range(n_perm_images)]
            col.null_pairs[0] += sum(int((lab_b[sg] == lab_b).sum()) for sg in sigmas)
            col.null_pairs[1] += bsz * len(sigmas)
        else:
            sigmas = [image_derangement(bsz, gen) for _ in range(n_perm_images)] if do_perm else []
        for j in blocks:
            i = frames[j].source_block
            delta: dict[str, Tensor] = {}
            for tag in states:
                key = str(i) if tag[0] == "m" else str(j)
                base_v = v[("b", tag[0], id(bt[tag]))][key]
                delta[tag] = (v[("w", tag)][key] - base_v).to(dtype)
            for tag in native_tags:
                _feed_native(col.native[j], tag, delta[tag])
            if collect_base:
                _feed_native(col.native[j], "base_m", v[("b", "m", id(bt["m_t"]))][str(i)].to(dtype))
                _feed_native(col.native[j], "base_k", v[("b", "k", id(bt["k_t"]))][str(j)].to(dtype))
            enc = {tag: match_patch_grid(delta[tag], n_ref) @ pt[j][0 if tag[0] == "m" else 1] for tag in need}
            if pairs or side_tags:
                dchunks[c][j].update(enc, side_tags, pairs)
            if collect_means:
                store = col.means.setdefault(j, {})
                t_j = frames[j].n_anchors
                feed = {tag: enc[tag] for tag in side_tags}
                if collect_base:
                    feed["base_m"] = match_patch_grid(v[("b", "m", id(bt["m_t"]))][str(i)].to(dtype), n_ref) @ pt[j][0]
                    feed["base_k"] = match_patch_grid(v[("b", "k", id(bt["k_t"]))][str(j)].to(dtype), n_ref) @ pt[j][1]
                for tag, x in feed.items():
                    lst = store.setdefault(tag, [])
                    while len(lst) <= c:
                        lst.append(MeanAcc(t_j))
                    lst[c].update(x)
                # the same sums of the NATIVE (d-dimensional, not grid-matched, not encoded) update / base activation
                nfeed = {tag: delta[tag] for tag in side_tags}
                if collect_base:
                    nfeed["base_m"] = v[("b", "m", id(bt["m_t"]))][str(i)].to(dtype)
                    nfeed["base_k"] = v[("b", "k", id(bt["k_t"]))][str(j)].to(dtype)
                nstore = col.native_means.setdefault(j, {})
                for tag, x in nfeed.items():
                    lst = nstore.setdefault(tag, [])
                    while len(lst) <= c:
                        lst.append(MeanAcc(int(x.shape[-1])))
                    lst[c].update(x)
            if do_perm:
                tx, ty = next((tx, ty) for nm, tx, ty in pairs if nm == "true")
                col.perm[j].update(enc[tx], enc[ty], sigmas)
            for nm, tx, ty, chunked in row_pairs:
                acc = rchunks[nm][c if chunked else 0][j]
                rows_x = _rows(delta[tx], n_ref, "mean" in row_types)
                rows_y = _rows(delta[ty], n_ref, "mean" in row_types)
                use_perm = do_perm and nm == "true" and col.perm_cross is not None
                acc.update(
                    rows_x, rows_y, qs[j][0 if tx[0] == "m" else 1], qs[j][0 if ty[0] == "m" else 1],
                    patches_per_image=int(n_ref) - 1, perms=sigmas if use_perm else (),
                    perm_cross=col.perm_cross[j] if use_perm else None,
                )
        print(f"[update_kernel_cka] {name}: batch {b + 1}/{len(m_batches)}", flush=True)
    for nm, tx, ty in pairs:
        col.pair_tags[nm] = (tx, ty)
    for j in blocks:
        for tag in side_tags:
            col.sides[j][tag] = [d[j].sides[tag] for d in dchunks]
        for nm, _, _ in pairs:
            col.pairs[j][nm] = [d[j].pairs[nm] for d in dchunks]
    col.rows.update(rchunks)


def _feed_native(store: dict[str, NativeSide], tag: str, tokens: Tensor) -> None:
    side = store.get(tag)
    if side is None:
        side = store[tag] = NativeSide(int(tokens.shape[1]), int(tokens.shape[2]))
    side.update(tokens)


def _grid_plan(cfg: UpdateKernelCkaConfig, n_m: int, n_k: int, canonical: int) -> tuple[int, str]:
    if n_m == n_k:
        return n_m, "direct"
    if cfg.grid_match == "direct":
        raise ValueError(f"grid_match='direct' but the models have {n_m} vs {n_k} tokens")
    ref = {"m": n_m, "k": n_k, "canonical": canonical}[cfg.grid_reference]
    return ref, f"resample(bilinear, reference={cfg.grid_reference}, n_tokens={ref}; native m={n_m}, k={n_k})"


MEAN_VECTORS_FILE = "mean_vectors.pt"
SELECTED_IMAGES_FILE = "selected_images.json"
CONTROL_TAGS = {
    "other_task": "k_o", "other_task_other_images": "k_oi", "random_update": "k_rand", "transported": "k_tr",
}


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    from ...run_logging import _code_fingerprint

    cfg = parse_config(payload)
    out_dir = Path(cfg.out_dir)
    if (out_dir / "summary.json").exists() and not cfg.overwrite:
        raise FileExistsError(f"{out_dir / 'summary.json'} exists (set overwrite to replace)")
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 off: the diagnostics compare small differences of kernels
    torch.backends.cudnn.allow_tf32 = bool(cfg.cudnn_allow_tf32)
    device = cfg.device
    dtype = torch.float32 if cfg.compute_dtype == "float32" else torch.float64

    clf_m, clf_k = _build_model(cfg.source, device), _build_model(cfg.target, device)
    depth_m, depth_k = len(clf_m.model.visual.transformer.resblocks), len(clf_k.model.visual.transformer.resblocks)
    blocks = list(cfg.blocks) if cfg.blocks is not None else list(range(depth_k))
    frames, frame_json, pairing = load_anchor_frames(cfg, depth_m, depth_k, blocks)

    base_m = {k: v.detach().cpu().clone() for k, v in clf_m.model.state_dict().items()}
    base_k = {k: v.detach().cpu().clone() for k, v in clf_k.model.state_dict().items()}
    upd_m, info_m = expert_update(base_m, cfg.source.experts[cfg.task])
    upd_k, info_k = expert_update(base_k, cfg.target.experts[cfg.task])
    m_work, k_work = copy.deepcopy(clf_m.model), copy.deepcopy(clf_k.model)
    m_work2, k_work2 = copy.deepcopy(clf_m.model), copy.deepcopy(clf_k.model)  # sides of the other experts

    m_batches, meta_m = _eval_batches(cfg, cfg.task, clf_m)
    k_batches, meta_k = _eval_batches(cfg, cfg.task, clf_k)
    sel_m, sel_k = meta_m.pop("selection"), meta_k.pop("selection")
    if sel_m != sel_k:
        raise RuntimeError("the two models' loaders do not yield the same image selection / order")
    for (_, ym), (_, yk) in zip(m_batches, k_batches, strict=True):
        if not torch.equal(ym, yk):
            raise RuntimeError("the two models' loaders do not yield the same images (labels differ)")
    if meta_m["image_indices_sha256"] != meta_k["image_indices_sha256"]:
        raise RuntimeError("the two models' loaders do not yield the same image indices")
    if meta_m["raw_image_bytes_sha256"] != meta_k["raw_image_bytes_sha256"]:
        raise RuntimeError("the two models' loaders do not yield the same raw image bytes")
    n_tokens_m = int(clf_m.model.visual.positional_embedding.shape[0])
    n_tokens_k = int(clf_k.model.visual.positional_embedding.shape[0])
    canonical = int(frame_json["config"]["canonical_tokens"])
    n_ref, grid_mode = _grid_plan(cfg, n_tokens_m, n_tokens_k, canonical)

    ctx = {
        "device": device, "dtype": dtype, "m_base": clf_m.model, "m_work": m_work, "m_work2": m_work2,
        "k_base": clf_k.model, "k_work": k_work, "k_work2": k_work2, "m_batches": m_batches, "k_batches": k_batches,
    }
    row_types = ("cls", "patch") + (("mean",) if cfg.include_mean_rows else ())
    n_b = len(m_batches)
    run_null = "null" in cfg.controls
    col = Collector.empty(blocks)
    state_m, state_k = {**base_m, **upd_m}, {**base_k, **upd_k}
    chunked = lambda b: chunk_of_batch(b, n_b, cfg.jackknife_chunks)  # noqa: E731
    control_info: dict[str, Any] = {}
    n_chunks_eff = max(1, min(int(cfg.jackknife_chunks), n_b))
    chunk_check = check_equal_chunks([int(x.shape[0]) for x, _ in m_batches], n_chunks_eff)
    ram_estimate = estimate_accumulator_gb(
        frames, n_ref=n_ref, n_native_m=n_tokens_m, n_native_k=n_tokens_k, controls=cfg.controls, m2=cfg.m2,
        n_chunks=n_chunks_eff, n_perm_images=cfg.image_permutations if run_null else 0, n_row_types=len(row_types),
        random_draws=cfg.random_draws,
    )
    print(f"[update_kernel_cka] estimated accumulators: {ram_estimate['total_gb']:.2f} GB "
          f"(blocks {blocks}, m2={cfg.m2}); components {({k: round(v, 2) for k, v in ram_estimate['components_gb'].items()})}", flush=True)
    common = {"ctx": ctx, "col": col, "frames": frames, "n_ref": n_ref, "row_types": row_types, "seed": cfg.control_seed}

    # --- main pass: m(t) x k(t) and, with other_task, the other experts on the SAME images (all sides live) ----------
    states = {"m_t": state_m, "k_t": state_k}
    side_tags = ["m_t", "k_t"]
    pairs = [("true", "m_t", "k_t")]
    _rp = lambda x: [x] if cfg.m2 else []  # noqa: E731  (row statistics of m2 only with m2: true)
    row_pairs = _rp(("true", "m_t", "k_t", True))
    native_tags = ["m_t", "k_t"]
    if "other_task" in cfg.controls:
        # expert-only control: the experts of the other task applied to the SAME task-t images (rows correspond)
        upd_ko, info_ko = expert_update(base_k, cfg.target.experts[cfg.other_task])
        upd_mo, info_mo = expert_update(base_m, cfg.source.experts[cfg.other_task])
        states.update({"m_o": {**base_m, **upd_mo}, "k_o": {**base_k, **upd_ko}})
        side_tags += ["m_o", "k_o"]
        native_tags += ["m_o", "k_o"]
        # other_task = (m_t, k_o); other_task_rev = the mirror (m_o, k_t); the comparator is the mean of the two directions
        pairs += [
            ("other_task", "m_t", "k_o"), ("other_task_rev", "m_o", "k_t"), ("within_m", "m_t", "m_o"), ("within_k", "k_t", "k_o"),
        ]
        row_pairs += _rp(("other_task", "m_t", "k_o", False))
        control_info["other_task"] = {
            "expert_k": info_ko, "expert_m": info_mo, "images": "the task-t images (same rows as the main variant)",
            "uses": "D_m(t) x D_k(t') for the direct measure, m1, m2; D(t) x D(t') within each model for the "
                    "within-model cross-task reference",
        }
    _stream_pass(
        name="main", states=states, native_tags=native_tags, side_tags=side_tags, pairs=pairs, row_pairs=row_pairs,
        collect_base=True, chunk_fn=chunked, n_perm_images=cfg.image_permutations if run_null else 0, **common,
        collect_means=cfg.mean_shift, null_across_classes=cfg.sampling == "stratified",
    )

    # --- remaining controls ---------------------------------------------------------------------------------------
    if "other_task_other_images" in cfg.controls:
        upd_oi, info_oi = expert_update(base_k, cfg.target.experts[cfg.other_task])
        kb_other, meta_other = _eval_batches(cfg, cfg.other_task, clf_k)
        meta_other.pop("selection", None)
        if [int(x.shape[0]) for x, _ in kb_other] != [int(x.shape[0]) for x, _ in m_batches]:
            raise RuntimeError("other_task other-images set has a different number/size of batches than the task's")
        control_info["other_task_other_images"] = {
            "expert": info_oi, "images": meta_other, "pairing": "image i of t with image i of t' (different images)",
        }
        _stream_pass(
            name="other_task_other_images", states={"m_t": state_m, "k_oi": {**base_k, **upd_oi}},
            batches={"k_oi": kb_other}, native_tags=["k_oi"], side_tags=["k_oi"],
            pairs=[("other_task_other_images", "m_t", "k_oi")],
            row_pairs=_rp(("other_task_other_images", "m_t", "k_oi", False)), chunk_fn=lambda b: 0, **common,
        )
    if "random_update" in cfg.controls:
        plan = random_draw_plan(cfg.control_seed, cfg.random_draws, cfg.random_antithetic)
        draws_info: list[dict[str, Any]] = []
        for dr in plan:  # one draw at a time: only one extra copy of model k's changed tensors is resident
            pair_name, tag = random_draw_names(dr["draw"])
            rand, norms = random_update_state(base_k, upd_k, seed=dr["seed"], sign=dr["sign"])
            draws_info.append({
                **dr, "pair": pair_name, "n_tensors": len(norms),
                "max_rel_norm_error_after_cast": max(v["rel_norm_error"] for v in norms.values()),
                "max_rel_norm_error_float64_r": max(abs(v["random_norm"] - v["tau_norm"]) / v["tau_norm"] for v in norms.values()),
                "total_tau_norm": sum(v["tau_norm"] ** 2 for v in norms.values()) ** 0.5,
            })
            first = dr["draw"] == 0  # draw 0 also feeds m1 / m2
            _stream_pass(
                name=pair_name, states={"m_t": state_m, tag: {**base_k, **rand}}, native_tags=[tag] if first else [],
                side_tags=[tag], pairs=[(pair_name, "m_t", tag)],
                row_pairs=_rp((pair_name, "m_t", tag, False)) if first else [], chunk_fn=chunked if first else (lambda b: 0),
                collect_means=cfg.mean_shift and first,  # the mean-shift vectors of draw 0 (chunked) as a chance comparator
                **common,  # draws >= 1: one chunk, only the all-images cosine (null distribution)
            )
            del rand
        control_info["random_update"] = {
            "seed": cfg.control_seed, "n_draws": len(plan), "antithetic": cfg.random_antithetic,
            "n_tensors": draws_info[0]["n_tensors"],
            "max_rel_norm_error_after_cast": max(d["max_rel_norm_error_after_cast"] for d in draws_info),
            "max_rel_norm_error_float64_r": max(d["max_rel_norm_error_float64_r"] for d in draws_info),
            "total_tau_norm": draws_info[0]["total_tau_norm"], "draws": draws_info,
            "m1_m2_comparator": "draw 0 only (random_update and true_minus_random_draw0 are draw 0; null stats, z and p use all draws)",
        }
    if "transported" in cfg.controls:
        from ...io.ckpt import load_ckpt

        tv_path = _find_transported(cfg.transported_run_dir, cfg.task)
        tv = load_ckpt(str(tv_path))
        missing = [k for k in tv if k not in base_k]
        if missing:
            raise RuntimeError(f"{tv_path}: keys not in the target model: {missing[:3]}")
        state = dict(base_k)
        for key, value in tv.items():
            state[key] = base_k[key] + float(cfg.transported_alpha) * value.to(base_k[key].dtype)
        control_info["transported"] = {"tv": str(tv_path), "alpha": cfg.transported_alpha}
        _stream_pass(
            name="transported", states={"m_t": state_m, "k_tr": state}, native_tags=["k_tr"], side_tags=["k_tr"],
            pairs=[("transported", "m_t", "k_tr")], row_pairs=_rp(("transported", "m_t", "k_tr", False)),
            chunk_fn=lambda b: 0, **common,
        )

    # --- finalisation --------------------------------------------------------------------------------------------
    groups_rows = ("cls", "patch", "all") + (("mean",) if cfg.include_mean_rows else ())
    main_opts = FinalizeOptions(
        groups_rows=groups_rows, debiased=cfg.debiased, ci=cfg.ci, null=run_null, n_perm=cfg.permutations,
        seed=cfg.control_seed, device=device, center_rows_m1=cfg.m1_center_rows,
        random_antithetic=cfg.random_antithetic,
    )
    ctl_opts = FinalizeOptions(
        groups_rows=groups_rows, debiased=cfg.debiased, device=device, center_rows_m1=cfg.m1_center_rows
    )
    ctl_names = [n for n in ("other_task", "other_task_other_images", "random_update", "transported") if n in cfg.controls]
    tags = {name: CONTROL_TAGS[name] for name in ctl_names}
    block_results = []
    n_chunks_actual = len(col.pairs[blocks[0]]["true"])
    vec_out: dict[int, Any] = {}
    for j in blocks:
        fr = frames[j]
        opts = copy.copy(main_opts)
        opts.seed = cfg.control_seed * 1000 + j
        res = finalize_pair(
            [c[j] for c in col.rows.get("true", [])], fr, opts, perm_cross=None if col.perm_cross is None else col.perm_cross[j],
            native=col.native[j], control_tags=tags, direct=col.direct_block(j),
        )
        if cfg.mean_shift and col.means.get(j):
            parts = mean_vectors(col.means[j])
            res["mean_shift"] = mean_shift_block(parts, ci=cfg.ci)
            vec_out[j] = {
                "tags": {tag: {"cls": v["cls"].float().contiguous(), "patch": v["patch"].float().contiguous()} for tag, v in parts.items()},
                "n_images": parts["m_t"]["n"].clone(), "n_anchors": fr.n_anchors, "source_block": fr.source_block,
            }
            col.means[j] = {}
            nat = mean_vectors(col.native_means.get(j, {}))
            vec_out[j]["native"] = {tag: {"cls": v["cls"].float().contiguous(), "patch": v["patch"].float().contiguous()}
                                    for tag, v in nat.items()}
            col.native_means[j] = {}
        for name in ctl_names:
            m2c: dict[str, float] = {}
            if cfg.m2:
                r = finalize_pair([c[j] for c in col.rows[name]], fr, ctl_opts)
                m2c = {g: v["value"] for g, v in r["m2"].items()}
            m1c = res["m1_controls"][name]
            res.setdefault("controls", {})[name] = {
                "direct": res["direct_controls"][name], "m1": m1c[opts.primary_variant], "m1_variants": m1c, "m2": m2c,
            }
        if "base_reference" in cfg.controls:
            res["base_reference"] = base_reference(fr, debiased=cfg.debiased, ci=cfg.ci, device=device)
        res.update({"target_block": j, "source_block": fr.source_block, "n_anchors": fr.n_anchors,
                    "c_source": fr.c_m, "c_target": fr.c_k})
        block_results.append(res)
        col.native[j], col.sides[j], col.pairs[j] = {}, {}, {}  # free the block's statistics
        print(f"[update_kernel_cka] finalised block {j} (source {fr.source_block})", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SELECTED_IMAGES_FILE).write_text(json.dumps(sel_m))
    if vec_out:
        torch.save({
            "meta": {
                "format": "mean_vectors_v1", "task": cfg.task, "other_task": cfg.other_task,
                "source": model_id(cfg.source), "target": model_id(cfg.target), "sampling": cfg.sampling,
                "n_chunks": n_chunks_actual, "images_per_chunk": [float(x) for x in next(iter(vec_out.values()))["n_images"]],
                "n_patch_positions": int(n_ref) - 1, "native_patch_positions": {"m": n_tokens_m - 1, "k": n_tokens_k - 1}, "split_seed": cfg.seed, "state_dir": cfg.state_dir,
                "contents": "blocks[j]['tags'][tag][group] = per-chunk SUM over images (and patch positions) of the encoded "
                            "rows (dH P^T for m_t k_t m_o k_o k_rand = update; H_base P^T for base_m base_k = position), "
                            "float32 [n_chunks, T]; group cls|patch (all = cls + patch). blocks[j]['native'][tag][group] = the same sums "
                            "in the native d-space (no grid matching, no encoding): float32 [n_chunks, d_model] (m_* tags: d of the "
                            "source, k_* tags: d of the target; base_m / base_k = base activations)",
            },
            "blocks": vec_out,
        }, out_dir / MEAN_VECTORS_FILE)
    setting = {
        "source": {"model": cfg.source.model, "pretrained": cfg.source.pretrained, "n_tokens": n_tokens_m,
                   "depth": depth_m, "expert": info_m, "base_checkpoint": cfg.source.base_checkpoint},
        "target": {"model": cfg.target.model, "pretrained": cfg.target.pretrained, "n_tokens": n_tokens_k,
                   "depth": depth_k, "expert": info_k, "base_checkpoint": cfg.target.base_checkpoint},
        "task": cfg.task, "other_task": cfg.other_task, "suite": cfg.suite, "state_dir": cfg.state_dir,
        "frame": frame_json["config"],
        "frame_kind": "pairwise frame of pair_frames (the decoder's two-model frame), NOT the gpa / consensus frame",
        "T_stored": int(frame_json["config"]["num_anchors"]), "anchor_filter": cfg.anchor_filter,
        "anchor_filter_k": cfg.anchor_filter_k, "pairing": list(pairing.pairing),
        "split": cfg.split, "images": {"task": meta_m, "task_target_loader": meta_k}, "val_size": cfg.val_size,
        "split_seed": cfg.seed, "sampling": cfg.sampling, "mean_shift": cfg.mean_shift,
        "mean_vectors_file": MEAN_VECTORS_FILE if vec_out else None, "selected_images_file": SELECTED_IMAGES_FILE,
        "image_null_pairs": (
            {"same_label": col.null_pairs[0], "total": col.null_pairs[1],
             "same_label_fraction": col.null_pairs[0] / max(1, col.null_pairs[1])}
            if cfg.sampling == "stratified" and run_null else None
        ),
        "batch_size": cfg.batch_size, "grid_mode": grid_mode, "grid_reference": cfg.grid_reference,
        "grid_reference_tokens": n_ref, "controls": list(cfg.controls), "permutations": cfg.permutations,
        "image_permutations": cfg.image_permutations, "ci": cfg.ci,
        "ci_method": "delete-one jackknife, t_{G-1}: direct measure and m2 over image chunks, m1 over anchor-image clusters",
        "jackknife_chunks_requested": cfg.jackknife_chunks, "jackknife_chunks_actual": n_chunks_actual,
        "m2": cfg.m2, "accumulation_dtype": str(ACC_DTYPE).replace("torch.", ""), "chunk_check": chunk_check,
        "ram_estimate_accumulators": ram_estimate, "blocks_run": blocks,
        "preprocess": {"source": meta_m["preprocess"], "target": meta_k["preprocess"]},
        "n_batches": n_b, "control_seed": cfg.control_seed, "random_draws": cfg.random_draws,
        "random_antithetic": cfg.random_antithetic,
        "random_seed_derivation": (
            f"draw i has Gaussian group q = i // 2 if random_antithetic else i; seed = control_seed + q * {RANDOM_SEED_STRIDE}; "
            "sign = -1 for odd i when antithetic (same g, opposite sign), else +1"
        ),
        "random_draw_seeds": [[d["seed"], d["sign"]] for d in control_info.get("random_update", {}).get("draws", [])], "debiased": cfg.debiased, "compute_dtype": cfg.compute_dtype,
        "tf32": {"matmul": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn": bool(torch.backends.cudnn.allow_tf32)},
        "include_mean_rows": cfg.include_mean_rows, "m1_center_rows": cfg.m1_center_rows,
        "m1_primary_variant": main_opts.primary_variant, "primary_measure": f"direct {DIRECT_PRIMARY}", "control_info": control_info,
        "config": dict(payload), "code": _code_fingerprint(), "seconds": time.perf_counter() - started,
    }
    summary = {"setting": setting, "blocks": block_results}
    write_outputs(summary, out_dir)
    return summary


# ----------------------------------------------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------------------------------------------

CONTROL_COLUMNS = ("other_task", "other_task_other_images", "random_update", "transported")
EXCESS_KEYS = (
    ("excess_other_task_mean", "excess_other_task_mean"),
    ("excess_other_task", "excess_other_task"),
    ("excess_other_task_rev", "excess_other_task_rev"),
    ("excess_within_model_cross_task", "excess_within_model"),
    ("true_minus_random_draw0", "true_minus_random_draw0"),
    ("excess_base_repr", "excess_base_repr"),
)
CSV_COLUMNS = (
    "target_block", "source_block", "measure", "group", "value", "lo", "hi", "se", "null_q95", "null_mean", "null_z",
    "null_p", "anchor_null_z", "anchor_null_p", *CONTROL_COLUMNS, "base_reference", "ref_within_model_cross_task",
    "ref_base_repr", "ref_other_task_rev", "ref_other_task_mean", "random_draws_used", "random_update_basis", "random_update_min", "random_update_max",
    "random_null_mean", "random_null_sd", "random_null_max_abs", "z_random", "p_random", "p_random_two_sided", *[f"{col}{suf}" for _, col in EXCESS_KEYS for suf in ("", "_lo", "_hi")],
)


def csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per block, measure and group; the PRIMARY (direct comparison) first, then m1 (Gram), then m2."""
    rows = []
    for blk in summary["blocks"]:
        measures = [(f"direct_{v}", "direct", v, blk["direct"][v]) for v in DIRECT_VARIANTS if "direct" in blk]
        measures += [(f"m1_{v}", "m1", v, blk["m1_variants"][v]) for v in M1_VARIANTS if "m1_variants" in blk]
        measures += [("m2", "m2", None, blk["m2"])] if blk.get("m2") else []
        for measure, family, variant, groups in measures:
            for group, e in groups.items():
                row = {"target_block": blk["target_block"], "source_block": blk["source_block"], "measure": measure,
                       "group": group, "value": e["value"], "lo": e.get("lo"), "hi": e.get("hi"), "se": e.get("se")}
                nl = e.get("image_null", e.get("null"))
                if nl is not None:
                    row["null_q95"], row["null_mean"] = nl["q95"], nl["mean"]
                    row["null_z"], row["null_p"] = nl.get("z"), nl.get("p_empirical")
                if "anchor_null" in e:
                    row["anchor_null_z"], row["anchor_null_p"] = e["anchor_null"].get("z"), e["anchor_null"].get("p_empirical")
                for name in CONTROL_COLUMNS:
                    ctl = blk.get("controls", {}).get(name)
                    if ctl is None:
                        continue
                    if family == "direct":
                        row[name] = ctl["direct"][variant].get(group)
                    elif family == "m1":
                        row[name] = ctl["m1_variants"][variant].get(group)
                    else:
                        row[name] = ctl["m2"].get(group)
                if family in ("direct", "m1"):
                    if family == "m1" and "base_reference" in blk:
                        row["base_reference"] = blk["base_reference"]["value"]
                    row["ref_within_model_cross_task"] = e.get("within_model_cross_task", {}).get("value")
                    row["ref_base_repr"] = e.get("base_repr", {}).get("value")
                    row["ref_other_task_rev"] = e.get("other_task_rev", {}).get("value")
                    row["ref_other_task_mean"] = e.get("other_task_mean", {}).get("value")
                    rd = e.get("random_update_draws")
                    if family == "direct" and rd is not None:  # direct: the draw mean / null distribution of ALL draws
                        row["random_draws_used"], row["random_update_basis"] = rd["n"], "draw0 (random_update); null stats over all draws"
                        row["random_update_min"], row["random_update_max"] = rd["min"], rd["max"]
                        for src, dst in (("null_mean", "random_null_mean"), ("null_sd", "random_null_sd"),
                                         ("null_max_abs", "random_null_max_abs"), ("z_random", "z_random"),
                                         ("p_random", "p_random"), ("p_random_two_sided", "p_random_two_sided")):
                            row[dst] = rd[src]
                    elif "random_update" in row:  # m1 / m2 rows: the random column is draw 0 only
                        row["random_draws_used"], row["random_update_basis"] = 1, "draw0 (random_update)"
                    for key, col in EXCESS_KEYS:
                        x = e.get(key)
                        if x is not None:
                            row[col], row[f"{col}_lo"], row[f"{col}_hi"] = x["value"], x.get("lo"), x.get("hi")
                rows.append(row)
        for kind in MS_KINDS:  # mean-shift agreement (cosine of the anchor-centred T-vectors), one row per group
            for group, e in blk.get("mean_shift", {}).get(kind, {}).items():
                row = {"target_block": blk["target_block"], "source_block": blk["source_block"],
                       "measure": f"meanshift_{kind}", "group": group, "value": e["value"], "lo": e.get("lo"),
                       "hi": e.get("hi"), "se": e.get("se")}
                row["other_task"] = e.get("other_task", {}).get("value")
                row["random_update"] = e.get("random_update", {}).get("value")
                row["ref_other_task_rev"] = e.get("other_task_rev", {}).get("value")
                row["ref_other_task_mean"] = e.get("other_task_mean", {}).get("value")
                for key, col in EXCESS_KEYS:
                    x = e.get(key)
                    if x is not None:
                        row[col], row[f"{col}_lo"], row[f"{col}_hi"] = x["value"], x.get("lo"), x.get("hi")
                rows.append(row)
    return rows


def _fmt(x: Any) -> str:
    return "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _ci(e: Mapping[str, Any] | None) -> str:
    return "" if not e else f"{_fmt(e.get('value'))} [{_fmt(e.get('lo'))}, {_fmt(e.get('hi'))}]"


def render_md(summary: Mapping[str, Any]) -> str:
    s = summary["setting"]
    src, tgt = s["source"], s["target"]
    blocks = summary["blocks"]
    primary = s.get("m1_primary_variant", "centered")
    head = (
        f"{src['model']}/{src['pretrained']} (m, {src['n_tokens']} token) vs {tgt['model']}/{tgt['pretrained']} "
        f"(k, {tgt['n_tokens']} token), task {s['task']}"
    )
    tf32 = s.get("tf32", {})
    lines = [
        f"# Update kernel CKA: {head}", "",
        "## Domanda", "",
        "Con lo stesso task t, i due modelli m e k cambiano la rappresentazione nello stesso modo quando si applica il "
        "proprio esperto, misurato nelle coordinate delle ancore condivise? E questo accordo e' piu' alto di quello "
        "ottenuto con un esperto di un altro task, con un update casuale o dentro lo stesso modello fra due task?", "",
        "## Procedura", "",
        f"- Modelli: m = {src['model']}/{src['pretrained']} (esperto {src['expert']['path']}), "
        f"k = {tgt['model']}/{tgt['pretrained']} (esperto {tgt['expert']['path']}).",
        f"- Task {s['task']}, {s['images']['task']['n_images']} immagini dello split `{s.get('split', '?')}` "
        f"(sha256 degli indici immagine {s['images']['task'].get('image_indices_sha256', '?')[:16]}, "
        f"delle etichette {s['images']['task'].get('labels_sha256', '?')[:16]}), seed {s['split_seed']}, batch {s['batch_size']}. "
        "Lo split `test` e' un sottoinsieme casuale con seme del test originale, disgiunto dai dati di training degli esperti.",
        f"- Identita' immagini: sha256 dei byte grezzi delle righe HF {s['images']['task'].get('raw_image_bytes_sha256', '?')[:16]}, "
        f"id di riga {s['images']['task'].get('row_ids_sha256')} (colonna {s['images']['task'].get('id_column')}); preprocess "
        f"sha256 m {s['images']['task'].get('preprocess_sha256', '?')[:16]}, k {s.get('images', {}).get('task_target_loader', {}).get('preprocess_sha256', '?')[:16]} "
        "(testo completo in `setting.preprocess`).",
        f"- Memoria: accumulatori stimati dal codice {s.get('ram_estimate_accumulators', {}).get('total_gb', float('nan')):.1f} GB, "
        f"m2={s.get('m2')}, somme in {s.get('accumulation_dtype', '?')}, blocchi eseguiti {s.get('blocks_run')}; "
        f"chunk del jackknife: {s.get('chunk_check')}.",
        f"- Ancore: state_dir `{s['state_dir']}`, T = {s['T_stored']} (dopo il filtro `{s['anchor_filter']}` k={s['anchor_filter_k']} "
        "T effettivo per blocco nel JSON). La diagnostica usa il frame A COPPIE (`pair_frames`, P divisa per c), "
        "NON il frame gpa/consensus.",
        f"- Blocchi accoppiati per profondita' relativa (j target -> i sorgente): {s['pairing']}.",
        "- dH = uscita del blocco (residuo) con esperto meno base; D = dH P^T (righe = token delle stesse immagini, colonne = "
        "le stesse ancore).",
        "- PRIMARIA (confronto diretto, numero grezzo): coseno NON centrato (`uncentered`) fra D_m e D_k sulle STESSE immagini, "
        "righe accoppiate per immagine e posizione canonica, per CLS, patch e all. E' il numero decoder-relevant: l'accordo di "
        "cio' che il decoder riceve davvero. Le altre varianti (`position_centered`, `centered`, `row_centered`, correlazione "
        "media per ancora) restano nel JSON come secondarie.",
        "- Comparatori del numero grezzo, appaiati per immagine/ancora con lo stesso jackknife: (1) altro task solo-esperto "
        "(k con l'esperto t' sulle STESSE immagini del task t), nelle due direzioni (m(t), k(t')) e lo speculare (m(t'), k(t)) "
        "e la loro media, con eccesso = vero - comparatore e IC jackknife sulle immagini; (2) update casuale (norma per tensore di tau_k): "
        "N estrazioni gaussiane indipendenti (o a coppie +g, -g con `random_antithetic`; allora il gruppo e' la coppia) formano una "
        "DISTRIBUZIONE NULLA del coseno; si riportano media, sd (ddof 1), z = (vero - media)/sd e p empirico (1 + #{coseno >= vero})/(1 + N), "
        "anche a due code centrato sulla media nulla; valori per estrazione e vero - estrazione 0 (con IC jackknife) nel JSON. "
        "L'estrazione 0 e' accumulata a chunk (IC, m1, m2), le altre in un solo chunk: z e p usano solo il coseno su tutte le immagini. "
        "Con `random_antithetic` le estrazioni a coppie (+g, -g) non sono indipendenti: z e p non sono calcolati (n/d), restano la media e le parti pari/dispari. "
        "Per m1 e m2 l'update casuale e' la sola estrazione 0. "
        "Null per permutazione delle ancore e per immagini disallineate (posizioni allineate): solo sanity.",
        f"- SECONDARIE: m1 = CKA debiased dei kernel di Gram K = D^T D sulle ancore (tutti i token nativi), variante "
        f"`{primary}` primaria fra le Gram, sempre anche `uncentered` (\"decoder-relevant (uncentred)\") e "
        "`position_centered`; riferimenti: cross-task nel modello, kernel della rappresentazione base, altro task solo-esperto, "
        "eccessi con jackknife su cluster di ancore della stessa immagine. m2 = CKA debiased con righe = token.",
        f"- Griglia patch: `{s['grid_mode']}` (riferimento `{s.get('grid_reference')}`). Stimatore "
        f"{'debiased' if s['debiased'] else 'biased'}, calcolo {s['compute_dtype']} con accumulo float64, "
        f"TF32 matmul={tf32.get('matmul')} cudnn={tf32.get('cudnn')} (disattivato).",
        f"- Intervalli: jackknife delete-one con t_(G-1): misura diretta e m2 su {s.get('jackknife_chunks_actual')} blocchi "
        f"di immagini (richiesti {s.get('jackknife_chunks_requested')}), m1 su cluster di ancore della stessa immagine.",
        (f"- Campionamento `stratified`: le {s['images']['task']['n_images']} immagini (stesso insieme del seme, split completo se "
         f"piu' piccolo) sono rimescolate con seme; i chunk del jackknife sono STRATIFICATI per classe (scarto massimo dalle "
         f"proporzioni di classe per chunk e classe: {s['images']['task'].get('chunk_class_balance', {}).get('max_abs_dev_from_proportional', '?')} "
         f"immagini, {s['images']['task'].get('chunk_class_balance', {}).get('n_classes', '?')} classi); il null a immagini disallineate "
         f"abbina ogni immagine a una di ALTRA classe nello stesso batch (coppie con stessa etichetta: "
         f"{(s.get('image_null_pairs') or {}).get('same_label', 'n/d')} su {(s.get('image_null_pairs') or {}).get('total', 'n/d')}). "
         f"Indici salvati in `{s.get('selected_images_file')}`."
         if s.get("sampling") == "stratified" else
         "- Campionamento `legacy`: immagini ordinate per indice del dataset (batch e chunk del jackknife nell'ordine delle classi se "
         "il dataset e' ordinato per classe: IC piu' stretti del dovuto, null immagini con partner spesso della stessa classe)."),
        "- Mean-shift: per ogni lato, somma sulle righe dell'update codificato dH P^T (`displacement`) e della posizione codificata "
        "H_base P^T / H_ft P^T (`base_position`, `ft_position`; H_ft = H_base + dH, la codifica e' lineare), per gruppo cls / patch / all, "
        "accumulata per chunk (file `" + str(s.get("mean_vectors_file")) + "`, float32). Misura: coseno dei vettori T di m e k dopo "
        "aver tolto la media sulle ancore (come `common_mode`), IC95 jackknife delete-one sui chunk (t_(G-1)); comparatori e eccessi "
        "come per la misura diretta (altro task: (m_t,k_o), (m_o,k_t), media; update casuale: estrazione 0).",
        f"- Controlli: {', '.join(s['controls'])}. Permutazioni ancore {s['permutations']}, permutazioni immagini "
        f"{s['image_permutations']}, seed controlli {s['control_seed']}, estrazioni update casuale {s.get('random_draws', 1)} "
        f"(antitetiche: {s.get('random_antithetic', False)}; seed: {s.get('random_seed_derivation', 'seed controlli, una estrazione')}).",
        f"- Codice: commit {s['code'].get('git_commit')} (dirty: {s['code'].get('git_dirty')}).", "",
        "## Risultato", "",
    ]

    def table(title, header, rowfn):
        lines.extend([f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)])
        for blk in blocks:
            cells = rowfn(blk)
            if cells is not None:
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    def bid(blk):
        return f"{blk['target_block']} ({blk['source_block']})"

    def de(blk, v, g):
        return blk.get("direct", {}).get(v, {}).get(g)

    def m1e(blk, v, g="all"):
        return blk.get("m1_variants", {}).get(v, {}).get(g)

    n_draws = s.get("random_draws", 1)

    def _rand_cells(rd):
        if not rd:
            return [""] * 5
        sd = rd["null_sd"]
        if rd.get("antithetic"):
            return [f"media {_fmt(rd['mean'])} (antitetiche)", f"{_fmt(rd['min'])} - {_fmt(rd['max'])}", "n/d", "n/d", "n/d"]
        return [f"{_fmt(rd['null_mean'])} ± {_fmt(sd) if sd is not None else 'n/d'}", f"{_fmt(rd['min'])} - {_fmt(rd['max'])}",
                _fmt(rd["z_random"]), _fmt(rd["p_random"]), _fmt(rd["p_random_two_sided"])]

    def _null_cell(e):
        nl = (e or {}).get("image_null")
        return "" if not nl else f"{_fmt(nl['mean'])} ± {_fmt(nl['std'])}"

    def ms(blk, kind, g):
        return blk.get("mean_shift", {}).get(kind, {}).get(g)

    if any("direct" in b for b in blocks):
        table(
            "PRIMARIA affiancata (gruppo all): coseno grezzo (`uncentered`), `position_centered`, null a immagini disallineate "
            "(media ± sd delle permutazioni) e coseno dello spostamento medio (centrato sulle ancore) con IC95 jackknife",
            ["blocco j (i)", "grezzo uncentered [IC]", "null immagini (uncentered)", "position_centered [IC]",
             "null immagini (position_centered)", "mean-shift spostamento [IC]", "mean-shift: eccesso vs altro task media [IC]"],
            lambda b: None if de(b, "uncentered", "all") is None else [
                bid(b), _ci(de(b, "uncentered", "all")), _null_cell(de(b, "uncentered", "all")),
                _ci(de(b, "position_centered", "all")), _null_cell(de(b, "position_centered", "all")),
                _ci(ms(b, "displacement", "all")), _ci((ms(b, "displacement", "all") or {}).get("excess_other_task_mean")),
            ],
        )
        for g in ("all", "patch", "cls"):
            table(
                f"PRIMARIA: coseno grezzo (`uncentered`) D_m vs D_k, gruppo {g}, comparatori ed eccessi (IC95 jackknife)",
                ["blocco j (i)", "valore [IC]", "altro task (m_t,k_o)", "altro task speculare (m_o,k_t)", "altro task media",
                 "eccesso vs altro task media", "eccesso vs altro task", "eccesso vs speculare",
                 f"update casuale: media ± sd ({n_draws} estrazioni)", "update casuale min-max", "z casuale",
                 "p casuale (1 coda)", "p casuale (2 code)"],
                lambda b, g=g: None if de(b, DIRECT_PRIMARY, g) is None else [
                    bid(b), _ci(de(b, DIRECT_PRIMARY, g)),
                    *[_fmt(de(b, DIRECT_PRIMARY, g).get(k, {}).get("value")) for k in
                      ("other_task", "other_task_rev", "other_task_mean")],
                    *[_ci(de(b, DIRECT_PRIMARY, g).get(k)) for k in
                      ("excess_other_task_mean", "excess_other_task", "excess_other_task_rev")],
                    *_rand_cells(de(b, DIRECT_PRIMARY, g).get("random_update_draws")),
                ],
            )
        table(
            "SECONDARIE nel JSON (gruppo all): `position_centered` [IC], `centered`, `row_centered`, corr. media per ancora",
            ["blocco j (i)", "position_centered [IC]", "altro task", "centered", "row_centered", "corr media per ancora",
             "corr media per ancora (per posizione)"],
            lambda b: None if de(b, "position_centered", "all") is None else [
                bid(b), _ci(de(b, "position_centered", "all")),
                _fmt(de(b, "position_centered", "all").get("other_task", {}).get("value")),
                _fmt(de(b, "centered", "all")["value"]), _fmt(de(b, "row_centered", "all")["value"]),
                _fmt(de(b, "mean_anchor_corr", "all")["value"]), _fmt(de(b, "mean_anchor_corr_position_centered", "all")["value"]),
            ],
        )
        if any("image_null" in (de(b, DIRECT_PRIMARY, "all") or {}) for b in blocks):
            table(
                "Sanity (primaria, all): null a immagini disallineate (posizioni allineate), z e p empirico; null delle ancore del coseno",
                ["blocco j (i)", "z immagini", "p immagini", "z ancore (uncentered)", "p ancore (uncentered)"],
                lambda b: [
                    bid(b), _fmt(de(b, DIRECT_PRIMARY, "all").get("image_null", {}).get("z")),
                    _fmt(de(b, DIRECT_PRIMARY, "all").get("image_null", {}).get("p_empirical")),
                    _fmt(de(b, "uncentered", "all").get("anchor_null", {}).get("z")),
                    _fmt(de(b, "uncentered", "all").get("anchor_null", {}).get("p_empirical")),
                ],
            )
    table(
        f"SECONDARIA m1 variante `{primary}`, gruppo all: valore ed eccessi sui riferimenti (IC95 jackknife)",
        ["blocco j (i)", "m1", "eccesso vs altro task", "eccesso vs cross-task nel modello", "eccesso vs base repr"],
        lambda b: None if m1e(b, primary) is None else [
            bid(b), _ci(m1e(b, primary)), _ci(m1e(b, primary).get("excess_other_task")),
            _ci(m1e(b, primary).get("excess_within_model_cross_task")), _ci(m1e(b, primary).get("excess_base_repr")),
        ],
    )
    table(
        "SECONDARIA m1 decoder-relevant (uncentred), gruppo all", ["blocco j (i)", "m1", "eccesso vs altro task", "eccesso vs base repr"],
        lambda b: None if m1e(b, "uncentered") is None else [
            bid(b), _ci(m1e(b, "uncentered")), _ci(m1e(b, "uncentered").get("excess_other_task")),
            _ci(m1e(b, "uncentered").get("excess_base_repr")),
        ],
    )
    table(
        "SECONDARIA m1 centrato per posizione, gruppo all", ["blocco j (i)", "m1", "eccesso vs altro task", "eccesso vs cross-task nel modello"],
        lambda b: None if m1e(b, "position_centered") is None else [
            bid(b), _ci(m1e(b, "position_centered")), _ci(m1e(b, "position_centered").get("excess_other_task")),
            _ci(m1e(b, "position_centered").get("excess_within_model_cross_task")),
        ],
    )
    table(
        "SECONDARIA m1 riferimenti (variante primaria Gram, all): altro task, cross-task nel modello (m, k), base repr",
        ["blocco j (i)", "altro task", "cross-task m", "cross-task k", "base repr"],
        lambda b: None if m1e(b, primary) is None else [
            bid(b), _fmt(m1e(b, primary).get("other_task", {}).get("value")),
            _fmt(m1e(b, primary).get("within_model_cross_task", {}).get("m")),
            _fmt(m1e(b, primary).get("within_model_cross_task", {}).get("k")),
            _fmt(m1e(b, primary).get("base_repr", {}).get("value")),
        ],
    )
    if any(b.get("m2") for b in blocks):
      table(
        "SECONDARIA m2 CKA sulle righe (gruppo patch)",
        ["blocco j (i)", "valore [IC95]", "altro task (solo esperto)", "altre immagini", "random", "trasportato", "z (immagini disallineate)"],
        lambda b: [
            bid(b), _ci(b["m2"]["patch"]),
            *[_fmt(b.get("controls", {}).get(n, {}).get("m2", {}).get("patch")) for n in
              ("other_task", "other_task_other_images", "random_update", "transported")],
            _fmt(b["m2"]["patch"].get("null", {}).get("z")),
        ],
      )
    if any("mean_shift" in b for b in blocks):
        for kind, title in (
            ("displacement", "spostamento medio dell'update dH (dove si sposta il dataset)"),
            ("ft_position", "posizione media del fine-tuned H_ft = H_base + dH (dove atterra il dataset)"),
        ):
            for g in ("all", "patch", "cls"):
                table(
                    f"Mean-shift, {title}: coseno fra m e k dei vettori T centrati sulle ancore, gruppo {g}, comparatori ed eccessi (IC95 jackknife)",
                    ["blocco j (i)", "coseno [IC]", "altro task (m_t,k_o) [IC]", "altro task speculare (m_o,k_t) [IC]",
                     "altro task media [IC]", "eccesso vs altro task media [IC]", "random (estrazione 0) [IC]"],
                    lambda b, kind=kind, g=g: None if ms(b, kind, g) is None else [
                        bid(b), _ci(ms(b, kind, g)), *[_ci(ms(b, kind, g).get(k)) for k in
                                                       ("other_task", "other_task_rev", "other_task_mean", "excess_other_task_mean", "random_update")],
                    ],
                )
        table(
            "Mean-shift, posizione media del base H_base (dipende solo dal modello e dalle immagini): coseno m vs k [IC]",
            ["blocco j (i)", "all", "patch", "cls"],
            lambda b: None if ms(b, "base_position", "all") is None else [
                bid(b), *[_ci(ms(b, "base_position", g)) for g in ("all", "patch", "cls")]],
        )
    if any("common_mode" in b for b in blocks):
        table(
            "Spostamento medio comune: frazione rho = n h^T Q h / tr(Q S) e coseno tra P h medi centrati sulle ancore",
            ["blocco j (i)", "rho m", "rho k", "cos medio (task vero)", "cos medio (altro task)"],
            lambda b: None if "common_mode" not in b else [
                bid(b), _fmt(b["common_mode"]["rho_m"]), _fmt(b["common_mode"]["rho_k"]),
                _fmt(b["common_mode"]["cos_mean_true_task"]), _fmt(b["common_mode"].get("cos_mean_other_task")),
            ],
        )
    if any("base_reference" in b for b in blocks):
        table(
            "Sanity: CKA dei kernel delle ancore P P^T (non e' un comparatore)", ["blocco", "CKA"],
            lambda b: None if "base_reference" not in b else [str(b["target_block"]), _ci(b["base_reference"])],
        )
    lines += [
        "CLS, mean-patch e le altre varianti per gruppo sono nel JSON e nel CSV (la primaria e' la prima chiave e le prime righe).", "",
        "## Come si legge", "",
        "- Il numero grezzo (coseno non centrato di D_m e D_k, stesse immagini e posizioni) e' l'accordo di cio' che il decoder "
        "riceve davvero; va letto contro il comparatore altro task (media delle due direzioni) tramite l'eccesso vero - comparatore "
        "con IC95: se l'IC include 0 non c'e' evidenza di accordo specifico del task. Contiene anche lo spostamento medio comune "
        "(vedi rho).",
        "- Il confronto con l'update casuale e' un test di livello di caso: z e p confrontano il coseno vero con la dispersione delle "
        "N estrazioni casuali (la media di estrazioni antitetiche e' ~0 per costruzione e le coppie non sono indipendenti: per questo con `random_antithetic` z e p non ci sono). L'IC "
        "jackknife sulle immagini NON include la varianza fra estrazioni, per questo l'eccesso vs casuale con IC non e' nella tabella "
        "primaria (nel JSON come `true_minus_random_draw0`, con la sola estrazione 0, senza interpretazione inferenziale). Il p minimo raggiungibile e' 1/(1 + N): "
        "con poche estrazioni non si puo' scendere sotto.",
        "- Le righe dei blocchi A_B non sono confronti indipendenti: blocchi target consecutivi condividono lo stesso blocco "
        "sorgente (accoppiamento [0,0,1,1,...] per profondita' relativa), quindi D_m e' la stessa per piu' righe della tabella.",
        "- La forma di Gram (m1, K = D^T D sulle ancore) e' dominata dall'accordo di frame: con aggiornamenti isotropi K ~ P P^T "
        "e due frame P_k = P_m Q danno gli stessi K a prescindere da quali update siano stati appresi; resta alto anche con "
        "update indipendenti. Per questo m1 e m2 sono secondarie e vanno lette solo tramite gli eccessi sui riferimenti.",
        "- `uncentered` contiene lo spostamento medio comune (se rho e' vicino a 1 domina quello); `position_centered`, "
        "`centered` e `row_centered` lo tolgono in modi diversi.",
        "- Null delle ancore (richiede `m2: true`), null delle immagini e CKA di P P^T sono solo sanity check, non comparatori.", "",
        "## Cautele", "",
        "- Il CKA debiased puo' essere leggermente negativo o fuori [0,1] vicino a zero.",
        "- Con griglie diverse le patch di k sono ricampionate bilinearmente: approssimazione (misura diretta, m2); m1 usa i token nativi.",
        "- Gli IC della misura diretta e di m2 sono jackknife su blocchi di immagini (dimensione uguale a meno di un batch); "
        "quelli di m1 su cluster di ancore della stessa immagine.",
        "- Il controllo altro task (solo esperto) usa le stesse immagini: cambia solo l'esperto. Il controllo altre immagini "
        "accoppia l'immagine i di t con l'immagine i di t' (indice, arbitrario).",
        ("- Il null per immagini e' un derangement fra classi diverse dentro ogni batch (immagini disallineate, posizioni allineate): "
         if s.get("sampling") == "stratified" else
         "- Il null per immagini e' un derangement dentro ogni batch (immagini disallineate, posizioni allineate; con lo split in ordine "
         "di classe molte coppie hanno la stessa classe, quindi il null e' troppo alto), 20 permutazioni: ") +
        "usare z e p empirico, non q95. Il null delle ancore per la variante `position_centered` non e' disponibile "
        "(servirebbero le somme per posizione fra ancore diverse).",
        "- Il controllo update casuale usa una direzione gaussiana per tensore con la stessa norma Frobenius di tau_k "
        "(errore relativo misurato dopo il cast al dtype del modello).",
        "- Diagnostica correlazionale: non dimostra che il trasporto funzioni.", "",
        "## File", "",
        "- `summary.json` (misura diretta per prima, poi m1 e m2, controlli, IC, setting completo), `blocks.csv`, `summary.md`.", "",
    ]
    return "\n".join(lines)


def write_outputs(summary: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    with open(out_dir / "blocks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(csv_rows(summary))
    (out_dir / "summary.md").write_text(render_md(summary))


_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")
MERGE_IGNORED_SETTING_KEYS = ("config", "seconds", "ram_estimate_accumulators", "blocks_run", "code", "control_info")


def _drop_keys(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return {k: _drop_keys(v, key) for k, v in value.items() if k != key}
    if isinstance(value, list):
        return [_drop_keys(v, key) for v in value]
    return value


def merge_summaries(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Concatenate the ``blocks`` of summaries of runs that differ only in ``blocks`` (A_B halves). The settings must
    agree (images, models, frame, controls, seeds, ...) except the keys of ``MERGE_IGNORED_SETTING_KEYS``; block
    indices must be disjoint. The merged setting is the first part's with ``merged_from`` (the blocks, configs and RAM
    estimate of each part)."""
    if not parts:
        raise ValueError("nothing to merge")
    # Object addresses in reprs (e.g. "<function _convert_to_rgb at 0x7d1a...>" in the preprocess text) differ between
    # processes; they are blanked before comparing so identical preprocessing is not reported as a difference.
    # The preprocess hashes of runs made before the hash ignored addresses differ for the same reason: they are
    # dropped here, and the (address-blanked) preprocess text itself is compared instead.
    def keep(st):
        out = json.loads(_ADDR_RE.sub(" at 0x?", json.dumps(
            {k: v for k, v in st.items() if k not in MERGE_IGNORED_SETTING_KEYS}, sort_keys=True, default=str)))
        return _drop_keys(out, "preprocess_sha256")
    ref = keep(parts[0]["setting"])
    for i, part in enumerate(parts[1:], 1):
        other = keep(part["setting"])
        diff = sorted(k for k in set(ref) | set(other) if ref.get(k) != other.get(k))
        if diff:
            raise ValueError(f"part {i} has a different setting: {diff}")
    blocks = [b for part in parts for b in part["blocks"]]
    ids = [b["target_block"] for b in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"overlapping target blocks: {sorted(ids)}")
    setting = copy.deepcopy(parts[0]["setting"])
    setting["merged_from"] = [
        {"blocks_run": p["setting"].get("blocks_run"), "config": p["setting"].get("config"),
         "ram_estimate_accumulators": p["setting"].get("ram_estimate_accumulators"), "code": p["setting"].get("code")}
        for p in parts
    ]
    setting["blocks_run"] = sorted(ids)
    setting["seconds"] = sum(float(p["setting"].get("seconds", 0.0)) for p in parts)
    return {"setting": setting, "blocks": sorted(blocks, key=lambda b: b["target_block"])}


def merge_vector_files(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Union of the blocks of ``mean_vectors.pt`` payloads of runs that differ only in ``blocks``; the meta must agree."""
    meta = parts[0]["meta"]
    blocks: dict[int, Any] = {}
    for i, part in enumerate(parts):
        if part["meta"] != meta:
            raise ValueError(f"part {i}: mean_vectors meta differs: {sorted(k for k in meta if part['meta'].get(k) != meta[k])}")
        dup = set(blocks) & set(part["blocks"])
        if dup:
            raise ValueError(f"overlapping blocks in mean_vectors: {sorted(dup)}")
        blocks.update(part["blocks"])
    return {"meta": meta, "blocks": dict(sorted(blocks.items()))}


def merge_dirs(dirs: Sequence[str | Path], out_dir: str | Path) -> dict[str, Any]:
    merged = merge_summaries([json.loads((Path(d) / "summary.json").read_text()) for d in dirs])
    out = Path(out_dir)
    write_outputs(merged, out)
    sel = [Path(d) / SELECTED_IMAGES_FILE for d in dirs]
    if all(f.exists() for f in sel):  # the image selection is part of the setting: every part must have used the same
        texts = {f.read_text() for f in sel}
        if len(texts) != 1:
            raise ValueError("the parts used different image selections (selected_images.json differs)")
        (out / SELECTED_IMAGES_FILE).write_text(texts.pop())
    vec = [Path(d) / MEAN_VECTORS_FILE for d in dirs]
    if all(f.exists() for f in vec):
        torch.save(merge_vector_files([torch.load(f, map_location="cpu", weights_only=False) for f in vec]), out / MEAN_VECTORS_FILE)
    return merged


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config")
    parser.add_argument("--merge", nargs="+", metavar="DIR", help="merge the summary.json of these part directories")
    parser.add_argument("--out", help="output directory of --merge / --matching")
    parser.add_argument("--matching", nargs="+", metavar="DIR",
                        help="cross-task mean-shift matching of the per-task run directories of one model pair "
                             "(see update_mean_matching)")
    args = parser.parse_args(argv)
    if args.matching:
        if not args.out:
            parser.error("--matching needs --out")
        from .update_mean_matching import run_matching

        run_matching(args.matching, args.out)
        return
    if args.merge:
        if not args.out:
            parser.error("--merge needs --out")
        merge_dirs(args.merge, args.out)
        return
    if not args.config:
        parser.error("--config, --merge or --matching is required")
    run(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
