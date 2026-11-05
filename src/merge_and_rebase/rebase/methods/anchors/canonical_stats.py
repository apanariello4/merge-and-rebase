"""Streaming anchor statistics of the canonical decoder: ``Sigma_mm``, ``Sigma_kk``, ``Sigma_mk`` of a block pair.

With ``P_a = (S_a - mu_a) / c_a`` the anchor frames of ``anchors/state.py`` (registration ``mu``, ``c``), the statistics
are ``Sigma_ab = P_a^T P_b`` (``[d_a, d_b]``, float64, NOT divided by the number of rows: ``eigh(Sigma_aa) = V S^2 V^T``).
They are what ``core.canonical_factors_from_sigma`` consumes, so the canonical correlations Gamma, and the canonical
decoders, can be built without ever holding the ``[T, d]`` frames.

Numerics. ``mu`` and ``c`` come from the registration set, which is fixed and disjoint from the anchors, so the rows
are centred and scaled BEFORE they are squared: every batch adds ``((S_m - mu_m) / c_m)^T ((S_k - mu_k) / c_k)`` in
float64 and no ``sum x x^T - n mu mu^T`` cancellation appears (the residual stream has a large common mean, see
``core.MomentAccumulator``). This is exact up to float64 rounding and independent of the batch sizes; it is the
"registration pass first, anchor pass second" form of the two-pass scheme. (Chan's merge of
``_ariadne.streaming._StreamingCrossCovariance`` centres on the stream's own mean, which is not what ``P`` uses.)

Halves. ``split_ids`` splits the image / document ids (never the rows) into two seeded halves; ``update`` takes the id of
every row (the first column of the frame rows ``[image, pos]`` / ``[seq, pos]``) and adds it to its half. The total is
the sum of the halves.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def split_ids(ids: Tensor | list[int], seed: int) -> tuple[Tensor, Tensor]:
    """Two disjoint halves of the distinct ids, seeded and independent of the order the ids come in.

    Returns ``(half_0, half_1)`` (sorted int64 tensors) with ``|half_0| = ceil(n / 2)``; the same ids and seed always
    give the same halves.
    """
    uniq = torch.unique(torch.as_tensor(ids, dtype=torch.long))  # sorted
    n = int(uniq.numel())
    if n < 2:
        raise ValueError("splitting in halves needs at least 2 distinct ids")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
    cut = (n + 1) // 2
    return torch.sort(uniq[perm[:cut]]).values, torch.sort(uniq[perm[cut:]]).values


class PairSigmaAccumulator:
    """Streaming ``Sigma_mm``, ``Sigma_kk``, ``Sigma_mk`` of one (source block, target block) pair, optionally per half.

    ``mu_*`` / ``c_*`` are the centring and scale of the frame the decoder uses: the registration ``mu`` / ``c`` or,
    under ``anchor_filter``, the filtered ones (``method.filtered_pair_inputs`` returns them together with the row mask
    of the flagged anchors). ``update(s_m, s_k, ids, mask)`` takes the RAW anchor rows ``[n, d_m]`` / ``[n, d_k]`` of
    the two models (same rows, same order), the id of each row and optionally a bool ``mask`` ``[n]`` (False rows are
    dropped before accumulation, e.g. the norm-filtered anchors). ``halves = (ids_0, ids_1)`` (from ``split_ids``) enables the split; a row whose id is
    in neither half raises.
    """

    def __init__(
        self,
        mu_m: Tensor,
        c_m: float,
        mu_k: Tensor,
        c_k: float,
        *,
        halves: tuple[Tensor, Tensor] | None = None,
    ) -> None:
        self.mu_m = mu_m.to(torch.float64)
        self.mu_k = mu_k.to(torch.float64)
        self.c_m = float(c_m)
        self.c_k = float(c_k)
        if not (self.c_m > 0.0 and self.c_k > 0.0):
            raise ValueError("c_m and c_k must be positive")
        if halves is not None and bool(torch.isin(halves[0], halves[1]).any()):
            raise ValueError("the two halves share ids")
        self.halves = halves
        d_m, d_k = int(self.mu_m.numel()), int(self.mu_k.numel())
        n_halves = 1 if halves is None else 2
        zero = torch.zeros
        self._mm = [zero(d_m, d_m, dtype=torch.float64) for _ in range(n_halves)]
        self._kk = [zero(d_k, d_k, dtype=torch.float64) for _ in range(n_halves)]
        self._mk = [zero(d_m, d_k, dtype=torch.float64) for _ in range(n_halves)]
        self.n = [0] * n_halves
        self._ids: list[set[int]] = [set() for _ in range(n_halves)]

    def update(self, s_m: Tensor, s_k: Tensor, ids: Tensor | None = None, mask: Tensor | None = None) -> None:
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
            if mask.shape[0] != s_m.reshape(-1, s_m.shape[-1]).shape[0]:
                raise ValueError("mask must give one flag per row")
            s_m = s_m.reshape(-1, s_m.shape[-1])[mask]
            s_k = s_k.reshape(-1, s_k.shape[-1])[mask]
            if ids is not None:
                ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)[mask]
        sm = s_m.reshape(-1, s_m.shape[-1]).to(torch.float64)
        sk = s_k.reshape(-1, s_k.shape[-1]).to(torch.float64)
        if sm.shape[0] != sk.shape[0]:
            raise ValueError(f"source and target batches must have the same rows, got {sm.shape[0]} vs {sk.shape[0]}")
        if sm.shape[0] == 0:
            return
        pm = (sm - self.mu_m) / self.c_m
        pk = (sk - self.mu_k) / self.c_k
        if self.halves is None:
            self._add(0, pm, pk, ids)
            return
        if ids is None:
            raise ValueError("ids are required when the statistics are split in halves")
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
        if ids.shape[0] != pm.shape[0]:
            raise ValueError("ids must give one id per row")
        masks = [torch.isin(ids, h) for h in self.halves]
        if not bool((masks[0] | masks[1]).all()):
            raise ValueError("a row id belongs to neither half")
        for h, mask in enumerate(masks):
            if bool(mask.any()):
                self._add(h, pm[mask], pk[mask], ids[mask])

    def _add(self, h: int, pm: Tensor, pk: Tensor, ids: Tensor | None) -> None:
        self._mm[h] += pm.T @ pm
        self._kk[h] += pk.T @ pk
        self._mk[h] += pm.T @ pk
        self.n[h] += int(pm.shape[0])
        if ids is not None:
            self._ids[h].update(int(i) for i in torch.unique(ids))

    def sigma(self, half: int | None = None) -> tuple[Tensor, Tensor, Tensor, int]:
        """``(Sigma_mm, Sigma_kk, Sigma_mk, n_rows)`` of one half (0 / 1) or of everything seen (``half=None``)."""
        idx = range(len(self.n)) if half is None else [half]
        if half is not None and not 0 <= half < len(self.n):
            raise ValueError(f"half must be None, 0 or 1 (1 only with halves), got {half}")
        n = sum(self.n[i] for i in idx)
        if n == 0:
            raise ValueError("no rows accumulated")
        return (
            sum(self._mm[i] for i in idx),
            sum(self._kk[i] for i in idx),
            sum(self._mk[i] for i in idx),
            n,
        )

    def ids_seen(self, half: int) -> set[int]:
        return set(self._ids[half])
