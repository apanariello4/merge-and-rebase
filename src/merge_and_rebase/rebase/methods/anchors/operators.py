"""Effect operators (docs/.local/AGENT_NEXT_PHASE.md, step O1): an expert as a linear operator on the residual stream.

In the source m, per block i, with h = (S_base - mu_m) / c_m and e = (S_ft - S_base) / c_m on the task calibration
rows (all tokens; e NOT centred, its mean goes to the intercept):

    e ~ h W_E + 1 b_E^T        ridge, Ariadne's empirical-Bayes rule lambda = tr(H~^T H~) / (n - 1), H~ centred

optionally truncated to rank r_E by a data-weighted projection (the top-r_E eigenvectors V of W_E^T H~^T H~ W_E,
i.e. of the fitted displacement's scatter; W_E <- W_E V V^T). Carried into the GPA common space with the rank-r
maps A (rows of a_m[:r], a_k[:r]; a row h of m is h A_m^T there):

    W_U = A_m W_E A_m^T,  b_U = b_E A_m^T
    D_k = scale * (h_k A_k^T W_U + 1 b_U^T) A_k,   h_k = (S_base,k - mu_k) / c_k on ANY target calibration rows

so the target only needs its own base activations: the library holds (W_U, b_U) per depth, not data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

Tensor = torch.Tensor


@dataclass
class EffectOperator:
    w: Tensor  # [d, d] float64: e ~ h w + b
    b: Tensor  # [d]
    info: dict[str, Any]


class _Moments:
    """Uncentred float64 sums of one regression ``e ~ h``: sum h h^T, sum h e^T, sum h, sum e, n."""

    def __init__(self) -> None:
        self.hh = self.he = self.sh = self.se = None
        self.n = 0

    def update(self, h: Tensor, e: Tensor) -> None:
        h = h.reshape(-1, h.shape[-1]).to(torch.float64)
        e = e.reshape(-1, e.shape[-1]).to(torch.float64)
        if self.hh is None:
            d, k = h.shape[1], e.shape[1]
            self.hh = torch.zeros(d, d, dtype=torch.float64)
            self.he = torch.zeros(d, k, dtype=torch.float64)
            self.sh = torch.zeros(d, dtype=torch.float64)
            self.se = torch.zeros(k, dtype=torch.float64)
        self.hh += (h.T @ h).cpu()
        self.he += (h.T @ e).cpu()
        self.sh += h.sum(0).cpu()
        self.se += e.sum(0).cpu()
        self.n += int(h.shape[0])


def ridge_lambda(centred_gram: Tensor, n: int) -> float:
    """Ariadne's empirical-Bayes ridge (``ResidualSufficientStatistics.solve``, identity output map)."""
    return float(torch.trace(centred_gram).item()) / max(n - 1, 1)


def fit_effect_operator(
    h_batches, e_batches, *, rank: int | None = None, lam: float | None = None
) -> EffectOperator:
    """Ridge ``e ~ h W + 1 b^T`` over the given row batches (float64; intercept unpenalised).

    ``lam`` None: Ariadne's automatic rule; ``lam = 0`` is the unregularised least squares (pseudo-inverse).
    ``rank``: data-weighted truncation of W (module docstring); None keeps full rank.
    """
    mom = _Moments()
    for h, e in zip(h_batches, e_batches, strict=True):
        mom.update(h, e)
    return fit_from_moments(mom, rank=rank, lam=lam)


def fit_from_moments(mom: _Moments, *, rank: int | None = None, lam: float | None = None) -> EffectOperator:
    """``fit_effect_operator`` from accumulated moments (streamed rows)."""
    n = mom.n
    mu_h, mu_e = mom.sh / n, mom.se / n
    sc = mom.hh - torch.outer(mom.sh, mu_h)
    sc = 0.5 * (sc + sc.T)
    bc = mom.he - torch.outer(mom.sh, mu_e)
    lam_used = ridge_lambda(sc, n) if lam is None else float(lam)
    if lam_used > 0:
        w = torch.linalg.solve(sc + lam_used * torch.eye(sc.shape[0], dtype=torch.float64), bc)
    else:
        w = torch.linalg.pinv(sc) @ bc
    kept = None
    if rank is not None and rank < min(w.shape):
        m = w.T @ sc @ w
        ev, v = torch.linalg.eigh(0.5 * (m + m.T))
        v = v[:, ev.argsort(descending=True)[: int(rank)]]
        kept = float(ev.sort(descending=True).values[: int(rank)].sum() / ev.clamp_min(0).sum())
        w = w @ v @ v.T
    b = mu_e - w.T @ mu_h
    info = {"n_rows": n, "lambda": lam_used, "rank": None if rank is None else int(rank),
            "rank_energy_kept": kept, "w_norm": float(torch.linalg.norm(w)), "b_norm": float(torch.linalg.norm(b))}
    return EffectOperator(w, b, info)


def r_squared(op: EffectOperator, h_batches, e_batches) -> float:
    """1 - SSE / SST of the operator on the given rows (SST about the rows' own mean of e)."""
    sse = sst = 0.0
    sum_e = None
    n = 0
    sum_e2 = 0.0
    for h, e in zip(h_batches, e_batches, strict=True):
        h = h.reshape(-1, h.shape[-1]).to(torch.float64).cpu()
        e = e.reshape(-1, e.shape[-1]).to(torch.float64).cpu()
        sse += float(((h @ op.w + op.b - e) ** 2).sum())
        sum_e = e.sum(0) if sum_e is None else sum_e + e.sum(0)
        sum_e2 += float((e**2).sum())
        n += e.shape[0]
    sst = sum_e2 - float((sum_e**2).sum()) / n
    return 1.0 - sse / sst if sst > 0 else float("nan")


def to_common_space(op: EffectOperator, a_m: Tensor, rank: int | None) -> tuple[Tensor, Tensor]:
    """``(W_U, b_U)`` with ``A_m = a_m[:rank]`` (``[r, d_m]``, orthonormal rows of the GPA map)."""
    a = a_m.to(torch.float64)
    a = a if rank is None else a[: int(rank)]
    return a @ op.w @ a.T, op.b @ a.T


def decode_operator(
    h_k: Tensor, w_u: Tensor, b_u: Tensor, a_k: Tensor, rank: int | None, scale: float,
    input_map: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """``D = scale (u W_U + 1 b_U^T) A_k`` row-wise, ``u = h_k A_k^T`` (``h_k [..., d_k]``, already centred and
    scaled); ``input_map (M, m)``: ``u <- u M + m``, the target's common coordinates carried onto the source's."""
    a = a_k.to(torch.float64)
    a = a if rank is None else a[: int(rank)]
    shape = h_k.shape
    rows = h_k.reshape(-1, shape[-1]).to(torch.float64)
    u = rows @ a.T
    if input_map is not None:
        u = u @ input_map[0] + input_map[1]
    out = float(scale) * ((u @ w_u + b_u) @ a)
    return out.reshape(*shape[:-1], a.shape[1])


def common_rows(x: Tensor, a: Tensor, rank: int | None) -> Tensor:
    """``x A^T`` with ``A = a[:rank]`` (rows of ``x`` flattened to ``[n, d]``), float64."""
    a = a.to(torch.float64)
    a = a if rank is None else a[: int(rank)]
    return x.reshape(-1, x.shape[-1]).to(torch.float64) @ a.T


def explained(pred_batches, true_batches) -> tuple[float, float]:
    """``(R^2, cosine)`` of predicted against true rows over all batches (R^2 about the true rows' mean; cosine
    uncentred, of the stacked rows)."""
    sse = dot = pp = tt = 0.0
    s = None
    n = 0
    for p, t in zip(pred_batches, true_batches, strict=True):
        p, t = p.to(torch.float64), t.to(torch.float64)
        sse += float(((p - t) ** 2).sum())
        dot += float((p * t).sum())
        pp += float((p**2).sum())
        tt += float((t**2).sum())
        s = t.sum(0) if s is None else s + t.sum(0)
        n += t.shape[0]
    sst = tt - float((s**2).sum()) / n
    return (1.0 - sse / sst if sst > 0 else float("nan")), dot / max((pp * tt) ** 0.5, 1e-300)


def operator_nbytes(w_u: Tensor, rank: int | None) -> int:
    """float32 storage of one depth's W_U (as factors U, V^T when the operator rank is below the side) plus b_U."""
    r_side = int(w_u.shape[0])
    if rank is not None and rank < r_side:
        return 4 * (2 * r_side * int(rank) + r_side)
    return 4 * (r_side * r_side + r_side)


def save_library(path: Path, ops: dict[int, tuple[Tensor, Tensor]], meta: dict[str, Any]) -> None:
    """One expert: ``{depth index: (W_U, b_U)}`` plus metadata that must name the frame (universe) version."""
    if "universe_version" not in meta:
        raise ValueError("an operator library entry must record the GPA frame version it lives in")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ops": {int(j): (w.float(), b.float()) for j, (w, b) in ops.items()}, "meta": dict(meta)}, path)


def load_library(path: Path) -> tuple[dict[int, tuple[Tensor, Tensor]], dict[str, Any]]:
    stored = torch.load(path, map_location="cpu", weights_only=False)
    return {int(j): (w.double(), b.double()) for j, (w, b) in stored["ops"].items()}, stored["meta"]
