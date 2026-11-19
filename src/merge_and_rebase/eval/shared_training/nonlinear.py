"""Non-linear shared-space experiments (AGENT_SHARED_TRAINING.md section 4): NL-2 local, NL-1, NL-2 global.

NL-2 local (closed form on activations, extends O1 + O2). For each student m with its expert on the task, at the grid
points of S IN ORDER (sequential): with the adapter already fitted at the earlier grid points applied (hooks), the
current stream ``S`` and the expert stream ``S_e`` at block ``pi_m(j)``:

    u = ((S - mu) / c) Phi^T,   U = ((S_e - S) / c) Phi^T            (every token)
    min_{W_j, b_j}  sum_m w_m mean_rows ||u W + 1 b^T - U||^2 + lam ||W||^2 + lam_P (1/T) ||G W + 1 b^T||^2

(float64 normal equations on ``[u 1]``, intercept unpenalised in the ridge). The sequential order and the current
stream make the fit match the adapter at inference (u from the current stream) and avoid counting a displacement
twice across depths. Several students = fusion by regression (``2b``: experts of A and C).

NL-1 (task loss through the true models) and NL-2 global (KL from teacher predictions): Adam on (W, b), one student
model per step (round robin), the stream after the first block of S cached per student, the adapter applied at that
block explicitly and at the later blocks by the hooks. Loss ``CE`` (labels) or ``KL(p_teacher || p_student)``,
``+ (lam_P/2) R_P + (lam/2) ||W||^2``. Initialised from a linear solution (``init``: a ``theta.pt``). The epoch with
the best students' mean validation accuracy (true model, adapter) is kept.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from merge_and_rebase.rebase.methods.anchors.jacobian import VisionTail
from merge_and_rebase.rebase.methods.anchors.shared_space import (
    ModelFrame,
    SharedAdapter,
    adapter_hooks,
    anchor_conservation,
)

from .evaluate import adapter_accuracy, adapter_from_theta, frame_on
from .lin import Theta

Tensor = torch.Tensor


# ------------------------------------------------------------------ NL-2 local


class _Normal:
    """Weighted sums of ``[u 1]^T [u 1]`` and ``[u 1]^T U`` (float64)."""

    def __init__(self, r: int, device) -> None:
        self.xx = torch.zeros(r + 1, r + 1, dtype=torch.float64, device=device)
        self.xy = torch.zeros(r + 1, r, dtype=torch.float64, device=device)
        self.trace_u = 0.0
        self.weight = 0.0

    def add(self, u: Tensor, target: Tensor, w: float) -> None:
        u = u.reshape(-1, u.shape[-1]).double()
        t = target.reshape(-1, target.shape[-1]).double()
        x = torch.cat([u, torch.ones(u.shape[0], 1, dtype=u.dtype, device=u.device)], dim=1)
        self.xx += w * (x.T @ x)
        self.xy += w * (x.T @ t)
        uc = u - u.mean(0, keepdim=True)
        self.trace_u += w * float((uc**2).sum())
        self.weight += w * u.shape[0]


def solve_local(stats: _Normal, g: Tensor, *, lam_rel: float, lam_p_rel: float) -> tuple[Tensor, Tensor, dict]:
    """Ridge + anchor conservation on the accumulated normal equations. ``lam = lam_rel * tr(u~^T u~)/(n r)`` (mean
    variance of the centred coordinates), ``lam_P = lam_p_rel * lam`` on ``(1/T) [G 1]^T [G 1]``."""
    r = stats.xy.shape[1]
    n = stats.weight
    a = stats.xx / n
    rhs = stats.xy / n
    scale = stats.trace_u / (n * r)
    lam = lam_rel * scale
    a = a.clone()
    idx = torch.arange(r, device=a.device)
    a[idx, idx] += lam
    if lam_p_rel > 0:
        gg = g.to(a.device, torch.float64)
        xg = torch.cat([gg, torch.ones(gg.shape[0], 1, dtype=gg.dtype, device=gg.device)], dim=1)
        a = a + lam_p_rel * lam * (xg.T @ xg) / gg.shape[0]
    theta = torch.linalg.solve(a, rhs)
    w, b = theta[:r], theta[r]
    return w, b, {"lam": lam, "scale": scale}


@torch.no_grad()
def fit_nl2_local(students: Sequence[dict[str, Any]], grid: Sequence[int], r: int, g: dict[int, Tensor], *,
                  lam_rel: float, lam_p_rel: float, device: str, log=print) -> tuple[Theta, dict]:
    """``students``: dicts with ``model`` (base nn.Module), ``expert`` (nn.Module), ``frame`` (ModelFrame),
    ``batches`` (list of image tensors). Sequential over ``grid``; every student weighs equally."""
    started = time.time()
    grid = sorted(int(j) for j in grid)
    theta: Theta = {}
    info: dict[str, Any] = {}
    for j in grid:
        stats = _Normal(r, device)
        for st in students:
            fdev = frame_on(st["frame"], device)
            block = st["frame"].blocks[st["frame"].index(j)]
            base_tail = VisionTail(st["model"], torch.zeros(1, 1))
            exp_tail = VisionTail(st["expert"], torch.zeros(1, 1))
            done = {k: v for k, v in theta.items()}
            ad = adapter_from_theta(done, device) if done else None
            n_rows = sum(int(x.shape[0]) for x in st["batches"])
            for x in st["batches"]:
                if ad is not None:
                    with adapter_hooks(st["model"], fdev, ad):
                        cur = base_tail.streams(x, [block])[block]
                else:
                    cur = base_tail.streams(x, [block])[block]
                se = exp_tail.streams(x, [block])[block]
                u = fdev.encode(cur, j)
                tgt = fdev.encode_delta(se - cur, j)
                # each student weighs equally: its rows (images x tokens) average to weight 1 / M
                stats.add(u, tgt, 1.0 / (len(students) * n_rows * cur.shape[1]))
        w, b, sinfo = solve_local(stats, g[j], lam_rel=lam_rel, lam_p_rel=lam_p_rel)
        theta[j] = (w, b)
        info[str(j)] = sinfo
        log(f"[NL-2 local] grid {j}: lam {sinfo['lam']:.3e} |W| {float(w.norm()):.3f} |b| {float(b.norm()):.3f}")
    info["seconds"] = time.time() - started
    return theta, info


# ------------------------------------------------------------------ NL-1 / NL-2 global


def adapted_logits(tail: VisionTail, frame: ModelFrame, adapter: SharedAdapter, h0: Tensor, start: int) -> Tensor:
    """Logits of the true model with the adapter, from the cached stream after ``start``: the displacement at
    ``start`` (if it is a block of S) applied here, the later ones by the hooks."""
    h = h0
    for j, b in zip(frame.grid, frame.blocks, strict=True):
        if b == start and j in adapter.grid:
            h = h + frame.decode(adapter.displacement(frame.encode(h, j), j), j)
    with adapter_hooks(tail.visual, frame, _AfterStart(adapter, frame, start)):
        return tail.logits(h, start)


class _AfterStart(SharedAdapter):
    """View of an adapter without the grid points at ``start`` (applied explicitly by ``adapted_logits``)."""

    def __init__(self, adapter: SharedAdapter, frame: ModelFrame, start: int) -> None:  # noqa: D107
        torch.nn.Module.__init__(self)
        self._inner = adapter
        self.grid = [j for j, b in zip(frame.grid, frame.blocks, strict=True) if b != start and j in adapter.grid]
        self.dim = adapter.dim
        self.has_bias = adapter.has_bias

    def displacement(self, u: Tensor, j: int) -> Tensor:
        return self._inner.displacement(u, j)


def train_through_models(students: Sequence[dict[str, Any]], grid: Sequence[int], r: int, g: dict[int, Tensor], *,
                         init: Theta | None, loss: str, lr: float, epochs: int, batch: int, lam: float,
                         lam_p: float, device: str, val_fn, seed: int = 0, log=print) -> tuple[Theta, dict]:
    """Adam on (W, b) through the true students. ``students``: dicts with ``tail`` (VisionTail of the base model),
    ``frame`` (ModelFrame on device), ``start``, ``h0`` (CPU [n, N, d] cached streams), ``labels`` [n], and for
    ``loss='kl'`` ``teacher`` [n, C] logits. ``val_fn(theta) -> float`` (mean students' val accuracy)."""
    started = time.time()
    torch.manual_seed(seed)
    ad = SharedAdapter(sorted(grid), r, dtype=torch.float32).to(device)
    if init is not None:
        ad.set_({j: init[j][0] for j in ad.grid}, {j: init[j][1] for j in ad.grid})
    opt = torch.optim.Adam(ad.parameters(), lr=lr)
    gdev = {j: g[j].to(device, torch.float32) for j in ad.grid}
    history = []
    best = (val_fn(_theta_of(ad)), _theta_of(ad), 0)
    history.append({"epoch": 0, "val": best[0]})
    log(f"[NL] init val {best[0]:.4f}")
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        perms = [torch.randperm(st["h0"].shape[0], generator=gen) for st in students]
        steps = max(-(-st["h0"].shape[0] // batch) for st in students)
        tot, cnt = 0.0, 0
        for step in range(steps):
            for st, perm in zip(students, perms, strict=True):  # one model per optimiser step, round robin
                idx = perm[(step * batch) % len(perm):][:batch]
                if len(idx) == 0:
                    continue
                h0 = st["h0"][idx].to(device)
                z = adapted_logits(st["tail"], st["frame"], ad, h0, st["start"])
                if loss == "ce":
                    data = F.cross_entropy(z, st["labels"][idx].to(device))
                else:
                    pt = F.log_softmax(st["teacher"][idx].to(device, z.dtype), -1)
                    data = F.kl_div(F.log_softmax(z, -1), pt, log_target=True, reduction="batchmean")
                reg = 0.5 * lam * sum((ad.weight(j) ** 2).sum() for j in ad.grid)
                if lam_p > 0:
                    reg = reg + 0.5 * lam_p * anchor_conservation(ad, gdev)
                total = data + reg
                opt.zero_grad(set_to_none=True)
                total.backward()
                opt.step()
                tot += float(data)
                cnt += 1
        val = val_fn(_theta_of(ad))
        history.append({"epoch": epoch, "train_loss": tot / max(cnt, 1), "val": val})
        log(f"[NL] epoch {epoch} loss {tot / max(cnt, 1):.4f} val {val:.4f} ({time.time() - started:.0f}s)")
        if val > best[0]:
            best = (val, _theta_of(ad), epoch)
    return best[1], {"history": history, "best_epoch": best[2], "best_val": best[0], "seconds": time.time() - started}


def _theta_of(ad: SharedAdapter) -> Theta:
    st = ad.state()
    return {j: (st["w"][j], st["b"][j]) for j in st["grid"]}


__all__ = ["adapted_logits", "fit_nl2_local", "solve_local", "train_through_models", "adapter_accuracy"]
