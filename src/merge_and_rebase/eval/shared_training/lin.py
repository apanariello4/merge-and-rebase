"""Linear experiments in the shared space (AGENT_SHARED_TRAINING.md section 3): LIN-1 and LIN-2.

Parameters ``theta = {j: (W_j [r, r], b_j [r])}`` over the grid points ``S``, float64. For a student model k and an
example x the linearised change of the logits is

    dz_k(x; theta) = J_k(x) . vec(u_k(x) W + 1 b^T)          (sum over j in S; u_k = base-stream coordinates)

with two linearisation points (``lin_point``):

* ``task``: ``J_k(x)`` of the model on x, applied matrix-free (``common_jvp`` / ``common_vjp`` from the cached stream
  after the first block of S; ``u`` recomputed from it by a no-grad tail pass);
* ``anchors``: the universal ``Jbar(j)`` (C x canonical tokens x r, from the anchors, diagnostics H3), the same for
  every model: ``dz = sum_j <Jbar_j, to_canonical(u_j) W_j + 1 b_j^T>`` (resampling commutes with ``W`` and keeps
  constants, so ``to_canonical(u W + 1 b^T) = to_canonical(u) W + 1 b^T``).

Every logit change is centred over the classes (softmax shift invariance). Models are weighted equally (``w_k =
1 / (n_k M)``).

LIN-1 (closed form, no labels): ``min sum_k w_k sum_x ||P(dz_k(x) - t_k(x))||^2 + lam ||W||^2 + lam_P R_P``,
``t_k(x) = mean_teachers z_T(x) - z_k(x)`` (``P`` centring), solved by conjugate gradient on the normal equations.
LIN-2 (labels): ``sum_k w_k sum_x CE(z_k(x) + dz_k(x), y) + (lam/2) ||W||^2 + (lam_P/2) R_P`` by L-BFGS (convex).
``lam = lam_rel * s``, ``lam_P = lam_P_rel * s / s_P`` with ``s`` the mean diagonal of the data Hessian on the W
block (Hutchinson, ``data_scale``) and ``s_P = tr(H_P)/dim`` the one of ``R_P``: relative values are comparable
across tasks and models.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from merge_and_rebase.rebase.methods.anchors.jacobian import VisionTail, common_jvp, common_vjp
from merge_and_rebase.rebase.methods.anchors.shared_space import ModelFrame
from merge_and_rebase.rebase.methods.anchors.state import to_canonical

Tensor = torch.Tensor
Theta = dict[int, tuple[Tensor, Tensor]]

CANON = 197


# ------------------------------------------------------------------ parameter algebra


def theta_zeros(grid: Sequence[int], r: int, device) -> Theta:
    return {j: (torch.zeros(r, r, dtype=torch.float64, device=device), torch.zeros(r, dtype=torch.float64,
                                                                               device=device)) for j in grid}


def theta_axpy(a: float, x: Theta, y: Theta) -> Theta:
    return {j: (a * x[j][0] + y[j][0], a * x[j][1] + y[j][1]) for j in x}


def theta_scale(a: float, x: Theta) -> Theta:
    return {j: (a * w, a * b) for j, (w, b) in x.items()}


def theta_dot(x: Theta, y: Theta) -> float:
    return float(sum((x[j][0] * y[j][0]).sum() + (x[j][1] * y[j][1]).sum() for j in x))


def theta_randn(grid, r, device, gen) -> Theta:
    return {j: (torch.randn(r, r, generator=gen, dtype=torch.float64).to(device),
                torch.randn(r, generator=gen, dtype=torch.float64).to(device)) for j in grid}


def theta_flat(x: Theta) -> Tensor:
    return torch.cat([torch.cat([w.reshape(-1), b]) for _, (w, b) in sorted(x.items())])


def theta_unflat(v: Tensor, grid: Sequence[int], r: int) -> Theta:
    out, i = {}, 0
    for j in sorted(grid):
        w = v[i : i + r * r].reshape(r, r)
        i += r * r
        b = v[i : i + r]
        i += r
        out[j] = (w, b)
    return out


def centre(z: Tensor) -> Tensor:
    return z - z.mean(dim=-1, keepdim=True)


# ------------------------------------------------------------------ data and operators


@dataclass
class StudentBatch:
    """One batch of one student model (CPU tensors; moved to the device per pass)."""

    model: str
    h0: Tensor | None  # [B, N, d] stream after the start block (lin_point 'task')
    ucan: dict[int, Tensor] | None  # j -> [B, CANON, r] canonical coordinates (lin_point 'anchors')
    z0: Tensor  # [B, C] base logits
    target: Tensor | None = None  # [B, C] teacher change (LIN-1)
    labels: Tensor | None = None  # [B] (LIN-2)


@dataclass
class Student:
    model: str
    frame: ModelFrame
    tail: VisionTail | None
    start: int
    batches: list[StudentBatch] = field(default_factory=list)

    @property
    def n(self) -> int:
        return sum(len(b.z0) for b in self.batches)


@torch.no_grad()
def tail_streams(tail: VisionTail, h: Tensor, start: int, blocks: Sequence[int]) -> dict[int, Tensor]:
    """Base streams after ``blocks`` (>= start) from the stream after ``start``."""
    want = {int(b) for b in blocks}
    out = {start: h} if start in want else {}
    x = h
    for j in range(start + 1, max(want) + 1):
        x = tail._run_block(tail.blocks[j], x)
        if j in want:
            out[j] = x
    return out


class LinearizedProblem:
    """The linear map ``theta -> dz`` over all students' batches and its adjoint, for one ``lin_point``."""

    def __init__(self, students: list[Student], grid: Sequence[int], r: int, *, lin_point: str, device,
                 jbar: dict[int, Tensor] | None = None) -> None:
        if lin_point not in ("task", "anchors"):
            raise ValueError(f"lin_point must be 'task' or 'anchors', got {lin_point!r}")
        if lin_point == "anchors" and jbar is None:
            raise ValueError("lin_point='anchors' needs the universal Jacobian jbar")
        self.students = students
        self.grid = [int(j) for j in grid]
        self.r = int(r)
        self.lin_point = lin_point
        self.device = device
        self.jbar = None if jbar is None else {j: jbar[j].to(device=device, dtype=torch.float64) for j in self.grid}
        n_models = len(students)
        self.weights = {s.model: 1.0 / (s.n * n_models) for s in students}

    # -- per batch
    def _coords(self, st: Student, mb: StudentBatch) -> tuple[dict[int, Tensor], Tensor | None]:
        if self.lin_point == "anchors":
            return {j: mb.ucan[j].to(self.device, torch.float64) for j in self.grid}, None
        h0 = mb.h0.to(self.device)
        blocks = [st.frame.blocks[st.frame.index(j)] for j in self.grid]
        streams = tail_streams(st.tail, h0, st.start, blocks)
        u = {j: st.frame.encode(streams[b], j) for j, b in zip(self.grid, blocks, strict=True)}
        return u, h0

    def _apply(self, st, theta: Theta, u, h0) -> Tensor:
        du = {j: u[j] @ theta[j][0].to(u[j].dtype) + theta[j][1].to(u[j].dtype) for j in self.grid}
        if self.lin_point == "anchors":
            return sum(torch.einsum("cnr,bnr->bc", self.jbar[j], du[j]) for j in self.grid).double()
        _, dz = common_jvp(st.tail, st.frame, h0, st.start, du)
        return dz.double()

    def _adjoint(self, st, g: Tensor, u, h0) -> Theta:
        if self.lin_point == "anchors":
            gu = {j: torch.einsum("bc,cnr->bnr", g.double(), self.jbar[j]) for j in self.grid}
        else:
            gu = common_vjp(st.tail, st.frame, h0, st.start, self.grid, g.to(h0.dtype))
        return {j: (torch.einsum("bns,bnr->sr", u[j].double(), gu[j].double()), gu[j].double().sum((0, 1)))
                for j in self.grid}

    def sweep(self, theta: Theta | None, cot: Callable[[Tensor, StudentBatch, float], tuple[Tensor, float]]
              ) -> tuple[Theta, float]:
        """``sum_batches J^T g`` with ``g, loss = cot(dz, batch, w)`` (``dz = J theta``; ``theta=None``: dz = 0, no
        forward). Returns the accumulated adjoint and loss."""
        acc = theta_zeros(self.grid, self.r, self.device)
        total = 0.0
        for st in self.students:
            w = self.weights[st.model]
            for mb in st.batches:
                u, h0 = self._coords(st, mb)
                if theta is None:
                    dz = torch.zeros_like(mb.z0, dtype=torch.float64, device=self.device)
                else:
                    dz = self._apply(st, theta, u, h0)
                g, loss = cot(dz, mb, w)
                total += loss
                adj = self._adjoint(st, g, u, h0)
                acc = theta_axpy(1.0, adj, acc)
        return acc, total

    def predict(self, theta: Theta) -> dict[str, Tensor]:
        """``dz`` per student, concatenated over its batches (CPU)."""
        out = {}
        for st in self.students:
            parts = []
            for mb in st.batches:
                u, h0 = self._coords(st, mb)
                parts.append(self._apply(st, theta, u, h0).cpu())
            out[st.model] = torch.cat(parts)
        return out

    def gauss_newton(self, p: Theta) -> Theta:
        """``A^T W P A p`` (the LIN-1 data Hessian, centring included)."""

        def cot(dz, mb, w):
            return w * centre(dz), 0.0

        return self.sweep(p, cot)[0]


def anchor_hessian(theta: Theta, g: dict[int, Tensor]) -> Theta:
    """``H_P theta`` for ``R_P/2``: W part ``(1/T) G^T (G W + 1 b^T)``, b part ``(1/T) 1^T (G W + 1 b^T)``."""
    out = {}
    for j, (w, b) in theta.items():
        gj = g[j].to(device=w.device, dtype=torch.float64)
        res = (gj @ w + b) / gj.shape[0]
        out[j] = (gj.T @ res, res.sum(0))
    return out


def anchor_hessian_scale(g: dict[int, Tensor]) -> float:
    """``tr(H_P) / dim`` (mean diagonal of the anchor term's Hessian)."""
    tr, dim = 0.0, 0
    for gj in g.values():
        gj = gj.to(torch.float64)
        t, r = gj.shape
        tr += float((gj**2).sum()) / t * r + r  # W block: r * ||G||^2 / T ; b block: r * T / T
        dim += r * r + r
    return tr / dim


def data_scale(problem: LinearizedProblem, probes: int = 2, seed: int = 0) -> float:
    """Hutchinson estimate of the mean diagonal of ``A^T W P A`` restricted to the W block."""
    gen = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(probes):
        p = theta_randn(problem.grid, problem.r, problem.device, gen)
        p = {j: (w, torch.zeros_like(b)) for j, (w, b) in p.items()}
        hp = problem.gauss_newton(p)
        vals.append(theta_dot(p, hp) / theta_dot(p, p))
    return float(sum(vals) / len(vals))


def _reg(theta: Theta, lam: float, lam_p: float, g: dict[int, Tensor] | None) -> Theta:
    out = {j: (lam * w, torch.zeros_like(b)) for j, (w, b) in theta.items()}
    if lam_p > 0 and g is not None:
        out = theta_axpy(lam_p, anchor_hessian(theta, g), out)
    return out


# ------------------------------------------------------------------ LIN-1: conjugate gradient


def solve_lin1(problem: LinearizedProblem, *, lam: float, lam_p: float = 0.0, g: dict[int, Tensor] | None = None,
               x0: Theta | None = None, tol: float = 1e-4, max_iter: int = 200, log=print) -> tuple[Theta, dict]:
    """CG on ``(A^T W P A + lam I_W + lam_P H_P) theta = A^T W P t`` (P: class centring)."""
    started = time.time()

    def rhs_cot(dz, mb, w):
        t = mb.target.to(dz.device, torch.float64)
        return w * centre(t), 0.0

    rhs = problem.sweep(None, rhs_cot)[0]

    def matvec(p):
        return theta_axpy(1.0, problem.gauss_newton(p), _reg(p, lam, lam_p, g))

    x = theta_zeros(problem.grid, problem.r, problem.device) if x0 is None else x0
    rvec = theta_axpy(-1.0, matvec(x), rhs) if x0 is not None else rhs
    p = rvec
    rs = theta_dot(rvec, rvec)
    rhs_norm = math.sqrt(theta_dot(rhs, rhs))
    history = []
    it = 0
    for it in range(1, max_iter + 1):
        ap = matvec(p)
        alpha = rs / theta_dot(p, ap)
        x = theta_axpy(alpha, p, x)
        rvec = theta_axpy(-alpha, ap, rvec)
        rs_new = theta_dot(rvec, rvec)
        rel = math.sqrt(rs_new) / max(rhs_norm, 1e-300)
        history.append(rel)
        if it % 10 == 0:
            log(f"[LIN-1] cg it {it} rel residual {rel:.2e} ({time.time() - started:.0f}s)")
        if rel < tol:
            break
        p = theta_axpy(rs_new / rs, p, rvec)
        rs = rs_new
    return x, {"cg_iters": it, "cg_rel_residual": history[-1] if history else 0.0, "seconds": time.time() - started,
               "cg_history": history[:: max(1, len(history) // 50)]}


def lin1_fit_quality(problem: LinearizedProblem, theta: Theta) -> dict[str, float]:
    """Relative residual ``||P(dz - t)|| / ||P t||`` per student on the training examples (linear model)."""
    pred = problem.predict(theta)
    out = {}
    for st in problem.students:
        t = torch.cat([mb.target for mb in st.batches]).double()
        out[st.model] = float((centre(pred[st.model]) - centre(t)).norm() / centre(t).norm())
    return out


# ------------------------------------------------------------------ LIN-2: L-BFGS on the linearised CE


def solve_lin2(problem: LinearizedProblem, *, lam: float, lam_p: float = 0.0, g: dict[int, Tensor] | None = None,
               x0: Theta | None = None, max_iter: int = 100, history_size: int = 20, tol: float = 1e-7,
               log=print) -> tuple[Theta, dict]:
    started = time.time()
    grid, r = problem.grid, problem.r
    x = theta_flat(x0 if x0 is not None else theta_zeros(grid, r, problem.device)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=1.0, max_iter=max_iter, history_size=history_size, tolerance_grad=tol,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    evals = {"n": 0, "loss": []}

    def ce_cot(dz, mb, w):
        logits = mb.z0.to(dz.device, torch.float64) + dz
        y = mb.labels.to(dz.device)
        logp = torch.log_softmax(logits, dim=-1)
        loss = -float(logp.gather(1, y[:, None]).sum()) * w
        grad = torch.softmax(logits, dim=-1)
        grad[torch.arange(len(y)), y] -= 1.0
        return w * grad, loss

    def closure():
        opt.zero_grad()
        theta = theta_unflat(x.detach(), grid, r)
        adj, loss = problem.sweep(theta, ce_cot)
        reg = _reg(theta, lam, lam_p, g)
        loss += 0.5 * theta_dot(theta, reg)
        gvec = theta_flat(theta_axpy(1.0, reg, adj))
        x.grad = gvec.to(x.dtype)
        evals["n"] += 1
        evals["loss"].append(loss)
        if evals["n"] % 10 == 0:
            log(f"[LIN-2] eval {evals['n']} loss {loss:.5f} |g| {float(gvec.norm()):.2e} ({time.time() - started:.0f}s)")
        return torch.tensor(loss, dtype=torch.float64)

    opt.step(closure)
    theta = theta_unflat(x.detach(), grid, r)
    return theta, {"lbfgs_evals": evals["n"], "loss_first": evals["loss"][0], "loss_last": evals["loss"][-1],
                   "seconds": time.time() - started}


def lin2_train_accuracy(problem: LinearizedProblem, theta: Theta) -> dict[str, float]:
    pred = problem.predict(theta)
    out = {}
    for st in problem.students:
        z0 = torch.cat([mb.z0 for mb in st.batches]).double()
        y = torch.cat([mb.labels for mb in st.batches])
        out[st.model] = float(((z0 + pred[st.model]).argmax(-1) == y).float().mean())
    return out


# ------------------------------------------------------------------ data collection


@torch.no_grad()
def collect_student(model_name: str, tail: VisionTail, frame: ModelFrame, grid: Sequence[int],
                    batches: list[tuple[Tensor, Tensor]], *, lin_point: str,
                    teacher_logits: Tensor | None = None, canonical_tokens: int = CANON) -> Student:
    """Cache what the linear model needs for a student: the stream after the first block of S (``task``) or the
    canonical coordinates (``anchors``), the base logits, the teacher change (LIN-1) and labels."""
    start = min(frame.blocks[frame.index(j)] for j in grid)
    st = Student(model_name, frame, tail if lin_point == "task" else None, start)
    offset = 0
    for x, y in batches:
        h = tail.prefix(x, start)
        z0 = tail.logits(h, start).double().cpu()
        ucan = None
        if lin_point == "anchors":
            blocks = [frame.blocks[frame.index(j)] for j in grid]
            streams = tail_streams(tail, h, start, blocks)
            ucan = {j: to_canonical(frame.encode(streams[b].double(), j).float(), canonical_tokens).cpu()
                    for j, b in zip(grid, blocks, strict=True)}
        target = None
        if teacher_logits is not None:
            target = teacher_logits[offset : offset + len(x)].double() - z0
        st.batches.append(StudentBatch(model_name, h.cpu() if lin_point == "task" else None, ucan, z0, target,
                                       torch.as_tensor(y)))
        offset += len(x)
    return st


def theta_to_payload(theta: Theta) -> dict[str, Any]:
    return {"grid": sorted(theta), "w": {j: w.detach().cpu() for j, (w, _) in theta.items()},
            "b": {j: b.detach().cpu() for j, (_, b) in theta.items()}}


def payload_to_theta(payload: dict[str, Any], device) -> Theta:
    return {int(j): (payload["w"][j].to(device, torch.float64), payload["b"][j].to(device, torch.float64))
            for j in payload["grid"]}


def project_rank(theta: Theta, rank: int) -> Theta:
    """Final low-rank projection of each ``W_j`` (truncated SVD); ``b`` unchanged."""
    out = {}
    for j, (w, b) in theta.items():
        u, s, vt = torch.linalg.svd(w, full_matrices=False)
        out[j] = ((u[:, :rank] * s[:rank]) @ vt[:rank], b)
    return out
