"""LIN-1 / LIN-2 solvers (eval/shared_training/lin.py) against explicit design matrices on tiny OpenCLIP towers."""

from __future__ import annotations

import torch
from open_clip.transformer import VisionTransformer

from merge_and_rebase.eval.shared_training.lin import (
    LinearizedProblem,
    centre,
    collect_student,
    solve_lin1,
    solve_lin2,
    theta_flat,
    theta_randn,
    theta_unflat,
)
from merge_and_rebase.rebase.methods.anchors.jacobian import VisionTail, canonical_adjoint, jacobian_rows
from merge_and_rebase.rebase.methods.anchors.shared_space import ModelFrame

R, C, TOK = 4, 3, 5


def _model(width, depth, seed):
    torch.manual_seed(seed)
    m = torch.nn.Module()
    m.visual = VisionTransformer(image_size=8, patch_size=4, width=width, layers=depth, heads=2, mlp_ratio=2.0,
                                 output_dim=8)
    m = m.double().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _frame(width, depth, grid, n_grid, seed):
    g = torch.Generator().manual_seed(seed)
    blocks = [round(j * (depth - 1) / (n_grid - 1)) for j in grid]
    phis = [torch.linalg.qr(torch.randn(width, width, generator=g, dtype=torch.float64))[0][:R].contiguous()
            for _ in grid]
    return ModelFrame(f"m{seed}", list(grid), blocks, [0.1 * torch.randn(width, generator=g, dtype=torch.float64)
                                                      for _ in grid], [0.7, 1.3][: len(grid)], phis)


def _text(seed):
    return torch.nn.functional.normalize(torch.randn(C, 8, generator=torch.Generator().manual_seed(seed),
                                                     dtype=torch.float64), dim=-1)


def _setup(width, seed, n=4):
    model = _model(width, 4, seed)
    tail = VisionTail(model, _text(seed))
    frame = _frame(width, 4, [2, 3], 4, seed)
    g = torch.Generator().manual_seed(seed + 50)
    x = torch.randn(n, 3, 8, 8, generator=g, dtype=torch.float64)
    y = torch.randint(0, C, (n,), generator=g)
    return model, tail, frame, [(x[:2], y[:2]), (x[2:], y[2:])]


def _design(tail, frame, batches, grid):
    """Explicit rows: dz[b, c] = sum_j <rows_j[b, c], u_j[b] W_j + 1 b_j^T> as a matrix on theta_flat."""
    start = frame.blocks[0]
    mats = []
    for x, _ in batches:
        h = tail.prefix(x, start)
        rows = jacobian_rows(tail, frame, h, start, grid, num_classes=C)
        streams = {}
        hh = h
        for b in range(start, tail.depth):
            if b > start:
                hh = tail._run_block(tail.blocks[b], hh)
            streams[b] = hh
        blocks = []
        for j in sorted(grid):
            u = frame.encode(streams[frame.blocks[frame.index(j)]], j)  # [B, N, R]
            cw = torch.einsum("bcnr,bns->bcsr", rows[j], u).reshape(*rows[j].shape[:2], R * R)
            cb = rows[j].sum(2)
            blocks.append(torch.cat([cw, cb], dim=-1))
        mats.append(torch.cat(blocks, dim=-1))
    return torch.cat(mats)  # [n, C, P]


def _wmask(grid):
    return torch.cat([torch.cat([torch.ones(R * R), torch.zeros(R)]) for _ in grid]).double()


def test_lin1_task_matches_explicit_ridge():
    grid = [2, 3]
    _, tail, frame, batches = _setup(8, 1)
    teacher = torch.randn(4, C, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    st = collect_student("s", tail, frame, grid, batches, lin_point="task", teacher_logits=teacher)
    prob = LinearizedProblem([st], grid, R, lin_point="task", device="cpu")
    lam = 0.05
    theta, info = solve_lin1(prob, lam=lam, tol=1e-12, max_iter=2000, log=lambda *_: None)
    d = _design(tail, frame, batches, grid)  # [n, C, P]
    d = d - d.mean(1, keepdim=True)
    t = torch.cat([mb.target for mb in st.batches])
    t = centre(t)
    w = 1.0 / 4
    dm = d.reshape(-1, d.shape[-1])
    a = w * dm.T @ dm + lam * torch.diag(_wmask(grid))
    ref = torch.linalg.solve(a, w * dm.T @ t.reshape(-1))
    torch.testing.assert_close(theta_flat(theta), ref, rtol=1e-6, atol=1e-8)


def test_lin1_anchors_equals_task_for_one_model_on_its_anchors():
    """Jbar from one model on the training images themselves (one image) = J(x): both lin_points coincide."""
    grid = [2, 3]
    _, tail, frame, batches = _setup(8, 2, n=1)
    batches = [batches[0]] if len(batches[0][0]) else batches
    batches = [(batches[0][0][:1], batches[0][1][:1])]
    teacher = torch.randn(1, C, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    start = frame.blocks[0]
    rows = jacobian_rows(tail, frame, tail.prefix(batches[0][0], start), start, grid, num_classes=C)
    jbar = {j: canonical_adjoint(rows[j], TOK).mean(0) for j in grid}
    sols = {}
    for lp in ("task", "anchors"):
        st = collect_student("s", tail, frame, grid, batches, lin_point=lp, teacher_logits=teacher,
                             canonical_tokens=TOK)
        prob = LinearizedProblem([st], grid, R, lin_point=lp, device="cpu", jbar=jbar)
        sols[lp] = theta_flat(solve_lin1(prob, lam=0.1, tol=1e-12, max_iter=2000, log=lambda *_: None)[0])
    torch.testing.assert_close(sols["anchors"], sols["task"], rtol=1e-5, atol=1e-7)


def test_lin2_gradient_matches_autograd():
    grid = [2, 3]
    _, tail, frame, batches = _setup(8, 3)
    st = collect_student("s", tail, frame, grid, batches, lin_point="task")
    prob = LinearizedProblem([st], grid, R, lin_point="task", device="cpu")
    theta = theta_randn(grid, R, "cpu", torch.Generator().manual_seed(8))
    theta = {j: (0.1 * w, 0.1 * b) for j, (w, b) in theta.items()}
    d = _design(tail, frame, batches, grid)
    z0 = torch.cat([mb.z0 for mb in st.batches])
    y = torch.cat([mb.labels for mb in st.batches])
    v = theta_flat(theta).clone().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(z0 + torch.einsum("bcp,p->bc", d, v), y, reduction="sum") / 4
    (g_ref,) = torch.autograd.grad(loss, v)

    def ce_cot(dz, mb, w):
        logits = mb.z0.double() + dz
        p = torch.softmax(logits, -1)
        p[torch.arange(len(mb.labels)), mb.labels] -= 1
        return w * p, 0.0

    g = theta_flat(prob.sweep(theta, ce_cot)[0])
    torch.testing.assert_close(g, g_ref, rtol=1e-7, atol=1e-9)
    sol, info = solve_lin2(prob, lam=0.01, max_iter=50, log=lambda *_: None)
    assert info["loss_last"] < info["loss_first"]


def test_theta_flat_roundtrip():
    grid = [0, 3]
    th = theta_randn(grid, R, "cpu", torch.Generator().manual_seed(0))
    back = theta_unflat(theta_flat(th), grid, R)
    for j in grid:
        assert torch.equal(back[j][0], th[j][0]) and torch.equal(back[j][1], th[j][1])


def test_compiled_reproduces_reported_fit_and_restores_weights():
    """compile_theta: after applying the c_proj vector, the first adapted block's output moves by H dW + beta, whose
    misfit to D = c (u W + 1 b^T) Phi on the calibration rows is the reported residual; applied() restores the base."""
    from merge_and_rebase.eval.shared_training.evaluate import applied, compile_theta, frame_on
    from merge_and_rebase.rebase.methods._ariadne.capture import iter_capture_tokens

    grid = [2, 3]
    model, tail, frame, batches = _setup(8, 5, n=4)
    model = model.float()
    frame32 = frame_on(frame, "cpu")
    theta = theta_randn(grid, R, "cpu", torch.Generator().manual_seed(2))
    theta = {j: (0.3 * w, 0.3 * b) for j, (w, b) in theta.items()}
    calib = [(x.float(), y) for x, y in batches]
    tv, diag = compile_theta(model_wrap(model), frame, theta, calib, [x for x, _ in calib], norm_filter=None,
                             rho_g=0.0, preserve_tokens=64, seed=0, device="cpu")
    b0 = frame.blocks[0]
    req = {"out": (b0, "boundary")}
    before = torch.cat([v["out"] for v in iter_capture_tokens(model, calib, req, "cpu")]).double()
    sd0 = {k: v.clone() for k, v in model.state_dict().items()}
    with applied(model, tv):
        after = torch.cat([v["out"] for v in iter_capture_tokens(model, calib, req, "cpu")]).double()
    for k, v in model.state_dict().items():
        assert torch.equal(v, sd0[k])
    w, b = theta[grid[0]]
    d = frame.decode(frame.encode(before, grid[0]) @ w + b, grid[0])
    rel = float((after - before - d).norm() / d.norm())
    assert abs(rel - diag[str(grid[0])]["relative_residual"]) < 1e-3
    assert diag[str(grid[0])]["relative_residual"] < 1.0
    del frame32


class model_wrap:  # noqa: N801 - minimal stand-in for OpenClipClassifier (.model)
    def __init__(self, model):
        self.model = model


def test_nl2_local_recovers_a_planted_adapter():
    """Expert = base + a known adapter (hooks): the sequential local fit recovers (W*, b*) at lam -> 0."""
    from copy import deepcopy

    from merge_and_rebase.eval.shared_training.evaluate import frame_on
    from merge_and_rebase.eval.shared_training.nonlinear import fit_nl2_local
    from merge_and_rebase.rebase.methods.anchors.shared_space import SharedAdapter, adapter_hooks

    grid = [2, 3]
    model, tail, frame, batches = _setup(8, 6, n=8)
    model = model.float()
    # the frame's Phi has R < width rows: plant W* inside the common space
    planted = theta_randn(grid, R, "cpu", torch.Generator().manual_seed(4))
    planted = {j: (0.2 * w, 0.2 * b) for j, (w, b) in planted.items()}
    ad = SharedAdapter(grid, R, dtype=torch.float32)
    ad.set_({j: planted[j][0] for j in grid}, {j: planted[j][1] for j in grid})
    expert = deepcopy(model)
    ctx = adapter_hooks(expert, frame_on(frame, "cpu"), ad)
    ctx.__enter__()
    try:
        g = {j: torch.randn(7, R, dtype=torch.float64) for j in grid}
        st = [{"model": model, "expert": expert, "frame": frame, "batches": [x.float() for x, _ in batches]}]
        theta, _ = fit_nl2_local(st, grid, R, g, lam_rel=1e-9, lam_p_rel=0.0, device="cpu", log=lambda *_: None)
    finally:
        ctx.__exit__(None, None, None)
    for j in grid:
        torch.testing.assert_close(theta[j][0], planted[j][0], rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(theta[j][1], planted[j][1], rtol=1e-3, atol=1e-4)


def test_adapted_logits_match_hooked_full_forward():
    from merge_and_rebase.eval.shared_training.evaluate import frame_on
    from merge_and_rebase.eval.shared_training.nonlinear import adapted_logits
    from merge_and_rebase.rebase.methods.anchors.shared_space import SharedAdapter, adapter_hooks

    grid = [2, 3]
    model, tail, frame, batches = _setup(8, 7, n=2)
    ad = SharedAdapter(grid, R, dtype=torch.float64)
    th = theta_randn(grid, R, "cpu", torch.Generator().manual_seed(5))
    ad.set_({j: 0.3 * th[j][0] for j in grid}, {j: 0.3 * th[j][1] for j in grid})
    x = batches[0][0]
    with adapter_hooks(model, frame, ad):
        ref = tail.full_logits(x)
    start = frame.blocks[0]
    out = adapted_logits(tail, frame, ad, tail.prefix(x, start), start)
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)
    del frame_on
