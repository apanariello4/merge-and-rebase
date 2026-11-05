"""Shared adapter and common-space Jacobian (AGENT_SHARED_TRAINING.md section 5) on tiny real OpenCLIP towers.

The towers are ``open_clip.transformer.VisionTransformer`` instances (8x8 images, 4x4 patches: CLS + 2x2 grid = 5
tokens; or 12x12 images: 10 tokens), so attention, pooling and projection are the production code paths.
"""

from __future__ import annotations

import pytest
import torch
from open_clip.transformer import VisionTransformer

from merge_and_rebase.rebase.methods.anchors.jacobian import (
    VisionTail,
    canonical_adjoint,
    common_jvp,
    common_vjp,
    jacobian_rows,
    jvp_logits,
    to_tokens,
)
from merge_and_rebase.rebase.methods.anchors.shared_space import (
    ModelFrame,
    SharedAdapter,
    adapter_hooks,
    anchor_conservation,
    resolve_depths,
)

R = 6  # common dimension of the toys
C = 3  # classes


class _Clip(torch.nn.Module):
    def __init__(self, width, depth, image_size=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.visual = VisionTransformer(
            image_size=image_size, patch_size=4, width=width, layers=depth, heads=2, mlp_ratio=2.0, output_dim=8
        )


def _model(width, depth, image_size=8, seed=0, dtype=torch.float64):
    m = _Clip(width, depth, image_size, seed).to(dtype).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _text(seed=1, dtype=torch.float64):
    t = torch.randn(C, 8, generator=torch.Generator().manual_seed(seed), dtype=dtype)
    return torch.nn.functional.normalize(t, dim=-1)


def _frame(width, depth, grid, n_grid, seed=3, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    blocks = [round(j * (depth - 1) / (n_grid - 1)) for j in grid]
    phis = []
    for _ in grid:
        q, _ = torch.linalg.qr(torch.randn(width, width, generator=g, dtype=dtype))
        phis.append(q[:R].contiguous())  # [r, d], orthonormal rows
    mus = [0.1 * torch.randn(width, generator=g, dtype=dtype) for _ in grid]
    cs = [0.5 + float(torch.rand(1, generator=g)) for _ in grid]
    return ModelFrame("toy", list(grid), blocks, mus, cs, phis)


def _images(n=3, size=8, seed=5, dtype=torch.float64):
    return torch.randn(n, 3, size, size, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _random_adapter(grid, scale, seed=7, dtype=torch.float64):
    ad = SharedAdapter(grid, R, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    ad.set_({j: scale * torch.randn(R, R, generator=g, dtype=dtype) for j in grid},
            {j: scale * torch.randn(R, generator=g, dtype=dtype) for j in grid})
    return ad


def test_resolve_depths():
    assert resolve_depths("last4", 12) == [8, 9, 10, 11]
    assert resolve_depths("all", 3) == [0, 1, 2]
    assert resolve_depths([2, 0, 2], 4) == [0, 2]
    with pytest.raises(ValueError):
        resolve_depths("last5", 4)


@pytest.mark.parametrize("width", [8, 12])
def test_zero_adapter_is_base(width):
    model = _model(width, 3)
    tail = VisionTail(model, _text())
    frame = _frame(width, 3, [1, 2], 3)
    ad = SharedAdapter(frame.grid, R, dtype=torch.float64)
    x = _images()
    ref = tail.full_logits(x)
    with adapter_hooks(model, frame, ad):
        out = tail.full_logits(x)
    assert torch.equal(ref, out)


@pytest.mark.parametrize("width,size", [(8, 8), (12, 12)])
def test_hook_matches_manual_insertion(width, size):
    """The hook adds c (u W + 1 b^T) Phi after block pi(s), u from the current stream, at any width/token count."""
    depth = 4
    model = _model(width, depth, image_size=size)
    tail = VisionTail(model, _text())
    frame = _frame(width, depth, [2, 3], 4)
    ad = _random_adapter(frame.grid, 0.3)
    x = _images(size=size)
    with adapter_hooks(model, frame, ad):
        hooked = tail.full_logits(x)
    h = tail.prefix(x, -1)
    for j in range(depth):
        h = tail._run_block(tail.blocks[j], h)
        for s, b in zip(frame.grid, frame.blocks, strict=True):
            if b == j:
                u = ((h - frame.mu[frame.index(s)]) / frame.c[frame.index(s)]) @ frame.phi[frame.index(s)].T
                du = u @ ad.weight(s) + ad.bias(s)
                h = h + frame.c[frame.index(s)] * du @ frame.phi[frame.index(s)]
    manual = tail.logits(h, depth - 1)
    torch.testing.assert_close(hooked, manual, rtol=1e-10, atol=1e-10)


def test_cache_equals_full_forward():
    model = _model(8, 4)
    tail = VisionTail(model, _text())
    x = _images()
    ref = tail.full_logits(x)
    for start in range(-1, 4):
        torch.testing.assert_close(tail.logits(tail.prefix(x, start), start), ref, rtol=1e-12, atol=1e-12)


def test_jvp_matches_finite_differences():
    model = _model(8, 4)
    tail = VisionTail(model, _text())
    x = _images()
    start = 1
    h = tail.prefix(x, start)
    g = torch.Generator().manual_seed(9)
    tangents = {1: torch.randn(h.shape, generator=g, dtype=h.dtype), 3: torch.randn(h.shape, generator=g, dtype=h.dtype)}
    z, dz = jvp_logits(tail, h, start, tangents)
    eps = 1e-6
    zp = tail.logits(h, start, {b: eps * t for b, t in tangents.items()})
    zm = tail.logits(h, start, {b: -eps * t for b, t in tangents.items()})
    torch.testing.assert_close(z, tail.logits(h, start))
    torch.testing.assert_close(dz, (zp - zm) / (2 * eps), rtol=1e-6, atol=1e-7)


def test_vjp_is_adjoint_of_jvp_in_common_space():
    model = _model(8, 4)
    tail = VisionTail(model, _text())
    frame = _frame(8, 4, [1, 2, 3], 4)
    x = _images()
    h = tail.prefix(x, frame.blocks[0])
    g = torch.Generator().manual_seed(4)
    du = {j: torch.randn(*h.shape[:2], R, generator=g, dtype=h.dtype) for j in frame.grid}
    cot = torch.randn(h.shape[0], C, generator=g, dtype=h.dtype)
    _, dz = common_jvp(tail, frame, h, frame.blocks[0], du)
    grads = common_vjp(tail, frame, h, frame.blocks[0], frame.grid, cot)
    lhs = (cot * dz).sum()
    rhs = sum((grads[j] * du[j]).sum() for j in frame.grid)
    torch.testing.assert_close(lhs, rhs, rtol=1e-9, atol=1e-10)
    rows = jacobian_rows(tail, frame, h, frame.blocks[0], frame.grid, num_classes=C)
    explicit = sum(torch.einsum("bcnr,bnr->bc", rows[j], du[j]) for j in frame.grid)
    torch.testing.assert_close(explicit, dz, rtol=1e-9, atol=1e-10)


def test_linearized_matches_true_for_small_w():
    """z(x; W) - z(x) = J . vec(u W + 1 b^T) + O(||W||^2), with u of the base stream."""
    model = _model(8, 4)
    tail = VisionTail(model, _text())
    frame = _frame(8, 4, [2, 3], 4)
    x = _images()
    start = frame.blocks[0]
    h = tail.prefix(x, start)
    base_streams = {}
    hh = h
    for j in range(start, 4):
        if j > start:
            hh = tail._run_block(tail.blocks[j], hh)
        base_streams[j] = hh
    errs = []
    for scale in (1e-2, 1e-3):
        ad = _random_adapter(frame.grid, scale)
        du = {s: ad.displacement(frame.encode(base_streams[frame.blocks[frame.index(s)]], s), s) for s in frame.grid}
        z, dz = common_jvp(tail, frame, h, start, du)
        with adapter_hooks(model, frame, ad):
            true = tail.full_logits(x)
        errs.append(float((true - z - dz).norm() / dz.norm()))
    assert errs[1] < 0.2 * errs[0]  # second-order remainder: ~10x smaller per 10x smaller W
    assert errs[1] < 1e-2


def test_canonical_adjoint():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, 5, R, generator=g, dtype=torch.float64)  # canonical: CLS + 2x2
    y = torch.randn(2, 10, R, generator=g, dtype=torch.float64)  # model: CLS + 3x3
    lhs = (to_tokens(x, 10) * y).sum()
    rhs = (x * canonical_adjoint(y, 5)).sum()
    torch.testing.assert_close(lhs, rhs)
    assert torch.equal(canonical_adjoint(x, 5), x)


def test_anchor_conservation_and_parametrisations():
    grid = [0, 1]
    g = {j: torch.randn(7, R, generator=torch.Generator().manual_seed(j), dtype=torch.float64) for j in grid}
    ad = _random_adapter(grid, 0.5)
    expected = sum(((g[j] @ ad.weight(j) + ad.bias(j)) ** 2).sum() / 7 for j in grid)
    torch.testing.assert_close(anchor_conservation(ad, g), expected)
    low = SharedAdapter(grid, R, rank=2, dtype=torch.float64)
    assert torch.count_nonzero(low.weight(0)) == 0 and low.num_parameters() == 2 * (2 * R * 2 + R)
    q, _ = torch.linalg.qr(torch.randn(R, 3, dtype=torch.float64))
    sub = SharedAdapter(grid, R, subspace={j: q for j in grid}, dtype=torch.float64)
    with torch.no_grad():
        sub.w["0"].copy_(torch.eye(3, dtype=torch.float64))
    torch.testing.assert_close(sub.weight(0), q @ q.T)
    st = ad.state()
    back = SharedAdapter.from_state(st, dtype=torch.float64)
    torch.testing.assert_close(back.weight(1), ad.weight(1))
