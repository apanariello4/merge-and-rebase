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


# ---- anchor-kernel frames (frame_map anchor_zca / anchor_tikhonov) ----

from merge_and_rebase.rebase.methods.anchors.shared_space import SharedFrame, anchor_kernel_maps  # noqa: E402


def _anchor_rows(t=12, d=20, seed=11):
    return torch.randn(t, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _rotation(d, seed=12):
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    return q


def test_psi_none_is_unchanged():
    base = _frame(8, 3, [1, 2], 3)
    same = ModelFrame(base.model_id, base.grid, base.blocks, base.mu, base.c, base.phi, None)
    twin = ModelFrame(base.model_id, base.grid, base.blocks, base.mu, base.c, base.phi, list(base.phi))
    du, gh = torch.randn(4, R, dtype=torch.float64), torch.randn(4, 8, dtype=torch.float64)
    for j in base.grid:
        i = base.index(j)
        assert torch.equal(same.decode(du, j), base.c[i] * (du @ base.phi[i]))
        assert torch.equal(same.encode_grad(gh, j), base.c[i] * (gh @ base.phi[i].T))
        assert torch.equal(twin.decode(du, j), same.decode(du, j))
        assert torch.equal(twin.encode_grad(gh, j), same.encode_grad(gh, j))


def test_decode_encode_grad_adjoint_with_psi():
    g = torch.Generator().manual_seed(3)
    phi, psi = (torch.randn(R, 9, generator=g, dtype=torch.float64) for _ in range(2))
    fr = ModelFrame("t", [0], [0], [torch.zeros(9, dtype=torch.float64)], [1.7], [phi], [psi])
    du, gh = torch.randn(5, R, generator=g, dtype=torch.float64), torch.randn(5, 9, generator=g, dtype=torch.float64)
    lhs = (fr.decode(du, 0) * gh).sum()
    rhs = (du * fr.encode_grad(gh, 0)).sum()
    torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)
    # encode uses phi, decode uses psi
    h = torch.randn(5, 9, generator=g, dtype=torch.float64)
    torch.testing.assert_close(fr.encode(h, 0), (h / 1.7) @ phi.T)
    assert not torch.allclose(fr.decode(du, 0), 1.7 * du @ phi)


@pytest.mark.parametrize("variant", ["anchor_zca", "anchor_tikhonov"])
def test_anchor_frame_zero_adapter_is_base(variant):
    width, depth = 16, 3
    model = _model(width, depth)
    tail = VisionTail(model, _text())
    t = 8
    phis, psis = [], []
    for s in (1, 2):
        phi, psi, _ = anchor_kernel_maps(_anchor_rows(t, width, seed=s), variant, 1e-2)
        phis.append(phi)
        psis.append(psi)
    base = _frame(width, depth, [1, 2], 3)
    frame = ModelFrame("toy", base.grid, base.blocks, base.mu, base.c, phis, psis)
    ad = SharedAdapter(frame.grid, t, dtype=torch.float64)
    x = _images()
    ref = tail.full_logits(x)
    with adapter_hooks(model, frame, ad):
        out = tail.full_logits(x)
    assert torch.equal(ref, out)
    ad2 = SharedAdapter(frame.grid, t, dtype=torch.float64)
    ad2.set_({j: 0.1 * torch.randn(t, t, dtype=torch.float64) for j in frame.grid})
    with adapter_hooks(model, frame, ad2):
        assert not torch.allclose(tail.full_logits(x), ref)


def test_zca_rotation_and_scale_invariance():
    p, rot = _anchor_rows(), _rotation(20)
    mu, c_m, s = torch.randn(20, dtype=torch.float64), 0.8, 2.5
    h = torch.randn(7, 20, dtype=torch.float64)
    phi_m, _, cond_m = anchor_kernel_maps(p, "anchor_zca", 1e-2)
    # (a) anchors s P R: the map is Q_k = Q_m R, whatever s (eps is relative to the trace)
    phi_k, _, cond_k = anchor_kernel_maps(s * p @ rot, "anchor_zca", 1e-2)
    torch.testing.assert_close(phi_k, phi_m @ rot, rtol=1e-9, atol=1e-9)
    assert abs(cond_k - cond_m) < 1e-8 * cond_m
    fm = ModelFrame("m", [0], [0], [mu], [c_m], [phi_m])
    # (b) consistent model k: tokens h_k = s h_m R, mu_k = s mu R, c_k = s c_m  =>  h~_k = h~_m R and P_k = P_m R
    c_k = s * c_m
    phi_k2, _, _ = anchor_kernel_maps(p @ rot, "anchor_zca", 1e-2)
    fk = ModelFrame("k", [0], [0], [s * mu @ rot], [c_k], [phi_k2])
    torch.testing.assert_close(fk.encode(s * h @ rot, 0), fm.encode(h, 0), rtol=1e-9, atol=1e-9)
    du = torch.randn(7, 12, dtype=torch.float64)
    torch.testing.assert_close(fk.decode(du, 0), (c_k / c_m) * fm.decode(du, 0) @ rot, rtol=1e-9, atol=1e-9)
    # (c) literal spec case: tokens AND anchors scaled by s at equal c: u_k = s u_m, decode(du) = Delta h_m R
    fk3 = ModelFrame("k3", [0], [0], [s * mu @ rot], [c_m], [phi_k])
    torch.testing.assert_close(fk3.encode(s * h @ rot, 0), s * fm.encode(h, 0), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(fk3.decode(du, 0), fm.decode(du, 0) @ rot, rtol=1e-9, atol=1e-9)


def test_tikhonov_anchor_relation():
    p, eps_rel = _anchor_rows(), 1e-2
    phi, psi, cond = anchor_kernel_maps(p, "anchor_tikhonov", eps_rel)
    k = p @ p.T
    eps = eps_rel * float(torch.trace(k)) / p.shape[0]
    assert torch.equal(phi, p)
    fr = ModelFrame("t", [0], [0], [torch.zeros(20, dtype=torch.float64)], [1.0], [phi], [psi])
    out = fr.decode(fr.encode(p, 0), 0)
    eye = torch.eye(p.shape[0], dtype=torch.float64)
    torch.testing.assert_close(out, k @ torch.linalg.solve(k + eps * eye, p), rtol=1e-9, atol=1e-9)
    ev = torch.linalg.eigvalsh(k)
    assert abs(cond - float((ev[-1] + eps) / (ev[0] + eps))) < 1e-9 * cond
    with pytest.raises(ValueError):
        anchor_kernel_maps(p, "gpa", eps_rel)


def test_zca_whitens_kernel():
    p = _anchor_rows()
    phi, psi, _ = anchor_kernel_maps(p, "anchor_zca", 1e-2)
    assert psi is phi
    k = p @ p.T
    eps = 1e-2 * float(torch.trace(k)) / p.shape[0]
    torch.testing.assert_close(phi @ phi.T, k @ torch.linalg.inv(k + eps * torch.eye(12, dtype=torch.float64)),
                               rtol=1e-9, atol=1e-9)


class _Reg:
    def __init__(self, depth, d, t, seed, k=3.0):
        g = torch.Generator().manual_seed(seed)
        self.depth = depth
        self.p = torch.randn(depth, t, d, generator=g, dtype=torch.float64)
        self.c = [1.0 + 0.1 * i for i in range(depth)]
        self.mu = torch.randn(depth, d, generator=g, dtype=torch.float64)
        flags = torch.zeros(depth, t, dtype=torch.bool)
        flags[:, seed::7] = True
        self.extras = {f"norm_k{k:g}": {"mu": self.mu * 0.5, "c": [2.0] * depth, "anchor_flags": flags}}


class _State:
    def __init__(self, regs):
        self.regs = regs

    def model(self, m):
        return self.regs[m]

    def load_norm_filter(self, m, k):
        return self.regs[m].extras[f"norm_k{k:g}"]


@pytest.mark.parametrize("variant", ["anchor_zca", "anchor_tikhonov"])
def test_from_state_anchor_path(monkeypatch, variant):
    import merge_and_rebase.rebase.methods.anchors.shared_space as ss

    regs = {"a": _Reg(4, 10, 40, 1), "b": _Reg(6, 14, 40, 2)}
    monkeypatch.setattr(ss, "AnchorState", lambda cfg, d: _State(regs))
    monkeypatch.setattr(ss, "fit_universe", lambda *a, **k: pytest.fail("anchor path must not fit a universe"))
    kw = dict(fit_models=["a"], add_models=["b"], n_grid=3, depths="last2", filter_k=3.0, frame_map=variant,
              anchor_rows=16, anchor_subset_seed=5)
    fr = SharedFrame.from_state("unused", **kw)
    fr2 = SharedFrame.from_state("unused", **kw)
    assert fr.universe is None and fr.rank == 16 and fr.grid == [1, 2]
    assert fr.meta["frame_map"] == variant and fr.meta["universe_version"] is None
    assert set(fr.meta["kept_anchors"]) == {1, 2} and all(v >= 16 for v in fr.meta["kept_anchors"].values())
    assert fr.meta["kernel_cond"]["a"][0] >= 1.0
    assert fr["a"].width == 10 and fr["b"].width == 14
    assert (fr["a"].psi is None) == (variant == "anchor_zca")
    for j in fr.grid:
        assert fr.g[j].shape == (16, 16)
        torch.testing.assert_close(fr["b"].phi[fr["b"].index(j)], fr2["b"].phi[fr2["b"].index(j)])
    # encode(anchor row) = h~ P^T with the filtered frame: the kept rows of model a at depth 1 (block 1)
    from merge_and_rebase.rebase.methods.anchors.method import filtered_frame

    ba, bb = fr["a"].blocks[0], fr["b"].blocks[0]
    assert (ba, bb) == (ss._block(_State(regs), "a", 3, 1), ss._block(_State(regs), "b", 3, 1))
    keep = ~(regs["a"].extras["norm_k3"]["anchor_flags"][ba] | regs["b"].extras["norm_k3"]["anchor_flags"][bb])
    p_a, c_a = filtered_frame(regs["a"], ba, keep, 3.0)
    flt = regs["a"].extras["norm_k3"]
    sel = torch.randperm(int(keep.sum()), generator=torch.Generator().manual_seed(5))[:16].sort().values
    raw = p_a[sel] * c_a + flt["mu"][ba]
    u = fr["a"].encode(raw, 1)
    phi = fr["a"].phi[fr["a"].index(1)]
    torch.testing.assert_close(u, p_a[sel] @ phi.T, rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError, match="only"):
        SharedFrame.from_state("unused", **{**kw, "anchor_rows": 500})
