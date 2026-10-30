"""Whitened kernel decodes of ``anchors`` (docs/.local/AGENT_KERNEL.md section 3): V2 'zca', V3 'agreement', and the
V1 limit (spectral_filter 'truncate_target'). Synthetic frames, float64; ``T <= d`` unless stated."""

from __future__ import annotations

import hashlib

import pytest
import torch

from merge_and_rebase.rebase.methods.anchors import parse_anchors_config
from merge_and_rebase.rebase.methods.anchors.core import (
    decode_rows,
    encode_rows,
    kernel,
    resolve_eps,
    spectral_transport_map,
    transport_map_primal,
    whitened_transport_map,
)
from merge_and_rebase.rebase.methods.anchors.method import transport_map
from merge_and_rebase.rebase.methods.anchors.state import RegisteredModel


def _frames(t, *dims, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(t, d, generator=g, dtype=torch.float64) for d in dims]


def _orthogonal(d, seed):
    q, r = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


def _registered(name, p, c):
    d = p.shape[1]
    return RegisteredModel(name, {}, [p.float()], [torch.zeros(d, dtype=torch.float64)], [c])


def _e(n, d, seed=5):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_parse_kernel_decode_options():
    assert parse_anchors_config({}).kernel_decode == "tikhonov"
    assert parse_anchors_config({"kernel_decode": "zca"}).agreement_tau is None
    assert parse_anchors_config({"kernel_decode": "agreement", "agreement_tau": 2.0}).agreement_tau == 2.0
    for bad in (
        {"kernel_decode": "nope"},
        {"kernel_decode": "agreement"},  # tau missing
        {"kernel_decode": "zca", "agreement_tau": 2.0},  # tau without agreement
        {"kernel_decode": "agreement", "agreement_tau": 0.5},
        {"kernel_decode": "zca", "frame_map": "gpa"},
        {"kernel_decode": "zca", "spectral_filter": "procrustes_cross"},
    ):
        with pytest.raises(ValueError):
            parse_anchors_config(bad)


def test_default_decode_is_unchanged():
    """V0: with the default kernel_decode the map is the Tikhonov encode/decode, in both the kernel and primal forms."""
    for t, d_m, d_k in ((6, 8, 10), (40, 6, 9)):
        p_m, p_k = _frames(t, d_m, d_k, seed=1)
        src, tgt = _registered("a", p_m, 1.7), _registered("b", p_k, 2.3)
        cfg = parse_anchors_config({"eps_rel": 0.05})
        fn, eps, info = transport_map(src, 0, tgt, 0, config=cfg)
        assert info == {}
        pm, pk = src.p[0].double(), tgt.p[0].double()
        e = _e(7, d_m)
        ref = decode_rows(encode_rows(e, pm, 1.7), pk, kernel(pk), scale=2.3, eps=resolve_eps(kernel(pk), 0.05))
        torch.testing.assert_close(fn(e), ref, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("t,d", [(6, 10), (40, 9)])
@pytest.mark.parametrize("eps_rel", [1e-3, 0.1, 1.0])
def test_zca_equals_tikhonov_when_the_kernels_coincide(t, d, eps_rel):
    """P_k = P_m O (same kernel): V2 and V0 give the same D (both = V diag(s^2/(s^2+eps)) V^T up to the rotation)."""
    (p_m,) = _frames(t, d, seed=2)
    p_k = p_m @ _orthogonal(d, 3)
    phi, info = whitened_transport_map(p_m, 1.3, p_k, scale=0.7, eps_rel=eps_rel, mode="zca")
    eps = resolve_eps(kernel(p_k), eps_rel)
    assert info["kernel_eps_source"] == pytest.approx(info["kernel_eps_target"], rel=1e-12)
    torch.testing.assert_close(phi, transport_map_primal(p_m, 1.3, p_k, scale=0.7, eps=eps), rtol=1e-8, atol=1e-10)


def test_zca_does_not_amplify():
    """eps = 0, T <= d: ||D||_F / scale <= ||E||_F / c_m for any E (Q_m^T Q_k has spectral norm <= 1)."""
    p_m, p_k = _frames(8, 12, 15, seed=4)
    p_k = p_k * torch.linspace(0.01, 10.0, 15, dtype=torch.float64)  # very uneven target spectrum
    phi, info = whitened_transport_map(p_m, 2.0, p_k, scale=3.0, eps_rel=0.0, mode="zca")
    assert info["kernel_map_spectral_norm"] <= 1.0 + 1e-10
    for seed in range(5):
        e = _e(20, 12, seed=seed)
        assert torch.linalg.norm(e @ phi) / 3.0 <= torch.linalg.norm(e) / 2.0 + 1e-10
    # Contrast: a target whose frame is the source's shrunk 10x (K_k = K_m / 100). Tikhonov at eps -> 0 multiplies
    # by 10 (the defect V2 removes); ZCA stays an isometry on span(P).
    (p_s,) = _frames(8, 12, seed=14)
    p_t = 0.1 * p_s @ _orthogonal(12, 15)
    tik = transport_map_primal(p_s, 1.0, p_t, scale=1.0, eps=1e-14)
    zca, _ = whitened_transport_map(p_s, 1.0, p_t, scale=1.0, eps_rel=0.0, mode="zca")
    assert torch.linalg.matrix_norm(tik, ord=2) == pytest.approx(10.0, rel=1e-4)
    assert torch.linalg.matrix_norm(zca, ord=2) == pytest.approx(1.0, rel=1e-9)


def test_zca_chain_equals_direct():
    """Three unrelated frames, T <= d, eps = 0: A -> C -> B through the anchor coordinates equals A -> B."""
    p_a, p_b, p_c = _frames(7, 9, 11, 13, seed=6)
    ac, _ = whitened_transport_map(p_a, 1.0, p_c, scale=1.0, eps_rel=0.0, mode="zca")
    cb, _ = whitened_transport_map(p_c, 1.0, p_b, scale=1.0, eps_rel=0.0, mode="zca")
    ab, _ = whitened_transport_map(p_a, 1.0, p_b, scale=1.0, eps_rel=0.0, mode="zca")
    torch.testing.assert_close(ac @ cb, ab, rtol=1e-9, atol=1e-10)


def test_agreement_with_large_tau_is_zca():
    """T <= d and eps > 0 (M full rank): tau -> infinity keeps every direction, so V3 = V2."""
    p_m, p_k = _frames(8, 12, 10, seed=8)
    zca, _ = whitened_transport_map(p_m, 1.1, p_k, scale=0.9, eps_rel=0.01, mode="zca")
    agr, info = whitened_transport_map(p_m, 1.1, p_k, scale=0.9, eps_rel=0.01, mode="agreement", tau=1e12)
    assert info["agreement_kept"] == 8
    torch.testing.assert_close(agr, zca, rtol=1e-8, atol=1e-10)
    # tau = 1 with different kernels keeps (almost) nothing; a band keeps a subset.
    _, tight = whitened_transport_map(p_m, 1.1, p_k, scale=0.9, eps_rel=0.01, mode="agreement", tau=1.0)
    _, band = whitened_transport_map(p_m, 1.1, p_k, scale=0.9, eps_rel=0.01, mode="agreement", tau=2.0)
    assert tight["agreement_kept"] <= band["agreement_kept"] <= 8


def test_agreement_with_equal_kernels_keeps_the_agreeing_directions():
    """K_m = K_k: gamma_i = lambda_i / (lambda_i + eps) <= 1, near 1 on the strong directions."""
    (p_m,) = _frames(10, 14, seed=9)
    p_k = p_m @ _orthogonal(14, 10)
    zca, _ = whitened_transport_map(p_m, 1.0, p_k, scale=1.0, eps_rel=1e-6, mode="zca")
    agr, info = whitened_transport_map(p_m, 1.0, p_k, scale=1.0, eps_rel=1e-6, mode="agreement", tau=1.25)
    assert info["agreement_kept"] == 10
    torch.testing.assert_close(agr, zca, rtol=1e-8, atol=1e-10)


def test_truncate_target_full_rank_is_tikhonov_eps_to_zero_kernel_form():
    """V1 with k = rank equals V0 with eps -> 0 (T <= d; the T > d case is test_spectral_filters)."""
    p_m, p_k = _frames(6, 8, 10, seed=11)
    full, info = spectral_transport_map(p_m, 1.5, p_k, scale=2.0, mode="truncate_target", rank=None)
    assert info["spectral_rank_used"] == 6
    e = _e(5, 8)
    ref = decode_rows(encode_rows(e, p_m, 1.5), p_k, kernel(p_k), scale=2.0, eps=1e-12)
    torch.testing.assert_close(e @ full, ref, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("mode,tau", [("zca", None), ("agreement", 1.5), ("agreement", 4.0)])
def test_pseudo_inverses_when_t_exceeds_d(mode, tau):
    """T > d (rank-deficient kernels), eps = 0: finite maps, null eigenvalues excluded, still non-amplifying (zca)."""
    p_m, p_k = _frames(50, 6, 9, seed=12)
    phi, info = whitened_transport_map(p_m, 1.0, p_k, scale=1.0, eps_rel=0.0, mode=mode, tau=tau)
    assert phi.shape == (6, 9) and torch.isfinite(phi).all()
    if mode == "zca":
        assert info["kernel_map_spectral_norm"] <= 1.0 + 1e-10
    else:
        assert info["agreement_nonzero"] <= 6  # rank of M <= rank K_m
    phi_eps, _ = whitened_transport_map(p_m, 1.0, p_k, scale=1.0, eps_rel=0.1, mode=mode, tau=tau)
    assert torch.isfinite(phi_eps).all()


@pytest.mark.parametrize("decode", [{"kernel_decode": "zca", "eps_rel": 0.01},
                                    {"kernel_decode": "agreement", "agreement_tau": 2.0, "eps_rel": 0.01}])
def test_map_depends_only_on_the_pair(decode):
    """Independence: the A -> B map is a function of A's and B's frames only; other registered models play no role."""
    p_a, p_b, p_c = _frames(30, 6, 9, 7, seed=13)
    a, b = _registered("a", p_a, 1.2), _registered("b", p_b, 0.8)
    cfg = parse_anchors_config({**decode, "anchor_filter": None})
    e = _e(4, 6)

    def digest():
        fn, _, _ = transport_map(a, 0, b, 0, config=cfg)
        return hashlib.sha256(fn(e).numpy().tobytes()).hexdigest()

    before = digest()
    _ = _registered("c", p_c, 1.0)  # a third model exists; nothing about the pair changes
    assert digest() == before
    fn, eps, info = transport_map(a, 0, b, 0, config=cfg)
    assert eps == info["kernel_eps_target"]


# ---------------------------------------------------------------- K1b: R2 dedup and R3 row normalisation


def test_row_normalize_and_dedup_mask():
    from merge_and_rebase.rebase.methods.anchors.core import dedup_mask, row_normalize

    (p,) = _frames(40, 6, seed=20)
    p[3] = 0.0
    n = row_normalize(p)
    norms = torch.linalg.vector_norm(n, dim=1)
    assert torch.allclose(norms[torch.arange(40) != 3], torch.ones(39, dtype=torch.float64))
    assert norms[3] == 0
    dup = torch.cat([p, 2.5 * p[10:20]], dim=0)  # 10 exact duplicates (cosine 1) of nonzero rows
    keep = dedup_mask(dup, 0.999, seed=1)
    kept = row_normalize(dup)[keep]
    cos = kept @ kept.T - torch.eye(int(keep.sum()), dtype=torch.float64)
    assert cos.max() < 0.999 and int(keep.sum()) <= 40
    assert int(dedup_mask(dup, 0.999, max_count=7, seed=1).sum()) == 7
    assert torch.equal(dedup_mask(dup, 0.5, seed=4), dedup_mask(dup, 0.5, seed=4))  # seeded


def test_row_norm_leaves_an_exact_rotation_unchanged():
    """R3: with models related by an exact rotation, the transport is unchanged by the row normalisation."""
    (p_m,) = _frames(30, 8, seed=21)
    o = _orthogonal(8, 22)
    a, b = _registered("a", p_m, 1.0), _registered("b", (p_m.float().double() @ o), 1.0)
    e = _e(5, 8)
    for decode in ({"eps_rel": 1e-9}, {"kernel_decode": "zca", "eps_rel": 0.0}, {"spectral_filter": "procrustes_cross"}):
        plain, _, _ = transport_map(a, 0, b, 0, config=parse_anchors_config(decode))
        normed, _, _ = transport_map(a, 0, b, 0, config=parse_anchors_config({**decode, "anchor_row_norm": True}))
        torch.testing.assert_close(normed(e), plain(e), rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(plain(e), e @ o, rtol=1e-6, atol=1e-8)


def test_dedup_is_measured_in_the_source_and_applied_to_both():
    from merge_and_rebase.rebase.methods.anchors.core import dedup_mask
    from merge_and_rebase.rebase.methods.anchors.method import pair_frames

    p_m, p_k = _frames(50, 6, 9, seed=23)
    a, b = _registered("a", p_m, 1.0), _registered("b", p_k, 1.0)
    cfg = parse_anchors_config({"anchor_dedup_cos": 0.3, "anchor_dedup_max": 20, "anchor_seed": 5})
    pm, _, pk, _ = pair_frames(a, 0, b, 0, config=cfg)
    sel = dedup_mask(a.p[0].double(), 0.3, max_count=20, seed=5)
    assert torch.equal(pm, a.p[0].double()[sel]) and torch.equal(pk, b.p[0].double()[sel])
    with pytest.raises(ValueError):
        parse_anchors_config({"anchor_dedup_max": 5})
    with pytest.raises(ValueError):
        parse_anchors_config({"anchor_dedup_cos": 1.5})


def test_defaults_keep_existing_universe_versions():
    from merge_and_rebase.rebase.methods.anchors.universe import universe_version

    class _S:
        class config:
            @staticmethod
            def frame():
                return {"x": 1}

        @staticmethod
        def model_meta_hash(m):
            return m

    v0 = universe_version(_S, ["a", "b"], 4, None, 3.0)
    assert universe_version(_S, ["a", "b"], 4, None, 3.0, None, False) == v0
    assert universe_version(_S, ["a", "b"], 4, None, 3.0, None, True) != v0
    assert universe_version(_S, ["a", "b"], 4, None, 3.0, ("a", 0.6, None, 0), False) != v0
