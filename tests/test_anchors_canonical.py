"""Phase 0.1 / 0.2 of the 'metodo K' plan: streaming anchor statistics (anchors/canonical_stats.py) and the unified
canonical decoder (core.canonical_*, kernel_decode canonical_hard / canonical_soft). Synthetic frames, float64, CPU."""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.rebase.methods.anchors import AnchorsConfig, parse_anchors_config
from merge_and_rebase.rebase.methods.anchors import core as C
from merge_and_rebase.rebase.methods.anchors.canonical_stats import PairSigmaAccumulator, split_ids
from merge_and_rebase.rebase.methods.anchors.method import transport_map
from merge_and_rebase.rebase.methods.anchors.state import AnchorState, RegisteredModel


def _frames(t, *dims, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(t, d, generator=g, dtype=torch.float64) for d in dims]


def _e(n, d, seed=5):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _registered(name, p, c):
    d = p.shape[1]
    return RegisteredModel(name, {}, [p.float()], [torch.zeros(d, dtype=torch.float64)], [c])


def _correlated(t, d_m, d_k, seed=0):
    """Two frames sharing a 16-dim latent, with decaying spectra (non-trivial Gamma, non-flat S)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(t, 16, generator=g, dtype=torch.float64)

    def one(d, decay):
        p = z @ torch.randn(16, d, generator=g, dtype=torch.float64) + 0.5 * torch.randn(t, d, generator=g, dtype=torch.float64)
        return p * torch.logspace(0, -decay, d, dtype=torch.float64)

    return one(d_m, 1.0), one(d_k, 1.5)


def _raw_rows(p, mu_scale, c, seed):
    """Raw rows S with P = (S - mu) / c, mu a large common mean (the cancellation hazard)."""
    mu = mu_scale * torch.randn(p.shape[1], generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return p * c + mu, mu


def _rel(a, b):
    return float((a - b).norm() / b.norm())


WIDTHS = [(32, 32), (24, 32), (32, 24)]
T = 80


# ---------------------------------------------------------------- 0.1 streaming statistics


@pytest.mark.parametrize("d_m,d_k", WIDTHS)
def test_gamma_from_streaming_sigma_equals_gamma_from_frames(d_m, d_k):
    t_rows = 160  # each half (80 rows) must exceed r_m + r_k = 64
    p_m, p_k = _correlated(t_rows, d_m, d_k, seed=1)
    c_m, c_k = 3.0, 0.5
    s_m, mu_m = _raw_rows(p_m, 1e3, c_m, 10)
    s_k, mu_k = _raw_rows(p_k, 1e3, c_k, 11)
    ids = torch.arange(t_rows) // 4  # 4 rows per image
    gamma_p = C.canonical_correlations(C.canonical_factors_from_frames(p_m, p_k))

    # batches of different sizes, rows streamed in order
    acc = PairSigmaAccumulator(mu_m, c_m, mu_k, c_k)
    for lo, hi in ((0, 7), (7, 8), (8, 40), (40, 159), (159, 160)):
        acc.update(s_m[lo:hi], s_k[lo:hi], ids[lo:hi])
    smm, skk, smk, n = acc.sigma()
    assert n == t_rows
    torch.testing.assert_close(smm, p_m.T @ p_m, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(smk, p_m.T @ p_k, rtol=1e-9, atol=1e-9)
    gamma_s = C.canonical_correlations(C.canonical_factors_from_sigma(smm, skk, smk, n))
    assert _rel(gamma_s, gamma_p) < 1e-9
    assert 0.0 < float(gamma_p[-1]) < float(gamma_p[0]) <= 1.0 + 1e-12  # non-degenerate

    # the two-half split by image id: each half's Gamma equals the Gamma of that half's rows of P
    halves = split_ids(ids, seed=3)
    acc = PairSigmaAccumulator(mu_m, c_m, mu_k, c_k, halves=halves)
    order = torch.randperm(t_rows, generator=torch.Generator().manual_seed(0))  # unordered stream
    for chunk in (order[:13], order[13:14], order[14:55], order[55:]):
        acc.update(s_m[chunk], s_k[chunk], ids[chunk])
    total = acc.sigma()
    torch.testing.assert_close(total[2], smk, rtol=1e-9, atol=1e-9)
    for h in (0, 1):
        sel = torch.isin(ids, halves[h])
        smm_h, skk_h, smk_h, n_h = acc.sigma(h)
        assert n_h == int(sel.sum())
        torch.testing.assert_close(smk_h, p_m[sel].T @ p_k[sel], rtol=1e-9, atol=1e-9)
        # a half has 80 rows > r_m + r_k <= 64: canonical factors are valid
        g_s = C.canonical_correlations(C.canonical_factors_from_sigma(smm_h, skk_h, smk_h, n_h))
        g_p = C.canonical_correlations(C.canonical_factors_from_frames(p_m[sel], p_k[sel]))
        assert _rel(g_s, g_p) < 1e-8
        assert acc.ids_seen(h) == {int(i) for i in halves[h]}


def test_phi_from_sigma_equals_phi_from_frames_for_both_decoders():
    p_m, p_k = _correlated(T, 24, 32, seed=2)
    sig = (p_m.T @ p_m, p_k.T @ p_k, p_m.T @ p_k)
    for mode, kw in (("canonical_hard", {"rank": 9}), ("canonical_hard", {}), ("canonical_soft", {"null_threshold": 0.3})):
        a, _ = C.canonical_transport_map(p_m, 1.7, p_k, scale=2.3, mode=mode, **kw)
        b, _ = C.canonical_transport_map_from_sigma(*sig, T, 1.7, scale=2.3, mode=mode, **kw)
        assert _rel(b, a) < 1e-9, (mode, kw)


def test_split_by_id_is_disjoint_reproducible_and_never_by_token():
    ids = torch.arange(100).repeat_interleave(3)  # 100 images, 3 token rows each
    h0, h1 = split_ids(ids, seed=7)
    assert int(h0.numel()) + int(h1.numel()) == 100 and not bool(torch.isin(h0, h1).any())
    h0b, h1b = split_ids(ids.flip(0), seed=7)  # independent of the order the ids arrive in
    assert torch.equal(h0, h0b) and torch.equal(h1, h1b)
    assert not torch.equal(h0, split_ids(ids, seed=8)[0])
    # every row of an image lands in the same half
    in0 = torch.isin(ids, h0)
    for i in range(100):
        assert len({bool(v) for v in in0[ids == i]}) == 1
    with pytest.raises(ValueError, match="neither half"):
        acc = PairSigmaAccumulator(torch.zeros(2), 1.0, torch.zeros(2), 1.0, halves=(h0, h1))
        acc.update(torch.ones(1, 2), torch.ones(1, 2), torch.tensor([1000]))
    with pytest.raises(ValueError, match="share ids"):
        PairSigmaAccumulator(torch.zeros(2), 1.0, torch.zeros(2), 1.0, halves=(h0, h0))


def test_k_per_image_and_k_per_document_sampling():
    cfg = AnchorsConfig(num_anchors=32, anchor_batches=2, canonical_tokens=17, anchor_sampling="k_per_image",
                        anchor_tokens_per_image=4)
    state = AnchorState(cfg)
    rows = state._image_anchor_rows([6, 6])
    assert len(rows) == 32 and rows == sorted(rows)
    by_img: dict[int, list[int]] = {}
    for img, pos in rows:
        by_img.setdefault(img, []).append(pos)
    assert len(by_img) == 8 and all(len(v) == 4 and len(set(v)) == 4 for v in by_img.values())
    assert rows == AnchorState(cfg)._image_anchor_rows([6, 6])  # seeded
    assert cfg.frame()["anchor_tokens_per_image"] == 4
    assert "anchor_tokens_per_image" not in AnchorsConfig().frame()  # default leaves existing frames unchanged

    cand = [[s, p] for s in range(10) for p in range(3 if s % 2 else 12)]  # odd docs are too short for k=4
    tcfg = AnchorsConfig(num_anchors=12, anchor_sampling="k_per_document", anchor_tokens_per_image=4)
    drawn = AnchorState(tcfg)._draw_text_rows(cand, "test rows")
    docs: dict[int, list[int]] = {}
    for s, p in drawn:
        docs.setdefault(s, []).append(p)
    assert len(docs) == 3 and all(len(v) == 4 and s % 2 == 0 for s, v in docs.items())
    with pytest.raises(ValueError, match="documents"):
        AnchorState(AnchorsConfig(num_anchors=24, anchor_sampling="k_per_document", anchor_tokens_per_image=4)
                    )._draw_text_rows(cand, "test rows")
    with pytest.raises(ValueError, match="k_per_document"):
        AnchorState(tcfg)._image_anchor_rows([6, 6])


# ---------------------------------------------------------------- 0.2 unified decoder


@pytest.mark.parametrize("d_m,d_k", WIDTHS)
def test_unified_core_reproduces_v0_zca_cross_and_procrustes(d_m, d_k):
    p_m, p_k = _correlated(T, d_m, d_k, seed=3)
    c_m, scale = 2.0, 3.0
    f = C.canonical_factors_from_frames(p_m, p_k)
    eps = C.resolve_eps_from_frame(p_k, 1e-3)
    # V0 Tikhonov
    phi = C.canonical_phi(f, C.control_core(f, "tikhonov", eps=eps), scale=scale, c_m=c_m)
    assert _rel(phi, C.transport_map_primal(p_m, c_m, p_k, scale=scale, eps=eps)) < 1e-10
    # ZCA at eps 0, and canonical_soft with a zero threshold
    zca, _ = C.whitened_transport_map(p_m, c_m, p_k, scale=scale, eps_rel=0.0, mode="zca")
    assert _rel(C.canonical_phi(f, C.control_core(f, "zca"), scale=scale, c_m=c_m), zca) < 1e-12
    soft, _ = C.canonical_transport_map(p_m, c_m, p_k, scale=scale, mode="canonical_soft")
    assert _rel(soft, zca) < 1e-10
    # cross-covariance: truncate_cross at full rank is C / gbar, gbar = tr(G_k) / d_k
    gbar = float(torch.trace(p_k.T @ p_k)) / d_k
    cross, _ = C.spectral_transport_map(p_m, c_m, p_k, scale=scale, mode="truncate_cross", rank=None)
    assert _rel(C.canonical_phi(f, C.control_core(f, "cross", eps=gbar), scale=scale, c_m=c_m), cross) < 1e-10
    # Procrustes on the anchors
    proc, _ = C.spectral_transport_map(p_m, c_m, p_k, scale=scale, mode="procrustes_cross", rank=None)
    proc_u = C.canonical_phi(f, C.control_core(f, "procrustes_cross"), scale=scale, c_m=c_m)
    assert _rel(proc_u, proc) < 1e-9
    # GPA with M = 2 (fixed point reached in a few iterations; tolerance 1e-9 is ~1e5 above the observed 1e-14)
    _, (a_m, a_k), info = C.gpa([p_m, p_k], tol=1e-14, max_iter=2000)
    assert _rel(C.gpa_transport_map(a_m, a_k, c_m, scale=scale), proc_u) < 1e-9
    assert info["gpa_iters"] < 2000


@pytest.mark.parametrize("d_m,d_k", WIDTHS)
def test_hard_decoder_is_the_polar_factor_and_matches_zca_exactly_when_gamma_is_one(d_m, d_k):
    p_m, p_k = _correlated(T, d_m, d_k, seed=4)
    f = C.canonical_factors_from_frames(p_m, p_k)
    q = min(d_m, d_k)
    core, gamma, info = C.canonical_core(f, "canonical_hard", rank=q)
    u, _, vt = torch.linalg.svd(f.m, full_matrices=False)
    torch.testing.assert_close(core, u @ vt, rtol=1e-10, atol=1e-12)  # Y Z^T = polar(M)
    assert info["canonical_pairs_used"] == q and float(gamma[-1]) < 0.9  # M itself is NOT its polar factor
    assert C.canonical_core(f, "canonical_hard", rank=None)[2]["canonical_pairs_used"] == q
    assert C.canonical_core(f, "canonical_hard", rank=3)[2]["canonical_pairs_used"] == 3
    # Plan statement "hard with r = min(d) equals ZCA at eps 0" holds iff Gamma == 1, i.e. in the degenerate regime
    # T <= max(d), which the decoders refuse (check=False here to build it).
    p_m, p_k = _frames(20, d_m, d_k, seed=5)
    fd = C.canonical_factors_from_frames(p_m, p_k, check=False)
    hard = C.canonical_phi(fd, C.canonical_core(fd, "canonical_hard", rank=min(fd.m.shape))[0], scale=1.0, c_m=1.0)
    zca, _ = C.whitened_transport_map(p_m, 1.0, p_k, scale=1.0, eps_rel=0.0, mode="zca")
    assert float(C.canonical_correlations(fd).min()) == pytest.approx(1.0, abs=1e-10)
    assert _rel(hard, zca) < 1e-10


def test_soft_threshold_scalar_and_array():
    p_m, p_k = _correlated(T, 24, 32, seed=6)
    f = C.canonical_factors_from_frames(p_m, p_k)
    gamma = C.canonical_correlations(f)
    thr = float(gamma[5]) - 1e-9  # keeps exactly the first 6 pairs
    core, _, info = C.canonical_core(f, "canonical_soft", null_threshold=thr)
    assert info["canonical_pairs_used"] == 6
    y, g, zt = torch.linalg.svd(f.m, full_matrices=False)
    torch.testing.assert_close(core, (y[:, :6] * g[:6]) @ zt[:6], rtol=1e-10, atol=1e-12)
    arr = torch.zeros(24, dtype=torch.float64)
    arr[5:] = 2.0  # an explicit per-pair threshold above every Gamma from pair 5 on
    core_a, _, info_a = C.canonical_core(f, "canonical_soft", null_threshold=arr)
    assert info_a["canonical_pairs_used"] == 5
    with pytest.raises(ValueError, match="one entry per canonical pair"):
        C.canonical_core(f, "canonical_soft", null_threshold=torch.zeros(3, dtype=torch.float64))


def test_canonical_correlations_are_invariant_to_the_basis_of_the_frames():
    p_m, p_k = _correlated(T, 24, 32, seed=7)
    q_m = torch.linalg.qr(torch.randn(24, 24, generator=torch.Generator().manual_seed(1), dtype=torch.float64))[0]
    g1 = C.canonical_correlations(C.canonical_factors_from_frames(p_m, p_k))
    g2 = C.canonical_correlations(C.canonical_factors_from_frames(p_m @ q_m, p_k))
    assert _rel(g2, g1) < 1e-10


def test_dispatch_through_transport_map_and_config_validation():
    p_m, p_k = _correlated(T, 24, 32, seed=8)
    src, tgt = _registered("a", p_m, 1.7), _registered("b", p_k, 2.3)
    pm, pk = src.p[0].double(), tgt.p[0].double()
    e = _e(5, 24)
    for params, mode, kw in (
        ({"kernel_decode": "canonical_hard", "canonical_rank": 7}, "canonical_hard", {"rank": 7}),
        ({"kernel_decode": "canonical_soft", "canonical_null_threshold": 0.4}, "canonical_soft", {"null_threshold": 0.4}),
        ({"kernel_decode": "canonical_soft"}, "canonical_soft", {}),
    ):
        fn, eps, info = transport_map(src, 0, tgt, 0, config=parse_anchors_config(params))
        phi, _ = C.canonical_transport_map(pm, 1.7, pk, scale=2.3, mode=mode, **kw)
        torch.testing.assert_close(fn(e), e @ phi, rtol=1e-12, atol=1e-12)
        assert eps is None and "canonical_gamma_max" in info
    for bad in (
        {"kernel_decode": "canonical_soft", "canonical_rank": 3},  # rank is hard-only
        {"kernel_decode": "canonical_hard", "canonical_null_threshold": 0.1},  # threshold is soft-only
        {"kernel_decode": "canonical_hard", "canonical_rank": 0},
        {"kernel_decode": "canonical_soft", "canonical_null_threshold": 1.5},
        {"kernel_decode": "canonical_hard", "frame_map": "gpa"},
        {"kernel_decode": "canonical_hard", "spectral_filter": "procrustes_cross"},
        {"kernel_decode": "canonical_hard", "agreement_tau": 2.0},
        {"canonical_rank": 3},  # without a canonical decoder
        {"anchor_sampling": "k_per_image"},  # k missing
        {"anchor_tokens_per_image": 4},  # k without a k sampling
        {"anchor_sampling": "k_per_image", "anchor_tokens_per_image": 5, "num_anchors": 512},  # k must divide T
    ):
        with pytest.raises(ValueError):
            parse_anchors_config(bad)
    assert "k_per_image" == parse_anchors_config(
        {"anchor_sampling": "k_per_image", "anchor_tokens_per_image": 16}
    ).anchor_sampling


@pytest.mark.parametrize("t,d_m,d_k", [(24, 24, 32), (32, 24, 32), (24, 16, 24), (20, 32, 16)])
def test_canonical_decoders_reject_t_not_above_the_widths(t, d_m, d_k):
    """T <= d_m (or <= d_k): M is orthogonal, Gamma == 1, canonical_hard == canonical_soft == zca: refused, in the
    config-level width check, in the decoders and in the Sigma path."""
    p_m, p_k = _frames(t, d_m, d_k, seed=9)
    with pytest.raises(ValueError, match="canonical decoders need more anchors"):
        C.check_canonical_widths(t, d_m, d_k)
    for mode in C.CANONICAL_DECODES:
        with pytest.raises(ValueError, match="canonical decoders need more anchors"):
            C.canonical_transport_map(p_m, 1.0, p_k, scale=1.0, mode=mode)
        with pytest.raises(ValueError, match="canonical decoders need more anchors"):
            C.canonical_transport_map_from_sigma(
                p_m.T @ p_m, p_k.T @ p_k, p_m.T @ p_k, t, 1.0, scale=1.0, mode=mode
            )
    cfg = parse_anchors_config({"kernel_decode": "canonical_hard", "num_anchors": t})
    src, tgt = _registered("a", p_m, 1.0), _registered("b", p_k, 1.0)
    with pytest.raises(ValueError, match="canonical decoders need more anchors"):
        transport_map(src, 0, tgt, 0, config=cfg)


def test_existing_kernel_decode_names_are_unchanged():
    assert C.KERNEL_DECODES[:3] == ("tikhonov", "zca", "agreement") and set(C.CANONICAL_DECODES) <= set(C.KERNEL_DECODES)
    p_m, p_k = _frames(6, 8, 10, seed=1)
    with pytest.raises(ValueError, match="whitened decode"):
        C.whitened_transport_map(p_m, 1.0, p_k, scale=1.0, eps_rel=0.0, mode="canonical_hard")


# ---------------------------------------------------------------- review fixes: H1 degeneracy rule, M1 held-out weights, M2 filtered centring


def test_forced_unit_pairs_between_max_d_and_sum_d_are_refused():
    """T = 50, d = 24 / 32: T > max(d) (old rule passes) but T < r_m + r_k = 56, so span(U_m) and span(U_k) meet in at
    least 6 dimensions of R^50: >= 6 canonical correlations are exactly 1 from independent noise. Now refused."""
    t, d_m, d_k = 50, 24, 32
    p_m, p_k = _frames(t, d_m, d_k, seed=11)
    fd = C.canonical_factors_from_frames(p_m, p_k, check=False)
    gamma = C.canonical_correlations(fd)
    forced = C.forced_unit_pairs(t, d_m, d_k)
    assert forced == 6
    assert int((gamma > 1 - 1e-9).sum()) >= forced  # the reviewer's reproduction: Gamma == 1 from pure noise
    _, _, info = C.canonical_core(fd, "canonical_soft")
    assert info["forced_unit_pairs"] == forced and info["canonical_rank_source"] == d_m
    with pytest.raises(ValueError, match=r"T=50 <= r_m \+ r_k = 24 \+ 32"):
        C.canonical_factors_from_frames(p_m, p_k)
    sig = (p_m.T @ p_m, p_k.T @ p_k, p_m.T @ p_k)
    with pytest.raises(ValueError, match="canonical decoders need more anchors"):
        C.canonical_factors_from_sigma(*sig, t)
    for mode in C.CANONICAL_DECODES:
        with pytest.raises(ValueError, match="canonical decoders need more anchors"):
            C.canonical_transport_map(p_m, 1.0, p_k, scale=1.0, mode=mode)


def test_degeneracy_boundary_is_t_equal_sum_of_ranks_and_t80_passes():
    with pytest.raises(ValueError):
        C.check_canonical_widths(56, 24, 32)  # T == r_m + r_k: refused
    C.check_canonical_widths(57, 24, 32)
    p_m, p_k = _correlated(80, 24, 32, seed=12)  # T = 80 > 56
    f = C.canonical_factors_from_frames(p_m, p_k)
    _, gamma, info = C.canonical_core(f, "canonical_soft")
    assert info["forced_unit_pairs"] == 0 and float(gamma[-1]) < 1 - 1e-3
    # a rank-deficient frame counts with its numerical rank: d_k = 32 columns of rank 8 pass at T = 40 (40 > 24 + 8)
    p_k8 = torch.randn(40, 8, generator=torch.Generator().manual_seed(1), dtype=torch.float64) @ torch.randn(
        8, 32, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    p_m24 = torch.randn(40, 24, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    f = C.canonical_factors_from_frames(p_m24, p_k8)
    assert f.s_k.numel() == 8 and f.n_rows == 40
    # the same rule per half in the Sigma path: 80 rows pass, 40 rows of d = 24 / 32 are refused
    sig = (p_m.T @ p_m, p_k.T @ p_k, p_m.T @ p_k)
    C.canonical_factors_from_sigma(*sig, 80)
    with pytest.raises(ValueError, match="canonical decoders need more anchors"):
        C.canonical_factors_from_sigma(*sig, 40)


def _sig(a, b):
    return a.T @ a, b.T @ b, a.T @ b


def test_gamma_out_equals_in_sample_gamma_on_identical_halves():
    p_m, p_k = _correlated(T, 24, 32, seed=13)
    f = C.canonical_factors_from_frames(p_m, p_k)
    out = C.canonical_correlations_out(f, _sig(p_m, p_k))
    torch.testing.assert_close(out, C.canonical_correlations(f), rtol=1e-9, atol=1e-10)


def test_gamma_out_of_independent_noise_is_near_zero_while_in_sample_is_not():
    n, d = 100, 16  # each half: n = 100 > 2 d
    p_m, p_k = _frames(2 * n, d, d, seed=14)
    fa = C.canonical_factors_from_frames(p_m[:n], p_k[:n])
    g_in = C.canonical_correlations(fa)
    g_out = C.canonical_correlations_out(fa, _sig(p_m[n:], p_k[n:]))
    # out-of-sample correlation of a fixed direction pair of independent data: sd ~ 1/sqrt(n) = 0.1 per pair
    assert float(g_out.abs().mean()) < 2.0 / n**0.5
    assert float(g_out.abs().max()) < 5.0 / n**0.5
    assert float(g_in.mean()) > 0.25 and float(g_in.mean()) > 3.0 * float(g_out.abs().mean())


def test_gamma_out_recovers_a_planted_correlation_within_the_sampling_error():
    rho, d, n = 0.6, 8, 4000
    g = torch.Generator().manual_seed(15)
    s = torch.randn(2 * n, 1, generator=g, dtype=torch.float64)
    x = rho**0.5 * s + (1 - rho) ** 0.5 * torch.randn(2 * n, 1, generator=g, dtype=torch.float64)
    y = rho**0.5 * s + (1 - rho) ** 0.5 * torch.randn(2 * n, 1, generator=g, dtype=torch.float64)
    p_m = torch.cat([x, torch.randn(2 * n, d - 1, generator=g, dtype=torch.float64)], 1)
    p_k = torch.cat([y, torch.randn(2 * n, d - 1, generator=g, dtype=torch.float64)], 1)
    # corr(x, y) = rho. Sampling sd of a correlation ~ (1 - rho^2) / sqrt(n) = 0.0101; tolerance 5 sd (+ direction error)
    tol = 5 * (1 - rho**2) / n**0.5
    fa = C.canonical_factors_from_frames(p_m[:n], p_k[:n])
    out = C.canonical_correlations_out(fa, _sig(p_m[n:], p_k[n:]))
    assert abs(float(out[0]) - rho) < tol
    assert float(out[1:].abs().max()) < 5.0 / n**0.5  # the noise pairs: sd 1/sqrt(n) = 0.016
    # swapping the roles of the halves gives the same answer within the same tolerance
    fb = C.canonical_factors_from_frames(p_m[n:], p_k[n:])
    assert abs(float(C.canonical_correlations_out(fb, _sig(p_m[:n], p_k[:n]))[0]) - rho) < tol


def test_soft_with_explicit_weights():
    p_m, p_k = _correlated(T, 24, 32, seed=16)
    f = C.canonical_factors_from_frames(p_m, p_k)
    gamma = C.canonical_correlations(f)
    # weights = Gamma (all pairs) is the in-sample soft decode with threshold 0
    a, _, _ = C.canonical_core(f, "canonical_soft", weights=gamma)
    b, _, _ = C.canonical_core(f, "canonical_soft", null_threshold=0.0)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
    # held-out weights thresholded by the caller: differ from the in-sample ones and zero where the caller says
    out = C.canonical_correlations_out(f, _sig(*_correlated(T, 24, 32, seed=17)))
    g = out * (out >= 0.2)
    core, _, info = C.canonical_core(f, "canonical_soft", weights=g)
    y, _, zt = torch.linalg.svd(f.m, full_matrices=False)
    torch.testing.assert_close(core, (y * g) @ zt, rtol=1e-12, atol=1e-14)
    assert info["canonical_pairs_used"] == float((g != 0).sum())
    phi, _ = C.canonical_transport_map(p_m, 1.0, p_k, scale=1.0, mode="canonical_soft", weights=g)
    assert phi.shape == (24, 32)
    with pytest.raises(ValueError, match="one entry per canonical pair"):
        C.canonical_core(f, "canonical_soft", weights=torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="mutually exclusive"):
        C.canonical_core(f, "canonical_soft", weights=gamma, null_threshold=0.5)
    with pytest.raises(ValueError, match="canonical_soft only"):
        C.canonical_core(f, "canonical_hard", weights=gamma)


def test_accumulator_with_filtered_mu_c_and_mask_reproduces_the_decoder_frame():
    """anchor_filter 'norm': the decoder's P is ``filtered_frame`` (filtered mu / c, flagged rows dropped); the
    accumulator built from ``filtered_pair_inputs`` equals P^T P of exactly that frame."""
    from merge_and_rebase.rebase.methods.anchors.method import filtered_frame, filtered_pair_inputs

    t, d_m, d_k, k = 60, 5, 7, 2.0
    g = torch.Generator().manual_seed(18)
    s_m = 3.0 + torch.randn(t, d_m, generator=g, dtype=torch.float64)
    s_k = -2.0 + 2.0 * torch.randn(t, d_k, generator=g, dtype=torch.float64)
    regs = []
    for s, mu_shift, c_f, flag_rows in ((s_m, 0.3, 1.7, [3, 10]), (s_k, -0.2, 2.9, [10, 41, 42])):
        mu0, c0 = s.mean(0), 1.25  # registration statistics (differ from the filtered ones)
        flags = torch.zeros(t, dtype=torch.bool)
        flags[flag_rows] = True
        extras = {f"norm_k{k:g}": {"mu": [mu0 + mu_shift], "c": [c_f], "anchor_flags": [flags]}}
        regs.append(RegisteredModel("m", {}, [((s - mu0) / c0).float()], [mu0], [c0], extras))
    src, tgt = regs
    mu_m, c_m, mu_k, c_k, keep = filtered_pair_inputs(src, 0, tgt, 0, k)
    assert int((~keep).sum()) == 4  # rows 3, 10, 41, 42
    p_m, c_m_ref = filtered_frame(src, 0, keep, k)
    p_k, c_k_ref = filtered_frame(tgt, 0, keep, k)
    assert (c_m, c_k) == (c_m_ref, c_k_ref)
    # raw rows as the stored float32 frame represents them (P c + mu), streamed in uneven batches with the row mask
    raw_m = src.p[0].double() * src.c[0] + src.mu[0]
    raw_k = tgt.p[0].double() * tgt.c[0] + tgt.mu[0]
    acc = PairSigmaAccumulator(mu_m, c_m, mu_k, c_k)
    for lo, hi in ((0, 11), (11, 40), (40, 60)):
        acc.update(raw_m[lo:hi], raw_k[lo:hi], mask=keep[lo:hi])
    smm, skk, smk, n = acc.sigma()
    assert n == int(keep.sum()) == p_m.shape[0]
    torch.testing.assert_close(smm, p_m.T @ p_m, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(skk, p_k.T @ p_k, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(smk, p_m.T @ p_k, rtol=1e-10, atol=1e-10)
    # centring with the registration mu / c instead would NOT match (the bug this fixes)
    acc0 = PairSigmaAccumulator(src.mu[0], src.c[0], tgt.mu[0], tgt.c[0])
    acc0.update(raw_m, raw_k)
    assert _rel(acc0.sigma()[0], p_m.T @ p_m) > 1e-2
    with pytest.raises(ValueError, match="one flag per row"):
        acc.update(raw_m[:5], raw_k[:5], mask=keep[:4])


def test_soft_rejects_negative_weights():
    p_m, p_k = _correlated(T, 24, 32, seed=18)
    f = C.canonical_factors_from_frames(p_m, p_k)
    g = C.canonical_correlations(f).clone()
    g[3] = -0.1
    with pytest.raises(ValueError, match="weights must be >= 0"):
        C.canonical_core(f, "canonical_soft", weights=g)
    # the documented mapping of a signed Gamma_out gives zero for the flipped pair
    out = torch.tensor([0.5] * (g.numel() - 1) + [-0.3], dtype=torch.float64)
    ok = out * (out >= 0.0)
    _, _, info = C.canonical_core(f, "canonical_soft", weights=ok)
    assert info["canonical_soft_weights"] == "explicit"


def test_in_sample_soft_warns_and_flags_the_info():
    p_m, p_k = _correlated(T, 24, 32, seed=19)
    f = C.canonical_factors_from_frames(p_m, p_k)
    with pytest.warns(UserWarning, match="NOT plan decoder D3"):
        _, _, info = C.canonical_core(f, "canonical_soft", null_threshold=0.1)
    assert info["canonical_soft_weights"] == "in_sample"
    _, _, info_h = C.canonical_core(f, "canonical_hard")
    assert "canonical_soft_weights" not in info_h


def _planted(n, rho, d, g):
    s = torch.randn(n, 1, generator=g, dtype=torch.float64)
    x = rho**0.5 * s + (1 - rho) ** 0.5 * torch.randn(n, 1, generator=g, dtype=torch.float64)
    y = rho**0.5 * s + (1 - rho) ** 0.5 * torch.randn(n, 1, generator=g, dtype=torch.float64)
    return (torch.cat([x, torch.randn(n, d - 1, generator=g, dtype=torch.float64)], 1),
            torch.cat([y, torch.randn(n, d - 1, generator=g, dtype=torch.float64)], 1))


def test_gamma_out_unequal_halves_and_scaled_sigma_b():
    rho, d, n_a = 0.6, 8, 3000
    n_b = n_a // 3
    g = torch.Generator().manual_seed(20)
    a_m, a_k = _planted(n_a, rho, d, g)
    b_m, b_k = _planted(n_b, rho, d, g)
    fa = C.canonical_factors_from_frames(a_m, a_k)
    sm, sk, smk = _sig(b_m, b_k)
    base = C.canonical_correlations_out(fa, (sm, sk, smk))
    # a (1 - rho^2) / sqrt(n_b) sampling sd; 5 sd tolerance
    assert abs(float(base[0]) - rho) < 5 * (1 - rho**2) / n_b**0.5
    # positive rescaling of half B's Sigma (all three blocks): Gamma_out unchanged
    for c in (0.01, 7.0, 1e4):
        torch.testing.assert_close(C.canonical_correlations_out(fa, (c * sm, c * sk, c * smk)), base, rtol=1e-10, atol=1e-12)
    # an unnormalised half B (Sigma not divided by n_b) equals the per-row average up to the scale n_b
    torch.testing.assert_close(C.canonical_correlations_out(fa, (sm / n_b, sk / n_b, smk / n_b)), base, rtol=1e-10, atol=1e-12)


def test_gamma_out_refuses_rank_deficient_half_b():
    p_m, p_k = _correlated(T, 24, 32, seed=21)
    f = C.canonical_factors_from_frames(p_m, p_k)
    b_m, b_k = _correlated(10, 24, 32, seed=22)  # 10 rows < 24, 32: rank-deficient, but a_i may still see variance
    b_m = torch.zeros_like(b_m)
    with pytest.raises(ValueError, match="rank-deficient"):
        C.canonical_correlations_out(f, _sig(b_m, b_k))
