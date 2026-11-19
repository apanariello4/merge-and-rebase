"""Plan decoder D3 of 'metodo K' (PIANO_METODO_K.md Fase 2): ``kernel_decode='canonical_soft'`` with held-out weights
``g = clip(Gamma_out, 0, 1)`` (``canonical_soft_weights='heldout'``). Synthetic frames, CPU, float64 unless stated."""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.rebase.methods.anchors import AnchorsConfig, parse_anchors_config
from merge_and_rebase.rebase.methods.anchors import core as C
from merge_and_rebase.rebase.methods.anchors.canonical_stats import heldout_gamma_out, split_ids
from merge_and_rebase.rebase.methods.anchors.method import transport_map
from merge_and_rebase.rebase.methods.anchors.state import RegisteredModel

D_M, D_K, N_IMG, K_ROWS, R_SHARED = 20, 24, 64, 4, 4  # T = 256 rows; each half has 128 > d_m + d_k = 44 rows

HELDOUT = {"kernel_decode": "canonical_soft", "canonical_soft_weights": "heldout",
           "anchor_sampling": "k_per_image", "anchor_tokens_per_image": 4, "num_anchors": 256}


def _planted(seed=0, noise=0.4):
    """Two frames sharing R_SHARED latent directions (variance 9 against noise variance (noise + 0.5)^2) plus independent
    noise in every other direction; ids: K_ROWS consecutive rows per image."""
    g = torch.Generator().manual_seed(seed)
    t = N_IMG * K_ROWS
    z = torch.randn(t, R_SHARED, generator=g, dtype=torch.float64)

    def one(d):
        a = torch.linalg.qr(torch.randn(d, R_SHARED, generator=g, dtype=torch.float64))[0].T  # [r, d] orthonormal rows
        return 3.0 * (z @ a) + (noise + 0.5) * torch.randn(t, d, generator=g, dtype=torch.float64)

    ids = torch.arange(t) // K_ROWS
    return one(D_M), one(D_K), ids


def _registered(name, p, ids, c=1.0):
    d = p.shape[1]
    reg = RegisteredModel(name, {}, [p.double()], [torch.zeros(d, dtype=torch.float64)], [c])
    reg.anchor_ids = ids
    return reg


# ---------------------------------------------------------------- config


def test_default_is_in_sample_and_not_recorded():
    cfg = parse_anchors_config({"kernel_decode": "canonical_soft"})
    assert cfg.canonical_soft_weights == "in_sample"
    assert "canonical_soft_weights" not in cfg.to_dict()
    assert "canonical_soft_weights" not in AnchorsConfig().to_dict()
    assert parse_anchors_config(HELDOUT).to_dict()["canonical_soft_weights"] == "heldout"


def test_config_validation():
    assert parse_anchors_config(HELDOUT).canonical_soft_weights == "heldout"
    assert parse_anchors_config({**HELDOUT, "anchor_sampling": "k_per_document"}).anchor_sampling == "k_per_document"
    assert parse_anchors_config({**HELDOUT, "canonical_null_threshold": 0.2}).canonical_null_threshold == 0.2
    for bad, match in (
        ({**HELDOUT, "kernel_decode": "canonical_hard"}, "canonical_soft"),
        ({**HELDOUT, "kernel_decode": "tikhonov"}, "canonical_soft"),
        ({**HELDOUT, "anchor_sampling": "random", "anchor_tokens_per_image": None}, "k_per_image"),
        ({**HELDOUT, "anchor_sampling": "one_per_image", "anchor_tokens_per_image": None}, "k_per_image"),
        ({**HELDOUT, "anchor_dedup_cos": 0.9}, "anchor_dedup_cos"),
        ({**HELDOUT, "canonical_soft_weights": "bogus"}, "canonical_soft_weights"),
        ({**HELDOUT, "frame_map": "gpa"}, "pairwise"),
    ):
        with pytest.raises(ValueError, match=match):
            parse_anchors_config(bad)


# ---------------------------------------------------------------- held-out weights on a planted subspace


def test_heldout_weights_are_one_on_shared_directions_and_zero_on_noise_while_in_sample_is_inflated():
    p_m, p_k, ids = _planted(seed=1)
    gamma_in = C.canonical_correlations(C.canonical_factors_from_frames(p_m, p_k))
    gamma_out, info = heldout_gamma_out(p_m, p_k, ids, seed=3)
    r = R_SHARED
    assert float(gamma_out[:r].min()) > 0.8  # the planted directions: high out of sample (signal / (signal + noise))
    assert float(gamma_out[:r].min()) <= 1.0 + 1e-9
    noise_out = gamma_out[r:].clamp(0.0, 1.0)
    assert float(noise_out.mean()) < 0.15  # noise directions: ~0 out of sample (signed, clipped at 0)
    assert float(gamma_in[r:].mean()) > 0.2  # in-sample noise directions are inflated by the fit
    assert float(gamma_in[r:].mean()) > 4 * float(noise_out.mean())
    assert info["heldout_rows_half_a"] + info["heldout_rows_half_b"] == N_IMG * K_ROWS


def test_split_is_by_id_never_by_row():
    p_m, p_k, ids = _planted(seed=2)
    ref, _ = heldout_gamma_out(p_m, p_k, ids, seed=5)
    perm = torch.randperm(ids.numel(), generator=torch.Generator().manual_seed(0))
    got, _ = heldout_gamma_out(p_m[perm], p_k[perm], ids[perm], seed=5)  # row order does not matter
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-10)
    # all rows of an id sit in the same half: the halves of the split partition the ids and every id is whole
    a, b = split_ids(ids, 5)
    assert not bool(torch.isin(a, b).any()) and a.numel() + b.numel() == N_IMG
    # the same data with ids that cut the images (a row-level split) gives a different estimate
    row_ids = torch.arange(ids.numel())
    cut, _ = heldout_gamma_out(p_m, p_k, row_ids, seed=5)
    assert not torch.allclose(cut[: ref.numel()], ref[: cut.numel()], atol=1e-6)
    with pytest.raises(ValueError, match="one id per"):
        heldout_gamma_out(p_m, p_k, ids[:-1], seed=5)


def test_heldout_is_deterministic_for_a_fixed_seed_and_depends_on_the_seed():
    p_m, p_k, ids = _planted(seed=3)
    a, _ = heldout_gamma_out(p_m, p_k, ids, seed=11)
    b, _ = heldout_gamma_out(p_m, p_k, ids, seed=11)
    c, _ = heldout_gamma_out(p_m, p_k, ids, seed=12)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    cfg = parse_anchors_config(HELDOUT)
    src, tgt = _registered("a", p_m, ids), _registered("b", p_k, ids)
    e = torch.randn(7, D_M, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    fn1, _, _ = transport_map(src, 0, tgt, 0, config=cfg)
    fn2, _, _ = transport_map(src, 0, tgt, 0, config=cfg)
    assert torch.equal(fn1(e), fn2(e))


def test_too_few_rows_in_a_half_is_refused():
    p_m, p_k, ids = _planted(seed=4)
    few = ids < 8  # 8 images x 4 rows = 32 rows, a half has 16 <= d_m + d_k
    with pytest.raises(ValueError, match="canonical decoders need more anchors"):
        heldout_gamma_out(p_m[few], p_k[few], ids[few], seed=0)


# ---------------------------------------------------------------- the decoder through transport_map


def test_transport_map_uses_heldout_weights_and_records_them():
    p_m, p_k, ids = _planted(seed=5)
    cfg = parse_anchors_config(HELDOUT)
    src, tgt = _registered("a", p_m, ids, 1.5), _registered("b", p_k, ids, 2.0)
    fn, eps, info = transport_map(src, 0, tgt, 0, config=cfg)
    assert eps is None and info["canonical_soft_weights"] == "heldout"
    gamma_out, _ = heldout_gamma_out(p_m, p_k, ids, seed=cfg.anchor_seed)
    assert info["canonical_gamma_out_mean"] == pytest.approx(float(gamma_out.mean()))
    assert info["canonical_gamma_out_min"] == pytest.approx(float(gamma_out.min()))
    assert info["canonical_gamma_out_max"] == pytest.approx(float(gamma_out.max()))
    # equals the explicit-weights decoder on the same factors
    factors = C.canonical_factors_from_frames(p_m, p_k)
    g = torch.zeros(min(factors.s_m.numel(), factors.s_k.numel()), dtype=torch.float64)
    g[: gamma_out.numel()] = gamma_out.clamp(0.0, 1.0)
    f, _, _ = C.canonical_core(factors, "canonical_soft", weights=g)
    phi = C.canonical_phi(factors, f, scale=2.0, c_m=1.5)
    e = torch.randn(5, D_M, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    torch.testing.assert_close(fn(e), e @ phi, rtol=1e-10, atol=1e-12)
    # D3 transports less than the in-sample soft decode (whose noise-direction weights are inflated)
    cfg_in = parse_anchors_config({"kernel_decode": "canonical_soft"})
    with pytest.warns(UserWarning, match="in-sample"):
        fn_in, _, info_in = transport_map(src, 0, tgt, 0, config=cfg_in)
    assert info_in["canonical_soft_weights"] == "in_sample"
    assert float(torch.linalg.norm(fn(e))) < float(torch.linalg.norm(fn_in(e)))
    # null threshold keeps its meaning: weights below it are zeroed
    cfg_thr = parse_anchors_config({**HELDOUT, "canonical_null_threshold": 0.5})
    _, _, info_thr = transport_map(src, 0, tgt, 0, config=cfg_thr)
    assert info_thr["canonical_pairs_used"] <= info["canonical_pairs_used"]
    assert info_thr["canonical_pairs_used"] == float((g >= 0.5).sum())


def test_heldout_without_row_ids_is_refused():
    p_m, p_k, ids = _planted(seed=6)
    src, tgt = _registered("a", p_m, None), _registered("b", p_k, None)
    with pytest.raises(ValueError, match="row ids"):
        transport_map(src, 0, tgt, 0, config=parse_anchors_config(HELDOUT))


def test_existing_decoders_do_not_read_the_ids():
    p_m, p_k, ids = _planted(seed=7)
    src, tgt = _registered("a", p_m, None), _registered("b", p_k, None)  # no ids at all
    e = torch.randn(3, D_M, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    for params in ({"kernel_decode": "canonical_hard"}, {"kernel_decode": "tikhonov"}, {"kernel_decode": "zca"}):
        fn, _, _ = transport_map(src, 0, tgt, 0, config=parse_anchors_config(params))
        assert torch.isfinite(fn(e)).all()


# ---------------------------------------------------------------- combinations (point 2 of the brief)


@pytest.mark.parametrize(
    "extra",
    [
        {"depth_fit": "tracking", "synthesis": "ridge_preserve", "sequential_impl": "incremental"},  # vision O2 closed loop
        {"depth_fit": "tracking", "synthesis": "ridge_preserve"},
        {"depth_fit": "sequential", "synthesis": "ridge"},  # decoder LLM: plain ridge
    ],
)
@pytest.mark.parametrize("sampling", ["k_per_image", "k_per_document"])
def test_kernel_decoder_combinations_pass_config_validation(extra, sampling):
    base = {k: v for k, v in HELDOUT.items() if k not in ("kernel_decode", "canonical_soft_weights")}
    for decode in ({"kernel_decode": "canonical_soft"}, {"kernel_decode": "canonical_hard"}, HELDOUT):
        cfg = parse_anchors_config({**base, **decode, "anchor_sampling": sampling, **extra})
        assert cfg.kernel_decode.startswith("canonical") and cfg.frame_map == "pairwise"


def test_toy_end_to_end_heldout_with_tracking_and_o2():
    from tests.test_anchors_method import _cfg, _prepare, _setup  # noqa: PLC0415

    setup = _setup(2, 4, source_width=3, target_width=4)
    kw = dict(num_anchors=20, anchor_sampling="k_per_image", anchor_tokens_per_image=5, kernel_decode="canonical_soft")
    plain = _prepare(_cfg(**kw), setup)
    held = _prepare(_cfg(**kw, canonical_soft_weights="heldout"), setup)
    trk = _prepare(_cfg(**kw, canonical_soft_weights="heldout", depth_fit="tracking", synthesis="ridge_preserve",
                        rho_g=0.1, preserve_tokens=6), setup)
    seq = _prepare(_cfg(**kw, canonical_soft_weights="heldout", depth_fit="sequential"), setup)
    for res in (plain, held, trk, seq):
        assert res.unit_task_vector and all(torch.isfinite(v).all() for v in res.unit_task_vector.values())
    assert any(r.get("anchors_canonical_soft_weights") == "heldout" for r in held.diagnostics)
    assert all(r.get("anchors_canonical_soft_weights") == "in_sample" for r in plain.diagnostics)


# ---------------------------------------------------------------- streamed Sigma path (all tokens, no stored rows)


def test_sigma_decoder_equals_the_row_decoder_on_the_same_rows():
    """Streamed per-half Sigma (several batches, raw rows with a large mean, registration mu / c) -> the same Phi as the
    row-based D3 on the stored frames P = (S - mu) / c."""
    from merge_and_rebase.rebase.methods.anchors.canonical_stats import (
        PairSigmaAccumulator,
        heldout_soft_phi_from_sigma,
    )

    p_m, p_k, ids = _planted(seed=8)
    c_m, c_k, scale, seed = 1.5, 2.0, 2.0, 7
    mu_m, mu_k = 50.0 * torch.randn(D_M, dtype=torch.float64), 30.0 * torch.randn(D_K, dtype=torch.float64)
    s_m, s_k = p_m * c_m + mu_m, p_k * c_k + mu_k
    acc = PairSigmaAccumulator(mu_m, c_m, mu_k, c_k, halves=split_ids(ids, seed))
    for lo in range(0, ids.numel(), 37):  # uneven batches
        acc.update(s_m[lo:lo + 37], s_k[lo:lo + 37], ids[lo:lo + 37])
    phi_sigma, info_s = heldout_soft_phi_from_sigma(acc.sigma(0), acc.sigma(1), c_m=c_m, scale=scale,
                                                    null_threshold=0.1)
    cfg = parse_anchors_config({**HELDOUT, "anchor_seed": seed, "canonical_null_threshold": 0.1})
    src, tgt = _registered("a", p_m, ids, c_m), _registered("b", p_k, ids, c_k)
    fn, _, info_r = transport_map(src, 0, tgt, 0, config=cfg)
    e = torch.randn(9, D_M, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    # the registered "scale" is c_k (rescale_target default), the same value used above
    phi_rows = fn(torch.eye(D_M, dtype=torch.float64))
    c_k_scale = cfg.rescale_target and c_k or c_m
    torch.testing.assert_close(phi_sigma * (c_k_scale / scale), phi_rows, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(fn(e), e @ phi_rows)
    for k in ("canonical_gamma_out_mean", "canonical_gamma_out_min", "canonical_gamma_out_max", "canonical_pairs_used"):
        assert info_s[k] == pytest.approx(info_r[k], rel=1e-6)


def test_sigma_all_tokens_config_default_unrecorded_and_validation():
    cfg = parse_anchors_config(HELDOUT)
    assert cfg.canonical_source == "rows" and "canonical_source" not in cfg.to_dict()
    assert "canonical_images" not in cfg.to_dict()
    ok = {"kernel_decode": "canonical_soft", "canonical_soft_weights": "heldout", "canonical_source": "sigma_all_tokens",
          "canonical_images": 8}
    assert parse_anchors_config(ok).to_dict()["canonical_source"] == "sigma_all_tokens"
    for bad in ({**ok, "canonical_images": None}, {**ok, "canonical_soft_weights": "in_sample"},
                {**ok, "anchor_filter": "norm"}, {**ok, "canonical_source": "bogus"},
                {"kernel_decode": "canonical_soft", "canonical_images": 8}):
        with pytest.raises(ValueError):
            parse_anchors_config(bad)


def test_streamed_sigma_toy_end_to_end_equals_the_row_path_on_the_same_tokens_and_the_cache_is_reused(tmp_path):
    """Whole images (all TOKENS canonical positions) through the toy models: the streamed path equals the row path run on
    the same rows (anchor_sampling='whole_images' stored rows), split by image id with the same seed; the second run
    loads the cache and does no forward pass."""
    import merge_and_rebase.rebase.methods.anchors.canonical_stream as cs  # noqa: PLC0415
    from merge_and_rebase.rebase.methods.anchors.state import AnchorState  # noqa: PLC0415
    from tests.test_anchors_method import _cfg, _prepare, _setup  # noqa: PLC0415

    setup = _setup(2, 4, source_width=3, target_width=4)
    kw = dict(kernel_decode="canonical_soft", canonical_soft_weights="heldout", num_anchors=20)
    rows = _prepare(_cfg(**kw, anchor_sampling="whole_images"), setup)
    stream_cfg = _cfg(**kw, canonical_source="sigma_all_tokens", canonical_images=4)
    sd = tmp_path / "state"
    first = _prepare(stream_cfg.__class__(**{**stream_cfg.__dict__, "state_dir": str(sd)}), setup)
    assert (sd / "sigma").is_dir() and len(list((sd / "sigma").glob("*.pt"))) == 1
    calls = []
    orig = cs.stream_sigma
    cs.stream_sigma = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        second = _prepare(stream_cfg.__class__(**{**stream_cfg.__dict__, "state_dir": str(sd)}), setup)
    finally:
        cs.stream_sigma = orig
    assert calls == []  # cache reuse
    for k, v in first.unit_task_vector.items():
        torch.testing.assert_close(second.unit_task_vector[k], v)
        assert torch.isfinite(v).all()
    assert any(r.get("anchors_canonical_source") == "sigma_all_tokens" for r in second.diagnostics)
    assert rows.unit_task_vector.keys() == first.unit_task_vector.keys()
    for k, v in rows.unit_task_vector.items():  # same tokens, same id split: the two sources give the same decoder
        torch.testing.assert_close(first.unit_task_vector[k], v, rtol=1e-4, atol=1e-6)
    del AnchorState
