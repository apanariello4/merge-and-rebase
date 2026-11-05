"""eval/diagnostics/update_kernel_cka.py: synthetic, CPU, small. The pipeline (models, data, GPU) is not run here,
except `_stream_pass` with fake models (rows correspond, only the expert differs)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from merge_and_rebase.analysis.cka import cka_bootstrap, linear_cka
from merge_and_rebase.eval.diagnostics import update_kernel_cka as uk

DT = torch.float64
N_IMG, N_PATCH = 40, 4  # 40 images, CLS + 4 patches (2 x 2 grid)
PAIR_DEFS = (
    ("other_task", "m_t", "k_o"), ("other_task_rev", "m_o", "k_t"), ("within_m", "m_t", "m_o"), ("within_k", "k_t", "k_o"), ("random_update", "m_t", "k_rand"),
)


def _tokens(n_img, n_patch, d, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(n_img, 1 + n_patch, d, generator=g, dtype=DT)


def _ortho(d, seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=DT))
    return q


def _frame(p_m, p_k, images=None):
    return uk.PairFrame(0, 0, p_m, p_k, 1.0, 1.0, images)


def _stream(hm, gk, p_m, p_k, *, batch=7, chunk_batches=2, n_perm=0, seed=0, extra=None):
    """Feed ``[N, 1 + P, d]`` update tokens through every accumulator in batches. ``extra``: ``tag -> (tokens,
    'm' | 'k')`` for the sides ``m_o``, ``k_o``, ``k_rand`` (pairs ``other_task``, ``within_m``, ``within_k``,
    ``random_update`` are added when both of their tags are present)."""
    toks = {"m_t": (hm, "m"), "k_t": (gk, "k"), **(extra or {})}
    frames_p = {"m": p_m, "k": p_k}
    pairs = [("true", "m_t", "k_t")] + [pd for pd in PAIR_DEFS if pd[1] in toks and pd[2] in toks]
    n, n_tok, dm = hm.shape
    dk = gk.shape[2]
    t = p_m.shape[0]
    q_m, q_k = p_m.T @ p_m, p_k.T @ p_k
    chunks: list[uk.PairAcc] = []
    dchunks: list[uk.DirectAcc] = []
    perm_cross = {rt: [uk.CrossStats(dm, dk) for _ in range(n_perm)] for rt in ("cls", "patch")} if n_perm else None
    perm = uk.PermCross(n_perm, n_tok, t) if n_perm else None
    for b, start in enumerate(range(0, n, batch)):
        c = b // chunk_batches
        while len(chunks) <= c:
            chunks.append(uk.PairAcc(dm, dk, ("cls", "patch")))
            dchunks.append(uk.DirectAcc(n_tok, t))
        sl = slice(start, start + batch)
        h, g = hm[sl], gk[sl]
        gen = torch.Generator().manual_seed(seed * 1000 + b)
        sigmas = [uk.image_derangement(h.shape[0], gen) for _ in range(n_perm)]
        chunks[c].update(
            uk._rows(h, n_tok, False), uk._rows(g, n_tok, False), q_m, q_k, patches_per_image=n_tok - 1,
            perms=sigmas, perm_cross=perm_cross,
        )
        enc = {tag: x[sl] @ frames_p[kind].T for tag, (x, kind) in toks.items()}
        dchunks[c].update(enc, list(toks), pairs)
        if perm is not None:
            perm.update(enc["m_t"], enc["k_t"], sigmas)
    block = uk.DirectBlock(
        {tag: [d.sides[tag] for d in dchunks] for tag in toks},
        {nm: [d.pairs[nm] for d in dchunks] for nm, _, _ in pairs},
        {nm: (tx, ty) for nm, tx, ty in pairs}, perm,
    )
    native = {}
    for tag, (x, _) in toks.items():
        native[tag] = uk.NativeSide(x.shape[1], x.shape[2])
        native[tag].update(x)
    return SimpleNamespace(chunks=chunks, perm_cross=perm_cross, direct=block, native=native)


def _fin(hm, gk, p_m, p_k, opts, *, frame=None, **kw):
    st = _stream(hm, gk, p_m, p_k, **kw)
    return uk.finalize_pair(
        st.chunks, frame or _frame(p_m, p_k), opts, perm_cross=st.perm_cross, native=st.native, direct=st.direct
    )


def _opts(**kw):
    return uk.FinalizeOptions(**{"debiased": True, **kw})


def _related_models(seed=0, d=8, t=60):
    hm = _tokens(N_IMG, N_PATCH, d, seed)
    q = _ortho(d, seed + 1)
    g = torch.Generator().manual_seed(seed + 2)
    p_m = torch.randn(t, d, generator=g, dtype=DT)
    return hm, hm @ q, p_m, p_m @ q  # dH_k = dH_m Q and P_k = P_m Q: D_k == D_m exactly


def test_exact_orthogonal_map_gives_one_everywhere():
    hm, gk, p_m, p_k = _related_models()
    res = _fin(hm, gk, p_m, p_k, _opts())
    assert next(iter(res)) == "direct"  # the primary first
    for g in ("all", "cls", "patch"):
        for v in uk.DIRECT_VARIANTS:
            assert res["direct"][v][g]["value"] == pytest.approx(1.0, abs=1e-9), (v, g)
        assert res["m1"][g]["value"] == pytest.approx(1.0, abs=1e-9)
        for v in uk.M1_VARIANTS:
            assert res["m1_variants"][v][g]["value"] == pytest.approx(1.0, abs=1e-9)
        assert res["m2"][g]["value"] == pytest.approx(1.0, abs=1e-9)


def test_independent_updates_sit_at_the_null_level():
    d, t = 10, 120
    hm = _tokens(60, N_PATCH, d, 10)
    gk = _tokens(60, N_PATCH, d, 11)
    g = torch.Generator().manual_seed(12)
    p_m, p_k = torch.randn(t, d, generator=g, dtype=DT), torch.randn(t, d, generator=g, dtype=DT)
    res = _fin(hm, gk, p_m, p_k, _opts(null=True, n_perm=60), n_perm=60)
    for g_ in ("cls", "patch", "all"):
        assert abs(res["m2"][g_]["value"]) < 0.25
        assert res["m2"][g_]["value"] < res["m2"][g_]["null"]["q95"] + 0.1
        assert abs(res["m2"][g_]["null"]["z"]) < 4.0 and res["m2"][g_]["null"]["p_empirical"] > 0.01
        for v in ("position_centered", "uncentered", "centered", "row_centered"):
            assert abs(res["direct"][v][g_]["value"]) < 0.15
        nl = res["direct"]["position_centered"][g_]["image_null"]
        assert nl["n_perm"] == 60 and abs(nl["z"]) < 4.0
    assert abs(res["m1"]["all"]["value"]) < 0.5
    assert res["m1"]["all"]["value"] < res["m1"]["all"]["null"]["q95"] + 0.15
    assert "anchor_null" in res["direct"]["uncentered"]["all"]


def test_measure1_invariant_to_tokens_per_image_and_image_order():
    hm, gk, p_m, p_k = _related_models(seed=3)
    gk = gk + 0.5 * _tokens(N_IMG, N_PATCH, gk.shape[2], 99)  # not exactly related: value below 1
    base = _fin(hm, gk, p_m, p_k, _opts())["m1_variants"]["uncentered"]
    base_c = _fin(hm, gk, p_m, p_k, _opts())["m1"]
    dup = lambda x: torch.cat([x, x[:, 1:]], dim=1)  # noqa: E731
    twice = _fin(dup(hm), dup(gk), p_m, p_k, _opts())["m1_variants"]["uncentered"]
    perm = torch.randperm(N_IMG, generator=torch.Generator().manual_seed(5))
    perm2 = torch.randperm(N_IMG, generator=torch.Generator().manual_seed(6))
    shuffled = _fin(hm[perm], gk[perm2], p_m, p_k, _opts())["m1_variants"]["uncentered"]
    imgs2 = _fin(hm.repeat(2, 1, 1), gk.repeat(2, 1, 1), p_m, p_k, _opts())["m1_variants"]["uncentered"]
    shuffled_c = _fin(hm[perm], gk[perm2], p_m, p_k, _opts())["m1"]
    for g in ("all", "cls", "patch"):
        assert 0.0 < base[g]["value"] < 0.999
        assert shuffled[g]["value"] == pytest.approx(base[g]["value"], rel=1e-9)
        assert shuffled_c[g]["value"] == pytest.approx(base_c[g]["value"], rel=1e-9)
        assert imgs2[g]["value"] == pytest.approx(base[g]["value"], rel=1e-9)
    for g in ("cls", "patch"):
        assert twice[g]["value"] == pytest.approx(base[g]["value"], rel=1e-9)


def test_anchor_column_permutation_destroys_m1_and_direct_not_m2():
    hm, gk, p_m, p_k = _related_models(seed=4, d=8, t=80)
    perm = torch.randperm(p_k.shape[0], generator=torch.Generator().manual_seed(7))
    ok = _fin(hm, gk, p_m, p_k, _opts())
    bad = _fin(hm, gk, p_m, p_k[perm], _opts())  # columns of D_k permuted
    assert ok["m1"]["all"]["value"] == pytest.approx(1.0, abs=1e-9)
    assert bad["m1"]["all"]["value"] < 0.5
    assert abs(bad["direct"]["uncentered"]["patch"]["value"]) < 0.3
    assert abs(bad["direct"]["mean_anchor_corr"]["patch"]["value"]) < 0.3
    # CKA over rows is invariant to the order of the feature columns (why the image null is used for m2)
    assert bad["m2"]["patch"]["value"] == pytest.approx(1.0, abs=1e-9)


def test_random_update_control_preserves_per_tensor_norm():
    g = torch.Generator().manual_seed(0)
    base = {"a": torch.randn(5, 7, generator=g), "b": torch.randn(11, generator=g), "c": torch.randn(3, 3, generator=g)}
    tuned = {"a": base["a"] + 0.3 * torch.randn(5, 7, generator=g), "b": base["b"] + 2.0 * torch.randn(11, generator=g),
             "c": base["c"].clone()}  # c unchanged: tau = 0 stays at base
    out, norms = uk.random_update_state(base, tuned, seed=3)
    assert set(out) == {"a", "b"}
    for k in ("a", "b"):
        tau = (tuned[k] - base[k]).double().norm()
        rnd = (out[k] - base[k]).double().norm()
        assert float(rnd) == pytest.approx(float(tau), rel=1e-5)
        assert norms[k]["random_norm"] == pytest.approx(norms[k]["tau_norm"], rel=1e-12)
        cos = ((out[k] - base[k]).flatten() @ (tuned[k] - base[k]).flatten()) / (rnd * tau)
        assert abs(float(cos)) < 0.9
    again, _ = uk.random_update_state(base, tuned, seed=3)
    other, _ = uk.random_update_state(base, tuned, seed=4)
    assert torch.equal(again["a"], out["a"]) and not torch.equal(other["a"], out["a"])


def test_random_update_norm_error_is_measured_after_the_cast():
    g = torch.Generator().manual_seed(1)
    base = {"w": torch.randn(64, 64, generator=g).to(torch.bfloat16)}
    tuned = {"w": (base["w"].float() + 0.01 * torch.randn(64, 64, generator=g)).to(torch.bfloat16)}
    out, norms = uk.random_update_state(base, tuned, seed=2)
    n = norms["w"]
    applied = float((out["w"].to(torch.float64) - base["w"].to(torch.float64)).norm())
    assert n["cast_norm"] == pytest.approx(applied, rel=1e-12)
    assert n["rel_norm_error"] == pytest.approx(abs(applied - n["tau_norm"]) / n["tau_norm"], rel=1e-9)
    # the float64 r has an exact norm; the update actually applied (after the bf16 cast) does not
    assert abs(n["random_norm"] - n["tau_norm"]) / n["tau_norm"] < 1e-12
    assert n["rel_norm_error"] > 1e-5


def _explicit(hm, gk, p_m, p_k):
    """In-memory D matrices per row group (rows image-major, CLS then patches)."""
    d_m, d_k = hm @ p_m.T, gk @ p_k.T
    cls = (d_m[:, 0], d_k[:, 0])
    patch = (d_m[:, 1:].reshape(-1, p_m.shape[0]), d_k[:, 1:].reshape(-1, p_k.shape[0]))
    return {"cls": cls, "patch": patch, "all": (torch.cat([cls[0], patch[0]]), torch.cat([cls[1], patch[1]]))}


def _cos(x, y):
    return float((x * y).sum() / (x.norm() * y.norm()))


@pytest.mark.parametrize("debiased", [True, False])
def test_streaming_accumulation_equals_in_memory(debiased):
    d_m, d_k, t = 7, 9, 50
    hm, gk = _tokens(N_IMG, N_PATCH, d_m, 20), _tokens(N_IMG, N_PATCH, d_k, 21)
    gk = gk + 0.7 * (hm @ torch.randn(d_m, d_k, generator=torch.Generator().manual_seed(22), dtype=DT))
    g = torch.Generator().manual_seed(23)
    p_m, p_k = torch.randn(t, d_m, generator=g, dtype=DT), torch.randn(t, d_k, generator=g, dtype=DT)
    st = _stream(hm, gk, p_m, p_k, batch=6, chunk_batches=3)
    res = uk.finalize_pair(
        st.chunks, _frame(p_m, p_k), _opts(debiased=debiased, center_rows_m1=False), native=st.native, direct=st.direct
    )
    ex = _explicit(hm, gk, p_m, p_k)
    dm3, dk3 = hm @ p_m.T, gk @ p_k.T  # [n, tok, T]
    pc_m, pc_k = dm3 - dm3.mean(0, keepdim=True), dk3 - dk3.mean(0, keepdim=True)  # position-centred
    sl = {"cls": slice(0, 1), "patch": slice(1, None), "all": slice(None)}
    for grp, (x, y) in ex.items():
        assert res["m2"][grp]["value"] == pytest.approx(linear_cka(x, y, debiased=debiased), rel=1e-8, abs=1e-10)
        d = res["direct"]
        assert d["uncentered"][grp]["value"] == pytest.approx(_cos(x, y), rel=1e-9)
        xc, yc = x - x.mean(0), y - y.mean(0)
        assert d["centered"][grp]["value"] == pytest.approx(_cos(xc, yc), rel=1e-9)
        corr = float(((xc * yc).sum(0) / (xc.norm(dim=0) * yc.norm(dim=0))).mean())
        assert d["mean_anchor_corr"][grp]["value"] == pytest.approx(corr, rel=1e-9)
        px, py = pc_m[:, sl[grp]].reshape(-1, t), pc_k[:, sl[grp]].reshape(-1, t)
        assert d["position_centered"][grp]["value"] == pytest.approx(_cos(px, py), rel=1e-9)
        corr_p = float(((px * py).sum(0) / (px.norm(dim=0) * py.norm(dim=0))).mean())
        assert d["mean_anchor_corr_position_centered"][grp]["value"] == pytest.approx(corr_p, rel=1e-9)
        rx, ry = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)  # each row centred over the anchors
        assert d["row_centered"][grp]["value"] == pytest.approx(_cos(rx, ry), rel=1e-9)
        # the d-space statistics (used for the anchor null) agree with the anchor-space ones
        sx, sy, cr = uk.combine_chunks(st.chunks, [1.0] * len(st.chunks)).bundle2(grp)
        c_, r_, cc_ = uk.agreement_from_stats(sx, sy, cr, p_m, p_k)
        assert c_ == pytest.approx(d["uncentered"][grp]["value"], rel=1e-9)
        assert cc_ == pytest.approx(d["centered"][grp]["value"], rel=1e-9)
        assert r_ == pytest.approx(d["mean_anchor_corr"][grp]["value"], rel=1e-9)
        # m1 (uncentred Gram): rows = anchors (features = task rows) with the existing estimator
        assert res["m1_variants"]["uncentered"][grp]["value"] == pytest.approx(
            linear_cka(x.T, y.T, debiased=debiased), rel=1e-8, abs=1e-10)
        # m1 on Grams of rows centred over the task rows
        want = linear_cka((x - x.mean(0)).T, (y - y.mean(0)).T, debiased=debiased)
        assert res["m1_variants"]["centered"][grp]["value"] == pytest.approx(want, rel=1e-7, abs=1e-10)
        want_p = linear_cka(px.T, py.T, debiased=debiased)  # mean over images removed per position
        assert res["m1_variants"]["position_centered"][grp]["value"] == pytest.approx(want_p, rel=1e-7, abs=1e-10)
    assert res["n_rows"]["all"] == N_IMG * (1 + N_PATCH)
    # centring is the default of the primary Gram variant
    assert uk.FinalizeOptions().center_rows_m1 is True and uk.FinalizeOptions().primary_variant == "centered"


def test_gram_cka_and_bootstrap_follow_the_existing_functions():
    g = torch.Generator().manual_seed(30)
    x, y = torch.randn(40, 6, generator=g, dtype=DT), torch.randn(40, 9, generator=g, dtype=DT)
    y[:, :3] += x[:, :3]
    for deb in (True, False):
        assert uk.gram_cka(x @ x.T, y @ y.T, debiased=deb) == pytest.approx(linear_cka(x, y, debiased=deb), rel=1e-10)
    ref = cka_bootstrap(x, y, n_boot=20, seed=5)
    got = uk.gram_cka_bootstrap(x @ x.T, y @ y.T, n_boot=20, seed=5)
    assert got["value"] == pytest.approx(ref["value"], rel=1e-10)
    assert got["lo"] == pytest.approx(ref["lo"], rel=1e-9) and got["hi"] == pytest.approx(ref["hi"], rel=1e-9)


def test_image_permutation_null_matches_explicit_permuted_rows():
    d_m, d_k, t = 6, 6, 40
    hm, gk = _tokens(24, N_PATCH, d_m, 40), _tokens(24, N_PATCH, d_k, 41)
    gk = gk + hm  # related, so the null differs visibly from the value
    g = torch.Generator().manual_seed(42)
    p_m, p_k = torch.randn(t, d_m, generator=g, dtype=DT), torch.randn(t, d_k, generator=g, dtype=DT)
    st = _stream(hm, gk, p_m, p_k, batch=24, n_perm=3, seed=1)
    gen = torch.Generator().manual_seed(1 * 1000 + 0)
    sigmas = [uk.image_derangement(24, gen) for _ in range(3)]
    q_m, q_k = p_m.T @ p_m, p_k.T @ p_k
    total = uk.combine_chunks(st.chunks, [1.0])
    sx_d = uk.SideStats.lincomb(st.direct.sides["m_t"])
    sy_d = uk.SideStats.lincomb(st.direct.sides["k_t"])
    dm3, dk3 = hm @ p_m.T, gk @ p_k.T
    for p, sigma in enumerate(sigmas):
        assert not bool((sigma == torch.arange(24)).any())  # derangement
        ex = _explicit(hm, gk[sigma], p_m, p_k)  # image i of m paired with image sigma(i) of k
        for grp in ("cls", "patch"):
            sx, sy, _ = total.bundle2(grp)
            got = uk.cka_rows_from_stats(sx, sy, st.perm_cross[grp][p], q_m, q_k)
            assert got == pytest.approx(linear_cka(*ex[grp]), rel=1e-8, abs=1e-10)
        # direct measure with mismatched images, same positions: uncentred and position-centred
        pc = uk.PairCross.__new__(uk.PairCross)
        pc.c, pc.rc = st.direct.perm.c[p], st.direct.perm.rc[p]
        vals = uk.direct_values(sx_d, sy_d, pc, "patch")
        x, y = ex["patch"]
        assert vals["uncentered"] == pytest.approx(_cos(x, y), rel=1e-9)
        px = (dm3 - dm3.mean(0, keepdim=True))[:, 1:].reshape(-1, t)
        # the marginals of k are those of the unpermuted images (the permutation stays inside the single batch)
        py = (dk3 - dk3.mean(0, keepdim=True))[:, 1:].reshape(-1, t)
        py_perm = (dk3[sigma] - dk3.mean(0, keepdim=True))[:, 1:].reshape(-1, t)
        want = float((px * py_perm).sum() / (px.norm() * py.norm()))
        assert vals["position_centered"] == pytest.approx(want, rel=1e-9)


def test_bilinear_grid_matching_is_identity_for_equal_grids():
    x = _tokens(3, 16, 5, 50)  # 4 x 4 grid
    assert torch.equal(uk.match_patch_grid(x, 17), x)
    y = uk.match_patch_grid(x, 10)  # 4 x 4 -> 3 x 3
    assert y.shape == (3, 10, 5)
    assert torch.equal(y[:, 0], x[:, 0])  # CLS untouched
    const = torch.ones(2, 17, 3, dtype=DT)
    assert torch.allclose(uk.match_patch_grid(const, 10), torch.ones(2, 10, 3, dtype=DT))


def test_measure1_uses_native_tokens_when_grids_differ():
    d_m, d_k, t = 6, 6, 40
    hm, gk = _tokens(20, 9, d_m, 60), _tokens(20, 16, d_k, 61)  # 3 x 3 vs 4 x 4 grids
    g = torch.Generator().manual_seed(62)
    p_m, p_k = torch.randn(t, d_m, generator=g, dtype=DT), torch.randn(t, d_k, generator=g, dtype=DT)
    n_ref = 10
    acc = uk.PairAcc(d_m, d_k, ("cls", "patch"))
    rows_m, rows_k = uk._rows(hm, n_ref, False), uk._rows(gk, n_ref, False)
    acc.update(rows_m, rows_k, p_m.T @ p_m, p_k.T @ p_k, patches_per_image=n_ref - 1)
    native = {"m_t": uk.NativeSide(hm.shape[1], d_m), "k_t": uk.NativeSide(gk.shape[1], d_k)}
    native["m_t"].update(hm)
    native["k_t"].update(gk)
    res = uk.finalize_pair([acc], _frame(p_m, p_k), _opts(center_rows_m1=False), native=native)
    x = (hm @ p_m.T).reshape(-1, t)  # all native tokens of m
    y = (gk @ p_k.T).reshape(-1, t)  # all native tokens of k (a different count of rows)
    assert x.shape[0] != y.shape[0]
    assert res["m1"]["all"]["value"] == pytest.approx(linear_cka(x.T, y.T), rel=1e-8)
    xs, ys = (rows_m["patch"] @ p_m.T), (rows_k["patch"] @ p_k.T)
    assert res["m2"]["patch"]["value"] == pytest.approx(linear_cka(xs, ys), rel=1e-8)


# ----------------------------------------------------------------------------------------------------------------
# interpretation toys: frame dominance and common mode
# ----------------------------------------------------------------------------------------------------------------

TOY_IMG = 200


def _toy_frames(d=8, t=100, seed=100, noise=0.01):
    g = torch.Generator().manual_seed(seed)
    p_m = torch.randn(t, d, generator=g, dtype=DT)
    q = _ortho(d, seed + 1)
    p_k = p_m @ q + noise * torch.randn(t, d, generator=g, dtype=DT)
    return p_m, p_k, q


def _toy_run(hm, gk, g_other, hm_other, p_m, p_k, images):
    extra = {"k_o": (g_other, "k"), "m_o": (hm_other, "m")}
    return _fin(
        hm, gk, p_m, p_k, _opts(ci=True), frame=_frame(p_m, p_k, images), batch=10, chunk_batches=2, extra=extra
    )


def test_frame_dominance_toy_gram_is_high_but_direct_is_zero_and_excess_vanishes():
    """P_k = P_m Q + noise with isotropic independent updates for the true and the other task: the Gram m1 is high
    (uncentred and centred) and so is the other-task m1; the excess is ~ 0; the primary direct measure is ~ 0."""
    p_m, p_k, _ = _toy_frames()
    d = p_m.shape[1]
    images = torch.arange(p_m.shape[0]) // 4
    hm, gk = _tokens(TOY_IMG, N_PATCH, d, 1), _tokens(TOY_IMG, N_PATCH, d, 2)
    g_other, hm_other = _tokens(TOY_IMG, N_PATCH, d, 3), _tokens(TOY_IMG, N_PATCH, d, 4)
    res = _toy_run(hm, gk, g_other, hm_other, p_m, p_k, images)
    for v in ("centered", "uncentered"):
        e = res["m1_variants"][v]["all"]
        assert e["value"] > 0.9 and e["other_task"]["value"] > 0.9
        ex = e["excess_other_task"]
        assert abs(ex["value"]) < 0.02
        assert ex["lo"] - 0.01 <= 0.0 <= ex["hi"] + 0.01  # includes 0 (up to a 1e-2 tolerance on a ~1e-3 quantity)
    prim = res["direct"][uk.DIRECT_PRIMARY]["all"]
    assert abs(prim["value"]) < 0.1
    assert prim["lo"] <= prim["value"] <= prim["hi"]
    ex = prim["excess_other_task"]
    assert ex["lo"] <= 0.0 <= ex["hi"]  # excess of the primary over the other task: includes 0


def test_shared_low_rank_update_gives_positive_primary_and_positive_excess():
    p_m, p_k, q = _toy_frames(seed=200)
    d = p_m.shape[1]
    images = torch.arange(p_m.shape[0]) // 4
    r = 2
    g = torch.Generator().manual_seed(7)
    w_m = torch.randn(r, d, generator=g, dtype=DT)
    z_true = _tokens(TOY_IMG, N_PATCH, r, 11)
    hm = z_true @ w_m + 0.3 * _tokens(TOY_IMG, N_PATCH, d, 12)
    gk = z_true @ (w_m @ q) + 0.3 * _tokens(TOY_IMG, N_PATCH, d, 13)  # the SAME latent: shared update
    # another task: another latent and another subspace (in each model)
    g_other = _tokens(TOY_IMG, N_PATCH, r, 21) @ (torch.randn(r, d, generator=g, dtype=DT) @ q) + 0.3 * _tokens(TOY_IMG, N_PATCH, d, 22)
    hm_other = _tokens(TOY_IMG, N_PATCH, r, 23) @ torch.randn(r, d, generator=g, dtype=DT) + 0.3 * _tokens(TOY_IMG, N_PATCH, d, 24)
    res = _toy_run(hm, gk, g_other, hm_other, p_m, p_k, images)
    prim = res["direct"][uk.DIRECT_PRIMARY]["all"]
    assert prim["value"] > 0.5
    assert prim["other_task"]["value"] < 0.15 and prim["excess_other_task"]["lo"] > 0.3
    assert prim["excess_within_model_cross_task"]["lo"] > 0.3
    assert res["m1_variants"]["centered"]["all"]["excess_other_task"]["lo"] > 0.1
    assert prim["excess_other_task"]["value"] == pytest.approx(prim["value"] - prim["other_task"]["value"], rel=1e-9)
    # control values of the direct measure are exported per control
    assert res["direct_controls"]["other_task"][uk.DIRECT_PRIMARY]["all"] == pytest.approx(prim["other_task"]["value"])


def test_common_mode_shift_dominates_uncentred_but_not_centred():
    p_m, p_k, q = _toy_frames(seed=300)
    d = p_m.shape[1]
    images = torch.arange(p_m.shape[0]) // 4
    r = 2
    g = torch.Generator().manual_seed(9)
    delta = 6.0 * torch.randn(d, generator=g, dtype=DT)  # a task-independent mean shift of the residual stream
    w_m = torch.randn(r, d, generator=g, dtype=DT)
    z = _tokens(TOY_IMG, N_PATCH, r, 31)
    hm = z @ w_m + delta + 0.2 * _tokens(TOY_IMG, N_PATCH, d, 32)
    gk = z @ (w_m @ q) + delta @ q + 0.2 * _tokens(TOY_IMG, N_PATCH, d, 33)
    z_o = _tokens(TOY_IMG, N_PATCH, r, 34)
    g_other = z_o @ (torch.randn(r, d, generator=g, dtype=DT) @ q) + delta @ q + 0.2 * _tokens(TOY_IMG, N_PATCH, d, 35)
    hm_other = _tokens(TOY_IMG, N_PATCH, r, 36) @ torch.randn(r, d, generator=g, dtype=DT) + delta + 0.2 * _tokens(TOY_IMG, N_PATCH, d, 37)
    res = _toy_run(hm, gk, g_other, hm_other, p_m, p_k, images)
    unc = res["m1_variants"]["uncentered"]["all"]
    cen = res["m1_variants"]["centered"]["all"]
    assert unc["value"] > 0.9 and unc["other_task"]["value"] > 0.9  # true ~ other when the mean shift dominates
    assert abs(unc["value"] - unc["other_task"]["value"]) < 0.05
    assert cen["value"] - cen["other_task"]["value"] > 0.2  # the centred one keeps them apart
    cm = res["common_mode"]
    assert cm["rho_m"] > 0.9 and cm["rho_k"] > 0.9
    assert cm["cos_mean_true_task"] > 0.95 and cm["cos_mean_other_task"] > 0.95  # the mean shift is task independent
    du = res["direct"]["uncentered"]["all"]
    assert du["value"] > 0.9 and du["other_task"]["value"] > 0.9  # direct uncentred cosine also contains the shift
    dp = res["direct"]["position_centered"]["all"]  # secondary variant: removes the shift
    assert dp["value"] > 0.4 and dp["other_task"]["value"] < 0.15  # the position-centred one removes it


# ----------------------------------------------------------------------------------------------------------------
# jackknife, clusters, chunks
# ----------------------------------------------------------------------------------------------------------------


def test_jackknife_interval_contains_the_point_estimate_on_unrelated_updates():
    d, t = 8, 60
    hm, gk = _tokens(100, N_PATCH, d, 70), _tokens(100, N_PATCH, d, 71)
    g = torch.Generator().manual_seed(72)
    p_m, p_k = torch.randn(t, d, generator=g, dtype=DT), torch.randn(t, d, generator=g, dtype=DT)
    res = _fin(hm, gk, p_m, p_k, _opts(ci=True), batch=10, chunk_batches=1)  # 10 chunks
    for grp in ("cls", "patch", "all"):
        for e in (res["m2"][grp], *(res["direct"][v][grp] for v in uk.DIRECT_VARIANTS)):
            assert e["n_groups"] == 10
            assert e["lo"] <= e["value"] <= e["hi"]
        assert res["m1"][grp]["lo"] <= res["m1"][grp]["value"] <= res["m1"][grp]["hi"]


def test_jackknife_ci_formula():
    loo = [0.1, 0.2, 0.3, 0.4]
    out = uk.jackknife_ci(0.25, loo)
    mean = sum(loo) / 4
    se = math.sqrt(3 / 4 * sum((x - mean) ** 2 for x in loo))
    assert out["se"] == pytest.approx(se)
    from scipy.stats import t as student_t

    assert out["hi"] - 0.25 == pytest.approx(student_t.ppf(0.975, 3) * se)
    assert out["n_groups"] == 4 and math.isnan(uk.jackknife_ci(0.25, [0.1])["lo"])


@pytest.mark.parametrize("n_images,batch,g", [(1000, 32, 16), (1000, 32, 10), (1000, 32, 32), (100, 10, 7)])
def test_number_of_chunks_equals_the_requested_one(n_images, batch, g):
    n_b = math.ceil(n_images / batch)
    ids = [uk.chunk_of_batch(b, n_b, g) for b in range(n_b)]
    assert len(set(ids)) == g and ids == sorted(ids) and ids[0] == 0 and ids[-1] == g - 1
    assert len({uk.chunk_of_batch(b, 3, 16) for b in range(3)}) == 3  # capped by the number of batches
    assert uk.UpdateKernelCkaConfig.__dataclass_fields__["jackknife_chunks"].default == 16


def test_cluster_jackknife_groups_anchors_by_image():
    images = torch.tensor([5, 5, 5, 9, 9, 2, 2, 2, 2, 7])  # 4 images; the anchors of an image stay together
    w = uk.cluster_weights(images, 10, "cpu")
    assert w.shape == (1 + 4, 10) and bool((w[0] == 1).all())
    deleted = {tuple(torch.nonzero(row == 0).flatten().tolist()) for row in w[1:]}
    assert deleted == {(5, 6, 7, 8), (0, 1, 2), (3, 4), (9,)}
    assert uk.cluster_weights(None, 10, "cpu").shape == (11, 10)
    # the masked HSIC equals the explicit estimator on the remaining anchors
    g = torch.Generator().manual_seed(80)
    a, b = torch.randn(10, 3, generator=g, dtype=DT), torch.randn(10, 3, generator=g, dtype=DT)
    ka, kb = uk._zero_diag(a @ a.T), uk._zero_diag(b @ b.T)
    got = uk._masked_hsic(ka, kb, w)
    for row, val in zip(w, got.tolist(), strict=True):
        idx = row.nonzero().flatten()
        want = float(uk._hsic_unbiased_zd(ka[idx][:, idx], kb[idx][:, idx]))
        assert val == pytest.approx(want, rel=1e-9)
    # m1: the number of clusters in the block result is the number of distinct anchor images
    d, t = 6, 40
    hm, gk = _tokens(30, N_PATCH, d, 81), _tokens(30, N_PATCH, d, 82)
    gg = torch.Generator().manual_seed(83)
    p_m, p_k = torch.randn(t, d, generator=gg, dtype=DT), torch.randn(t, d, generator=gg, dtype=DT)
    res = _fin(hm, gk, p_m, p_k, _opts(ci=True), frame=_frame(p_m, p_k, torch.arange(t) // 5))
    assert res["m1_n_clusters"] == 8 and res["m1"]["all"]["n_groups"] == 8


# ----------------------------------------------------------------------------------------------------------------
# expert-only control: the SAME rows, only the expert differs (fake models)
# ----------------------------------------------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, w):
        self.w = w

    def load_state_dict(self, sd, strict=True):
        self.w = sd["w"]


def _fake_capture(model, batches, requests, device, *, family_adapter=None, store_device="cpu"):
    for x, _ in batches:
        yield {key: x @ model.w for key in requests}


def test_expert_only_control_uses_the_same_rows_and_only_the_expert_differs(monkeypatch):
    from merge_and_rebase.rebase.methods._ariadne import capture

    monkeypatch.setattr(capture, "iter_capture_tokens", _fake_capture)
    d, t, n_tok = 6, 20, 1 + N_PATCH
    g = torch.Generator().manual_seed(90)
    batches = [(torch.randn(5, n_tok, d, generator=g, dtype=DT), torch.zeros(5)) for _ in range(4)]
    eye = torch.eye(d, dtype=DT)
    w = {tag: eye + 0.5 * torch.randn(d, d, generator=g, dtype=DT) for tag in ("m_t", "m_o", "k_t", "k_o")}
    p_m, p_k = torch.randn(t, d, generator=g, dtype=DT), torch.randn(t, d, generator=g, dtype=DT)
    frames = {0: uk.PairFrame(0, 0, p_m, p_k, 1.0, 1.0, torch.arange(t) // 2)}
    ctx = {
        "device": "cpu", "dtype": DT, "m_batches": batches, "k_batches": batches, "m_base": _FakeModel(eye),
        "k_base": _FakeModel(eye), "m_work": _FakeModel(eye), "m_work2": _FakeModel(eye), "k_work": _FakeModel(eye),
        "k_work2": _FakeModel(eye),
    }
    col = uk.Collector.empty([0])
    uk._stream_pass(
        name="t", ctx=ctx, col=col, states={tag: {"w": wt} for tag, wt in w.items()}, frames=frames, n_ref=n_tok,
        row_types=("cls", "patch"), native_tags=list(w), side_tags=list(w),
        pairs=[("true", "m_t", "k_t"), ("other_task", "m_t", "k_o"), ("within_m", "m_t", "m_o"), ("within_k", "k_t", "k_o")],
        row_pairs=[("true", "m_t", "k_t", True), ("other_task", "m_t", "k_o", False)], collect_base=True,
        chunk_fn=lambda b: b // 2, seed=0,
    )
    x = torch.cat([b[0] for b in batches])  # the same 20 images for every side
    enc = {tag: (x @ (wt - eye)) @ (p_m if tag[0] == "m" else p_k).T for tag, wt in w.items()}
    assert len(col.pairs[0]["true"]) == 2  # 4 batches in chunks of 2
    for name, (tx, ty) in col.pair_tags.items():
        got = uk.PairCross.lincomb(col.pairs[0][name]).c
        assert torch.allclose(got, (enc[tx] * enc[ty]).sum(0), atol=1e-9), name
    # the m side of the control is the main m side (same images, same expert); only the k expert changed
    main_rows = uk.combine_chunks([c[0] for c in col.rows["true"]], [1.0, 1.0]).bundle2("all")
    ctl_rows = col.rows["other_task"][0][0].bundle2("all")
    assert torch.allclose(main_rows[0].S, ctl_rows[0].S) and torch.allclose(main_rows[0].s, ctl_rows[0].s)
    assert not torch.allclose(main_rows[1].S, ctl_rows[1].S)
    dm_t, dk_o = x @ (w["m_t"] - eye), x @ (w["k_o"] - eye)
    want = dm_t.reshape(-1, d).T @ dk_o.reshape(-1, d)
    assert torch.allclose(ctl_rows[2].M, want, atol=1e-9)  # row i of m(t) with row i of k(t') of the same image
    nat = col.native[0]
    assert torch.allclose(nat["k_o"].S_cls, dk_o[:, 0].T @ dk_o[:, 0])
    assert not torch.allclose(nat["k_o"].S_cls, nat["k_t"].S_cls)
    assert torch.allclose(nat["base_m"].s_pos, x.sum(0)) and torch.allclose(nat["base_k"].s_pos, x.sum(0))


def _fake_ctx(batches_m, batches_k, d, eye):
    return {
        "device": "cpu", "dtype": DT, "m_batches": batches_m, "k_batches": batches_k, "m_base": _FakeModel(eye),
        "k_base": _FakeModel(eye), "m_work": _FakeModel(eye), "m_work2": _FakeModel(eye), "k_work": _FakeModel(eye),
        "k_work2": _FakeModel(eye),
    }


def test_stream_pass_unequal_grids_pairs_rows_after_resampling_and_exact_map_gives_one(monkeypatch):
    """m with 197 tokens (14 x 14), k with 257 (16 x 16), reference grid = k: m is resampled up to 257. With
    x_k = resample(x_m) Q, A_k = Q^T A Q and P_k = P_m Q the two encodings coincide row by row (image, position)."""
    from merge_and_rebase.rebase.methods._ariadne import capture

    monkeypatch.setattr(capture, "iter_capture_tokens", _fake_capture)
    d, t, n_img, bs = 4, 16, 12, 6
    g = torch.Generator().manual_seed(7)
    q = _ortho(d, 8)
    a = torch.randn(d, d, generator=g, dtype=DT)
    eye = torch.eye(d, dtype=DT)
    x_m = torch.randn(n_img, 197, d, generator=g, dtype=DT)
    x_k = uk.match_patch_grid(x_m, 257) @ q
    assert x_k.shape[1] == 257 and not torch.equal(x_m[:, :1], x_k[:, :1])
    bm = [(x_m[i:i + bs], torch.zeros(bs)) for i in range(0, n_img, bs)]
    bk = [(x_k[i:i + bs], torch.zeros(bs)) for i in range(0, n_img, bs)]
    p_m = torch.randn(t, d, generator=g, dtype=DT)
    frames = {0: uk.PairFrame(0, 0, p_m, p_m @ q, 1.0, 1.0, torch.arange(t) // 2)}
    col = uk.Collector.empty([0])
    uk._stream_pass(
        name="t", ctx=_fake_ctx(bm, bk, d, eye), col=col,
        states={"m_t": {"w": eye + a}, "k_t": {"w": eye + q.T @ a @ q}}, frames=frames, n_ref=257, row_types=("cls", "patch"),
        side_tags=["m_t", "k_t"], pairs=[("true", "m_t", "k_t")], row_pairs=[("true", "m_t", "k_t", True)],
        chunk_fn=lambda b: b, seed=0,
    )
    enc = uk.match_patch_grid(x_m @ a, 257) @ p_m.T  # [n_img, 257, T]
    c = uk.PairCross.lincomb(col.pairs[0]["true"]).c
    assert torch.allclose(c, (enc * enc).sum(0), atol=1e-9)
    res = uk.finalize_pair(
        [ch[0] for ch in col.rows["true"]], frames[0], _opts(), direct=col.direct_block(0)
    )
    for grp in ("cls", "patch", "all"):
        for v in ("uncentered", "position_centered"):
            assert res["direct"][v][grp]["value"] == pytest.approx(1.0, abs=1e-9), (v, grp)
        assert res["m2"][grp]["value"] == pytest.approx(1.0, abs=1e-8)


def _two_block_pass(monkeypatch, blocks, m2):
    from merge_and_rebase.rebase.methods._ariadne import capture

    monkeypatch.setattr(capture, "iter_capture_tokens", _fake_capture)
    d, t, n_tok = 5, 24, 1 + N_PATCH
    g = torch.Generator().manual_seed(11)
    eye = torch.eye(d, dtype=DT)
    batches = [(torch.randn(6, n_tok, d, generator=g, dtype=DT), torch.zeros(6)) for _ in range(4)]
    w = {tag: eye + 0.4 * torch.randn(d, d, generator=g, dtype=DT) for tag in ("m_t", "k_t", "m_o", "k_o")}
    all_frames = {
        j: uk.PairFrame(j, j, torch.randn(t, d, generator=g, dtype=DT), torch.randn(t, d, generator=g, dtype=DT), 1.0, 1.0,
                        torch.arange(t) // 3)
        for j in (0, 1)
    }
    frames = {j: all_frames[j] for j in blocks}
    col = uk.Collector.empty(list(blocks))
    row_pairs = [("true", "m_t", "k_t", True)] if m2 else []
    uk._stream_pass(
        name="t", ctx=_fake_ctx(batches, batches, d, eye), col=col, states={tag: {"w": wt} for tag, wt in w.items()},
        frames=frames, n_ref=n_tok, row_types=("cls", "patch"), native_tags=list(w), side_tags=list(w), collect_base=True,
        pairs=[("true", "m_t", "k_t"), ("other_task", "m_t", "k_o"), ("other_task_rev", "m_o", "k_t"),
               ("within_m", "m_t", "m_o"), ("within_k", "k_t", "k_o")],
        row_pairs=row_pairs, chunk_fn=lambda b: b // 2, n_perm_images=3, seed=0,
    )
    blocks_out = []
    for j in blocks:
        res = uk.finalize_pair(
            [c[j] for c in col.rows.get("true", [])], frames[j], _opts(ci=True, null=True, n_perm=4),
            perm_cross=None if col.perm_cross is None else col.perm_cross[j], native=col.native[j], direct=col.direct_block(j),
        )
        blocks_out.append({"target_block": j, "source_block": j, **res})
    return blocks_out


def _summary(blocks, run_blocks, cfg_tag):
    return {"setting": {"task": "T", "images": {"sha": "x"}, "blocks_run": run_blocks, "config": {"tag": cfg_tag},
                        "seconds": 1.0}, "blocks": blocks}


def test_split_blocks_equal_a_full_run_and_summaries_merge(monkeypatch):
    full = _two_block_pass(monkeypatch, (0, 1), m2=False)
    part1, part2 = _two_block_pass(monkeypatch, (0,), m2=False), _two_block_pass(monkeypatch, (1,), m2=False)
    merged = uk.merge_summaries([_summary(part1, [0], "p1"), _summary(part2, [1], "p2")])
    assert json.dumps(merged["blocks"], sort_keys=True, default=str) == json.dumps(full, sort_keys=True, default=str)
    assert merged["setting"]["blocks_run"] == [0, 1] and len(merged["setting"]["merged_from"]) == 2
    assert all(b["m2"] == {} for b in full) and "other_task_mean" in full[0]["direct"]["uncentered"]["all"]
    with pytest.raises(ValueError, match="different setting"):
        bad = _summary(part2, [1], "p2")
        bad["setting"]["task"] = "U"
        uk.merge_summaries([_summary(part1, [0], "p1"), bad])
    with pytest.raises(ValueError, match="overlapping"):
        uk.merge_summaries([_summary(part1, [0], "p1"), _summary(part1, [0], "p2")])


def test_m2_switch_drops_row_statistics_but_keeps_the_direct_measure(monkeypatch):
    on = _two_block_pass(monkeypatch, (0,), m2=True)[0]
    off = _two_block_pass(monkeypatch, (0,), m2=False)[0]
    assert on["m2"] and not off["m2"]
    for v in ("uncentered", "position_centered"):
        for grp in ("cls", "patch", "all"):
            assert off["direct"][v][grp]["value"] == on["direct"][v][grp]["value"]
            assert off["direct"][v][grp]["lo"] == on["direct"][v][grp]["lo"]
    assert "anchor_null" in on["direct"]["uncentered"]["all"] and "anchor_null" not in off["direct"]["uncentered"]["all"]
    assert "image_null" in off["direct"]["uncentered"]["all"]
    assert uk.parse_config({"source": {"model": "m", "pretrained": "p", "experts": {"T": "x"}},
                            "target": {"model": "m", "pretrained": "p", "experts": {"T": "x"}}, "task": "T",
                            "state_dir": "s", "out_dir": "o", "controls": ["null"]}).m2 is False


def test_mirror_other_task_comparator_and_mean(monkeypatch):
    res = _two_block_pass(monkeypatch, (0,), m2=False)[0]
    e = res["direct"]["uncentered"]["all"]
    assert e["other_task_mean"]["value"] == pytest.approx((e["other_task"]["value"] + e["other_task_rev"]["value"]) / 2)
    assert e["excess_other_task_mean"]["value"] == pytest.approx(e["value"] - e["other_task_mean"]["value"])
    assert e["excess_other_task_rev"]["value"] == pytest.approx(e["value"] - e["other_task_rev"]["value"])
    assert e["other_task"]["value"] != e["other_task_rev"]["value"]
    assert res["direct_controls"]["other_task_rev"]["uncentered"]["all"] == pytest.approx(e["other_task_rev"]["value"])
    # the mirror pair is (m_o, k_t): explicit check from the fake-model encodings
    assert e["other_task_mean"]["n_groups"] == 2


def test_direct_sums_are_float64_even_for_float32_encodings():
    g = torch.Generator().manual_seed(3)
    x = (3000.0 + 0.01 * torch.randn(8, 5, 64, generator=g)).to(torch.float32)  # large offset: float32 sums cancel
    y = (3000.0 + 0.01 * torch.randn(8, 5, 64, generator=g)).to(torch.float32)
    st, pc = uk.SideStats(5, 64), uk.PairCross(5, 64)
    st.update(x)
    pc.update(x, y)
    x64, y64 = x.double(), y.double()
    assert torch.equal(st.q, (x64 * x64).sum(0)) and torch.equal(st.s, x64.sum(0))
    assert torch.equal(pc.c, (x64 * y64).sum(0))
    assert uk.ACC_DTYPE == torch.float64


def test_chunk_check_ram_estimate_and_image_identity():
    chk = uk.check_equal_chunks([32] * 32, 16)
    assert chk["equal"] and chk["images_per_chunk"] == [64] * 16
    with pytest.warns(UserWarning, match="unequal jackknife chunks"):
        bad = uk.check_equal_chunks([32] * 31 + [8], 16)
    assert not bad["equal"]
    fr = {0: uk.PairFrame(0, 0, torch.zeros(10, 3), torch.zeros(10, 4), 1.0, 1.0, None)}
    kw = dict(n_ref=5, n_native_m=5, n_native_k=7, n_chunks=2, n_perm_images=0)
    off = uk.estimate_accumulator_gb(fr, controls=("other_task",), m2=False, **kw)
    on = uk.estimate_accumulator_gb(fr, controls=("other_task",), m2=True, **kw)
    side, pair = 2 * 5 * 10 + 5 + 1, 5 * 10 + 5
    assert off["components_gb"]["m2"] == 0.0 and on["total_gb"] > off["total_gb"]
    assert off["components_gb"]["direct"] == pytest.approx(8.0 * (4 * side + 5 * pair) * 2 / 1e9, abs=1e-9)
    assert uk.estimate_accumulator_gb(fr, controls=("null",), m2=False, **{**kw, "n_perm_images": 3})["components_gb"]["direct"] > 0
    from PIL import Image

    imgs = [Image.new("RGB", (4, 4), (i, i, i)) for i in range(4)]
    ds = SimpleNamespace(split=[{"image": im, "id": f"r{i}"} for i, im in enumerate(imgs)], image_key="image")
    a = uk.image_identity(ds, [0, 2])
    assert a == uk.image_identity(ds, [0, 2]) and a["raw_image_bytes_sha256"] != uk.image_identity(ds, [0, 3])["raw_image_bytes_sha256"]
    assert a["row_ids_sha256"] is None  # a plain list has no column_names: no id column


def test_images_record_split_indices():
    ds = SimpleNamespace(split=SimpleNamespace(_indices=None))
    assert uk._absolute_indices(ds, [3, 1])[0] == [3, 1]
    import pyarrow as pa

    ds2 = SimpleNamespace(split=SimpleNamespace(_indices=pa.table({"indices": [10, 20, 30, 40]})))
    got, source = uk._absolute_indices(ds2, [3, 0])
    assert got == [40, 10] and "_indices" in source


# ----------------------------------------------------------------------------------------------------------------
# outputs and configuration
# ----------------------------------------------------------------------------------------------------------------


def test_bootstrap_and_outputs_smoke(tmp_path):
    hm, gk, p_m, p_k = _related_models(seed=8)
    frame = _frame(p_m, p_k, torch.arange(p_m.shape[0]) // 3)
    res = _fin(hm, gk, p_m, p_k, _opts(ci=True, null=True, n_perm=5), frame=frame, batch=5, chunk_batches=2, n_perm=4)
    e = res["m2"]["patch"]
    assert e["lo"] <= e["value"] + 1e-9 and e["hi"] >= e["value"] - 1e-9
    assert math.isfinite(res["m1"]["all"]["lo"])
    assert res["m1"]["all"]["null"]["n_perm"] == 5 and res["m2"]["patch"]["null"]["n_perm"] == 4
    assert res["direct"]["position_centered"]["patch"]["image_null"]["n_perm"] == 4
    res.update({"target_block": 0, "source_block": 0, "n_anchors": 60, "c_source": 1.0, "c_target": 1.0})
    res["base_reference"] = uk.base_reference(frame, debiased=True, ci=True, device="cpu")
    setting = {
        "source": {"model": "A", "pretrained": "x", "n_tokens": 5, "expert": {"path": "a"}},
        "target": {"model": "B", "pretrained": "y", "n_tokens": 5, "expert": {"path": "b"}},
        "task": "T", "state_dir": "s", "T_stored": 60, "anchor_filter": "norm", "anchor_filter_k": 3.0,
        "pairing": [0], "split": "test", "images": {"task": {"n_images": N_IMG, "image_indices_sha256": "ab" * 32}},
        "val_size": 1000, "split_seed": 89, "batch_size": 32, "grid_mode": "direct", "debiased": True,
        "compute_dtype": "float64", "controls": ["null"], "permutations": 5, "image_permutations": 4,
        "jackknife_chunks_actual": 4, "jackknife_chunks_requested": 4, "control_seed": 0,
        "tf32": {"matmul": False, "cudnn": False}, "m1_primary_variant": "centered",
        "code": {"git_commit": "x", "git_dirty": False},
    }
    uk.write_outputs({"setting": setting, "blocks": [res]}, tmp_path)
    md = (tmp_path / "summary.md").read_text()
    for section in ("## Domanda", "## Procedura", "## Risultato", "## Come si legge", "## Cautele", "## File"):
        assert section in md
    assert "NON il frame gpa/consensus" in md and "TF32 matmul=False" in md
    assert "update casuale: media ± sd (1 estrazioni)" in md
    assert md.index("PRIMARIA: coseno grezzo") < md.index("SECONDARIA m1")  # the primary first
    rows = (tmp_path / "blocks.csv").read_text().splitlines()
    assert rows[0].startswith("target_block,source_block,measure") and rows[1].split(",")[2] == "direct_uncentered"
    assert (tmp_path / "blocks.csv").read_text().count("\n") > 5
    block0 = json.loads((tmp_path / "summary.json").read_text())["blocks"][0]
    assert next(iter(block0)) == "direct"


def test_config_parsing_rejects_unknown_keys_and_missing_experts():
    spec = {"model": "ViT-B-16", "pretrained": "p", "experts": {"EuroSAT": "x.pt"}}
    ok = {"source": spec, "target": spec, "task": "EuroSAT", "state_dir": "s", "out_dir": "o", "controls": ["null"]}
    cfg = uk.parse_config(ok)
    assert cfg.num_images == 1000 and cfg.val_size == 1000 and cfg.seed == 89 and cfg.permutations == 200
    assert cfg.split == "test" and cfg.grid_reference == "canonical" and cfg.m1_center_rows is True
    assert cfg.jackknife_chunks == 16
    with pytest.raises(ValueError, match="unknown config keys"):
        uk.parse_config({**ok, "bogus": 1})
    with pytest.raises(ValueError, match="split"):
        uk.parse_config({**ok, "split": "train"})
    with pytest.raises(ValueError, match="experts"):
        uk.parse_config({**ok, "task": "GTSRB"})
    with pytest.raises(ValueError, match="other_task"):
        uk.parse_config({**ok, "controls": ["other_task"]})


def test_example_configs_parse_with_canonical_grid():
    from pathlib import Path

    paths = sorted((Path(__file__).resolve().parents[1] / "configs/examples/update_kernel_cka").glob("*.json"))
    assert paths
    for p in paths:
        cfg = uk.parse_config(json.loads(p.read_text()))
        assert cfg.grid_reference == "canonical" and cfg.split == "test"


def test_example_configs_equal_chunks_parts_and_smoke():
    from pathlib import Path

    d = Path(__file__).resolve().parents[1] / "configs/examples/update_kernel_cka"
    cfgs = {p.stem: uk.parse_config(json.loads(p.read_text())) for p in sorted(d.glob("*.json"))}
    assert not [n for n in cfgs if n.startswith("A_B") and "part" not in n and "smoke" not in n]
    for name, c in cfgs.items():
        if "smoke" in name:
            continue
        assert c.num_images == 1024 and c.num_images % (c.batch_size * c.jackknife_chunks) == 0 and c.m2 is False, name
    for task in ("dtd", "eurosat", "gtsrb"):
        p1, p2 = cfgs[f"A_B_{task}_part1"], cfgs[f"A_B_{task}_part2"]
        assert p1.blocks == tuple(range(12)) and p2.blocks == tuple(range(12, 24)) and p1.out_dir != p2.out_dir
    s = cfgs["smoke_self_L14"]
    assert s.source == s.target and s.blocks == (0, 12, 23) and s.num_images == 64 and s.jackknife_chunks == 4
    assert set(s.controls) == {"null", "other_task", "random_update", "base_reference"}
    assert "other_task_other_images" in cfgs["smoke_A_B_dtd"].controls and cfgs["smoke_A_B_dtd"].blocks == (0, 11, 23)


# ----------------------------------------------------------------------------------------------------------------
# random_update with several (antithetic) draws
# ----------------------------------------------------------------------------------------------------------------


def _random_draw_block(monkeypatch, n_draws, antithetic, control_seed=5, d=5, t=24):
    """Fake linear models (dH = x (W - I), linear in the weight perturbation): the real k update tau_k and the random
    draws of ``random_draw_plan`` through ``random_update_state`` and ``_stream_pass``; returns the direct block."""
    from merge_and_rebase.rebase.methods._ariadne import capture

    monkeypatch.setattr(capture, "iter_capture_tokens", _fake_capture)
    n_tok = 1 + N_PATCH
    g = torch.Generator().manual_seed(31)
    eye = torch.eye(d, dtype=DT)
    batches = [(torch.randn(6, n_tok, d, generator=g, dtype=DT), torch.zeros(6)) for _ in range(4)]
    w_m, w_k = eye + 0.4 * torch.randn(d, d, generator=g, dtype=DT), eye + 0.4 * torch.randn(d, d, generator=g, dtype=DT)
    frames = {0: uk.PairFrame(0, 0, torch.randn(t, d, generator=g, dtype=DT), torch.randn(t, d, generator=g, dtype=DT), 1.0, 1.0,
                              torch.arange(t) // 3)}
    col = uk.Collector.empty([0])
    common = dict(ctx=_fake_ctx(batches, batches, d, eye), col=col, frames=frames, n_ref=n_tok, row_types=("cls", "patch"),
                  chunk_fn=lambda b: b // 2, seed=0)
    uk._stream_pass(name="main", states={"m_t": {"w": w_m}, "k_t": {"w": w_k}}, side_tags=["m_t", "k_t"],
                    pairs=[("true", "m_t", "k_t")], **common)
    plan = uk.random_draw_plan(control_seed, n_draws, antithetic)
    for dr in plan:
        name, tag = uk.random_draw_names(dr["draw"])
        rand, _ = uk.random_update_state({"w": eye}, {"w": w_k}, seed=dr["seed"], sign=dr["sign"])
        # as in run(): draw 0 chunked, draws >= 1 one chunk
        uk._stream_pass(name=name, states={"m_t": {"w": w_m}, tag: {"w": rand["w"]}}, side_tags=[tag],
                        pairs=[(name, "m_t", tag)], **{**common, "chunk_fn": common["chunk_fn"] if dr["draw"] == 0 else (lambda b: 0)})
    return col.direct_block(0), plan


def test_antithetic_draws_cancel_in_the_mean_on_a_linear_model(monkeypatch):
    blk, plan = _random_draw_block(monkeypatch, 4, True)
    assert [p["sign"] for p in plan] == [1.0, -1.0, 1.0, -1.0] and plan[0]["seed"] == plan[1]["seed"] != plan[2]["seed"]
    entries, controls = uk.direct_block(blk, _opts(ci=True, random_antithetic=True))
    for v in uk.DIRECT_VARIANTS:
        for g in uk.DIRECT_GROUPS:
            e = entries[v][g]
            vals = e["random_update_draws"]["values"]
            assert len(vals) == 4 and e["random_update_draws"]["n"] == 4
            # dH(-g) = -dH(g): the per-chunk cross numerators cancel; the norms are equal, so the cosines are opposite
            assert vals[0] == pytest.approx(-vals[1], abs=1e-12) and vals[2] == pytest.approx(-vals[3], abs=1e-12)
            assert e["random_update"]["value"] == vals[0] and abs(e["random_update_draws"]["mean"]) < 1e-12
            assert e["random_update_draws"]["min"] == min(vals) and e["random_update_draws"]["max"] == max(vals)
            assert e["true_minus_random_draw0"]["value"] == pytest.approx(e["value"] - vals[0], abs=1e-12)
            assert "excess_random_update" not in e and "true_minus_random_mean" not in e
            rd = e["random_update_draws"]
            assert rd["z_random"] is None and rd["p_random"] is None and rd["p_random_two_sided"] is None and rd["z_p_note"]
            assert rd["n_groups"] == 2 and rd["values"][0::2] == [vals[0], vals[2]]
            assert all(abs(x) < 1e-12 for x in rd["even"]) and rd["odd"] == pytest.approx(vals[0::2], abs=1e-12)
            assert e["random_update"]["lo"] <= e["random_update"]["hi"]
            assert controls["random_update"][v][g] == e["random_update"]["value"]
    assert "random_update_d1" not in controls and "random_update_d3" not in controls
    # the mean is over per-replicate cosines: the jackknife replicates of the comparator are the draw means too
    assert e["random_update_draws"]["null_sd"] > 0 and e["random_update_draws"]["sd_ddof"] == 1


def test_random_draw_seeds_are_deterministic_and_values_reproducible(monkeypatch):
    assert uk.random_draw_plan(7, 4, True) == uk.random_draw_plan(7, 4, True)
    s = [p["seed"] for p in uk.random_draw_plan(7, 4, True)]
    assert s == [7, 7, 7 + uk.RANDOM_SEED_STRIDE, 7 + uk.RANDOM_SEED_STRIDE]
    assert [p["seed"] for p in uk.random_draw_plan(7, 3, False)] == [7, 7 + uk.RANDOM_SEED_STRIDE, 7 + 2 * uk.RANDOM_SEED_STRIDE]
    assert uk.random_draw_plan(8, 4, True)[0]["seed"] != s[0]
    a, _ = _random_draw_block(monkeypatch, 4, True, control_seed=7)
    b, _ = _random_draw_block(monkeypatch, 4, True, control_seed=7)
    c, _ = _random_draw_block(monkeypatch, 4, True, control_seed=8)
    va, vb, vc = (uk.direct_block(x, _opts(ci=False))[0]["uncentered"]["all"]["random_update_draws"]["values"] for x in (a, b, c))
    assert va == vb and va != vc
    g = torch.Generator().manual_seed(0)
    base, tuned = {"w": torch.randn(6, 6, generator=g)}, {"w": torch.randn(6, 6, generator=g)}
    plus, n1 = uk.random_update_state(base, tuned, seed=3, sign=1.0)
    minus, n2 = uk.random_update_state(base, tuned, seed=3, sign=-1.0)
    assert torch.allclose(plus["w"] - base["w"], -(minus["w"] - base["w"]), atol=1e-6)
    assert n1["w"]["random_norm"] == pytest.approx(n1["w"]["tau_norm"], rel=1e-12) == pytest.approx(n2["w"]["random_norm"], rel=1e-12)


def test_odd_antithetic_draw_count_raises():
    with pytest.raises(ValueError, match="even"):
        uk.random_draw_plan(0, 3, True)
    with pytest.raises(ValueError, match="random_draws"):
        uk.random_draw_plan(0, 0, False)
    spec = {"model": "ViT-B-16", "pretrained": "p", "experts": {"T": "x.pt"}}
    ok = {"source": spec, "target": spec, "task": "T", "state_dir": "s", "out_dir": "o", "controls": ["null", "random_update"]}
    cfg = uk.parse_config(ok)
    assert cfg.random_draws == 19 and cfg.random_antithetic is False
    with pytest.raises(ValueError, match="even"):
        uk.parse_config({**ok, "random_draws": 3, "random_antithetic": True})
    assert uk.parse_config({**ok, "random_draws": 3}).random_draws == 3  # independent draws: any count
    assert uk.parse_config({**ok, "random_draws": 3, "random_antithetic": False}).random_draws == 3


def test_single_draw_without_antithetic_reproduces_the_old_single_draw(monkeypatch):
    plan = uk.random_draw_plan(5, 1, False)
    assert plan == [{"draw": 0, "seed": 5, "sign": 1.0}] and uk.random_draw_names(0) == ("random_update", "k_rand")
    g = torch.Generator().manual_seed(2)
    base, tuned = {"w": torch.randn(6, 6, generator=g)}, {"w": torch.randn(6, 6, generator=g)}
    old, _ = uk.random_update_state(base, tuned, seed=5)
    assert torch.equal(uk.random_update_state(base, tuned, seed=5, sign=1.0)[0]["w"], old["w"])
    blk, _ = _random_draw_block(monkeypatch, 1, False, control_seed=5)
    entries, controls = uk.direct_block(blk, _opts(ci=True))
    g_chunks = len(blk.pairs["true"])
    for v in uk.DIRECT_VARIANTS:
        for grp in uk.DIRECT_GROUPS:
            e = entries[v][grp]
            # the old computation: the pair's cosine on all chunks, and per deleted chunk
            full = uk.direct_values(uk._comb(blk.sides["m_t"], None, g_chunks), uk._comb(blk.sides["k_rand"], None, g_chunks),
                                    uk._comb(blk.pairs["random_update"], None, g_chunks), grp)[v]
            true = e["value"]
            assert e["random_update"]["value"] == full and e["true_minus_random_draw0"]["value"] == true - full
            assert controls["random_update"][v][grp] == full
            assert e["random_update_draws"]["values"] == [full] and e["random_update_draws"]["null_sd"] is None and e["random_update_draws"]["z_random"] is None
            loo = [uk.direct_values(uk._comb(blk.sides["m_t"], d, g_chunks), uk._comb(blk.sides["k_rand"], d, g_chunks),
                                    uk._comb(blk.pairs["random_update"], d, g_chunks), grp)[v] for d in range(g_chunks)]
            ref = uk.jackknife_ci(full, loo)
            assert e["random_update"]["lo"] == ref["lo"] and e["random_update"]["hi"] == ref["hi"]


def test_summary_md_and_csv_report_the_draws(monkeypatch, tmp_path):
    blk, _ = _random_draw_block(monkeypatch, 4, True)
    blk, _ = _random_draw_block(monkeypatch, 8, False)
    res = {"direct": uk.direct_block(blk, _opts(ci=True))[0], "target_block": 0, "source_block": 0}
    setting = {
        "source": {"model": "A", "pretrained": "x", "n_tokens": 5, "expert": {"path": "a"}},
        "target": {"model": "B", "pretrained": "y", "n_tokens": 5, "expert": {"path": "b"}},
        "task": "T", "state_dir": "s", "T_stored": 24, "anchor_filter": "norm", "anchor_filter_k": 3.0, "pairing": [0],
        "split": "test", "images": {"task": {"n_images": 24}}, "split_seed": 1, "batch_size": 6, "grid_mode": "direct",
        "debiased": True, "compute_dtype": "float64", "controls": ["random_update"], "permutations": 0, "image_permutations": 0,
        "control_seed": 5, "random_draws": 8, "random_antithetic": False, "random_seed_derivation": "q = i // 2",
        "code": {}, "tf32": {},
    }
    uk.write_outputs({"setting": setting, "blocks": [res]}, tmp_path)
    md = (tmp_path / "summary.md").read_text()
    assert "update casuale: media ± sd (8 estrazioni)" in md and "z casuale" in md and "p casuale (1 coda)" in md
    assert "| eccesso vs casuale |" not in md and "test di livello di caso" in md
    rd = res["direct"]["uncentered"]["all"]["random_update_draws"]
    assert f"{rd['z_random']:.3f}" in md and f"{rd['p_random']:.3f}" in md
    rows = (tmp_path / "blocks.csv").read_text().splitlines()
    head = rows[0].split(",")
    assert {"random_draws_used", "random_update_basis", "random_update_min", "random_update_max"} <= set(head)
    assert rows[1].split(",")[head.index("random_draws_used")] == "8"
    assert rows[1].split(",")[head.index("random_update_basis")].startswith("draw0")
    assert {"random_null_mean", "random_null_sd", "z_random", "p_random", "p_random_two_sided"} <= set(head)


def test_ram_estimate_grows_with_random_draws_and_summary_md_labels_them():
    fr = {0: uk.PairFrame(0, 0, torch.zeros(10, 3), torch.zeros(10, 4), 1.0, 1.0, None)}
    kw = dict(n_ref=5, n_native_m=5, n_native_k=7, n_chunks=2, n_perm_images=0, controls=("random_update",), m2=False)
    one, four = uk.estimate_accumulator_gb(fr, **kw), uk.estimate_accumulator_gb(fr, random_draws=4, **kw)
    side, pair = 2 * 5 * 10 + 5 + 1, 5 * 10 + 5
    assert one["components_gb"]["direct"] == pytest.approx(8.0 * (2 * side + 1 * pair + (side + pair)) * 2 / 1e9, abs=1e-12)
    assert four["components_gb"]["direct"] - one["components_gb"]["direct"] == pytest.approx(8.0 * 3 * (side + pair) * 1 / 1e9, abs=1e-12)
    assert four["components_gb"]["native"] == one["components_gb"]["native"]


def test_random_null_z_and_p_on_a_hand_case():
    r = uk._draw_summary([0.1, -0.2, 0.3, 0.0], 0.25)
    sd = (0.13 / 3) ** 0.5
    assert r["n_groups"] == 4 and r["null_mean"] == pytest.approx(0.05) and r["null_sd"] == pytest.approx(sd)
    assert r["null_max_abs"] == pytest.approx(0.3) and r["min"] == -0.2 and r["max"] == 0.3
    assert r["z_random"] == pytest.approx(0.2 / sd)
    assert r["p_random"] == pytest.approx(2 / 5)
    assert r["p_random_two_sided"] == pytest.approx(3 / 5)  # |c - 0.05| >= 0.2 for -0.2 and 0.3
    assert uk._draw_summary([0.1, -0.2, 0.3, 0.0], -0.4)["p_random"] == pytest.approx(1.0)
    assert uk._draw_summary([0.1, -0.2, 0.3, 0.0], -0.4)["p_random_two_sided"] == pytest.approx(1 / 5)
    # antithetic: groups are the +g draws, even / odd parts per pair
    a = uk._draw_summary([0.3, -0.1, -0.2, 0.2], 0.25, antithetic=True)
    assert a["n"] == 4 and a["n_groups"] == 2 and a["null_mean"] == pytest.approx(0.05)
    assert a["even"] == pytest.approx([0.1, 0.0]) and a["odd"] == pytest.approx([0.2, -0.2])
    assert a["p_random"] is None and a["z_random"] is None  # mirrored draws: no z / p
    one = uk._draw_summary([0.4], 0.5)
    assert one["null_sd"] is None and one["z_random"] is None and one["p_random"] == pytest.approx(1 / 2)


def test_independent_draws_have_a_null_spread_and_the_md_shows_z_and_p(monkeypatch, tmp_path):
    blk, _ = _random_draw_block(monkeypatch, 8, False)
    ent = uk.direct_block(blk, _opts(ci=True))[0]
    for v in uk.DIRECT_VARIANTS:
        for g in uk.DIRECT_GROUPS:
            e, rd = ent[v][g], ent[v][g]["random_update_draws"]
            assert rd["n"] == rd["n_groups"] == 8 and rd["null_sd"] > 0 and "even" not in rd
            assert rd["z_random"] == pytest.approx((e["value"] - rd["null_mean"]) / rd["null_sd"])
            assert e["random_update"]["value"] == rd["values"][0]  # random_update = draw 0 (the chunked draw)
            assert len(blk.pairs["random_update"]) == 2 and len(blk.pairs["random_update_d3"]) == 1  # draws >= 1: one chunk
            assert rd["p_random_two_sided"] >= 1 / 9 and rd["p_random"] >= 1 / 9


def test_two_sided_p_is_centred_on_the_null_and_nonfinite_gives_none():
    shifted = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.65, 0.63, 0.67]
    r = uk._draw_summary(shifted, -0.19)
    assert r["z_random"] < -20 and r["p_random"] == pytest.approx(1.0)
    assert r["p_random_two_sided"] == pytest.approx(1 / 10)  # no draw is as far from the null mean as the true value
    assert uk._draw_summary(shifted, 0.65)["p_random_two_sided"] == pytest.approx(1.0, abs=0.11)  # in the middle of the null
    far = uk._draw_summary(shifted, 0.69 + 0.5)
    assert far["p_random_two_sided"] == pytest.approx(1 / 10) and far["p_random"] == pytest.approx(1 / 10)
    for bad_true, vals in ((float("nan"), shifted), (0.1, shifted[:3] + [float("nan")] + shifted[4:])):
        r = uk._draw_summary(vals, bad_true)
        assert r["z_random"] is None and r["p_random"] is None and r["p_random_two_sided"] is None
    assert uk._draw_summary(shifted, float("inf"))["p_random"] is None
    a = uk._draw_summary([0.3, -0.3, 0.2, -0.2], 0.9, antithetic=True)
    assert a["z_random"] is None and a["p_random"] is None and a["p_random_two_sided"] is None and a["even"] == [0.0, 0.0]


def test_single_chunk_null_draws_are_accepted_and_memory_counts_them(monkeypatch):
    blk, _ = _random_draw_block(monkeypatch=monkeypatch, n_draws=5, antithetic=False)
    assert len(blk.pairs["true"]) == 2 and len(blk.pairs["random_update"]) == 2
    assert all(len(blk.pairs[f"random_update_d{i}"]) == 1 for i in (1, 2, 3, 4))
    ent, controls = uk.direct_block(blk, _opts(ci=True))
    e = ent["uncentered"]["all"]
    assert e["random_update_draws"]["n"] == 5 and "lo" in e["random_update"] and "lo" in e["true_minus_random_draw0"]
    assert set(controls) == {"random_update"} and controls["random_update"]["uncentered"]["all"] == e["random_update"]["value"]
    fr = {0: uk.PairFrame(0, 0, torch.zeros(10, 3), torch.zeros(10, 4), 1.0, 1.0, None)}
    kw = dict(n_ref=5, n_native_m=5, n_native_k=7, n_chunks=16, n_perm_images=0, controls=("random_update",), m2=False)
    side, pair = 2 * 5 * 10 + 5 + 1, 5 * 10 + 5
    got = uk.estimate_accumulator_gb(fr, random_draws=19, **kw)["components_gb"]["direct"]
    assert got == pytest.approx(8.0 * ((2 * side + pair) * 16 + (side + pair) * (16 + 18)) / 1e9, abs=1e-12)
