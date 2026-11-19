"""Synthetic tests of the metodo K Phase 1 statistics (no model, no download)."""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.eval.diagnostics import metodoK_phase1 as p1
from merge_and_rebase.rebase.methods.anchors.canonical_stats import PairSigmaAccumulator
from merge_and_rebase.rebase.methods.anchors.core import canonical_correlations, canonical_factors_from_frames

F64 = torch.float64


def feed(acc, grid, pm, pk, valid, seed, bs, n_null):
    """Feed images ``[N, T, d]`` in batches of ``bs`` (a batch stays in one image stratum)."""
    n = pm.shape[0]
    halves = p1.half_labels(grid, seed)
    ranks = p1.token_ranks(torch.arange(n), grid.n_tokens, seed)
    levels = p1.token_levels(ranks, grid.ks)
    for b in range(n // bs):
        sl = slice(b * bs, (b + 1) * bs)
        pos = torch.arange(n)[sl]
        a = int(p1.image_stratum(pos[:1], grid))
        part = p1.null_partners(halves[sl], n_null, seed, b) if n_null else None
        acc.update(a, pm[sl], pk[sl], valid[sl], levels[sl], halves[sl], part)
    return halves, levels


def planted(n, rho, d_m, d_k, gen):
    """Rows with exactly known canonical correlations ``rho`` (then random invertible mixing, which keeps them)."""
    r = len(rho)
    z = torch.randn(n, r, dtype=F64, generator=gen)
    w = torch.randn(n, r, dtype=F64, generator=gen)
    rho_t = torch.as_tensor(rho, dtype=F64)
    x_sig, y_sig = z, rho_t * z + (1 - rho_t**2).sqrt() * w
    x = torch.cat([x_sig, torch.randn(n, d_m - r, dtype=F64, generator=gen)], 1)
    y = torch.cat([y_sig, torch.randn(n, d_k - r, dtype=F64, generator=gen)], 1)
    return x @ torch.randn(d_m, d_m, dtype=F64, generator=gen), y @ torch.randn(d_k, d_k, dtype=F64, generator=gen)


def one_cell(pm, pk, n_null, seed, bs=50, gram=None):
    n = pm.shape[0]
    grid = p1.Grid((n,), (1,))
    acc = p1.StratifiedSigma(pm.shape[1], pk.shape[1], grid, n_null)
    valid = torch.ones(n, 1, dtype=torch.bool)
    feed(acc, grid, pm[:, None], pk[:, None], valid, seed, bs, n_null)
    return p1.analyze_cell(acc.cell(0, 0), p1.AnalysisOptions(n_null=n_null), gram)


def test_planted_canonical_correlations_recovered():
    gen = torch.Generator().manual_seed(0)
    rho = [0.9, 0.7, 0.5]
    pm, pk = planted(3000, rho, 6, 5, gen)
    info, t = one_cell(pm, pk, n_null=19, seed=0)
    assert info["status"] == p1.STATUS_OK
    go = t["gamma_out"][:3]
    assert torch.allclose(go, torch.tensor(rho, dtype=F64), atol=0.06), go
    assert float(t["gamma_out"][3:].abs().max()) < 0.15
    # q = 5: the fourth (noise) index is above the pooled threshold in ~5 % of the draws, hence 3 or 4
    assert info["r_star"] in (3, 4) and info["r_star_contig"] >= 3
    # in-sample Gamma of the accumulated sums equals the one computed from the frames themselves
    g_frames = canonical_correlations(canonical_factors_from_frames(pm, pk))
    assert torch.allclose(t["gamma_in"], g_frames, atol=1e-8)


def test_independent_data_has_no_directions_above_null_most_of_the_time():
    zero = 0
    n_seeds = 40
    for s in range(n_seeds):
        gen = torch.Generator().manual_seed(100 + s)
        pm = torch.randn(600, 4, dtype=F64, generator=gen)
        pk = torch.randn(600, 3, dtype=F64, generator=gen)
        info, _ = one_cell(pm, pk, n_null=19, seed=s, bs=60)
        zero += int(info["r_star"] == 0)
    assert zero / n_seeds >= 0.8, zero


def test_nested_strata_equal_direct_sigma_and_null_sums():
    gen = torch.Generator().manual_seed(1)
    grid = p1.Grid((6, 12), (1, 3, 5))
    n, t, d_m, d_k, bs, r = 12, 5, 4, 3, 3, 2
    pm = torch.randn(n, t, d_m, dtype=F64, generator=gen)
    pk = torch.randn(n, t, d_k, dtype=F64, generator=gen)
    valid = torch.rand(n, t, generator=gen) < 0.85
    acc = p1.StratifiedSigma(d_m, d_k, grid, r)
    halves, levels = feed(acc, grid, pm, pk, valid, 7, bs, r)
    partners = [p1.null_partners(halves[b * bs:(b + 1) * bs], r, 7, b) for b in range(n // bs)]
    for ia, n_img in enumerate(grid.n_images):
        for il in range(len(grid.ks)):
            cs = acc.cell(ia, il)
            sel = valid & (levels <= il) & (torch.arange(n)[:, None] < n_img)
            for h in (0, 1):
                img = ((halves == h) & (torch.arange(n) < n_img)).nonzero().flatten()
                ref = PairSigmaAccumulator(torch.zeros(d_m), 1.0, torch.zeros(d_k), 1.0,
                                           halves=(img, torch.tensor([-1] * 0 + [10_000])))
                m = sel[img]
                ids = img[:, None].expand(-1, t)[m]
                ref.update(pm[img][m], pk[img][m], ids)
                mm, kk, mk, nn = ref.sigma(0)
                assert torch.allclose(cs.mm[h], mm) and torch.allclose(cs.kk[h], kk) and torch.allclose(cs.mk[h], mk)
                assert int(cs.n[h]) == nn
                for rr in range(r):  # explicit null: partner (same position) of another image, same batch and half
                    tot, cnt = torch.zeros(d_m, d_k, dtype=F64), 0
                    for p in img.tolist():
                        q_loc = int(partners[p // bs][rr, p % bs])
                        if q_loc < 0:  # a half group of one image in this batch: no partner, rows skipped
                            continue
                        q = q_loc + (p // bs) * bs
                        assert q != p and halves[q] == h
                        ok = sel[p] & valid[q]
                        tot += pm[p][ok].T @ pk[q][ok]
                        cnt += int(ok.sum())
                    assert torch.allclose(cs.null_mk[rr, h], tot) and int(cs.null_n[rr, h]) == cnt


def test_halves_disjoint_nested_and_balanced():
    g1, g2 = p1.Grid((8, 16), (1, 4)), p1.Grid((8, 16, 32), (1, 4))
    h1, h2 = p1.half_labels(g1, 3), p1.half_labels(g2, 3)
    assert torch.equal(h1, h2[:16])  # same split of the common images
    assert int((h2[:8] == 0).sum()) == 4 and int((h2[8:16] == 0).sum()) == 4 and int((h2[16:] == 0).sum()) == 8
    assert set(h2.tolist()) == {0, 1}


def test_token_subsets_nested_per_image_and_batch_independent():
    ks = (1, 4, 9)
    r_all = p1.token_ranks(torch.arange(6), 9, 5)
    r_tail = p1.token_ranks(torch.arange(3, 6), 9, 5)
    assert torch.equal(r_all[3:], r_tail)  # depends on (seed, image) only
    lv = p1.token_levels(r_all, ks)
    for i in range(6):
        counts = [int((lv[i] <= level).sum()) for level in range(3)]
        assert counts == [1, 4, 9]
    # k = 1 is a random token: not always position 0
    assert len({int((lv[i] == 0).nonzero()) for i in range(6)}) > 1
    assert not torch.equal(r_all, p1.token_ranks(torch.arange(6), 9, 6))


def test_null_partners_are_derangements_per_half():
    half = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1])
    part = p1.null_partners(half, 5, 0, 2)
    for r in range(5):
        assert not bool((part[r] == torch.arange(8)).any())
        assert torch.equal(half[part[r]], half)
    single = p1.null_partners(torch.tensor([0, 0, 1]), 2, 0, 0)
    assert bool((single[:, 2] == -1).all())


def icc_cell(n_img, t, valid_p, sig_a, sig_e, gen):
    a = torch.randn(n_img, 1, 2, dtype=F64, generator=gen) * torch.tensor(sig_a, dtype=F64)
    x = a + torch.randn(n_img, t, 2, dtype=F64, generator=gen) * torch.tensor(sig_e, dtype=F64) + 3.0
    valid = torch.rand(n_img, t, generator=gen) < valid_p
    grid = p1.Grid((n_img,), (1, t))
    acc = p1.StratifiedSigma(2, 1, grid, 0)
    feed(acc, grid, x, torch.randn(n_img, t, 1, dtype=F64, generator=gen), valid, 0, 50, 0)
    return acc.cell(0, 1), valid


def test_icc_estimator_known_values_unbalanced():
    gen = torch.Generator().manual_seed(2)
    cs, valid = icc_cell(2000, 12, 0.6, [1.0, 0.5], [1.0, 1.0], gen)  # rho = 0.5 and 0.2, unequal image sizes
    assert len(set(valid.sum(1).tolist())) > 3
    rho, rho_tr = p1.icc_directions(cs, torch.eye(2, dtype=F64))
    assert abs(float(rho[0]) - 0.5) < 0.03 and abs(float(rho[1]) - 0.2) < 0.03, rho
    expect_tr = (1.0 + 0.25) / (2.0 + 1.25)  # variance-weighted
    assert abs(float(rho_tr) - expect_tr) < 0.03
    # k = 1: no within-image variation, ICC undefined
    grid = p1.Grid((100,), (1,))
    acc = p1.StratifiedSigma(2, 1, grid, 0)
    x = torch.randn(100, 1, 2, dtype=F64, generator=gen)
    feed(acc, grid, x, x[..., :1], torch.ones(100, 1, dtype=torch.bool), 0, 50, 0)
    assert bool(torch.isnan(p1.icc_directions(acc.cell(0, 0), torch.eye(2, dtype=F64))[0]).all())


def test_icc_balanced_matches_classic_formula():
    gen = torch.Generator().manual_seed(3)
    n_img, t = 400, 6
    x = torch.randn(n_img, 1, 1, dtype=F64, generator=gen) + torch.randn(n_img, t, 1, dtype=F64, generator=gen)
    grid = p1.Grid((n_img,), (1, t))
    acc = p1.StratifiedSigma(1, 1, grid, 0)
    feed(acc, grid, x, x, torch.ones(n_img, t, dtype=torch.bool), 0, 50, 0)
    rho, _ = p1.icc_directions(acc.cell(0, 1), torch.ones(1, 1, dtype=F64))
    xs = x[..., 0]
    msb = t * ((xs.mean(1) - xs.mean()) ** 2).sum() / (n_img - 1)
    msw = ((xs - xs.mean(1, keepdim=True)) ** 2).sum() / (n_img * (t - 1))
    assert abs(float(rho[0]) - float((msb - msw) / (msb + (t - 1) * msw))) < 1e-9


def test_status_flags_and_degenerate_cells_report_no_gamma():
    assert p1.classify_rows(128, 768, 1024) == p1.STATUS_DEGENERATE  # N = 256, k = 1 (half = 128)
    assert p1.classify_rows(512, 768, 1024) == p1.STATUS_DEGENERATE  # N = 1024, k = 1
    assert p1.classify_rows(1792, 768, 1024) == p1.STATUS_DEGENERATE  # <= d_m + d_k
    assert p1.classify_rows(2048, 768, 1024) == p1.STATUS_BORDERLINE  # N = 4096, k = 1
    assert p1.classify_rows(3583, 768, 1024) == p1.STATUS_BORDERLINE
    assert p1.classify_rows(3584, 768, 1024) == p1.STATUS_OK
    gen = torch.Generator().manual_seed(4)
    pm, pk = planted(40, [0.8], 6, 5, gen)  # half = 20 rows > 11: ok; borderline < 22
    info, t = one_cell(pm, pk, n_null=3, seed=0, bs=10)
    assert info["status"] == p1.STATUS_BORDERLINE and "gamma_in_top1" in info and t
    pm, pk = planted(20, [0.8], 6, 5, gen)  # half = 10 <= 11
    info, t = one_cell(pm, pk, n_null=3, seed=0, bs=10)
    assert info["status"] == p1.STATUS_DEGENERATE and t == {} and "gamma_in_top1" not in info and "r_star" not in info
    # rank-deficient frame (a duplicated column): handled by the pinv rule of the factors
    gen2 = torch.Generator().manual_seed(5)
    pm = torch.randn(400, 6, dtype=F64, generator=gen2)
    pm[:, 5] = pm[:, 4]
    pk = torch.randn(400, 5, dtype=F64, generator=gen2)
    info, t = one_cell(pm, pk, n_null=3, seed=0)
    assert info["status"] in (p1.STATUS_OK, p1.STATUS_BORDERLINE)  # the duplicated direction is dropped by the pinv rule, the cell still computes


def test_coverage_in_subspace_is_one_and_isotropic_is_r_over_d():
    gen = torch.Generator().manual_seed(6)
    d, r_m, r = 200, 150, 30
    q, _ = torch.linalg.qr(torch.randn(d, r_m, dtype=F64, generator=gen))
    y, _ = torch.linalg.qr(torch.randn(r_m, r, dtype=F64, generator=gen))
    w = q @ y
    e_in = torch.randn(5000, r, dtype=F64, generator=gen) @ w.T
    cov = p1.task_coverage(e_in.T @ e_in, q, y)
    assert abs(cov["cov_total"] - 1) < 1e-9 and abs(cov["cov_span"] - 1) < 1e-9
    e_iso = torch.randn(20000, d, dtype=F64, generator=gen)
    cov = p1.task_coverage(e_iso.T @ e_iso, q, y)
    assert abs(cov["cov_total"] - r / d) < 0.01
    assert abs(cov["cov_span"] - r / r_m) < 0.015
    assert p1.task_coverage(e_iso.T @ e_iso, q, y[:, :0])["cov_total"] == 0.0


def test_analyze_cell_coverage_with_planted_subspace():
    gen = torch.Generator().manual_seed(7)
    pm, pk = planted(3000, [0.9, 0.8], 6, 5, gen)
    e = torch.randn(2000, 6, dtype=F64, generator=gen)
    info, _ = one_cell(pm, pk, 19, 0, gram={"t": e.T @ e})
    cov = info["coverage"]["t"]
    assert 0 <= cov["cov_total"] <= 1 and info["coverage_isotropic_reference"] == info["r_star"] / info["rank_m"]
    assert abs(cov["cov_total"] - info["r_star"] / 6) < 0.1  # isotropic E: r*/d


def _cell(n, k, r, vec, status="ok"):
    return {"N": n, "k": k, "T_nominal": n * k, "status": status, "r_star": r, "gamma_out_vec": torch.tensor(vec, dtype=F64)}


def _cells(last=(0.9,) * 10):
    return [
        _cell(256, 1, 0, [], "degenerate"),
        _cell(256, 16, 3, [0.7] * 10),  # Gamma_out far from the larger cells: unstable
        _cell(1024, 16, 10, [0.9] * 10),  # stable against every larger cell (r*_prefix of a cell is only a profile)
        _cell(256, 197, 10, [0.9] * 10),
        _cell(4096, 16, 10, [0.905] * 10),
        _cell(4096, 197, 10, list(last)),
    ]


def test_selection_helper_absolute_gamma_criterion():
    sel = p1.select_cell(_cells())
    assert sel["cell"] == {"N": 1024, "k": 16, "T_nominal": 16384} and sel["r_ref"] == 10 and sel["trivial"] is False
    assert {(c["N"], c["k"]) for c in sel["compared"]} == {(256, 197), (4096, 16), (4096, 197)}
    # a cell with a LOW r* but the same Gamma_out is not rejected for its r* (no dependence on the power of the null test)
    cells = _cells()
    cells[1] = _cell(256, 16, 0, [0.9] * 10)
    assert p1.select_cell(cells)["cell"]["T_nominal"] == 4096
    # r_ref = 0 at the largest cell: nothing to stabilise, no cell is selected
    cells0 = _cells()
    cells0[-1] = _cell(4096, 197, 0, [0.9] * 10)
    assert p1.select_cell(cells0)["cell"] is None and p1.select_cell(cells0)["r_ref"] == 0
    assert p1.select_cell([_cell(1, 1, 0, [], "degenerate")])["cell"] is None


def test_select_global_trivial_flag_and_compared_cells():
    cells = _cells()
    g = p1.select_global([cells, [dict(c) for c in cells]])
    assert g["cell"] == {"N": 1024, "k": 16, "T_nominal": 16384} and g["trivial"] is False
    assert len(g["compared"]) == 2 and len(g["compared"][0]) == 3 and g["r_ref_per_block"] == [10, 10]
    bad = _cells(last=(0.7,) * 10)  # the largest cell moves Gamma_out by 0.2: only the largest cell is stable
    gb = p1.select_global([bad, bad])
    assert gb["cell"] == {"N": 4096, "k": 197, "T_nominal": 4096 * 197} and gb["trivial"] is True and gb["compared"] == [[], []]
    degenerate = [dict(c) for c in cells]
    degenerate[2] = dict(degenerate[2], status="degenerate")
    assert p1.select_global([cells, degenerate])["cell"]["T_nominal"] > 16384
    no_dir = _cells()
    no_dir[-1] = _cell(4096, 197, 0, [0.9] * 10)
    gn = p1.select_global([cells, no_dir])  # the second block has no directions: no constraint, listed
    assert gn["blocks_without_directions"] == [1] and gn["cell"] is not None


def test_cost_estimate_config_validation_and_device_memory():
    g = p1.Grid((256, 1024, 4096), (1, 16, 197))
    c = p1.estimate_cost(768, 1024, g, 19, 24, ram_budget_gb=24.0)
    assert 2.0 < c["per_block_accumulator_gb"] < 3.0 and c["blocks_per_pass"] * c["per_block_accumulator_gb"] < 24
    assert c["n_passes"] * c["blocks_per_pass"] >= 24
    cc = p1.estimate_cost(768, 1024, g, 19, 24, ram_budget_gb=24.0, content_null=True)
    assert cc["per_block_accumulator_gb"] > 1.9 * c["per_block_accumulator_gb"] and cc["flops_per_block"] == 2 * c["flops_per_block"]
    # A -> B on a 16 GB card (free 14 GB, margin 4 GB): reduced to what fits; refused if not even one block fits
    gpu = p1.estimate_cost(768, 1024, g, 19, 24, acc_device="cuda", free_gb=14.0, blocks_per_pass=8)
    assert gpu["blocks_per_pass"] * gpu["per_block_accumulator_gb"] <= 14.0 - 4.0 and gpu["reduced_from"] == 8
    assert gpu["peak_accumulator_gb"] < 14.0
    with pytest.raises(ValueError, match="acc_device='cpu'"):
        p1.estimate_cost(768, 1024, g, 19, 24, acc_device="cuda", free_gb=6.0, content_null=True)
    assert p1.device_free_gb("cpu") is None
    spec = {"model": "m", "pretrained": "p", "experts": {"EuroSAT": "x"}}
    base = {"source": spec, "target": spec, "state_dir": "s", "out_dir": "o", "tasks": ["EuroSAT"]}
    cfg = p1.parse_config({**base, "n_images": [32, 64], "ks": [1, 4], "batch_size": 8})
    assert cfg.acc_device == "cpu" and cfg.cudnn_allow_tf32 is False and cfg.matmul_allow_tf32 is False
    assert p1.parse_config({**base, "n_images": [64, 128], "ks": [1, 4]}).batch_size == 64
    with pytest.raises(ValueError, match="divide"):
        p1.parse_config({**base, "n_images": [32, 60], "ks": [1, 4], "batch_size": 8})
    with pytest.raises(ValueError, match="unknown"):
        p1.parse_config({**base, "bogus": 1})


def positional_data(seed, pos_scale, rho_c, n=800, t=10, d=12):
    """Shared per-position means (4 dims, linearly related across models) + 4 content dims correlated at ``rho_c``."""
    g = torch.Generator().manual_seed(seed)
    pos_m = torch.randn(t, 4, dtype=F64, generator=g) * pos_scale
    pos_k = pos_m @ torch.randn(4, 4, dtype=F64, generator=g)
    z = torch.randn(n, t, 4, dtype=F64, generator=g)
    w = torch.randn(n, t, 4, dtype=F64, generator=g)
    noise = lambda *sh: torch.randn(*sh, dtype=F64, generator=g)  # noqa: E731
    xm = torch.cat([pos_m.expand(n, t, 4) + 0.3 * noise(n, t, 4), z, noise(n, t, d - 8)], -1)
    xk = torch.cat([pos_k.expand(n, t, 4) + 0.3 * noise(n, t, 4), rho_c * z + (1 - rho_c**2) ** 0.5 * w, noise(n, t, d - 8)], -1)
    return xm - xm.reshape(-1, d).mean(0), xk - xk.reshape(-1, d).mean(0), pos_m, pos_k


def run_null(xm, xk, kind, seed=0, n_null=19, bs=50):
    n, t, _ = xm.shape
    grid = p1.Grid((n,), (t,))
    acc = p1.StratifiedSigma(xm.shape[-1], xk.shape[-1], grid, n_null)
    halves = p1.half_labels(grid, seed)
    lv = p1.token_levels(p1.token_ranks(torch.arange(n), t, seed), grid.ks)
    for b in range(n // bs):
        sl = slice(b * bs, (b + 1) * bs)
        pos = torch.arange(n)[sl]
        plan = (p1.NullRandomPosition(halves[sl], pos, t, n_null, seed, b) if kind == "n1"
                else p1.NullPositionKept(halves[sl], n_null, seed, b))
        acc.update(0, xm[sl], xk[sl], torch.ones(bs, t, dtype=torch.bool), lv[sl], halves[sl], plan)
    return p1.analyze_cell(acc.cell(0, 0), p1.AnalysisOptions(icc=False))


def test_positional_structure_is_counted_under_the_primary_null():
    """Reviewer case: 4 real positional directions + 4 correlated content directions. The position-kept null keeps the
    shared per-position mean term and finds (almost) nothing; the primary null (random position) finds them."""
    xm, xk, _, _ = positional_data(0, 3.0, 0.7)
    info_n1, _ = run_null(xm, xk, "n1")
    info_old, _ = run_null(xm, xk, "kept")
    assert info_n1["r_star_prefix"] >= 4, info_n1  # planted content directions (the 4 positional ones are real too)
    assert info_n1["r_star_prefix"] >= 8, info_n1
    assert info_old["r_star_prefix"] < info_n1["r_star_prefix"]
    # the secondary null is valid on rows centred per position
    pos_mean = xm.mean(0), xk.mean(0)  # per-position means (here estimated on the data: enough for the unit test)
    info_c, _ = run_null(xm - pos_mean[0], xk - pos_mean[1], "kept")
    assert info_c["r_star_prefix"] >= 4, info_c


def test_null_calibration_with_intra_image_correlation():
    """k > 1 rows with an image effect in each model, models independent: false directions at the 5 % level."""
    frac, any_dir, n_seeds = [], 0, 40
    for s in range(n_seeds):
        g = torch.Generator().manual_seed(500 + s)
        n, t = 300, 6
        xm = torch.randn(n, 1, 4, dtype=F64, generator=g) + torch.randn(n, t, 4, dtype=F64, generator=g)
        xk = torch.randn(n, 1, 3, dtype=F64, generator=g) + torch.randn(n, t, 3, dtype=F64, generator=g)
        xm, xk = xm - xm.reshape(-1, 4).mean(0), xk - xk.reshape(-1, 3).mean(0)
        info, _ = run_null(xm, xk, "n1", seed=s, bs=60)
        frac.append(info["n_above"] / info["q"])
        any_dir += int(info["r_star_prefix"] > 0)
    assert 0.01 <= sum(frac) / n_seeds <= 0.11, sum(frac) / n_seeds
    assert any_dir / n_seeds <= 0.2


def test_random_position_null_sums_equal_explicit_computation():
    gen = torch.Generator().manual_seed(11)
    grid = p1.Grid((8,), (1, 4))
    n, t, d_m, d_k, bs, r = 8, 4, 3, 2, 4, 2
    pm = torch.randn(n, t, d_m, dtype=F64, generator=gen)
    pk = torch.randn(n, t, d_k, dtype=F64, generator=gen)
    valid = torch.rand(n, t, generator=gen) < 0.8
    halves = p1.half_labels(grid, 3)
    levels = p1.token_levels(p1.token_ranks(torch.arange(n), t, 3), grid.ks)
    acc = p1.StratifiedSigma(d_m, d_k, grid, r)
    plans = []
    for b in range(n // bs):
        sl = slice(b * bs, (b + 1) * bs)
        plans.append(p1.NullRandomPosition(halves[sl], torch.arange(n)[sl], t, r, 3, b))
        acc.update(0, pm[sl], pk[sl], valid[sl], levels[sl], halves[sl], plans[-1])
    cs = acc.cell(0, 1)
    for rr in range(r):
        for h in (0, 1):
            tot, cnt = torch.zeros(d_m, d_k, dtype=F64), 0
            for pimg in (halves == h).nonzero().flatten().tolist():
                b, loc = pimg // bs, pimg % bs
                q_loc = int(plans[b].partners[rr, loc])
                if q_loc < 0:
                    continue
                q = b * bs + q_loc
                sig = p1.position_permutation(rr, pimg, t, 3)
                assert not torch.equal(sig, torch.arange(t)) or t < 3
                ok = valid[pimg] & valid[q][sig]
                tot += pm[pimg][ok].T @ pk[q][sig][ok]
                cnt += int(ok.sum())
            assert torch.allclose(cs.null_mk[rr, h], tot) and int(cs.null_n[rr, h]) == cnt
    # independent sigma per (replicate, image), and a function of (seed, r, image) only
    assert not torch.equal(p1.position_permutation(0, 1, 50, 3), p1.position_permutation(1, 1, 50, 3))
    assert not torch.equal(p1.position_permutation(0, 1, 50, 3), p1.position_permutation(0, 2, 50, 3))
    assert torch.equal(p1.position_permutation(0, 1, 50, 3), p1.position_permutation(0, 1, 50, 3))
    # a bare partner tensor is still accepted (position-kept null)
    acc2 = p1.StratifiedSigma(d_m, d_k, grid, r)
    acc2.update(0, pm[:bs], pk[:bs], valid[:bs], levels[:bs], halves[:bs], p1.null_partners(halves[:bs], r, 3, 0))


def test_r_star_rules_pooled_threshold_prefix_excess_contig():
    null = torch.randn(19, 400, dtype=F64, generator=torch.Generator().manual_seed(0)) * 0.05
    gam = torch.full((400,), 0.0, dtype=F64)
    gam[:50] = 0.9
    gam[10] = 0.0  # a hole inside the leading block
    info, t = p1.r_star_rules(gam, null, p1.AnalysisOptions())
    assert info["r_star_contig"] == 10 and info["n_above"] == 49
    assert info["r_star_prefix"] == 50  # 49 / 51 = 96 % would admit 51, but the trailing failure is trimmed
    assert abs(info["r_star_excess"] - (49 - 0.05 * 351)) < 1e-9
    pooled = float(torch.quantile(null.reshape(-1), 0.95, interpolation="higher"))
    assert info["threshold"] == pytest.approx(pooled) and t["threshold"].numel() == 400
    assert bool((t["threshold"] >= pooled - 1e-12).all()) and len(info["threshold_windows"]) == 3
    gam2 = gam.clone()
    gam2[[3, 4]] = 0.0  # three holes in the leading block: the prefix falls back
    assert p1.r_star_rules(gam2, null, p1.AnalysisOptions())[0]["r_star_prefix"] < 50


def test_frame_check_helpers_on_synthetic_data():
    gen = torch.Generator().manual_seed(8)
    b, t, d = 6, 5, 7
    p = torch.randn(b, t, d, dtype=F64, generator=gen)
    rows = torch.tensor([[0, 1], [2, 4], [5, 0], [3, 3]])
    ref = torch.stack([p[i, j] for i, j in rows.tolist()])
    assert torch.equal(p1.gather_rows(p, rows[:, 0], rows[:, 1]), ref)
    assert p1.relative_errors(ref, ref)["rel_fro"] == 0.0
    noisy = ref * (1 + 1e-3 * torch.randn(ref.shape, dtype=F64, generator=gen))
    e = p1.relative_errors(noisy, ref)
    assert 5e-4 < e["rel_fro"] < 2e-3 and e["row_max"] >= e["row_median"] and e["n_rows"] == 4
    exact = p1.relative_errors(ref * 1.03, ref)  # a uniform 3 % scale error is reported as 3 %
    assert abs(exact["rel_fro"] - 0.03) < 1e-12 and abs(exact["row_median"] - 0.03) < 1e-12


def test_float32_products_with_float64_accumulation_match_float64():
    gen = torch.Generator().manual_seed(12)
    n, t, d = 120, 16, 24
    mix = torch.randn(d, d, dtype=F64, generator=gen)
    pm = torch.randn(n, t, d, dtype=F64, generator=gen) @ mix * 0.1
    pk = pm @ torch.randn(d, d, dtype=F64, generator=gen) * 0.1 + 0.01 * torch.randn(n, t, d, dtype=F64, generator=gen)
    out = {}
    for name, cd in (("f64", F64), ("f32", torch.float32)):
        grid = p1.Grid((n,), (1, t))
        acc = p1.StratifiedSigma(d, d, grid, 2, compute_dtype=cd)
        halves = p1.half_labels(grid, 0)
        lv = p1.token_levels(p1.token_ranks(torch.arange(n), t, 0), grid.ks)
        for b in range(n // 40):
            sl = slice(b * 40, (b + 1) * 40)
            acc.update(0, pm[sl], pk[sl], torch.ones(40, t, dtype=torch.bool), lv[sl], halves[sl],
                       p1.NullRandomPosition(halves[sl], torch.arange(n)[sl], t, 2, 0, b))
        out[name] = acc.cell(0, 1)
    assert out["f32"].mm.dtype == F64
    for f in ("mm", "kk", "mk", "null_mk"):
        a, b2 = getattr(out["f64"], f), getattr(out["f32"], f)
        assert float((a - b2).abs().max() / a.abs().max()) < 1e-5, f
    g64 = p1.analyze_cell(out["f64"], p1.AnalysisOptions(icc=False))[1]["gamma_out"]
    g32 = p1.analyze_cell(out["f32"], p1.AnalysisOptions(icc=False))[1]["gamma_out"]
    assert float((g64 - g32).abs().max()) < 1e-3


def test_coverage_non_isotropic_planted_in_canonical_subspace_detects_wrong_y():
    gen = torch.Generator().manual_seed(13)
    pm, pk = planted(3000, [0.9, 0.7, 0.5], 8, 6, gen)
    fac = canonical_factors_from_frames(pm, pk)
    y, gam, _ = torch.linalg.svd(fac.m, full_matrices=False)
    r = 3
    scales = torch.tensor([5.0, 1.0, 0.2], dtype=F64)  # anisotropic energy inside the r-dim subspace
    w = fac.v_m @ y[:, :r]
    e = (torch.randn(4000, r, dtype=F64, generator=gen) * scales) @ w.T
    g = e.T @ e
    ok = p1.task_coverage(g, fac.v_m, y[:, :r])
    assert abs(ok["cov_total"] - 1) < 1e-9 and abs(ok["cov_span"] - 1) < 1e-9
    wrong_cols = p1.task_coverage(g, fac.v_m, y[:, r:2 * r])  # the next canonical directions
    assert wrong_cols["cov_total"] < 0.05
    q, _ = torch.linalg.qr(torch.randn(y.shape[0], r, dtype=F64, generator=gen))  # a random rotation of the basis
    assert p1.task_coverage(g, fac.v_m, q)["cov_total"] < 0.9
    # a rank-1 sub-selection captures exactly the energy share of the strongest direction
    one = p1.task_coverage(g, fac.v_m, y[:, :1])["cov_total"]
    assert abs(one - 25.0 / (25.0 + 1.0 + 0.04)) < 0.03
    # through analyze_cell: r* = 3 on this planted data, so the coverage of the cell is the one of the planted subspace
    info, _ = one_cell(pm, pk, 19, 0, gram={"t": g})
    assert info["r_star"] >= 3  # a superset of the planted subspace still covers it entirely
    assert abs(info["coverage"]["t"]["cov_total"] - 1) < 1e-6


def test_window_thresholds_raise_the_top_indices():
    gen = torch.Generator().manual_seed(1)
    null = torch.randn(19, 200, dtype=F64, generator=gen) * 0.05
    null[:, :10] *= 3.0  # the leading null directions overfit more
    gam = torch.zeros(200, dtype=F64)
    gam[:30] = 0.2  # above the pooled threshold, below the leading window's
    info, t = p1.r_star_rules(gam, null, p1.AnalysisOptions())
    assert float(t["threshold"][0]) > 2 * info["threshold"] and float(t["threshold"][100]) >= info["threshold"]
    assert info["r_star_contig"] == 0  # the top ten are not above their own window threshold


def test_planted_r_star_is_exactly_three_in_most_draws():
    hit, n_seeds = 0, 40
    for s in range(n_seeds):
        gen = torch.Generator().manual_seed(900 + s)
        pm, pk = planted(1500, [0.9, 0.7, 0.5], 6, 5, gen)
        info, _ = one_cell(pm, pk, n_null=19, seed=s)
        hit += int(info["r_star"] == 3)
    assert hit / n_seeds >= 0.85, hit
