"""Shared-anchor algebra (``rebase/methods/anchors/core.py``): toy cases of the reference document and the
properties the method rests on (round trip, rotation invariance, cycle consistency, linearity)."""

from __future__ import annotations

import math

import pytest
import torch

from merge_and_rebase.analysis.cka import linear_cka
from merge_and_rebase.rebase.methods.anchors.core import (
    CoverageAccumulator,
    MomentAccumulator,
    decode_rows,
    encode_rows,
    kernel,
    kernel_diagnostics,
    kernel_spectrum,
    resolve_eps,
    round_trip_operator,
)

F64 = torch.float64
ROT90 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=F64)  # h -> h O rotates (1,0) to (0,1)


def _transport(e, p_a, p_b, eps=0.0):
    r = encode_rows(e, p_a, 1.0)
    return decode_rows(r, p_b, kernel(p_b), scale=1.0, eps=eps)


def _orthogonal(d, seed):
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed), dtype=F64))
    return q


def _randn(*shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=F64)


# ---------------------------------------------------------------- toy cases (c = 1, eps = 0)


def test_toy_case_1_rotation_recovered():
    p_a = torch.eye(2, dtype=F64)
    p_b = p_a @ ROT90
    e = torch.tensor([[2.0, 1.0]], dtype=F64)
    assert torch.allclose(encode_rows(e, p_a, 1.0), torch.tensor([[2.0, 1.0]], dtype=F64))
    d = _transport(e, p_a, p_b)
    assert torch.allclose(d, torch.tensor([[-1.0, 2.0]], dtype=F64))
    assert torch.allclose(d, e @ ROT90)


def test_toy_case_2_overlapping_anchors_need_kernel_inverse():
    p_a = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=F64)
    p_b = p_a @ ROT90
    e = torch.tensor([[2.0, 1.0]], dtype=F64)
    r = encode_rows(e, p_a, 1.0)
    assert torch.allclose(r, torch.tensor([[2.0, 3.0]], dtype=F64))
    w = torch.linalg.solve(kernel(p_b), r.T).T
    assert torch.allclose(w, torch.tensor([[1.0, 1.0]], dtype=F64))
    assert torch.allclose(_transport(e, p_a, p_b), e @ ROT90)
    # Without K^{-1} the shared horizontal direction is counted twice: (5, 3) O.
    assert torch.allclose(r @ p_b, torch.tensor([[5.0, 3.0]], dtype=F64) @ ROT90)


def test_toy_case_3_stretch_breaks_transport_and_diagnostic_flags_it():
    p_a = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=F64)
    p_b = p_a @ torch.diag(torch.tensor([1.0, 2.0], dtype=F64))
    e = torch.tensor([[2.0, 1.0]], dtype=F64)
    w = torch.linalg.solve(kernel(p_b), encode_rows(e, p_a, 1.0).T).T
    assert torch.allclose(w, torch.tensor([[1.75, 0.25]], dtype=F64))
    assert torch.allclose(_transport(e, p_a, p_b), torch.tensor([[2.0, 0.5]], dtype=F64))
    rel = kernel_diagnostics(kernel(p_a), kernel(p_b))["rel_err"]
    assert rel == pytest.approx(3.0 / math.sqrt(7.0))  # ~1.13
    assert rel == pytest.approx(1.13, abs=5e-3)


# ---------------------------------------------------------------- properties


def test_round_trip_is_projection_onto_anchor_span():
    t, d, n = 6, 10, 20
    p = _randn(t, d, seed=1)
    e = _randn(n, d, seed=2)
    back = _transport(e, p, p)
    # orthogonal projection of each row onto span(rows of P)
    coef = torch.linalg.lstsq(p.T, e.T).solution
    assert torch.allclose(back, (p.T @ coef).T, atol=1e-10)
    m = round_trip_operator(p, kernel(p), 0.0)
    assert torch.allclose(m @ m, m, atol=1e-10)
    assert torch.allclose(m, m.T, atol=1e-10)


def test_round_trip_is_identity_when_anchors_span_the_space():
    d = 5
    p = _randn(9, d, seed=3)  # T > d, full column rank
    e = _randn(4, d, seed=4)
    assert torch.allclose(_transport(e, p, p, eps=1e-12), e, atol=1e-6)


def test_encoding_is_invariant_to_orthogonal_change_of_basis():
    t, d, n = 7, 12, 15
    p = _randn(t, d, seed=5)
    e = _randn(n, d, seed=6)
    o = _orthogonal(d, seed=7)
    assert torch.allclose(encode_rows(e, p, 1.3), encode_rows(e @ o, p @ o, 1.3), atol=1e-12)
    assert torch.allclose(kernel(p), kernel(p @ o), atol=1e-12)
    # transport onto a rotated copy is exactly the rotation (within the anchor span)
    proj = _transport(e, p, p)
    assert torch.allclose(_transport(e, p, p @ o), proj @ o, atol=1e-10)


def test_cycle_consistency_when_t_le_d():
    t, n = 6, 9
    p_a, p_b, p_c = _randn(t, 10, seed=8), _randn(t, 14, seed=9), _randn(t, 8, seed=10)
    e = _randn(n, 10, seed=11)
    direct = _transport(e, p_a, p_c)
    via_b = _transport(_transport(e, p_a, p_b), p_b, p_c)
    assert torch.allclose(via_b, direct, atol=1e-8)


def test_cycle_with_eps_contracts_weak_directions_only_slightly():
    t, n = 6, 9
    p_a, p_b, p_c = _randn(t, 10, seed=8), _randn(t, 14, seed=9), _randn(t, 8, seed=10)
    e = _randn(n, 10, seed=11)
    eps = resolve_eps(kernel(p_b), 1e-3)
    direct = _transport(e, p_a, p_c)
    via_b = _transport(_transport(e, p_a, p_b, eps=eps), p_b, p_c)
    rel = float(torch.linalg.norm(via_b - direct) / torch.linalg.norm(direct))
    assert 0.0 < rel < 1e-2


def test_decode_is_linear_and_keeps_the_mean():
    t, d_a, d_b, n = 5, 9, 11, 13
    p_a, p_b = _randn(t, d_a, seed=12), _randn(t, d_b, seed=13)
    k_b = kernel(p_b)
    eps = resolve_eps(k_b, 1e-3)
    r1, r2 = _randn(n, t, seed=14), _randn(n, t, seed=15)
    lhs = decode_rows(r1 + r2, p_b, k_b, scale=2.0, eps=eps)
    rhs = decode_rows(r1, p_b, k_b, scale=2.0, eps=eps) + decode_rows(r2, p_b, k_b, scale=2.0, eps=eps)
    assert torch.allclose(lhs, rhs, atol=1e-10)
    # A pure shift (every row the same) is not removed by encode/decode: it survives for the ridge bias.
    e = _randn(1, d_a, seed=16).expand(n, d_a)
    d = decode_rows(encode_rows(e, p_a, 1.0), p_b, k_b, scale=1.0, eps=eps)
    assert torch.allclose(d - d[:1], torch.zeros_like(d), atol=1e-12)
    assert float(torch.linalg.norm(d[0])) > 0.0


def test_decode_formula_holds_when_t_exceeds_d():
    """``(PP^T + eps I)^{-1} P = P (P^T P + eps I)^{-1}``: the same decode serves T > d (narrow target)."""
    t, d = 12, 5
    p = _randn(t, d, seed=17)
    eps = resolve_eps(kernel(p), 1e-3)
    lhs = torch.linalg.solve(kernel(p) + eps * torch.eye(t, dtype=F64), p)
    rhs = p @ torch.linalg.inv(p.T @ p + eps * torch.eye(d, dtype=F64))
    assert torch.allclose(lhs, rhs, atol=1e-9)
    r = _randn(3, t, seed=18)
    assert torch.isfinite(decode_rows(r, p, kernel(p), scale=1.0, eps=eps)).all()


def test_kernel_cka_matches_biased_feature_cka():
    t = 32
    p_a, p_b = _randn(t, 10, seed=19), _randn(t, 17, seed=20)
    diag = kernel_diagnostics(kernel(p_a), kernel(p_b))
    assert diag["cka"] == pytest.approx(linear_cka(p_a, p_b, debiased=False), abs=1e-12)
    o = _orthogonal(10, seed=21)
    same = kernel_diagnostics(kernel(p_a), kernel(3.0 * p_a @ o))
    assert same["cka"] == pytest.approx(1.0, abs=1e-12)


def test_moment_accumulator_matches_direct_and_survives_a_large_mean():
    x = _randn(301, 7, seed=22) + 1e6
    acc = MomentAccumulator()
    for chunk in torch.split(x, 37):
        acc.update(chunk)
    mu, c = acc.finalize()
    assert torch.allclose(mu, x.mean(0), rtol=0, atol=1e-6)
    assert c == pytest.approx(float(torch.linalg.norm(x - x.mean(0)) / math.sqrt(x.shape[0])), rel=1e-9)


def test_coverage_accumulator_matches_direct_projection():
    t, d = 4, 9
    p = _randn(t, d, seed=23)
    e = _randn(50, d, seed=24) + 3.0
    acc = CoverageAccumulator()
    for chunk in torch.split(e, 11):
        acc.update(chunk)
    cov = acc.coverage(p, kernel(p), 0.0)
    ec = e - e.mean(0)
    rec = _transport(ec, p, p)
    expected = 1.0 - float(((ec - rec) ** 2).sum() / (ec**2).sum())
    assert cov["coverage"] == pytest.approx(expected, rel=1e-10)
    assert cov["reconstructed_energy_ratio"] == pytest.approx(expected, rel=1e-10)  # eps = 0: projection
    full = CoverageAccumulator()
    full.update(e)
    assert full.coverage(_randn(d + 3, d, seed=25), kernel(_randn(d + 3, d, seed=25)), 1e-12)["coverage"] == (
        pytest.approx(1.0, abs=1e-6)
    )


def test_kernel_spectrum_reports_rank_deficiency():
    p = _randn(10, 4, seed=26)  # T > d: rank 4
    spec = kernel_spectrum(kernel(p))
    assert spec["numerical_rank"] == 4
    assert spec["condition_number"] > 1e8


def test_gram_forms_equal_kernel_forms():
    """T > d: diagnostics, spectrum and round trip from the d x d Gram equal the T x T kernel versions."""
    from merge_and_rebase.rebase.methods.anchors.core import (
        kernel_diagnostics_from_frames,
        kernel_spectrum_from_frame,
        round_trip_operator_primal,
    )

    g = torch.Generator().manual_seed(5)
    t = 50
    p_m = torch.randn(t, 7, generator=g, dtype=torch.float64) + 0.3
    p_k = torch.randn(t, 9, generator=g, dtype=torch.float64) @ torch.randn(9, 9, generator=g, dtype=torch.float64)
    ref = kernel_diagnostics(kernel(p_m), kernel(p_k))
    got = kernel_diagnostics_from_frames(p_m, p_k)
    assert got["cka"] == pytest.approx(ref["cka"], rel=1e-10)
    assert got["rel_err"] == pytest.approx(ref["rel_err"], rel=1e-8)
    ref_s, got_s = kernel_spectrum(kernel(p_k)), kernel_spectrum_from_frame(p_k)
    assert got_s["effective_rank"] == pytest.approx(ref_s["effective_rank"], rel=1e-9)
    assert got_s["trace"] == pytest.approx(ref_s["trace"], rel=1e-12)
    assert got_s["numerical_rank"] == ref_s["numerical_rank"] == 9
    eps = 0.1
    torch.testing.assert_close(round_trip_operator_primal(p_k, eps), round_trip_operator(p_k, kernel(p_k), eps))


# ---------------------------------------------------------------- GPA (AGENT_NEXT.md 4.9)


def _rot(d, seed):
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    return q


def test_gpa_two_models_equal_width_is_pairwise_procrustes():
    from merge_and_rebase.rebase.methods.anchors.core import gpa

    g = torch.Generator().manual_seed(3)
    p1 = torch.randn(60, 6, generator=g, dtype=torch.float64)
    p2 = p1 @ _rot(6, 4) + 0.3 * torch.randn(60, 6, generator=g, dtype=torch.float64)
    _, (a1, a2), info = gpa([p1, p2], tol=1e-14, max_iter=500)
    u, _, vt = torch.linalg.svd(p1.T @ p2)
    torch.testing.assert_close(a1.T @ a2, u @ vt, rtol=1e-6, atol=1e-7)
    assert info["gpa_d_u"] == 6 and not info["gpa_underdetermined"]


def test_gpa_recovers_known_rotations_with_small_noise():
    from merge_and_rebase.rebase.methods.anchors.core import gpa

    g = torch.Generator().manual_seed(1)
    p = torch.randn(80, 5, generator=g, dtype=torch.float64)
    rots = [_rot(5, s) for s in (10, 11, 12)]
    frames = [p @ o + 1e-4 * torch.randn(80, 5, generator=g, dtype=torch.float64) for o in rots]
    _, maps, _ = gpa(frames)
    for m in range(3):
        for k in range(3):
            torch.testing.assert_close(maps[m].T @ maps[k], rots[m].T @ rots[k], rtol=0, atol=1e-3)


def test_gpa_chain_equals_direct_with_full_universe():
    from merge_and_rebase.rebase.methods.anchors.core import gpa, gpa_transport_map

    g = torch.Generator().manual_seed(2)
    frames = [torch.randn(40, 5, generator=g, dtype=torch.float64) for _ in range(3)]
    _, (a1, a2, a3), _ = gpa(frames)
    direct = gpa_transport_map(a1, a3, 1.0, scale=1.0)
    chain = gpa_transport_map(a1, a2, 1.0, scale=1.0) @ gpa_transport_map(a2, a3, 1.0, scale=1.0)
    torch.testing.assert_close(chain, direct, rtol=1e-9, atol=1e-10)


def test_gpa_gauge_rotation_changes_no_transport():
    from merge_and_rebase.rebase.methods.anchors.core import gpa, gpa_transport_map

    g = torch.Generator().manual_seed(6)
    frames = [
        torch.randn(30, 4, generator=g, dtype=torch.float64),
        torch.randn(30, 6, generator=g, dtype=torch.float64),
    ]
    _, (a1, a2), _ = gpa(frames)
    q = _rot(6, 30)  # a global rotation of G: A_m -> Q A_m
    torch.testing.assert_close(
        gpa_transport_map(q @ a1, q @ a2, 1.0, scale=1.0), gpa_transport_map(a1, a2, 1.0, scale=1.0)
    )


def test_gpa_nested_truncation_and_full_rank_equals_untruncated():
    from merge_and_rebase.rebase.methods.anchors.core import gpa, gpa_transport_map

    g = torch.Generator().manual_seed(7)
    frames = [
        torch.randn(50, 5, generator=g, dtype=torch.float64),
        torch.randn(50, 7, generator=g, dtype=torch.float64),
    ]
    universe, (a1, a2), info = gpa(frames)
    assert info["gpa_d_u"] == 7
    full = gpa_transport_map(a1, a2, 1.0, scale=1.0)
    torch.testing.assert_close(gpa_transport_map(a1, a2, 1.0, scale=1.0, rank=7), full)
    # after the gauge the axes of G are ordered by variance
    var = (universe**2).sum(0)
    assert bool((var[:-1] >= var[1:] - 1e-9).all())
    r3 = gpa_transport_map(a1, a2, 1.0, scale=1.0, rank=3)
    assert torch.linalg.matrix_rank(r3).item() <= 3


def test_gpa_different_widths_have_orthonormal_columns():
    from merge_and_rebase.rebase.methods.anchors.core import gpa

    g = torch.Generator().manual_seed(4)
    _, (a1, a2), _ = gpa(
        [torch.randn(30, 5, generator=g, dtype=torch.float64), torch.randn(30, 8, generator=g, dtype=torch.float64)]
    )
    assert a1.shape == (8, 5) and a2.shape == (8, 8)
    torch.testing.assert_close(a1.T @ a1, torch.eye(5, dtype=torch.float64))
    torch.testing.assert_close(a2.T @ a2, torch.eye(8, dtype=torch.float64))


def test_universe_incremental_add_keeps_g_and_existing_maps(tmp_path):
    import hashlib

    from merge_and_rebase.rebase.methods.anchors import AnchorsConfig, AnchorState
    from merge_and_rebase.rebase.methods.anchors.state import RegisteredModel
    from merge_and_rebase.rebase.methods.anchors.universe import fit_universe

    def digest(t):
        return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()

    g = torch.Generator().manual_seed(8)
    state = AnchorState(AnchorsConfig(num_anchors=40), tmp_path)
    for name, width, depth in (("A", 5, 2), ("B", 6, 4), ("C", 5, 2)):
        p = [torch.randn(40, width, generator=g) for _ in range(depth)]
        reg = RegisteredModel(
            name, {"depth": depth}, p, [torch.zeros(width, dtype=torch.float64)] * depth, [1.0] * depth
        )
        state.models[name] = reg
    base = fit_universe(state, ["A", "B"], n_grid=4)
    before = {m: [digest(a) for a in base.maps[m]] for m in ("A", "B")}
    g_before = [digest(x) for x in base.g]
    grown = fit_universe(state, ["A", "B"], n_grid=4, add_models=["C"])  # loaded from disk, then C added
    assert grown.version == base.version and grown.added == ["C"]
    assert [digest(x) for x in grown.g] == g_before
    assert {m: [digest(a) for a in grown.maps[m]] for m in ("A", "B")} == before
    joint = fit_universe(state, ["A", "B", "C"], n_grid=4)
    assert joint.version != base.version
