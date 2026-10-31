"""Effect operators of ``anchors`` (docs/.local/AGENT_NEXT_PHASE.md O1): float64 synthetic checks."""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics
from merge_and_rebase.rebase.methods.anchors import parse_anchors_config
from merge_and_rebase.rebase.methods.anchors.core import gpa, gpa_transport_map
from merge_and_rebase.rebase.methods.anchors.operators import (
    EffectOperator,
    decode_operator,
    fit_effect_operator,
    load_library,
    r_squared,
    save_library,
    to_common_space,
)


def _g(seed):
    return torch.Generator().manual_seed(seed)


def _orthogonal(d, seed):
    q, r = torch.linalg.qr(torch.randn(d, d, generator=_g(seed), dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


def _linear_effect(d=6, n_batches=3, rows=40, seed=0):
    w = torch.randn(d, d, generator=_g(seed), dtype=torch.float64)
    b = torch.randn(d, generator=_g(seed + 1), dtype=torch.float64)
    h = [torch.randn(rows, d, generator=_g(seed + 2 + k), dtype=torch.float64) for k in range(n_batches)]
    return w, b, h, [x @ w + b for x in h]


def test_exactly_linear_effect_is_recovered_with_r2_one():
    w, b, h, e = _linear_effect()
    op = fit_effect_operator(h, e, lam=0.0)
    torch.testing.assert_close(op.w, w, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(op.b, b, rtol=1e-9, atol=1e-9)
    assert r_squared(op, h, e) == pytest.approx(1.0, abs=1e-12)
    # held-out rows of the same linear law
    _, _, h2, _ = _linear_effect(seed=50)
    assert r_squared(op, h2, [x @ w + b for x in h2]) == pytest.approx(1.0, abs=1e-12)


def test_automatic_ridge_matches_ariadnes_shared_statistics():
    """Same lambda rule and centring as ResidualSufficientStatistics.solve (empirical Bayes, identity output map)."""
    _, _, h, _ = _linear_effect(d=5, seed=3)
    e = [x @ torch.randn(5, 5, generator=_g(9), dtype=torch.float64) + 0.3 * torch.randn(40, 5, generator=_g(k),
         dtype=torch.float64) for k, x in enumerate(h)]
    op = fit_effect_operator(h, e)
    stats = ResidualSufficientStatistics()
    for x, y in zip(h, e, strict=True):
        stats.update(x, y, None, torch.eye(5, dtype=torch.float64))
    xt, diag = stats.solve(ridge_relative=0.01, ridge_estimator="empirical_bayes")
    assert op.info["lambda"] == pytest.approx(diag["ridge"], rel=1e-10)
    torch.testing.assert_close(op.w.T.float(), xt, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(op.b.float(), diag["bias_correction"], rtol=1e-4, atol=1e-5)


def test_rank_truncation_keeps_the_dominant_displacement():
    w, b, h, e = _linear_effect(d=8, seed=4)
    full = fit_effect_operator(h, e, lam=0.0)
    low = fit_effect_operator(h, e, lam=0.0, rank=3)
    assert torch.linalg.matrix_rank(low.w).item() == 3
    assert 0.0 < low.info["rank_energy_kept"] < 1.0
    assert r_squared(low, h, e) < r_squared(full, h, e)


def test_operator_transport_between_rotated_models_is_exact():
    """B = A rotated (P_k = P_m O, h_k = h_m O): the operator decoded in B is the effect rotated, at full rank."""
    d, t = 6, 30
    p_m = torch.randn(t, d, generator=_g(5), dtype=torch.float64)
    o = _orthogonal(d, 6)
    _, (a_m, a_k), _ = gpa([p_m, p_m @ o])
    w, b, h, e = _linear_effect(d=d, seed=7)
    op = fit_effect_operator(h, e, lam=0.0)
    w_u, b_u = to_common_space(op, a_m, None)
    for x, y in zip(h, e, strict=True):
        torch.testing.assert_close(decode_operator(x @ o, w_u, b_u, a_k, None, 1.0), y @ o, rtol=1e-8, atol=1e-8)


def test_operator_equals_data_library_when_the_effect_is_linear():
    """lambda = 0, same rows, E linear in h, aligned frames: D_operator = D_data = E A_m^T A_k (scale c_m)."""
    d, t, c_m = 5, 25, 1.7
    p_m = torch.randn(t, d, generator=_g(11), dtype=torch.float64)
    o = _orthogonal(d, 12)
    _, (a_m, a_k), _ = gpa([p_m, p_m @ o])
    w, b, h, _ = _linear_effect(d=d, seed=13)
    e_raw = [c_m * (x @ w + b) for x in h]  # E / c_m is linear in h
    op = fit_effect_operator(h, [y / c_m for y in e_raw], lam=0.0)
    w_u, b_u = to_common_space(op, a_m, None)
    phi = gpa_transport_map(a_m, a_k, c_m, scale=c_m)
    for x, y in zip(h, e_raw, strict=True):
        torch.testing.assert_close(decode_operator(x @ o, w_u, b_u, a_k, None, c_m), y @ phi, rtol=1e-8, atol=1e-8)


def test_library_round_trip(tmp_path):
    op = EffectOperator(torch.randn(4, 4, dtype=torch.float64), torch.randn(4, dtype=torch.float64), {})
    a = _orthogonal(4, 1)
    ops = {0: to_common_space(op, a, None), 3: to_common_space(op, a, 2)}
    with pytest.raises(ValueError):
        save_library(tmp_path / "x.pt", ops, {})
    save_library(tmp_path / "x.pt", ops, {"universe_version": "abc", "source": "A", "task": "T"})
    back, meta = load_library(tmp_path / "x.pt")
    assert meta["universe_version"] == "abc"
    h = torch.randn(7, 4, dtype=torch.float64)
    for j, (w_u, b_u) in ops.items():
        rank = None if j == 0 else 2
        ref = decode_operator(h, w_u.float().double(), b_u.float().double(), a, rank, 1.0)
        torch.testing.assert_close(decode_operator(h, *back[j], a, rank, 1.0), ref)


def test_parse_operator_options():
    assert parse_anchors_config({}).library == "data"
    cfg = parse_anchors_config({"library": "operator", "frame_map": "gpa", "operator_rank": 16,
                                "target_calibration": "disjoint"})
    assert cfg.operator_rank == 16 and cfg.target_calibration == "disjoint"
    for bad in ({"library": "operator"}, {"operator_rank": 4}, {"target_calibration": "disjoint"},
                {"library": "operator", "frame_map": "gpa", "depth_fit": "sequential"}, {"library": "x"}):
        with pytest.raises(ValueError):
            parse_anchors_config(bad)
