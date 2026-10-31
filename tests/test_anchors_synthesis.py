"""D2 ridge with conservation (anchors/synthesis.py, AGENT_NEXT_PHASE.md O2): float64 synthetic checks."""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics
from merge_and_rebase.rebase.methods.anchors.synthesis import GenericInputStats, generic_change, solve_preserve


def _g(s):
    return torch.Generator().manual_seed(s)


def _problem(n=200, d_in=8, d_out=5, m=300, seed=0, g_rows=None):
    h = torch.randn(n, d_in, generator=_g(seed), dtype=torch.float64) + 0.5
    d = h @ torch.randn(d_in, d_out, generator=_g(seed + 1), dtype=torch.float64) + 0.3 + 0.2 * torch.randn(
        n, d_out, generator=_g(seed + 2), dtype=torch.float64)
    stats = ResidualSufficientStatistics()
    for k in range(4):  # several batches, as in the pipeline
        sl = slice(k * n // 4, (k + 1) * n // 4)
        stats.update(h[sl], d[sl], None, torch.eye(d_out, dtype=torch.float64))
    g = torch.randn(m, d_in, generator=_g(seed + 3), dtype=torch.float64) - 0.2 if g_rows is None else g_rows
    gen = GenericInputStats()
    gen.update(g)
    return h, d, g, stats, gen


def test_rho_zero_is_ariadnes_ridge():
    _, _, _, stats, gen = _problem()
    xt, diag = stats.solve(ridge_relative=0.01, ridge_estimator="empirical_bayes")
    dw, beta, info = solve_preserve(stats, gen, 0.0, diag["ridge"])
    torch.testing.assert_close(dw.T.float(), xt, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(beta.float(), diag["bias_correction"], rtol=1e-4, atol=1e-5)
    assert info["residual_norm_after"] == pytest.approx(diag["residual_norm_after"], rel=1e-6)


def test_large_rho_cancels_the_change_on_full_rank_generic_inputs():
    _, _, g, stats, gen = _problem()
    _, diag = stats.solve(ridge_relative=0.01, ridge_estimator="empirical_bayes")
    sizes = []
    for rho in (0.0, 1.0, 1e3, 1e7):
        dw, beta, _ = solve_preserve(stats, gen, rho, diag["ridge"])
        sizes.append(float(torch.linalg.norm(g @ dw + beta)))
    assert sizes[-1] < 1e-4 * sizes[0]
    assert all(a >= b - 1e-12 for a, b in zip(sizes, sizes[1:]))
    # generic_change agrees with the explicit computation
    dw, beta, _ = solve_preserve(stats, gen, 1.0, diag["ridge"])
    w = torch.randn(5, 8, generator=_g(9), dtype=torch.float64)
    ref = float(torch.linalg.norm(g @ dw + beta) / torch.linalg.norm(g @ w.T))
    assert generic_change(dw, beta, w, gen) == pytest.approx(ref, rel=1e-9)


def test_generic_equal_task_inputs_shrinks_the_update_monotonically():
    h, _, _, stats, _ = _problem()
    gen = GenericInputStats()
    gen.update(h)
    _, diag = stats.solve(ridge_relative=0.01, ridge_estimator="empirical_bayes")
    norms = [float(torch.linalg.norm(solve_preserve(stats, gen, rho, diag["ridge"])[0])) for rho in (0, 0.1, 1, 10, 100)]
    assert all(a > b for a, b in zip(norms, norms[1:]))


def test_layerscale_output_map_is_rejected():
    _, _, _, _, gen = _problem()
    stats = ResidualSufficientStatistics()
    stats.update(torch.randn(10, 8, dtype=torch.float64), torch.randn(10, 5, dtype=torch.float64), None,
                 2.0 * torch.eye(5, dtype=torch.float64))
    with pytest.raises(ValueError):
        solve_preserve(stats, gen, 1.0, 1.0)


def test_reduced_rank_projection():
    from merge_and_rebase.rebase.methods.anchors.synthesis import reduce_rank

    h, d, _, stats, _ = _problem(d_in=10, d_out=6)
    xt, diag = stats.solve(ridge_relative=0.01, ridge_estimator="empirical_bayes")
    dw = xt.T.double()
    full, beta_full, kept_full = reduce_rank(stats, dw, 6)
    torch.testing.assert_close(full, dw, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(beta_full.float(), diag["bias_correction"], rtol=1e-4, atol=1e-5)
    assert kept_full == pytest.approx(1.0)
    low, beta, kept = reduce_rank(stats, dw, 2)
    assert torch.linalg.matrix_rank(low).item() == 2 and 0 < kept < 1
    hc = h - h.mean(0)
    # the kept directions are the top principal directions of the fitted displacement H~ dW
    _, _, vt = torch.linalg.svd(hc @ dw, full_matrices=False)
    torch.testing.assert_close(hc @ low, hc @ dw @ vt[:2].T @ vt[:2], rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(beta, d.mean(0) - low.T @ h.mean(0), rtol=1e-8, atol=1e-8)
