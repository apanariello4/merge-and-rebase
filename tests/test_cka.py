"""Tests for merge_and_rebase.analysis.cka."""

from __future__ import annotations

import math

import torch

from merge_and_rebase.analysis.cka import cka_bootstrap, linear_cka


def _random_features(n: int, d: int, *, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64)


def _biased_naive(x: torch.Tensor, y: torch.Tensor) -> float:
    """Closed-form ``tr(Kc Lc) / (||Kc|| ||Lc||)`` on explicit centered Gram matrices."""
    x64, y64 = x.double(), y.double()
    n = x64.shape[0]
    h = torch.eye(n, dtype=torch.float64) - torch.ones(n, n, dtype=torch.float64) / n
    k = x64 @ x64.T
    l_ = y64 @ y64.T
    kc = h @ k @ h
    lc = h @ l_ @ h
    numerator = torch.trace(kc @ lc)
    denominator = torch.linalg.norm(kc, ord="fro") * torch.linalg.norm(lc, ord="fro")
    return float((numerator / denominator).item())


def test_self_cka_is_one():
    x = _random_features(50, 8, seed=0)
    for debiased in (True, False):
        value = linear_cka(x, x, debiased=debiased)
        assert math.isclose(value, 1.0, abs_tol=1e-10), (debiased, value)


def test_symmetry():
    x = _random_features(60, 6, seed=1)
    y = _random_features(60, 9, seed=2)
    for debiased in (True, False):
        assert math.isclose(linear_cka(x, y, debiased=debiased), linear_cka(y, x, debiased=debiased), abs_tol=1e-10)


def test_orthogonal_invariance():
    n, d = 80, 10
    x = _random_features(n, d, seed=3)
    y = _random_features(n, d, seed=4)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(5), dtype=torch.float64))
    y_rot = y @ q
    for debiased in (True, False):
        base = linear_cka(x, y, debiased=debiased)
        rotated = linear_cka(x, y_rot, debiased=debiased)
        assert math.isclose(base, rotated, abs_tol=1e-9), (debiased, base, rotated)


def test_isotropic_scale_invariance():
    n, d = 80, 10
    x = _random_features(n, d, seed=6)
    y = _random_features(n, d, seed=7)
    for debiased in (True, False):
        base = linear_cka(x, y, debiased=debiased)
        scaled = linear_cka(x, 3.7 * y, debiased=debiased)
        assert math.isclose(base, scaled, abs_tol=1e-9), (debiased, base, scaled)


def test_debiased_independent_gaussians_near_zero_biased_clearly_positive():
    n, d = 500, 20
    x = _random_features(n, d, seed=8)
    y = _random_features(n, d, seed=9)
    debiased_value = linear_cka(x, y, debiased=True)
    biased_value = linear_cka(x, y, debiased=False)
    assert abs(debiased_value) < 0.05, debiased_value
    # The biased estimator's positive bias for independent features scales like d/n
    # (here d/n = 0.04); it is unambiguously positive and well above the near-zero
    # debiased estimate, but not large in absolute terms at this n, d.
    assert biased_value > 0.02, biased_value
    assert biased_value > 5 * abs(debiased_value), (biased_value, debiased_value)


def test_biased_matches_naive_closed_form():
    x = _random_features(40, 5, seed=10)
    y = _random_features(40, 7, seed=11)
    fast = linear_cka(x, y, debiased=False)
    naive = _biased_naive(x, y)
    assert math.isclose(fast, naive, abs_tol=1e-9), (fast, naive)


def test_bootstrap_deterministic_and_interval_contains_value():
    x = _random_features(60, 6, seed=12)
    y = x @ torch.randn(
        6, 6, generator=torch.Generator().manual_seed(13), dtype=torch.float64
    ) + 0.05 * _random_features(60, 6, seed=14)
    result_a = cka_bootstrap(x, y, n_boot=50, seed=0)
    result_b = cka_bootstrap(x, y, n_boot=50, seed=0)
    assert result_a == result_b
    assert result_a["lo"] <= result_a["value"] <= result_a["hi"]


def test_debiased_requires_at_least_four_samples():
    x = _random_features(3, 4, seed=15)
    y = _random_features(3, 4, seed=16)
    try:
        linear_cka(x, y, debiased=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n < 4")
