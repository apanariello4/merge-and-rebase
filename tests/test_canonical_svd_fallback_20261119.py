"""gesdd -> gesvd fallback of the canonical SVD: unchanged when torch converges, same decomposition when it does not,
refusal on a non-finite M."""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.rebase.methods.anchors import core


def _m(seed=0, n=12, k=9):
    return torch.randn(n, k, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def test_default_path_is_torch_svd():
    m = _m()
    u, s, vt = core._svd_canonical(m)
    u0, s0, vt0 = torch.linalg.svd(m, full_matrices=False)
    assert torch.equal(u, u0) and torch.equal(s, s0) and torch.equal(vt, vt0)


def _failing_svd(*a, **k):
    raise torch.linalg.LinAlgError("forced")


@pytest.mark.parametrize("m", [_m(1), torch.eye(7, dtype=torch.float64)])
def test_fallback_reconstructs_and_matches_values(monkeypatch, m):
    s_ref = torch.linalg.svdvals(m)
    monkeypatch.setattr(torch.linalg, "svd", _failing_svd)
    with pytest.warns(RuntimeWarning, match="gesvd"):
        u, s, vt = core._svd_canonical(m)
    assert u.dtype == m.dtype and s.shape == s_ref.shape
    assert torch.allclose(s, s_ref, atol=1e-12)
    assert torch.allclose(u @ torch.diag(s) @ vt, m, atol=1e-10)


def test_fallback_refuses_non_finite(monkeypatch):
    m = _m(2)
    m[0, 0] = float("nan")
    monkeypatch.setattr(torch.linalg, "svd", _failing_svd)
    with pytest.raises(ValueError, match="non-finite"):
        core._svd_canonical(m)
