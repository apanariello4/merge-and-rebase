"""Pure helpers of the H1-H3 diagnostic (eval/diagnostics/anchors_jacobian.py)."""

from __future__ import annotations

import math

import torch

from merge_and_rebase.eval.diagnostics.anchors_jacobian import Pair, alignment, centred, functional_directions


def test_pair_metrics():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(5, 4, generator=g, dtype=torch.float64)
    p = Pair()
    p.add(a, a + 3.0)  # a common shift is invisible after centring
    r = p.result()
    assert r["rel_err"] < 1e-12 and abs(r["cos_mean"] - 1) < 1e-12 and abs(r["norm_ratio"] - 1) < 1e-12
    q = Pair()
    q.add(a, 2 * a)
    assert abs(q.result()["rel_err"] - 1) < 1e-12 and abs(q.result()["norm_ratio"] - 2) < 1e-12
    b = torch.randn(5, 4, generator=g, dtype=torch.float64)
    s = Pair()
    s.add(a, b)
    ac, bc = centred(a), centred(b)
    assert math.isclose(s.result()["rel_err"], float((ac - bc).norm() / ac.norm()), rel_tol=1e-12)


def test_alignment_identical_and_orthogonal():
    g = torch.Generator().manual_seed(1)
    j = torch.randn(3, 4, 5, 2, generator=g, dtype=torch.float64)
    out = alignment(j, 2 * j)
    assert abs(out["alignment_mean"] - 1) < 1e-12
    assert abs(alignment(j, -j)["alignment_mean"] + 1) < 1e-12


def test_functional_directions_agreement():
    r = 4
    e = torch.eye(r, dtype=torch.float64)
    # both models weigh axis 0 alike; axis 1 only model "a", axis 2 only model "b"
    ma = torch.diag(torch.tensor([4.0, 3.0, 0.0, 1.0], dtype=torch.float64))
    mb = torch.diag(torch.tensor([4.0, 0.0, 3.0, 1.0], dtype=torch.float64))
    out = functional_directions({"a": ma, "b": mb})
    top = out["eigvecs"][:, 0]
    assert abs(abs(float(top @ e[0])) - 1) < 1e-10
    assert abs(float(out["agreement"][0]) - 1) < 1e-10
    assert float(out["agreement"][1]) < 1e-10 and float(out["agreement"][2]) < 1e-10
