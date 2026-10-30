"""DriftAccumulator: norm ratio, cosine and relative error of realized vs desired block-end changes."""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.eval.diagnostics.depth_drift import DriftAccumulator


def test_drift_summary_separates_size_and_direction():
    g = torch.Generator().manual_seed(0)
    d = torch.randn(3, 5, 4, generator=g)
    acc = DriftAccumulator()
    acc.update(2.0 * d[:2], d[:2])  # streamed in two batches
    acc.update(2.0 * d[2:], d[2:])
    s = acc.summary()
    assert s["norm_ratio"] == pytest.approx(2.0)
    assert s["cosine"] == pytest.approx(1.0)
    assert s["relative_error"] == pytest.approx(1.0)
    orth = DriftAccumulator()
    orth.update(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
    assert orth.summary()["cosine"] == pytest.approx(0.0)
    assert orth.summary()["relative_error"] == pytest.approx(2**0.5)
