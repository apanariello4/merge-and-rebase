"""Pure parts of the step-2 artifact diagnostic: frame anchor draw, frame-based profile stats, cosine rows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from merge_and_rebase.eval.diagnostics.anchors_artifacts import (
    frame_anchor_rows,
    jaccard,
    profile_stats,
    unit_rows,
)
from merge_and_rebase.rebase.methods.anchors.core import kernel, kernel_diagnostics


def test_frame_anchor_rows_are_nested_prefixes():
    small = {tuple(r) for r in frame_anchor_rows(20, 197, 64, 0)}
    large = {tuple(r) for r in frame_anchor_rows(20, 197, 256, 0)}
    assert len(small) == 64 and small <= large


def test_frame_anchor_rows_match_a_stored_frame():
    frame = Path("results/anchors_state/tiny_imagenet_A16_T2048/frame.json")
    if not frame.exists():
        pytest.skip("local T-sweep frame not present")
    stored = json.loads(frame.read_text())
    n_images = sum(stored["pool"]["batch_sizes"][: stored["config"]["anchor_batches"]])
    assert frame_anchor_rows(n_images, 197, 2048, 0) == stored["anchor_rows"]


def test_profile_stats_match_kernel_forms_and_null_is_low():
    g = torch.Generator().manual_seed(1)
    p_a = torch.randn(80, 6, generator=g, dtype=torch.float64)
    p_b = p_a @ torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))[0]
    s = profile_stats(p_a, p_b, n_perm=10, seed=0)
    assert s["cka"] == pytest.approx(kernel_diagnostics(kernel(p_a), kernel(p_b))["cka"], rel=1e-10)
    assert s["cka"] == pytest.approx(1.0) and s["cka_null_q95"] < 0.5


def test_unit_rows_and_jaccard():
    p = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    torch.testing.assert_close(unit_rows(p), torch.tensor([[0.6, 0.8], [0.0, 0.0]]))
    assert jaccard(torch.tensor([True, True, False]), torch.tensor([True, False, False])) == pytest.approx(0.5)
