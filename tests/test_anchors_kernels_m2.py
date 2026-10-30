"""M2 kernel diagnostics (``eval/diagnostics/anchors_kernels.py``): the pure analysis, on synthetic frames."""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.eval.diagnostics.anchors_kernels import (
    anchor_subset,
    kernel_pair_stats,
    pair_profile,
    plot_profiles,
)
from merge_and_rebase.rebase.methods.anchors import RegisteredModel
from merge_and_rebase.rebase.methods.anchors.core import kernel


def _reg(p_blocks, model_id):
    depth = len(p_blocks)
    return RegisteredModel(
        model_id,
        {},
        [p.float() for p in p_blocks],
        [torch.zeros(p.shape[1], dtype=torch.float64) for p in p_blocks],
        [1.0] * depth,
    )


def _orthogonal(d, seed):
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    return q


def test_anchor_subset_is_seeded_sorted_and_nested_at_t_max():
    a = anchor_subset(50, 20, seed=3)
    assert torch.equal(a, anchor_subset(50, 20, seed=3))
    assert torch.equal(a, torch.sort(a).values) and a.unique().numel() == 20
    assert torch.equal(anchor_subset(50, 50, seed=3), torch.arange(50))
    with pytest.raises(ValueError):
        anchor_subset(50, 51, seed=0)


def test_identical_geometry_scores_one_and_beats_the_null():
    g = torch.Generator().manual_seed(0)
    p = torch.randn(40, 16, generator=g, dtype=torch.float64)
    k_a = kernel(p)
    k_b = kernel(2.0 * p @ _orthogonal(16, 1))  # same geometry up to rotation and global scale
    stats = kernel_pair_stats(k_a, k_b, n_perm=50, n_boot=50, seed=0)
    assert stats["cka"] == pytest.approx(1.0, abs=1e-12)
    assert stats["cka_null_q95"] < 0.9
    assert stats["cka_boot_lo"] == pytest.approx(1.0, abs=1e-9)
    assert stats["rel_err"] == pytest.approx(3.0)  # ||4K - K|| / ||K||: the scale is NOT normalised out by rel_err


def test_unrelated_geometries_sit_at_the_null():
    g = torch.Generator().manual_seed(1)
    k_a = kernel(torch.randn(60, 8, generator=g, dtype=torch.float64))
    k_b = kernel(torch.randn(60, 8, generator=g, dtype=torch.float64))
    stats = kernel_pair_stats(k_a, k_b, n_perm=100, n_boot=20, seed=0)
    assert stats["cka"] < stats["cka_null_q95"] + 0.05


def test_pair_profile_on_a_rotated_copy(tmp_path):
    g = torch.Generator().manual_seed(2)
    src_blocks = [torch.randn(30, 6, generator=g, dtype=torch.float64) for _ in range(3)]
    o = _orthogonal(6, 4)
    source = _reg(src_blocks, "A")
    target = _reg([p @ o for p in src_blocks], "B")
    prof = pair_profile(source, target, t_values=[10, 30], n_perm=20, n_boot=10, seed=0)
    assert set(prof["by_T"]) == {"10", "30"}
    for rows in prof["by_T"].values():
        assert [r["source_block"] for r in rows] == [0, 1, 2]
        assert all(r["cka"] == pytest.approx(1.0, abs=1e-6) for r in rows)
        assert all(r["rel_err"] < 1e-6 for r in rows)
    assert prof["relative_pairing_is_argmax_fraction"] == 1.0
    assert len(prof["cross_depth_cka"]) == 3 and len(prof["cross_depth_cka"][0]) == 3
    plot_profiles({"tiny_imagenet": {"pairs": {"A->B": prof}}}, tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 0
