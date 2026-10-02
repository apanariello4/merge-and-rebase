"""P5.13: ``spread_duplicate_pairing`` equals the realized layout of a real BlockExtender(skip_correction=True)."""

from __future__ import annotations

import pytest
import torch
from test_ariadne_depth_baselines import _make_loader, _TinyModel

from merge_and_rebase.eval.block_extension import BlockExtensionConfig, run_block_extension
from merge_and_rebase.rebase.depth_pairing import spread_duplicate_pairing
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.alignment import apply_depth_pairing_override
from merge_and_rebase.rebase.methods._ariadne.config import parse_direct_residual_config


def _realized(source_depth, target_depth):
    torch.manual_seed(0)
    base, ft = _TinyModel(depth=source_depth), _TinyModel(depth=source_depth)
    cfg = BlockExtensionConfig(
        target_layers_total=target_depth,
        n_batches_act=1,
        skip_correction=True,
        skip_final_ln=True,
        verbose=False,
        show_progress=False,
    )
    layout: dict = {}
    run_block_extension(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=_make_loader(),
        target_layers_total=None,
        config=cfg,
        device="cpu",
        layout_out=layout,
    )
    blocks = layout["final_blocks"]
    assert len(blocks) == target_depth
    if layout["direction"] == "extend":
        return tuple(b["source_orig_idx"] for b in blocks)
    return tuple(b["span_orig_idxs"][0] for b in blocks)  # first original index of each span


@pytest.mark.parametrize("depths", [(3, 5), (4, 9), (2, 7), (5, 6), (6, 4), (7, 3), (8, 5), (5, 4), (12, 3)])
def test_matches_realized_layout_with_default_schedule(depths):
    s, t = depths
    got = spread_duplicate_pairing(s, t)
    assert got.pairing == _realized(s, t)
    assert (got.source_depth, got.target_depth) == (s, t)


def test_equal_depth_is_identity():
    assert spread_duplicate_pairing(4, 4).pairing == (0, 1, 2, 3)


def test_config_and_override_wiring():
    cfg = parse_direct_residual_config({"depth_pairing": "spread_duplicate"})
    assert cfg.depth_pairing == "spread_duplicate"
    with pytest.raises(ValueError, match="renamed to 'spread_duplicate'"):
        parse_direct_residual_config({"depth_pairing": "brace_ancestry"})
    with pytest.raises(ValueError, match="unknown direct_residual fields"):
        parse_direct_residual_config({"depth_pairing": "spread_duplicate", "depth_pairing_brace": {}})
    rel = DiscreteLayerPairing.compute(3, 5)
    anc = spread_duplicate_pairing(3, 5)
    assert apply_depth_pairing_override(rel, "spread_duplicate", ancestry=anc) is anc
    assert apply_depth_pairing_override(rel, "relative") is rel
    with pytest.raises(ValueError, match="needs the ancestry"):
        apply_depth_pairing_override(rel, "spread_duplicate")
    with pytest.raises(ValueError, match="renamed to 'spread_duplicate'"):
        apply_depth_pairing_override(rel, "brace_ancestry")
