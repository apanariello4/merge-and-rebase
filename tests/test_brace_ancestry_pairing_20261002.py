"""P5.13: ``brace_ancestry_pairing`` equals the realized layout of a real BlockExtender(skip_correction=True)."""

from __future__ import annotations

import pytest
import torch
from test_ariadne_depth_baselines import _make_loader, _TinyModel

from merge_and_rebase.eval.block_extension import BlockExtensionConfig, run_block_extension
from merge_and_rebase.rebase.depth_pairing import brace_ancestry_pairing
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.alignment import apply_depth_pairing_override
from merge_and_rebase.rebase.methods._ariadne.config import parse_direct_residual_config


def _realized(source_depth, target_depth, **kw):
    torch.manual_seed(0)
    base, ft = _TinyModel(depth=source_depth), _TinyModel(depth=source_depth)
    cfg = BlockExtensionConfig(
        target_layers_total=target_depth,
        n_batches_act=1,
        skip_correction=True,
        skip_final_ln=True,
        verbose=False,
        show_progress=False,
        **kw,
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


@pytest.mark.parametrize("order", ["bottom-top", "top-bottom"])
@pytest.mark.parametrize("density", ["spread", "spread_mod", "clump"])
@pytest.mark.parametrize("depths", [(3, 5), (4, 9), (2, 7), (5, 6)])
def test_extension_matches_realized_layout(depths, density, order):
    s, t = depths
    got = brace_ancestry_pairing(s, t, order, density)
    assert got.pairing == _realized(s, t, insertion_order=order, extension_density=density)
    assert got.source_depth == s and got.target_depth == t


@pytest.mark.parametrize("schedule", ["cascade", "disjoint_spans"])
@pytest.mark.parametrize("order", ["bottom-top", "top-bottom"])
@pytest.mark.parametrize("density", ["spread", "spread_mod", "clump"])
@pytest.mark.parametrize("depths", [(6, 4), (7, 3), (8, 5), (5, 4)])
def test_shrink_matches_realized_layout(depths, density, order, schedule):
    s, t = depths
    got = brace_ancestry_pairing(s, t, order, density, schedule)
    want = _realized(s, t, insertion_order=order, extension_density=density, collapse_schedule=schedule)
    assert got.pairing == want


def test_equal_depth_is_identity_and_random_is_rejected():
    assert brace_ancestry_pairing(4, 4).pairing == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="random"):
        brace_ancestry_pairing(3, 5, "random")


def test_config_and_override_wiring():
    cfg = parse_direct_residual_config(
        {"depth_pairing": "brace_ancestry", "depth_pairing_brace": {"insertion_order": "top-bottom"}}
    )
    assert cfg.depth_pairing_brace == {"insertion_order": "top-bottom"}
    with pytest.raises(ValueError, match="requires depth_pairing='brace_ancestry'"):
        parse_direct_residual_config({"depth_pairing_brace": {}})
    with pytest.raises(ValueError, match="random"):
        parse_direct_residual_config(
            {"depth_pairing": "brace_ancestry", "depth_pairing_brace": {"insertion_order": "random"}}
        )
    rel = DiscreteLayerPairing.compute(3, 5)
    anc = brace_ancestry_pairing(3, 5)
    assert apply_depth_pairing_override(rel, "brace_ancestry", ancestry=anc) is anc
    assert apply_depth_pairing_override(rel, "relative") is rel
    with pytest.raises(ValueError, match="needs the ancestry"):
        apply_depth_pairing_override(rel, "brace_ancestry")
