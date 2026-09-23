"""Tests for Direct Residual's ``component_target='output_total'`` mode.

``'output_total'`` gives each fitted component a target built from the source
block's FULL realized output change between the fine-tuned and base
endpoints -- ``A_c(X^{s1}_i, W^{ft}_i) - A_c(X^{s0}_i, W^{base}_i)`` -- unlike
``'output_local'``, which evaluates BOTH endpoints on the base input ``X^{s0}``
and therefore never sees upstream-propagated input drift. The Procrustes
alignment map is unchanged between the two modes (always fit from the base
endpoint, ``A_c(X^{s0}_i, W^{base}_i)`` against the target's own pristine
``A_c^{t0}_j``); only the fine-tuned side of the delta differs.

``output_total``'s depth rule is deliberately matched to ``block_boundary``'s
(every position uses only its own paired source block, at weight 1.0), NOT to
``output_local``'s span/multiplicity-aware rule -- see
``direct_residual.position_paired_only_contributions``. This file checks:

(a) when the FT model differs from base only in block i's weights and i is
    the first block (X^{s1}_i == X^{s0}_i for that block), output_total's
    target at that position matches output_local's, bitwise;
(b) when an upstream block's weights change, output_total's target changes
    but output_local's does not;
(c) the paired-only weight rule: extend 2->4 and shrink 4->2 both give every
    position exactly one contribution, at weight 1.0.

Golden-hash preservation of the untouched block_boundary/None path (this
ablation touches no code any block_boundary run executes) is checked directly
against the golden hashes already pinned in
``tests/test_direct_residual_component_coverage.py``.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    capture_paired_boundary_activations,
    compute_desired_effects,
    fit_direct_residual,
    position_paired_only_contributions,
    position_source_contributions,
)
from merge_and_rebase.eval.target_residual_completion import order_components
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

# --------------------------------------------------------------------------
# Fixture (same shape as test_direct_residual_component_coverage.py's stock
# fixture -- a real nn.MultiheadAttention target, per that module's own
# no-cross-import convention this file follows too).
# --------------------------------------------------------------------------


class _StockAttentionBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = torch.nn.MultiheadAttention(width, 1, batch_first=True)
        self.ls_1 = torch.nn.Identity()
        self.ln_2 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("c_fc", torch.nn.Linear(width, width * 2)), ("gelu", torch.nn.GELU()),
            ("c_proj", torch.nn.Linear(width * 2, width)),
        ]))
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        normed = self.ln_1(x)
        x = x + self.ls_1(self.attn(normed, normed, normed, need_weights=False)[0])
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class _StockVisual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_StockAttentionBlock(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _StockModel(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _StockVisual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _stock_loader(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n)), batch_size=2, shuffle=False)


def _stock_setup(source_depth, target_depth, width=4, seed=41):
    torch.manual_seed(seed)
    source_base = _StockModel(width, source_depth).eval()
    target_base = _StockModel(width, target_depth).eval()
    data = _stock_loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, target_base, data, pairing, target_base_sd


def _tune_block(model, index, seed, scale=0.2, components=("attn.out_proj", "mlp.c_proj")):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        block = tuned.visual.transformer.resblocks[index]
        if "mlp.c_proj" in components:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
        if "attn.out_proj" in components:
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def _fit_mode(source_base, source_ft, target_base, data, pairing, target_base_sd, component_target, components):
    config = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, component_target=component_target, components=tuple(components),
    )
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
        component_inputs=order_components(config.components),
        capture_source_ft_component_inputs=component_target == "output_total",
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu",
    )
    return corrections, diagnostics, captured


# --------------------------------------------------------------------------
# (a) First-block-only FT change: X^{s1}_0 == X^{s0}_0 (block 0 has nothing
#     upstream), so output_total and output_local must agree bitwise at any
#     position paired 1:1 to block 0 (same-arch pairing.pairing[j] == j).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tuned_component, key_suffix",
    [
        ("attn.out_proj", "attn.out_proj.weight"),
        ("mlp.c_proj", "mlp.c_proj.weight"),
    ],
)
def test_output_total_matches_output_local_when_only_first_block_changes(tuned_component, key_suffix):
    """X^{s1}_{0,c} == X^{s0}_{0,c} exactly for a component c of the FIRST
    block whenever nothing feeding INTO c's own input changed: block 0 has no
    upstream block, and each of attn.out_proj/mlp.c_proj's own input only
    depends on computation strictly BEFORE it within the block's forward
    pass. Tuning only ONE component's weight (not both -- tuning
    attn.out_proj would also perturb mlp.c_proj's own input, since mlp
    follows attn within the block -- an intra-block dependency, not an
    "upstream block" one) isolates that component's target: output_total's
    ``A_c(X^{s1}_0, W^{ft}) - A_c(X^{s0}_0, W^{base})`` must then equal
    output_local's ``A_c(X^{s0}_0, W^{ft}) - A_c(X^{s0}_0, W^{base})``
    bitwise, since X^{s1}_{0,c} = X^{s0}_{0,c}.
    """
    source_depth = target_depth = 3
    source_base, target_base, data, pairing, target_base_sd = _stock_setup(source_depth, target_depth)
    source_ft = _tune_block(source_base, 0, seed=7, components=(tuned_component,))

    corr_local, _diag_local, _cap_local = _fit_mode(
        source_base, source_ft, deepcopy(target_base), data, pairing, deepcopy(target_base_sd),
        "output_local", (tuned_component,),
    )
    corr_total, _diag_total, _cap_total = _fit_mode(
        source_base, source_ft, deepcopy(target_base), data, pairing, deepcopy(target_base_sd),
        "output_total", (tuned_component,),
    )
    # Position 0 is paired 1:1 to source block 0 under same-arch identity pairing.
    assert pairing.pairing[0] == 0
    key = f"visual.transformer.resblocks.0.{key_suffix}"
    torch.testing.assert_close(corr_local[key], corr_total[key], rtol=0, atol=1e-6)


# --------------------------------------------------------------------------
# (b) Upstream weight change: output_total's target changes, output_local's
#     does not.
# --------------------------------------------------------------------------


def test_output_total_reacts_to_upstream_change_output_local_does_not():
    source_depth = target_depth = 3
    source_base, target_base, data, pairing, target_base_sd = _stock_setup(source_depth, target_depth)
    # Fine-tune block 1 (upstream of block 2) only.
    source_ft_a = _tune_block(source_base, 1, seed=11)
    source_ft_b = deepcopy(source_ft_a)
    with torch.no_grad():
        # A second, larger perturbation to the SAME upstream block: the
        # forward pass through block 2 will see a different X^{s1}_2 (a
        # different upstream block-1 output) between ft_a and ft_b, even
        # though block 2's own weights never change.
        block1 = source_ft_b.visual.transformer.resblocks[1]
        block1.mlp.c_proj.weight.add_(0.4 * torch.randn_like(block1.mlp.c_proj.weight))
        block1.attn.out_proj.weight.add_(0.4 * torch.randn_like(block1.attn.out_proj.weight))

    def _run(component_target, source_ft):
        return _fit_mode(
            source_base, source_ft, deepcopy(target_base), data, pairing, deepcopy(target_base_sd),
            component_target, ("mlp.c_proj",),
        )

    key_mlp = "visual.transformer.resblocks.2.mlp.c_proj.weight"

    corr_total_a, _d, _c = _run("output_total", source_ft_a)
    corr_total_b, _d, _c = _run("output_total", source_ft_b)
    assert not torch.equal(corr_total_a[key_mlp], corr_total_b[key_mlp]), (
        "output_total's target at position 2 must react to an upstream (block 1) weight change "
        "via X^{s1}_2 -- it evaluates the FT endpoint on the source FT model's own input"
    )

    corr_local_a, _d, _c = _run("output_local", source_ft_a)
    corr_local_b, _d, _c = _run("output_local", source_ft_b)
    torch.testing.assert_close(corr_local_a[key_mlp], corr_local_b[key_mlp], rtol=0, atol=0)


# --------------------------------------------------------------------------
# (c) Paired-only weights: extend 2->4 and shrink 4->2, exactly one weight-1
#     contribution per position, for output_total's depth rule.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
def test_output_total_paired_only_contributions_are_single_weight_one(source_depth, target_depth):
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    contributions = position_paired_only_contributions(pairing)
    assert set(contributions) == set(range(target_depth))
    for j in range(target_depth):
        assert contributions[j] == [(pairing.pairing[j], 1.0)]


def test_output_total_depth_rule_differs_from_output_local_under_extend():
    """Under extend (2 -> 4), output_local splits a source block's weight
    across the m_i target positions that share it (1/m_i); output_total's
    paired-only rule gives every position weight 1.0 regardless -- the
    deliberate, documented difference from output_local's depth handling."""
    pairing = DiscreteLayerPairing.compute(2, 4)
    local = position_source_contributions(pairing)
    total = position_paired_only_contributions(pairing)
    # Some source block must be shared by more than one target position
    # under a 2x extend (m_i == 2 for both blocks here).
    counts: dict[int, int] = {}
    for i in pairing.pairing:
        counts[i] = counts.get(i, 0) + 1
    assert any(m > 1 for m in counts.values())
    for j in range(4):
        assert local[j] == [(pairing.pairing[j], 1.0 / counts[pairing.pairing[j]])]
        assert total[j] == [(pairing.pairing[j], 1.0)]


def test_output_total_depth_rule_differs_from_output_local_under_shrink():
    """Under shrink (4 -> 2), output_local's span partition can give a
    position MULTIPLE contributions (every source block whose reverse
    pairing collapses onto it); output_total's paired-only rule always gives
    exactly one, at weight 1.0."""
    pairing = DiscreteLayerPairing.compute(4, 2)
    local = position_source_contributions(pairing)
    total = position_paired_only_contributions(pairing)
    assert any(len(terms) > 1 for terms in local.values())
    for j in range(2):
        assert len(total[j]) == 1
        assert total[j] == [(pairing.pairing[j], 1.0)]
