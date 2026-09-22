"""The correctness anchor: Direct Residual and ARIADNE's `target_scope="all"`
`direct_target` path must produce bit-identical fitted corrections whenever
they are handed the same alignment and the same captured activation banks.

This is what makes Part A's refactor (extracting
`complete_residuals_direct`'s per-position solver body into
`target_informed_runtime._fit_direct_target_position`) a structural
guarantee rather than a hope: both `direct_residual.fit_direct_residual` and
`complete_residuals_direct` call that identical helper. `i(j) =
round(j*(D_A-1)/(D_B-1))` does NOT equal ARIADNE's `2*i+1 -> i` ancestry in
general (Direct Residual fits every target position, not just the "inserted"
half), so this test does not rely on that coincidence -- it FORCES a
`DiscreteLayerPairing` and a hand-built ARIADNE layout to agree on the same
source-index assignment, then proves the shared solver treats the two
callers identically regardless of which one supplied the alignment.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    capture_paired_boundary_activations,
    compute_desired_effects,
    fit_direct_residual,
)
from merge_and_rebase.eval.target_informed_runtime import capture_residual_references, complete_residuals_direct
from merge_and_rebase.eval.target_residual_completion import ResidualCompletionConfig
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing


class _Attention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(x)


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = _Attention(width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, width * 2)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.attn(x) + self.ls_2(self.mlp(x))


class _Visual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _Visual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _fixture(source_depth=2, target_depth=4, width=5, seed=21):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    source_ft = deepcopy(source_base)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for block in source_ft.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(0.2 * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(0.2 * torch.randn_like(block.attn.out_proj.weight))
    generator = torch.Generator().manual_seed(seed + 2)
    images = torch.randn(8, 5, 4, generator=generator)
    data = DataLoader(TensorDataset(images, torch.arange(8)), batch_size=2, shuffle=False)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, target_base_sd


_SHARED_FIELDS = (
    "component",
    "position",
    "desired_norm",
    "effect_before_norm",
    "relative_residual_before",
    "relative_residual_after",
    "correction_rank",
    "residual_norm_before",
    "residual_norm_after",
    "reachable_residual_norm",
    "unreachable_residual_norm",
    "correction_norm",
    "bias_norm",
    "ridge",
    "n_rows",
)


def test_shared_solver_is_bit_identical_across_both_callers():
    source_depth, target_depth = 2, 4
    source_base, source_ft, target_base, data, target_base_sd = _fixture(source_depth, target_depth)

    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    # extend 2->4: i(j) = round(j*1/3) -> [0, 0, 1, 1].
    assert pairing.pairing == (0, 0, 1, 1)

    # Hand-build an ARIADNE "all scope" layout whose ancestry is FORCED equal
    # to the discrete pairing above, at every position (not just the odd
    # "inserted" ones ARIADNE's own doubling protocol would normally use).
    layout = {
        "final_blocks": [
            {"position": j, "source_orig_idx": pairing.pairing[j], "block_kind": "final"} for j in range(target_depth)
        ]
    }

    ariadne_config = ResidualCompletionConfig(
        enabled=True,
        mode="direct_target",
        target_scope="all",
        target_trajectory="step",
        cascade_order="independent",
        components=("attn.out_proj", "mlp.c_proj"),
        ridge_relative=0.05,
        num_batches=3,
    )
    references = capture_residual_references(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        num_batches=3,
        seed=17,
        device="cpu",
        target_scope="all",
    )
    ariadne_corrections, ariadne_diagnostics = complete_residuals_direct(
        target_base,
        target_base_sd,
        references,
        layout,
        data,
        config=ariadne_config,
        device="cpu",
    )

    direct_config = DirectResidualConfig(
        components=("attn.out_proj", "mlp.c_proj"),
        ridge_relative=0.05,
        num_batches=3,
        seed=17,
        cascade_order="bottom_top",  # deliberately different from ariadne_config's "independent":
        # proves cascade_order is a genuine no-op for the Direct Residual caller.
    )
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=3,
        seed=17,
        device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    direct_corrections, direct_diagnostics = fit_direct_residual(
        target_base,
        target_base_sd,
        captured,
        desired,
        pairing,
        config=direct_config,
        device="cpu",
    )

    # 1. Bit-identical fitted task vectors.
    assert set(ariadne_corrections) == set(direct_corrections)
    for key in ariadne_corrections:
        assert torch.equal(ariadne_corrections[key], direct_corrections[key]), key

    # 2. Bit-identical (to exact float equality on the diagnostic scalars the
    # two callers actually share -- ARIADNE's rows additionally carry
    # scope/trajectory/block_kind/target_coordinate/source_orig_idx, which
    # Direct Residual has no equivalent concept for and does not claim to
    # match; those extra fields are provenance bookkeeping, not solver output).
    assert len(ariadne_diagnostics) == len(direct_diagnostics)
    ariadne_by_key = {(row["position"], row["component"]): row for row in ariadne_diagnostics}
    direct_by_key = {(row["position"], row["component"]): row for row in direct_diagnostics}
    assert set(ariadne_by_key) == set(direct_by_key)
    for lookup_key, a_row in ariadne_by_key.items():
        d_row = direct_by_key[lookup_key]
        for field in _SHARED_FIELDS:
            a_val, d_val = a_row[field], d_row[field]
            if isinstance(a_val, torch.Tensor):
                assert torch.equal(a_val, d_val), (lookup_key, field)
            else:
                assert a_val == d_val, (lookup_key, field)
        torch.testing.assert_close(a_row["bias_correction"], d_row["bias_correction"], rtol=0, atol=0)
