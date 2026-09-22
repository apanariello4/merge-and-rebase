"""`DirectResidualConfig.cascade_order` is a documented, provable no-op.

Extends the backward-equivalence argument
`tests/test_cascade_order_backward_equivalence_20260922.py` makes for
ARIADNE's `target_residual_completion.cascade_order` (there: `top_bottom`
happens to coincide with `independent` because nothing downstream can have
moved an upstream block's input) one step further for Direct Residual's own
field: since Direct Residual never mounts a correction before fitting the
next position AT ALL -- not cross-position, not even within one position's
own attn.out_proj -> mlp.c_proj pair -- by construction (see the module
docstring in `direct_residual.py`), there is no cascade to order in the
first place. `fit_direct_residual` forces the shared solver
(`target_informed_runtime._fit_direct_target_position`) into independent
(no-mount) mode on every call, regardless of what `cascade_order` value the
caller's `DirectResidualConfig` carries. This test proves that directly:
fitted corrections and diagnostics are IDENTICAL across every legal
`cascade_order` value.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    capture_paired_boundary_activations,
    compute_desired_effects,
    fit_direct_residual,
)
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


def _fixture(source_depth=2, target_depth=4, width=5, seed=71):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    from copy import deepcopy

    source_ft = deepcopy(source_base)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for block in source_ft.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(0.2 * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(0.2 * torch.randn_like(block.attn.out_proj.weight))
    generator = torch.Generator().manual_seed(seed + 2)
    images = torch.randn(6, 5, 4, generator=generator)
    data = DataLoader(TensorDataset(images, torch.arange(6)), batch_size=2, shuffle=False)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, target_base_sd


def _fit_with_cascade_order(order):
    source_base, source_ft, target_base, data, target_base_sd = _fixture()
    pairing = DiscreteLayerPairing.compute(2, 4)
    config = DirectResidualConfig(
        components=("attn.out_proj", "mlp.c_proj"),
        ridge_relative=0.05,
        num_batches=3,
        seed=81,
        cascade_order=order,
    )
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=config.num_batches,
        seed=config.seed,
        device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    return fit_direct_residual(
        target_base,
        target_base_sd,
        captured,
        desired,
        pairing,
        config=config,
        device="cpu",
    )


@pytest.mark.parametrize("order", ["independent", "bottom_top", "top_bottom"])
def test_cascade_order_parses_to_a_distinct_config_value(order):
    """The config value itself stays distinct; only its effect on the fit is a no-op."""
    config = DirectResidualConfig(cascade_order=order)
    assert config.cascade_order == order


def test_cascade_order_is_a_no_op_on_fitted_corrections():
    baseline_corrections, baseline_diagnostics = _fit_with_cascade_order("independent")
    for order in ("bottom_top", "top_bottom"):
        corrections, diagnostics = _fit_with_cascade_order(order)
        assert set(corrections) == set(baseline_corrections)
        for key in corrections:
            assert torch.equal(corrections[key], baseline_corrections[key]), (order, key)
        assert len(diagnostics) == len(baseline_diagnostics)
        baseline_by_key = {(r["position"], r["component"]): r for r in baseline_diagnostics}
        for row in diagnostics:
            base_row = baseline_by_key[(row["position"], row["component"])]
            for field in (
                "relative_residual_before",
                "relative_residual_after",
                "residual_norm_before",
                "residual_norm_after",
                "correction_rank",
                "ridge",
                "correction_norm",
                "n_rows",
            ):
                assert row[field] == base_row[field], (order, row["position"], field)
