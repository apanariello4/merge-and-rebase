"""Shape/well-definedness smoke test for Direct Residual, at Vision8-like
proportions (ViT-B/16 -> ViT-L/14 style extend/shrink, plus same-arch), using
tiny synthetic CLIP-shaped models rather than real checkpoints. Confirms no
shape errors, finite outputs, and the correct component key set in
`target_corrections`, across all three depth/width regimes this method is
designed to cover.
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
                    ("c_fc", torch.nn.Linear(width, width * 4)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(width * 4, width)),
                ]
            )
        )
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.attn(x) + self.ls_2(self.mlp(x))


class _Visual(torch.nn.Module):
    def __init__(self, width, depth, in_dim=6):
        super().__init__()
        self.input = torch.nn.Linear(in_dim, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, width, depth, in_dim=6):
        super().__init__()
        self.visual = _Visual(width, depth, in_dim=in_dim)

    def encode_image(self, x):
        return self.visual(x)


def _tuned_copy(model, seed):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(0.1 * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(0.1 * torch.randn_like(block.attn.out_proj.weight))
    return tuned


# (source_width, source_depth, target_width, target_depth) -- Vision8-like
# proportions scaled down: ViT-B/16 (width 768, depth 12) -> ViT-L/14 (width
# 1024, depth 24) becomes a small width bump plus depth doubling here.
REGIMES = [
    pytest.param(6, 4, 8, 8, id="extend_b16_to_l14_like"),
    pytest.param(8, 8, 6, 4, id="shrink_l14_to_b16_like"),
    pytest.param(6, 4, 6, 4, id="same_arch"),
]


@pytest.mark.parametrize("source_width, source_depth, target_width, target_depth", REGIMES)
def test_shapes_and_finiteness(source_width, source_depth, target_width, target_depth):
    torch.manual_seed(31)
    source_base = _Model(source_width, source_depth).eval()
    target_base = _Model(target_width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed=32)
    generator = torch.Generator().manual_seed(33)
    images = torch.randn(8, 5, 6, generator=generator)
    data = DataLoader(TensorDataset(images, torch.arange(8)), batch_size=4, shuffle=False)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}

    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    assert len(pairing.pairing) == target_depth
    assert all(0 <= i < source_depth for i in pairing.pairing)

    config = DirectResidualConfig(
        components=("attn.out_proj", "mlp.c_proj"),
        ridge_relative=0.05,
        num_batches=2,
        seed=41,
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
    corrections, diagnostics = fit_direct_residual(
        target_base,
        target_base_sd,
        captured,
        desired,
        pairing,
        config=config,
        device="cpu",
    )

    expected_keys = set()
    for j in range(target_depth):
        expected_keys.add(f"visual.transformer.resblocks.{j}.mlp.c_proj.weight")
        expected_keys.add(f"visual.transformer.resblocks.{j}.mlp.c_proj.bias")
        expected_keys.add(f"visual.transformer.resblocks.{j}.attn.out_proj.weight")
        expected_keys.add(f"visual.transformer.resblocks.{j}.attn.out_proj.bias")
    assert set(corrections) == expected_keys

    for key, value in corrections.items():
        assert value.shape == target_base_sd[key].shape, key
        assert torch.isfinite(value).all(), key

    assert len(diagnostics) == 2 * target_depth
    for row in diagnostics:
        assert torch.isfinite(torch.tensor(row["residual_norm_after"]))
        assert row["position"] in range(target_depth)
