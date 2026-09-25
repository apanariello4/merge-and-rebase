"""Device-parametrized coverage for Direct Residual's block_boundary/backfit/
output_local fit paths.

This login node has no GPU (see the module-level skip below); every ``cuda``
case is collected but skipped here and is meant to be run on a GPU node. The
node id list this file produces (via ``pytest --collect-only -q``) is exactly
what ``gpu_tests.sh`` replays.

Motivation (see the reviewed bug in
``target_informed_runtime._fit_block_boundary_backfit``): ``local_block`` is
moved to ``device`` before ``_component_weight_bias`` reads its parameters,
so a naive implementation could let ``base[c]`` inherit that device while
every fitted delta (``deltas[c]``, always ``.cpu()``'d off
``ResidualSufficientStatistics.solve()``) stays on CPU. ``base[c2][0] + w2``
would then mix a CUDA tensor with a CPU one -- a ``RuntimeError`` that only
a CUDA run can ever surface (CPU + CPU never errors regardless of which
tensor came from where). The fix keeps every "delta"-shaped tensor
(``base[c]`` included) on CPU float32 throughout, moving onto ``device``
only inside ``_mount_component``. These tests exercise exactly the code
path that would have raised, on a real CUDA device, so a regression here is
caught rather than silently passing because CI is CPU-only.

No cross-test imports (repo convention for this test family): fixtures are
duplicated, not shared, with ``tests/test_direct_residual_component_coverage.py``
and ``tests/test_direct_residual_backfit.py``.
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
from merge_and_rebase.eval.target_residual_completion import order_components
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")),
]

# --------------------------------------------------------------------------
# Fixture A: plain (non-stock) attention wrapper -- used for block_boundary
# {none, backfit}, identical shape to test_direct_residual_backfit.py's.
# --------------------------------------------------------------------------


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


def _loader(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n)), batch_size=2, shuffle=False)


def _tuned_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def _setup(source_depth, target_depth, width=5, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


def _fit(source_depth, target_depth, config, device, **setup_kwargs):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(
        source_depth, target_depth, **setup_kwargs
    )
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device=device,
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device=device,
    )
    return corrections, diagnostics


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
def test_block_boundary_none_is_finite_on_device(source_depth, target_depth, device):
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"), block_split="none",
    )
    corrections, diagnostics = _fit(source_depth, target_depth, cfg, device)
    for value in corrections.values():
        assert torch.isfinite(value).all()
        assert value.device.type == "cpu"
    assert diagnostics


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("components", [("attn.out_proj",), ("mlp.c_proj",), ("attn.out_proj", "mlp.c_proj")])
def test_backfit_is_finite_on_device(components, device):
    """Regression for the base[c]-device bug: on CUDA, ``base[c2][0] + w2``
    inside ``_fit_block_boundary_backfit`` used to mix a CUDA tensor
    (``base``, read off the ``local_block`` that is moved ``.to(device)``)
    with a CPU tensor (``w2``, the fitted delta) whenever more than one
    component was requested (the single-component case never mounts another
    component, so it never hit the mixed add)."""
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=components, block_split="backfit", backfit_max_iters=3,
    )
    corrections, diagnostics = _fit(2, 4, cfg, device)
    for value in corrections.values():
        assert torch.isfinite(value).all()
        assert value.device.type == "cpu"
    for row in diagnostics:
        assert isinstance(row["backfit_n_sweeps"], int)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block_split", ["none", "backfit"])
def test_realization_diagnostics_finite_on_device(block_split, device):
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"),
        block_split=block_split, realization_diagnostics=True,
    )
    _corrections, diagnostics = _fit(2, 4, cfg, device)
    expected = {
        "fit_relative_residual", "target_norm", "update_norm", "relative_update_norm", "realized_target_norm_ratio",
    }
    for row in diagnostics:
        assert expected <= set(row), row.keys()
        for key in expected:
            assert torch.isfinite(torch.tensor(float(row[key])))


# --------------------------------------------------------------------------
# Fixture B: stock nn.MultiheadAttention -- used for component_target=
# 'output_local' (needed for internal q/k/v/c_fc components).
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
    source_ft = deepcopy(source_base)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for block in source_ft.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(0.2 * torch.randn_like(block.mlp.c_proj.weight))
            block.mlp.c_fc.weight.add_(0.15 * torch.randn_like(block.mlp.c_fc.weight))
            block.attn.out_proj.weight.add_(0.2 * torch.randn_like(block.attn.out_proj.weight))
            block.attn.in_proj_weight.add_(0.1 * torch.randn_like(block.attn.in_proj_weight))
    data = _stock_loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


_ALL_SIX = ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj")


def _stock_output_local_fit(source_depth, target_depth, components, device, **overrides):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _stock_setup(source_depth, target_depth)
    params = dict(num_batches=3, ridge_relative=0.05, component_target="output_local", components=tuple(components))
    params.update(overrides)
    config = DirectResidualConfig(**params)
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device=device,
        component_inputs=order_components(config.components),
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device=device,
    )
    return corrections, diagnostics


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
def test_output_local_all_six_components_finite_on_device(source_depth, target_depth, device):
    corrections, diagnostics = _stock_output_local_fit(source_depth, target_depth, _ALL_SIX, device)
    assert corrections
    for key, value in corrections.items():
        assert torch.isfinite(value).all(), key
        assert value.device.type == "cpu"
    assert {row["component"] for row in diagnostics} == set(_ALL_SIX)


@pytest.mark.parametrize("device", DEVICES)
def test_output_local_realization_diagnostics_finite_on_device(device):
    _corrections, diagnostics = _stock_output_local_fit(
        2, 4, _ALL_SIX, device, realization_diagnostics=True,
    )
    expected = {
        "fit_relative_residual", "target_norm", "update_norm", "relative_update_norm", "realized_target_norm_ratio",
    }
    for row in diagnostics:
        assert expected <= set(row), row.keys()
        for key in expected:
            assert torch.isfinite(torch.tensor(float(row[key])))
