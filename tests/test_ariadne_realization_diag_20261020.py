"""Realization-drift diagnostic (`eval/diagnostics/ariadne_realization`): toy CPU checks.

(a) alpha=0 mounts nothing: dT_j = 0 -> rel_error 1, ratio 0; (b) alpha=1 rows equal the existing
`measure_direct_residual_realization_streaming` norms for the same delta; (c) an existing out_dir is refused.
Fixture duplicated from `tests/test_direct_residual_streaming_parity.py` per the no-cross-test-import convention.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.diagnostics.ariadne_realization import measure_for_prepared, run
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_streaming,
    measure_streaming_realization_for,
    prepare_direct_residual_streaming,
)


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


N_CLASSES = 6


def _loader(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n) % N_CLASSES), batch_size=2, shuffle=False)


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


def _fit(source_depth, target_depth):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(source_depth, target_depth)
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, activation_storage="streaming")
    prepared = prepare_direct_residual_streaming(
        source_base,
        target_base,
        data,
        data,
        pairing,
        num_batches=3,
        seed=config.seed,
        device="cpu",
        source_ft_model=source_ft,
    )
    corr, _rows = fit_direct_residual_streaming(
        target_base, target_base_sd, source_base, source_ft, prepared, pairing, config=config, device="cpu"
    )
    return source_base, source_ft, target_base, target_base_sd, prepared, pairing, corr, config


def _measure(fit, alphas):
    source_base, source_ft, target_base, target_base_sd, prepared, pairing, corr, config = fit
    return measure_for_prepared(
        target_base,
        target_base_sd,
        source_base,
        source_ft,
        prepared,
        pairing,
        corr,
        alphas,
        residual_target=config.residual_target,
        device="cpu",
    )


@pytest.mark.parametrize("depths", [(2, 4), (4, 2)], ids=["extend", "shrink"])
def test_alpha_zero_realizes_nothing(depths):
    rows = _measure(_fit(*depths), [0.0])[0.0]
    for row in rows.values():
        assert row["delta_norm"] == 0.0 and row["ratio"] == 0.0 and row["cosine"] == 0.0
        assert row["rel_error"] == pytest.approx(1.0, abs=1e-9)
        assert row["desired_norm"] > 0


@pytest.mark.parametrize("depths", [(2, 4), (4, 2), (3, 3)], ids=["extend", "shrink", "same"])
def test_alpha_one_matches_existing_measurement(depths):
    fit = _fit(*depths)
    source_base, source_ft, target_base, target_base_sd, prepared, pairing, corr, config = fit
    ref = measure_streaming_realization_for(
        target_base, target_base_sd, source_base, source_ft, prepared, pairing, config=config, device="cpu"
    )(corr)
    rows = _measure(fit, [1.0, 0.5])
    for j, row in ref.items():
        mine = rows[1.0][j]
        assert mine["desired_norm"] == row["desired_norm"]
        assert mine["delta_norm"] == row["joint_delta_norm"]
        assert mine["ratio"] == row["joint_delta_norm_over_desired"]
        assert mine["rel_error"] == row["block_realized_target_error"]
        assert -1.0 - 1e-9 <= mine["cosine"] <= 1.0 + 1e-9
    assert any(rows[0.5][j]["delta_norm"] != rows[1.0][j]["delta_norm"] for j in rows[1.0])


def test_out_dir_refused(tmp_path):
    with pytest.raises(FileExistsError):
        run({"run_summary": "unused", "alphas": [1.0], "out_dir": str(tmp_path)}, "cpu")
