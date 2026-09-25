"""DirectResidualConfig.depth_pairing (dr_ablation_v2_20260925 ablation switch 1).

Covers: the pairing-tuple transform itself (reversed/shift_plus1/shift_minus1,
including edge clipping) at both an extend-like (12 -> 24) and a shrink-like
(24 -> 12) depth; config validation (component_target='block_boundary' only);
that a non-relative pairing actually changes both D_j's source AND Q_j's
alignment target -- i.e. actually changes the fitted task vector tau, not just
the recorded metadata; and that resident/streaming agree under every mode.

Fixture duplicated from `tests/test_direct_residual_streaming_parity.py` per
this suite's no-cross-test-import convention.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    apply_depth_pairing_override,
    capture_paired_boundary_activations,
    compute_desired_effects,
    fit_direct_residual,
    fit_direct_residual_streaming,
    parse_direct_residual_config,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

# ---- pure pairing-tuple transform (no model needed) --------------------------


@pytest.mark.parametrize("source_depth,target_depth", [(12, 24), (24, 12)])
def test_reversed_pairing_formula(source_depth, target_depth):
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    rev = apply_depth_pairing_override(pairing, "reversed")
    assert rev.source_depth == pairing.source_depth
    assert rev.target_depth == pairing.target_depth
    expected = tuple(source_depth - 1 - i for i in pairing.pairing)
    assert rev.pairing == expected
    assert all(0 <= i < source_depth for i in rev.pairing)


@pytest.mark.parametrize("source_depth,target_depth", [(12, 24), (24, 12)])
@pytest.mark.parametrize("mode,delta", [("shift_plus1", 1), ("shift_minus1", -1)])
def test_shift_pairing_formula_and_clipping(source_depth, target_depth, mode, delta):
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    shifted = apply_depth_pairing_override(pairing, mode)
    expected = tuple(min(source_depth - 1, max(0, i + delta)) for i in pairing.pairing)
    assert shifted.pairing == expected
    assert all(0 <= i < source_depth for i in shifted.pairing)
    # Edge clipping actually engages somewhere in a 12<->24 sweep (pairing
    # touches both 0 and source_depth-1 at the depth extremes).
    if delta == 1:
        assert (
            0 in pairing.pairing
        )  # some j has pi(j)=0, clipped at +1 only if it also equals source_depth-1 (never here)
        assert shifted.pairing[-1] == min(source_depth - 1, pairing.pairing[-1] + 1)
    else:
        assert shifted.pairing[0] == max(0, pairing.pairing[0] - 1)


def test_relative_is_identity():
    pairing = DiscreteLayerPairing.compute(3, 7)
    assert apply_depth_pairing_override(pairing, "relative") is pairing


def test_unknown_depth_pairing_rejected():
    pairing = DiscreteLayerPairing.compute(3, 7)
    with pytest.raises(ValueError):
        apply_depth_pairing_override(pairing, "bogus")


# ---- config validation ---------------------------------------------------------


def test_config_rejects_unknown_depth_pairing():
    with pytest.raises(ValueError):
        parse_direct_residual_config({"depth_pairing": "bogus"})


@pytest.mark.parametrize("depth_pairing", ["reversed", "shift_plus1", "shift_minus1"])
def test_config_rejects_non_block_boundary_with_non_relative_pairing(depth_pairing):
    with pytest.raises(ValueError):
        parse_direct_residual_config(
            {"depth_pairing": depth_pairing, "component_target": "output_local", "components": ["mlp.c_fc"]}
        )


@pytest.mark.parametrize("depth_pairing", ["relative", "reversed", "shift_plus1", "shift_minus1"])
def test_config_accepts_block_boundary_with_any_pairing(depth_pairing):
    cfg = parse_direct_residual_config({"depth_pairing": depth_pairing})
    assert cfg.depth_pairing == depth_pairing


# ---- end-to-end: a non-relative pairing changes both D_j's source and tau -----


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
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, target_base_sd


def _fit_resident(setup, pairing, config):
    source_base, source_ft, target_base, data, target_base_sd = setup
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
    desired = compute_desired_effects(captured, pairing, residual_target=config.residual_target)
    corr, rows = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu"
    )
    return corr, rows


def _fit_streaming(setup, pairing, config):
    source_base, source_ft, target_base, data, target_base_sd = setup
    prepared = prepare_direct_residual_streaming(
        source_base,
        target_base,
        data,
        data,
        pairing,
        num_batches=config.num_batches,
        seed=config.seed,
        device="cpu",
        source_ft_model=source_ft,
    )
    corr, rows = fit_direct_residual_streaming(
        target_base,
        target_base_sd,
        source_base,
        source_ft,
        prepared,
        pairing,
        config=config,
        device="cpu",
    )
    return corr, rows


REGIMES = [pytest.param(2, 4, id="extend"), pytest.param(4, 2, id="shrink")]


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
@pytest.mark.parametrize("depth_pairing", ["reversed", "shift_plus1", "shift_minus1"])
def test_non_relative_pairing_changes_tau_resident(source_depth, target_depth, depth_pairing):
    setup = _setup(source_depth, target_depth)
    relative = DiscreteLayerPairing.compute(source_depth, target_depth)
    overridden = apply_depth_pairing_override(relative, depth_pairing)
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, depth_pairing=depth_pairing)
    base_config = DirectResidualConfig(num_batches=3, ridge_relative=0.05)

    corr_default, rows_default = _fit_resident(setup, relative, base_config)
    corr_overridden, rows_overridden = _fit_resident(setup, overridden, config)

    if overridden.pairing == relative.pairing:
        pytest.skip("degenerate depth pair: override left the pairing unchanged")
    assert set(corr_default) == set(corr_overridden)
    changed = any(not torch.equal(corr_default[k], corr_overridden[k]) for k in corr_default)
    assert changed, "depth_pairing override must change the fitted task vector"

    # Every diagnostics row's source_position matches the overridden pairing,
    # not the relative one -- both which source block defined D_j AND which
    # one Q_j was fit against are the SAME index (see apply_depth_pairing_
    # override's docstring: one swap changes both consistently).
    for j, i in enumerate(overridden.pairing):
        matching = [r for r in rows_overridden if r.get("position") == j]
        assert matching, j
        for row in matching:
            assert row["source_position"] == pytest.approx(float(i))


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
@pytest.mark.parametrize("depth_pairing", ["reversed", "shift_plus1", "shift_minus1"])
def test_non_relative_pairing_resident_streaming_agree(source_depth, target_depth, depth_pairing):
    setup = _setup(source_depth, target_depth)
    relative = DiscreteLayerPairing.compute(source_depth, target_depth)
    overridden = apply_depth_pairing_override(relative, depth_pairing)
    config_r = DirectResidualConfig(num_batches=3, ridge_relative=0.05, depth_pairing=depth_pairing)
    config_s = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, depth_pairing=depth_pairing, activation_storage="streaming"
    )
    corr_r, _rows_r = _fit_resident(setup, overridden, config_r)
    corr_s, _rows_s = _fit_streaming(setup, overridden, config_s)
    assert set(corr_r) == set(corr_s)
    for key in corr_r:
        assert torch.allclose(corr_r[key], corr_s[key], rtol=1e-5, atol=1e-6), key
