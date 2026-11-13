"""Tests for the opt-in vision ``depth_fit="sequential"`` + ``block_split="backfit"`` Ariadne mode.

attn.out_proj + mlp.c_proj of block j are fitted jointly by the intra-block Gauss-Seidel backfit against the
sequential residual target ``E_j = D_j - U_j``. Fixture duplicated from the sibling test modules per this suite's
no-cross-test-import convention.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.alignment import compute_desired_effects
from merge_and_rebase.rebase.methods._ariadne.capture import capture_paired_boundary_activations
from merge_and_rebase.rebase.methods._ariadne.config import parse_direct_residual_config
from merge_and_rebase.rebase.methods._ariadne.fit import fit_direct_residual
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_sequential_streaming,
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


def _setup(source_depth, target_depth, width=5, seed=11):
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
    images = torch.randn(6, 5, 4, generator=generator)
    data = DataLoader(TensorDataset(images, torch.arange(6)), batch_size=2, shuffle=False)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, sd


OD = ("attn.out_proj", "mlp.c_proj")
SEQ_BF = dict(
    activation_storage="streaming",
    depth_fit="sequential",
    block_split="backfit",
    num_batches=3,
    ridge_relative=0.05,
    components=OD,
    backfit_max_iters=5,
)
RES_BF = dict(block_split="backfit", num_batches=3, ridge_relative=0.05, components=OD, backfit_max_iters=5)
REGIMES = [pytest.param(2, 4, id="extend"), pytest.param(4, 2, id="shrink"), pytest.param(3, 3, id="same_arch")]


def _sha(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(d[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _fit_seq(setup, overrides=None, **kwargs):
    source_base, source_ft, target_base, data, pairing, sd = setup
    cfg = parse_direct_residual_config({**SEQ_BF, **(overrides or {})})
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=cfg.num_batches, seed=cfg.seed, device="cpu"
    )
    return fit_direct_residual_sequential_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu", **kwargs
    )


def _fit_resident(setup):
    source_base, source_ft, target_base, data, pairing, sd = setup
    cfg = parse_direct_residual_config(RES_BF)
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    return fit_direct_residual(target_base, sd, captured, desired, pairing, config=cfg, device="cpu")


def test_config_accepts_od_sequential_backfit():
    cfg = parse_direct_residual_config(SEQ_BF)
    assert (cfg.depth_fit, cfg.block_split, cfg.activation_storage) == ("sequential", "backfit", "streaming")


@pytest.mark.parametrize(
    "overrides",
    [
        {"components": ("mlp.c_proj",)},
        {"components": ("attn.out_proj",)},
        {"components": ("attn.out_proj", "mlp.c_proj", "mlp.c_fc")},
        {"activation_storage": "resident"},
        {"pair_target_split": "interpolate"},
        {"sequential_points": "end_end"},
        {"block_split": "joint"},
        {"backfit_max_iters": 0},
        {"backfit_tol": 0.0},
        {"tv_scaling": "global"},
    ],
)
def test_config_rejects_other_combinations(overrides):
    with pytest.raises(ValueError):
        parse_direct_residual_config({**SEQ_BF, **overrides})


def test_independent_streaming_backfit_still_rejected():
    with pytest.raises(ValueError):
        parse_direct_residual_config({**SEQ_BF, "depth_fit": "independent"})


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_block0_matches_resident_independent_backfit(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    seq, _ = _fit_seq(setup)
    (
        res,
        _,
    ) = _fit_resident(setup)[:2]
    keys0 = [k for k in res if ".0." in k]
    assert keys0 and all(k in seq for k in keys0)
    bit_equal = all(torch.equal(seq[k], res[k]) for k in keys0)
    if not bit_equal:
        for k in keys0:
            scale = res[k].abs().max().item() + 1e-30
            assert (seq[k] - res[k]).abs().max().item() <= 1e-6 * scale, (
                k,
                "max abs diff",
                (seq[k] - res[k]).abs().max(),
            )
    print(f"block0 seq-backfit vs resident backfit ({source_depth}->{target_depth}): bit_equal={bit_equal}")


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_backfit_objective_non_increasing_and_rows(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    corrections, rows = _fit_seq(setup)
    assert {r["position"] for r in rows} == set(range(target_depth))
    for r in rows:
        assert r["depth_fit"] == "sequential" and r["block_split"] == "backfit"
        assert "upstream_effect_norm" in r and "source_position" in r
        assert r["block_replay_bitwise"] is True
        trace = r["backfit_j_trace"]
        j0 = r["desired_norm"] ** 2  # J(Delta=0) = ||E_j||^2
        assert trace[0] <= j0 * (1 + 1e-9)
        assert all(b <= a * (1 + 1e-9) for a, b in zip(trace, trace[1:], strict=False))
    assert next(r for r in rows if r["position"] == 0)["upstream_effect_norm"] == 0.0
    assert any(r["upstream_effect_norm"] > 0 for r in rows if r["position"] > 0)
    assert all(k.endswith((".weight", ".bias")) for k in corrections)
    assert sum(k.endswith(".bias") for k in corrections) == 2 * target_depth


def test_target_state_restored_and_deterministic():
    setup = _setup(2, 4)
    target_base, sd = setup[2], setup[5]
    first, _ = _fit_seq(setup)
    assert all(torch.equal(v, target_base.state_dict()[k]) for k, v in sd.items())
    second, _ = _fit_seq(setup)
    assert _sha(first) == _sha(second)


def test_decoder_family_adapter_raises():
    with pytest.raises(NotImplementedError):
        _fit_seq(_setup(2, 2), family_adapter=object())


def test_default_sequential_unchanged_by_new_field_values():
    cfg = parse_direct_residual_config({**SEQ_BF, "block_split": "none", "components": ("mlp.c_proj",)})
    assert cfg.block_split == "none"
