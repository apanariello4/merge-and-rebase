"""Tests for Direct Residual's ``component_target`` option.

Only ``component_target='block_boundary'`` (default) is supported: every
requested component is fit against the SAME block-boundary target. The former
``'output_local'``/``'output_total'`` modes were retired (closed dead ends) and
must raise at parse time.

The null control is pinned at sha256-hash level against golden values recorded
at HEAD ``d77bbce`` via a ``git worktree``, reusing the fixture of
``tests/test_direct_residual_fit.py``. This module has no cross-test imports.
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
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig, parse_direct_residual_config
from merge_and_rebase.rebase.methods._ariadne.fit import fit_direct_residual
from merge_and_rebase.utils.cost_accounting import PhaseCostRecorder, recording

# --------------------------------------------------------------------------
# Golden hashes for the untouched block_boundary path, recorded at HEAD
# d77bbce (clean tree, via `git worktree add ... d77bbce`) on the fixture of
# test_direct_residual_fit.py, BEFORE any of this ablation's code was written.
# --------------------------------------------------------------------------
_GOLDEN_EXTEND = "e6f02b6922df921f802cb02ca245a60f6f988d44a70a2425061c7cd10f2e6dbc"
_GOLDEN_SHRINK = "111eafdb947b940ebee0366366c5d726830ff6661d8f78222f0a1c67f13dfe94"
_GOLDEN_SAME_ARCH = "3b0600d6b14887867dc83d8d026903dea27bf0de20baaaa5055848fb0385c9d3"


def _state_dict_sha256(d: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(d.keys()):
        h.update(key.encode())
        h.update(d[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Shared synthetic fixture (copied from test_direct_residual_fit.py -- this
# module has no cross-test imports).
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


def _fit(source_depth, target_depth, config=None, **setup_kwargs):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(
        source_depth, target_depth, **setup_kwargs
    )
    config = config or DirectResidualConfig(num_batches=3, ridge_relative=0.05)
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu",
    )
    return corrections, diagnostics


# --------------------------------------------------------------------------
# 1. Golden-hash pinning: the block_boundary path is untouched.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth, golden", [
    (2, 4, _GOLDEN_EXTEND), (4, 2, _GOLDEN_SHRINK), (3, 3, _GOLDEN_SAME_ARCH),
])
def test_golden_hash_block_boundary(source_depth, target_depth, golden):
    corrections, _diag = _fit(source_depth, target_depth)
    assert _state_dict_sha256(corrections) == golden


@pytest.mark.parametrize("source_depth, target_depth, golden", [
    (2, 4, _GOLDEN_EXTEND), (4, 2, _GOLDEN_SHRINK), (3, 3, _GOLDEN_SAME_ARCH),
])
def test_golden_hash_block_boundary_under_cost_recording(source_depth, target_depth, golden):
    # Cost accounting only synchronizes and reads counters: the pinned default
    # path is bit-identical with an active recorder.
    recorder = PhaseCostRecorder("cpu")
    with recording(recorder):
        corrections, _diag = _fit(source_depth, target_depth)
    assert _state_dict_sha256(corrections) == golden
    phases = recorder.summary()["phases"]
    assert phases["activation_collection"]["segments"] > 0
    assert phases["transformation"]["segments"] > 0


def test_component_target_defaults_to_block_boundary():
    cfg = DirectResidualConfig()
    assert cfg.component_target == "block_boundary"
    assert cfg.realization_diagnostics is False
    parsed = parse_direct_residual_config(None)
    assert parsed.component_target == "block_boundary"


# --------------------------------------------------------------------------
# 2. Parser coverage.
# --------------------------------------------------------------------------


def test_internal_components_rejected_for_block_boundary():
    with pytest.raises(ValueError, match="unknown components"):
        parse_direct_residual_config({"components": ["attn.q_proj", "mlp.c_proj"]})


def test_out_proj_alone_is_allowed_for_block_boundary():
    """Direct Residual is always independent (no cascade), so the P1-era
    "c_proj anchor" restriction does not apply here; DT-O is a legal arm."""
    cfg = parse_direct_residual_config({"components": ["attn.out_proj"]})
    assert cfg.components == ("attn.out_proj",)


def test_c_proj_alone_is_still_allowed_for_block_boundary():
    cfg = parse_direct_residual_config({"components": ["mlp.c_proj"]})
    assert cfg.components == ("mlp.c_proj",)


@pytest.mark.parametrize("retired", ["output_local", "output_total"])
def test_retired_component_targets_raise_at_parse_time(retired):
    with pytest.raises(ValueError, match="retired"):
        parse_direct_residual_config({"component_target": retired})
    with pytest.raises(ValueError, match="retired"):
        parse_direct_residual_config({"component_target": retired, "components": ["attn.out_proj"]})

