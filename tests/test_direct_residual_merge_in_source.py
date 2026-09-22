"""Confirms `capture_paired_boundary_activations`/`fit_direct_residual` make
no hidden per-task assumption.

`merge_in_source_then_fit`'s actual merge-once orchestration lives in
`vision_rebase.py` (out of scope here, per the implementation plan -- this
module only has to support being CALLED once against any
(source_base_model, source_ft_model) pair, whether that pair is a genuine
single-task fine-tune or an already-merged (e.g. summed) pair). What this
test file CAN verify, entirely within `direct_residual.py`'s own surface, is
that nothing here special-cases N=1 tasks: calling the functions once against
a synthetic "merged" delta (the sum of two synthetic per-task deltas)
succeeds and produces finite output exactly as calling them once against an
unsummed, single-task delta does.
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


def _delta(model, seed, scale=0.15):
    """A per-task delta state dict: fresh Gaussian noise on every parameter."""
    torch.manual_seed(seed)
    return {k: scale * torch.randn_like(v) for k, v in model.state_dict().items()}


def _apply(model, base_state, delta):
    tuned = deepcopy(model)
    merged = {k: base_state[k] + delta.get(k, torch.zeros_like(v)) for k, v in base_state.items()}
    tuned.load_state_dict(merged, strict=True)
    return tuned


def _run(source_ft_model, source_depth=2, target_depth=4, width=5, seed=51):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    images = torch.randn(6, 5, 4, generator=generator)
    data = DataLoader(TensorDataset(images, torch.arange(6)), batch_size=2, shuffle=False)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, seed=61)

    captured = capture_paired_boundary_activations(
        source_base,
        source_ft_model,
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


def test_single_task_delta_runs_and_is_finite():
    torch.manual_seed(51)
    source_base = _Model(5, 2).eval()
    base_state = {k: v.clone() for k, v in source_base.state_dict().items()}
    single = _apply(source_base, base_state, _delta(source_base, seed=52))
    corrections, diagnostics = _run(single)
    assert corrections
    for value in corrections.values():
        assert torch.isfinite(value).all()
    assert diagnostics


def test_summed_two_task_delta_runs_identically_in_shape_and_finiteness():
    """A 'merge_in_source_then_fit'-style summed delta is just another
    (source_base, source_ft) pair to this module -- there is no code path
    here that inspects how many tasks contributed to it."""
    torch.manual_seed(51)
    source_base = _Model(5, 2).eval()
    base_state = {k: v.clone() for k, v in source_base.state_dict().items()}
    delta_a = _delta(source_base, seed=53)
    delta_b = _delta(source_base, seed=54)
    summed_delta = {k: delta_a[k] + delta_b[k] for k in base_state}
    merged = _apply(source_base, base_state, summed_delta)

    corrections, diagnostics = _run(merged)
    assert corrections
    for value in corrections.values():
        assert torch.isfinite(value).all()
    assert diagnostics
    # 2 components (default: attn.out_proj, mlp.c_proj) x 4 target positions.
    assert len(diagnostics) == 8
