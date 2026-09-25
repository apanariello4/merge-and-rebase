"""STEP 0 of the dr_ablation_v2_20260925 campaign: golden hashes of the DEFAULT
Direct Residual path (every new switch at its default value), resident and
streaming, at an extend-like depth (2 -> 4) and a shrink-like depth (4 -> 2).

These pin `fit_direct_residual`/`fit_direct_residual_streaming`'s transported
task-vector tensors (sha256 over sorted state-dict keys, dtype, shape and raw
bytes -- `target_informed_runtime._task_vector_sha256`) on a tiny synthetic
CPU fixture, captured BEFORE any of the depth_pairing / alignment_map=
'random_isometry' / ridge_estimator='none' / fidelity_holdout ablation
switches existed. Every new switch defaults to its historical value
(depth_pairing='relative', alignment_map='polar', ridge_estimator=
'fixed_relative', fidelity_holdout=False), so a config that never sets any of
them must still reproduce these hashes byte for byte after the ablation
switches land -- this is the regression gate for that claim.

Fixture duplicated from `tests/test_direct_residual_streaming_parity.py` per
this suite's no-cross-test-import convention.
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
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.eval.target_informed_runtime import _task_vector_sha256
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


# (source_depth, target_depth, resident_sha256, streaming_sha256), captured
# with `DirectResidualConfig(num_batches=3, ridge_relative=0.05)` (every other
# field at its default) on this exact fixture, before the ablation switches
# were added. Both hashes are identical for this tiny fixture (single-CPU,
# no nondeterminism source in the accumulation order at this scale); a
# platform on which they diverge is a genuine floating-point regression
# worth investigating, not something this test should paper over with a
# tolerance.
GOLDEN = {
    "extend": {
        "depths": (2, 4),
        "resident_sha256": "485999a6658380cf0266d5a510ef43ec69f40de5a0940bed817104e9af694aee",
        "streaming_sha256": "485999a6658380cf0266d5a510ef43ec69f40de5a0940bed817104e9af694aee",
    },
    "shrink": {
        "depths": (4, 2),
        "resident_sha256": "48a7491929b8c070af483496962c27ceddbdd2ed42749d188e967fec7400860f",
        "streaming_sha256": "48a7491929b8c070af483496962c27ceddbdd2ed42749d188e967fec7400860f",
    },
}


def _default_config(**overrides):
    return DirectResidualConfig(num_batches=3, ridge_relative=0.05, **overrides)


def _resident_hash(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    config = _default_config()
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
    corr, _rows = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu"
    )
    return _task_vector_sha256(corr)


def _streaming_hash(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    config = _default_config(activation_storage="streaming")
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
    corr, _rows = fit_direct_residual_streaming(
        target_base,
        target_base_sd,
        source_base,
        source_ft,
        prepared,
        pairing,
        config=config,
        device="cpu",
    )
    return _task_vector_sha256(corr)


def test_default_resident_golden_hashes():
    for name, spec in GOLDEN.items():
        sd, td = spec["depths"]
        assert _resident_hash(sd, td) == spec["resident_sha256"], name


def test_default_streaming_golden_hashes():
    for name, spec in GOLDEN.items():
        sd, td = spec["depths"]
        assert _streaming_hash(sd, td) == spec["streaming_sha256"], name


def test_default_resident_and_streaming_agree_with_each_other():
    """Sanity check on GOLDEN itself: the two recorded hashes per regime match."""
    for spec in GOLDEN.values():
        assert spec["resident_sha256"] == spec["streaming_sha256"]
