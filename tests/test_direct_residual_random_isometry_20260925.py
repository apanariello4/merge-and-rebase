"""DirectResidualConfig.alignment_map='random_isometry' (dr_ablation_v2_20260925
ablation switch 2): replaces the polar Procrustes map Q_j by a random partial
isometry of the SAME shape/orientation, seeded deterministically from
(alignment_seed, position). Keeps the same centering (mu_s/mu_t) as polar.

Covers: config validation (uniform row weighting only, rejected with
procrustes_source='gradient' or non-block_boundary component_target, rejected
with a non-default residual_target); the random map's shape/orthonormality and
determinism given the seed; that it actually changes the fitted tau; and
resident/streaming agreement (same per-block seed derivation on both paths).
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    _derive_block_seed,
    _random_isometry_map,
    capture_paired_boundary_activations,
    compute_desired_effects,
    fit_direct_residual,
    fit_direct_residual_streaming,
    parse_direct_residual_config,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

# ---- config validation -----------------------------------------------------------


def test_config_accepts_random_isometry():
    cfg = parse_direct_residual_config({"alignment_map": "random_isometry"})
    assert cfg.alignment_map == "random_isometry"
    assert cfg.alignment_seed == 0


def test_config_random_isometry_requires_uniform_row_weighting():
    with pytest.raises(ValueError):
        parse_direct_residual_config({"alignment_map": "random_isometry", "alignment_row_weighting": "cls_balanced"})


def test_config_random_isometry_rejected_with_gradient_procrustes():
    with pytest.raises(ValueError):
        parse_direct_residual_config({"alignment_map": "random_isometry", "procrustes_source": "gradient"})


def test_config_random_isometry_rejected_with_non_block_boundary_target():
    with pytest.raises(ValueError):
        parse_direct_residual_config(
            {"alignment_map": "random_isometry", "component_target": "output_local", "components": ["mlp.c_fc"]}
        )


def test_config_random_isometry_rejected_with_endpoint_target():
    with pytest.raises(ValueError):
        parse_direct_residual_config({"alignment_map": "random_isometry", "residual_target": "transported_endpoint"})


def test_config_alignment_seed_must_be_int():
    with pytest.raises(ValueError):
        parse_direct_residual_config({"alignment_seed": 1.5})


# ---- pure map construction ---------------------------------------------------------


def test_random_isometry_map_shape_and_orthonormality_tall():
    q = _random_isometry_map((3, 7), seed=42)
    assert q.shape == (3, 7)
    # d_source < d_target: rows are orthonormal (Q Q^T = I).
    assert torch.allclose(q @ q.T, torch.eye(3, dtype=q.dtype), atol=1e-9)


def test_random_isometry_map_shape_and_orthonormality_wide():
    q = _random_isometry_map((7, 3), seed=42)
    assert q.shape == (7, 3)
    # d_source > d_target: columns are orthonormal (Q^T Q = I).
    assert torch.allclose(q.T @ q, torch.eye(3, dtype=q.dtype), atol=1e-9)


def test_random_isometry_map_deterministic_given_seed():
    q1 = _random_isometry_map((5, 5), seed=7)
    q2 = _random_isometry_map((5, 5), seed=7)
    assert torch.equal(q1, q2)


def test_random_isometry_map_differs_across_seeds():
    q1 = _random_isometry_map((5, 5), seed=7)
    q2 = _random_isometry_map((5, 5), seed=8)
    assert not torch.equal(q1, q2)


def test_derive_block_seed_deterministic_and_position_sensitive():
    a = _derive_block_seed(3, 0)
    b = _derive_block_seed(3, 0)
    c = _derive_block_seed(3, 1)
    d = _derive_block_seed(4, 0)
    assert a == b
    assert a != c
    assert a != d
    assert isinstance(a, int) and a >= 0


# ---- end-to-end Direct Residual fit --------------------------------------------


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


REGIMES = [pytest.param(2, 4, id="extend"), pytest.param(4, 2, id="shrink")]


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_random_isometry_changes_tau_resident(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    base_config = DirectResidualConfig(num_batches=3, ridge_relative=0.05)
    ri_config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, alignment_map="random_isometry")

    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=base_config.num_batches,
        seed=base_config.seed,
        device="cpu",
    )
    desired_default = compute_desired_effects(
        captured, pairing, residual_target=base_config.residual_target, alignment_map=base_config.alignment_map
    )
    desired_ri = compute_desired_effects(
        captured,
        pairing,
        residual_target=ri_config.residual_target,
        alignment_map=ri_config.alignment_map,
        alignment_seed=ri_config.alignment_seed,
    )
    corr_default, _r1 = fit_direct_residual(
        target_base, target_base_sd, captured, desired_default, pairing, config=base_config, device="cpu"
    )
    corr_ri, _r2 = fit_direct_residual(
        target_base, target_base_sd, captured, desired_ri, pairing, config=ri_config, device="cpu"
    )
    assert set(corr_default) == set(corr_ri)
    changed = any(not torch.equal(corr_default[k], corr_ri[k]) for k in corr_default)
    assert changed


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_random_isometry_deterministic_given_alignment_seed(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, alignment_map="random_isometry", alignment_seed=5)
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
    desired1 = compute_desired_effects(
        captured,
        pairing,
        residual_target=config.residual_target,
        alignment_map=config.alignment_map,
        alignment_seed=config.alignment_seed,
    )
    desired2 = compute_desired_effects(
        captured,
        pairing,
        residual_target=config.residual_target,
        alignment_map=config.alignment_map,
        alignment_seed=config.alignment_seed,
    )
    for j in desired1:
        for a, b in zip(desired1[j], desired2[j], strict=True):
            assert torch.equal(a, b)
    corr1, _ = fit_direct_residual(
        target_base, target_base_sd, captured, desired1, pairing, config=config, device="cpu"
    )
    corr2, _ = fit_direct_residual(
        target_base, target_base_sd, captured, desired2, pairing, config=config, device="cpu"
    )
    for key in corr1:
        assert torch.equal(corr1[key], corr2[key])


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_random_isometry_resident_streaming_agree(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    config_r = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, alignment_map="random_isometry", alignment_seed=3
    )
    config_s = DirectResidualConfig(
        num_batches=3,
        ridge_relative=0.05,
        alignment_map="random_isometry",
        alignment_seed=3,
        activation_storage="streaming",
    )
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=config_r.num_batches,
        seed=config_r.seed,
        device="cpu",
    )
    desired = compute_desired_effects(
        captured,
        pairing,
        residual_target=config_r.residual_target,
        alignment_map=config_r.alignment_map,
        alignment_seed=config_r.alignment_seed,
    )
    corr_r, _rows_r = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config_r, device="cpu"
    )
    prepared = prepare_direct_residual_streaming(
        source_base,
        target_base,
        data,
        data,
        pairing,
        num_batches=config_s.num_batches,
        seed=config_s.seed,
        device="cpu",
        source_ft_model=source_ft,
        alignment_map=config_s.alignment_map,
        alignment_seed=config_s.alignment_seed,
    )
    corr_s, _rows_s = fit_direct_residual_streaming(
        target_base,
        target_base_sd,
        source_base,
        source_ft,
        prepared,
        pairing,
        config=config_s,
        device="cpu",
    )
    assert set(corr_r) == set(corr_s)
    for key in corr_r:
        assert torch.allclose(corr_r[key], corr_s[key], rtol=1e-5, atol=1e-6), key

    # The two paths must use the SAME per-position random map (same
    # alignment_seed -> same _derive_block_seed derivation).
    for j in range(pairing.target_depth):
        assert torch.allclose(prepared["q_by_position"][j], prepared["activation_q64_by_position"][j].float())
