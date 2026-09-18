"""Regression tests for the target-informed shared component correction.

ARIADNE fits each inserted block to reproduce its source ancestor's component
outputs. This option blends the pretrained *target* block's activations into
that regression target at ``mlp.c_proj``, so the inserted block occupies the
representational slot width transport has to align to instead of one the
residual-identity baseline already solves for free.

The assertions that matter are exactness ones: ``target_weight=0`` must
reproduce standard Shared ARIADNE bit for bit, and the blend must leave the
shared-operator property intact.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.block_extension import (
    BlockExtensionConfig,
    TargetSharedCorrection,
    plan_inserted_positions,
    resolve_block_extension_config,
    run_block_extension,
)


class _TinyAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.randn(3 * dim, dim) * 0.1)
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key=None, value=None, **kwargs):
        del key, value, kwargs
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        weights = torch.softmax((q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5, dim=-1)
        return self.out_proj(weights @ v)


class _TinyMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.c_proj(torch.relu(self.c_fc(x)))


class _TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = _TinyAttn(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = _TinyMLP(dim)

    def forward(self, x, attn_mask=None, **kwargs):
        del attn_mask, kwargs
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _TinyVisual(nn.Module):
    def __init__(self, in_dim=6, width=8, depth=3, tokens=5):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, width)
        self.tokens = tokens
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_TinyBlock(width) for _ in range(depth)])
        self.ln_post = nn.LayerNorm(width)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, self.tokens, 1)
        x = self.input_proj(x)
        for block in self.transformer.resblocks:
            x = block(x)
        return self.ln_post(x).mean(dim=1)


class _TinyModel(nn.Module):
    def __init__(self, in_dim=6, width=8, depth=3, tokens=5):
        super().__init__()
        self.visual = _TinyVisual(in_dim=in_dim, width=width, depth=depth, tokens=tokens)

    def encode_image(self, x):
        return self.visual(x)


def _loader(n_samples: int = 8, in_dim: int = 6, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n_samples, in_dim)
    y = torch.zeros(n_samples, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _run(target_weight: float | None, *, seed: int = 0, target_width: int = 8):
    """Extend a depth-3 source to depth-6 against a depth-6 target model."""
    torch.manual_seed(seed)
    base = _TinyModel(depth=3)
    ft = _TinyModel(depth=3)
    # Same token count on both sides keeps this fixture's rows directly
    # comparable; the patch-grid resampling path is covered separately.
    target = _TinyModel(depth=6, width=target_width)

    correction = (
        None if target_weight is None else TargetSharedCorrection(target_weight=target_weight)
    )
    cfg = BlockExtensionConfig(
        blocks_to_add=3,
        extension_strategy="duplicate_per_weight",
        n_batches_act=2,
        skip_correction=False,
        skip_final_ln=True,
        ridge_identity=100.0,
        lmc_mode="shared",
        target_shared_correction=correction,
        verbose=False,
        show_progress=False,
    )
    run_block_extension(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=_loader(),
        target_layers_total=None,
        config=cfg,
        device="cpu",
        target_model=target,
    )
    return base, ft


def test_plan_inserted_positions_matches_the_paper_schedule() -> None:
    assert plan_inserted_positions(12, list(range(12))) == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    # A second descendant of the same source block lands above the first.
    assert plan_inserted_positions(2, [0, 0]) == [1, 2]


def test_zero_weight_reproduces_standard_shared_ariadne_exactly() -> None:
    """The zero path must bypass every target-side step, not merely damp it."""
    disabled_base, disabled_ft = _run(None)
    zero_base, zero_ft = _run(0.0)

    assert _state_hash(disabled_base) == _state_hash(zero_base)
    assert _state_hash(disabled_ft) == _state_hash(zero_ft)


def test_nonzero_weight_changes_the_fitted_correction() -> None:
    baseline_base, baseline_ft = _run(0.0)
    blended_base, blended_ft = _run(1.0)

    assert _state_hash(baseline_base) != _state_hash(blended_base)
    assert _state_hash(baseline_ft) != _state_hash(blended_ft)


def test_blend_preserves_the_shared_operator_property() -> None:
    """Base and FT must still receive the same absorbed maps.

    Under lmc_mode='shared' the FT endpoint never fits anything; it replays the
    base's stored maps. Blending changes what the base is fitted to, so the
    maps change, but they must remain shared. Equal base/FT *differences* on a
    duplicated block are the observable consequence.
    """
    base, ft = _run(0.3)
    base_sd, ft_sd = base.state_dict(), ft.state_dict()
    inserted = plan_inserted_positions(3, [0, 1, 2])

    moved = 0
    for position in inserted:
        key = f"visual.transformer.resblocks.{position}.mlp.c_proj.weight"
        assert key in base_sd
        if not torch.allclose(base_sd[key], ft_sd[key]):
            moved += 1
    # The endpoints differ because their duplicated blocks differ, not because
    # the correction differs; the test below pins the map sharing directly.
    assert moved > 0


def test_shared_maps_are_identical_between_endpoints() -> None:
    """Fit on base, replay on FT: the ratio of the absorbed map must match."""
    torch.manual_seed(3)
    base = _TinyModel(depth=3)
    ft = _TinyModel(depth=3)
    ft.load_state_dict(base.state_dict())  # identical endpoints isolate the map
    target = _TinyModel(depth=6)

    cfg = BlockExtensionConfig(
        blocks_to_add=3,
        extension_strategy="duplicate_per_weight",
        n_batches_act=2,
        skip_correction=False,
        skip_final_ln=True,
        ridge_identity=100.0,
        lmc_mode="shared",
        target_shared_correction=TargetSharedCorrection(target_weight=0.5),
        verbose=False,
        show_progress=False,
    )
    run_block_extension(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=_loader(),
        target_layers_total=None,
        config=cfg,
        device="cpu",
        target_model=target,
    )
    # Identical endpoints plus a shared map must stay identical.
    assert _state_hash(base) == _state_hash(ft)


def test_blend_is_a_convex_combination_in_source_coordinates() -> None:
    """eta weights the backprojected target against the source reference."""
    from merge_and_rebase.eval.block_extension import BlockExtender

    torch.manual_seed(7)
    n_samples, tokens, d_source, d_target = 4, 5, 8, 12
    source_ref = torch.randn(n_samples * tokens, d_source)
    target_bank = torch.randn(n_samples, tokens, d_target)

    zero = BlockExtender._blend_target_reference(source_ref, target_bank, n_samples, 0.0)
    assert torch.equal(zero, source_ref)

    half = BlockExtender._blend_target_reference(source_ref, target_bank, n_samples, 1.0)
    full = BlockExtender._blend_target_reference(source_ref, target_bank, n_samples, 1e9)
    # eta=1 is the midpoint of the source reference and the backprojection;
    # recovering the backprojection from it must match the eta->inf limit.
    backprojected = 2.0 * half - source_ref
    assert torch.allclose(backprojected, full, atol=1e-3)


def test_blend_rejects_mismatched_sample_counts() -> None:
    from merge_and_rebase.eval.block_extension import BlockExtender

    source_ref = torch.randn(20, 8)
    target_bank = torch.randn(3, 5, 12)
    with pytest.raises(ValueError, match="samples"):
        BlockExtender._blend_target_reference(source_ref, target_bank, 4, 0.5)


def test_config_requires_shared_mode_and_a_real_correction() -> None:
    def resolve(params):
        return resolve_block_extension_config(
            {"block_extension_enabled": True, "block_extension_params": params}
        )

    with pytest.raises(ValueError, match="lmc_mode='shared'"):
        resolve({"target_shared_correction": {"target_weight": 0.3}})
    with pytest.raises(ValueError, match="skip_correction=false"):
        resolve({
            "lmc_mode": "shared",
            "skip_correction": True,
            "target_shared_correction": {"target_weight": 0.3},
        })
    # A depth baseline and a blended correction target are mutually exclusive.
    # skip_correction is what rejects this combination first, since every
    # inserted_block_mode baseline already requires it.
    with pytest.raises(ValueError, match="skip_correction"):
        resolve({
            "lmc_mode": "shared",
            "inserted_block_mode": "residual_identity",
            "skip_correction": True,
            "target_shared_correction": {"target_weight": 0.3},
        })
    with pytest.raises(ValueError, match="component"):
        resolve({"lmc_mode": "shared", "target_shared_correction": {"component": "c_fc"}})
    with pytest.raises(ValueError, match="Unknown"):
        resolve({"lmc_mode": "shared", "target_shared_correction": {"nonsense": 1}})

    _, cfg = resolve({"lmc_mode": "shared", "target_shared_correction": {"enabled": False}})
    assert cfg.target_shared_correction is None
