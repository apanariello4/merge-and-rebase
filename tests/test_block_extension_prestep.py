from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.block_extension import (
    BlockExtender,
    BlockExtensionConfig,
    block_extension_protocol,
    resolve_block_extension_config,
    run_block_extension,
    select_loader,
)


class _TinyAttn(nn.Module):
    """Minimal stand-in for CLIP's MultiheadAttention with a fused in_proj.

    BRACE's reference-capture hooks (`_capture_component_references`,
    `_capture_per_weight_reference_subset`) patch `attn.forward` and read
    `attn.in_proj_weight`/`in_proj_bias` directly, so the fixture must expose
    those attributes with the real interface, not just a generic linear.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.randn(3 * dim, dim) * 0.1)
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key=None, value=None, **kwargs) -> torch.Tensor:
        del key, value, kwargs
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = q.shape[-1] ** -0.5
        weights = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        return self.out_proj(weights @ v)


class _TinyMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(torch.relu(self.c_fc(x)))


class _TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = _TinyAttn(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = _TinyMLP(dim)

    def forward(self, x: torch.Tensor, attn_mask=None, **kwargs):
        del attn_mask, kwargs
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _TinyVisual(nn.Module):
    def __init__(self, in_dim: int = 6, width: int = 8, depth: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_TinyBlock(width) for _ in range(depth)])
        self.ln_post = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, 4, 1)
        x = self.input_proj(x)
        for block in self.transformer.resblocks:
            x = block(x)
        x = self.ln_post(x)
        return x.mean(dim=1)


class _TinyModel(nn.Module):
    def __init__(self, in_dim: int = 6, width: int = 8, depth: int = 3):
        super().__init__()
        self.visual = _TinyVisual(in_dim=in_dim, width=width, depth=depth)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _make_loader(n_samples: int = 16, in_dim: int = 6, batch_size: int = 4):
    x = torch.randn(n_samples, in_dim)
    y = torch.zeros(n_samples, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_resolve_block_extension_config_defaults() -> None:
    enabled, cfg = resolve_block_extension_config({})
    assert not enabled
    assert isinstance(cfg, BlockExtensionConfig)
    assert cfg.extension_strategy == "interpolate_per_weight"
    assert cfg.insertion_order == "bottom-top"
    assert cfg.ridge_weight == 1e-6
    assert not cfg.joint_blockwise_correction.enabled


def test_direct_target_can_use_layout_only_block_extension() -> None:
    """Direct P1 needs the realized layout, not source-side ARIADNE fits."""
    _, cfg = resolve_block_extension_config(
        {
            "block_extension_enabled": True,
            "block_extension_params": {
                "skip_correction": True,
                "lmc_mode": "independent",
                "target_residual_completion": {
                    "enabled": True,
                    "mode": "direct_target",
                    "target_scope": "all",
                },
            },
        }
    )
    assert cfg.skip_correction is True
    assert cfg.lmc_mode == "independent"
    semantics = block_extension_protocol(cfg)
    assert semantics["label"] == "direct_target_layout_only"
    assert semantics["initialization"] == "layout_only"


def test_resolve_block_extension_config_accepts_joint_blockwise_option3() -> None:
    _, cfg = resolve_block_extension_config(
        {
            "block_extension_enabled": True,
            "block_extension_params": {
                "lmc_mode": "shared",
                "extension_strategy": "duplicate_per_weight",
                "calibration_split": "val",
                "joint_blockwise_correction": {
                    "enabled": True,
                    "source_weight": 0.5,
                    "target_weight": 2.0,
                    "ridge_relative": 0.02,
                }
            },
        }
    )
    assert cfg.joint_blockwise_correction.enabled
    assert cfg.joint_blockwise_correction.source_weight == 0.5
    assert cfg.joint_blockwise_correction.target_weight == 2.0
    with pytest.raises(ValueError, match="requires lmc_mode='shared'"):
        resolve_block_extension_config(
            {"block_extension_params": {"joint_blockwise_correction": {"enabled": True}}}
        )
    with pytest.raises(ValueError, match="requires calibration_split='val'"):
        resolve_block_extension_config(
            {
                "block_extension_params": {
                    "lmc_mode": "shared",
                    "extension_strategy": "duplicate_per_weight",
                    "joint_blockwise_correction": {"enabled": True},
                }
            }
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_block_extension_config(
            {
                "block_extension_params": {
                    "target_residual_completion": {"enabled": True},
                    "joint_blockwise_correction": {"enabled": True},
                }
            }
        )


def test_resolve_block_extension_config_accepts_direct_p1_correction() -> None:
    _, cfg = resolve_block_extension_config(
        {
            "block_extension_enabled": True,
            "block_extension_params": {
                "lmc_mode": "shared",
                "extension_strategy": "duplicate_per_weight",
                "calibration_split": "val",
                "direct_p1_correction": {
                    "enabled": True,
                    "source_weight": 1.0,
                    "target_weight": 0.1,
                    "ridge_relative": 0.02,
                },
            },
        }
    )
    assert cfg.direct_p1_correction.enabled
    assert cfg.direct_p1_correction.target_weight == 0.1
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_block_extension_config(
            {
                "block_extension_params": {
                    "target_residual_completion": {"enabled": True},
                    "direct_p1_correction": {"enabled": True},
                }
            }
        )


def test_resolve_block_extension_config_accepts_two_ridge_coefficients() -> None:
    _, cfg = resolve_block_extension_config(
        {
            "block_extension_enabled": True,
            "block_extension_params": {"ridge_identity": 200.0, "ridge_weight": 1e-3},
        }
    )

    assert cfg.ridge_identity == 200.0
    assert cfg.ridge_weight == 1e-3


@pytest.mark.parametrize(
    "params",
    [
        {"n_batches_act": 0},
        {"ridge_identity": -1.0},
        {"ridge_weight": -1e-6},
    ],
)
def test_resolve_block_extension_config_rejects_invalid_calibration(params) -> None:
    with pytest.raises(ValueError):
        resolve_block_extension_config(
            {"block_extension_enabled": True, "block_extension_params": params}
        )


def test_spread_duplication_covers_the_whole_model() -> None:
    # Same fix as the decoder extender: "spread" distributes the added blocks
    # over the depth instead of stacking them at one end.
    sched = BlockExtender._build_duplication_schedule(28, 8, "bottom-top", "spread")
    assert len(sched) == 8
    assert len(set(sched)) == 8
    assert sched[0] == 0
    assert 27 not in sched
    gaps = [b - a for a, b in zip(sched, sched[1:], strict=False)]
    assert max(gaps) - min(gaps) <= 1
    # spread_mod stays on the old prefix schedule for reproducing earlier runs
    assert BlockExtender._build_duplication_schedule(28, 8, "bottom-top", "spread_mod") == list(range(8))


def test_spread_collapse_covers_the_whole_model() -> None:
    sched = BlockExtender._build_collapse_schedule(28, 8, "bottom-top", "spread")
    assert len(sched) == 8
    assert len(set(sched)) == 8
    assert all(0 <= a <= 26 for a in sched)


def test_select_loader_split_precedence() -> None:
    train = object()
    test = object()
    val = object()

    assert select_loader("train", train, test, val) is train
    assert select_loader("val", train, test, val) is val
    assert select_loader("val", train, test, None) is test
    assert select_loader("test", train, test, val) is test


def test_run_block_extension_increases_depth_without_wrappers() -> None:
    source_base = _TinyModel(depth=3)
    source_ft = _TinyModel(depth=3)
    loader = _make_loader()

    cfg = BlockExtensionConfig(
        blocks_to_add=2,
        insertion_order="bottom-top",
        extension_density="spread",
        extension_strategy="duplicate_per_weight",
        dampening_factor=1.0,
        n_batches_act=1,
        skip_correction=True,
        skip_final_ln=True,
    )

    final_depth = run_block_extension(
        source_base_model=source_base,
        source_ft_model=source_ft,
        calibration_loader=loader,
        target_layers_total=None,
        config=cfg,
        device="cpu",
    )

    assert final_depth == 5
    assert len(source_base.visual.transformer.resblocks) == 5
    assert len(source_ft.visual.transformer.resblocks) == 5
    assert all(not hasattr(block, "aligner") for block in source_base.visual.transformer.resblocks)
    assert not hasattr(source_base.visual.ln_post, "aligner")


def test_resolve_block_extension_config_accepts_shared_ft_mode() -> None:
    enabled, cfg = resolve_block_extension_config(
        {"block_extension_enabled": True, "block_extension_params": {"lmc_mode": "shared_ft"}}
    )

    assert enabled
    assert cfg.lmc_mode == "shared_ft"


def _make_per_weight_models(depth: int = 3, width: int = 8, in_dim: int = 6):
    torch.manual_seed(0)
    base = _TinyModel(in_dim=in_dim, width=width, depth=depth)
    ft = _TinyModel(in_dim=in_dim, width=width, depth=depth)
    with torch.no_grad():
        for p in ft.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return base, ft


_ORIGINAL_CAPTURE_PER_WEIGHT_SUBSET = BlockExtender._capture_per_weight_reference_subset


def _eager_reference_capture(self, reference_models, *, endpoints, block_indices, input_indices, loader, n_batches):
    """Test monkeypatch: emulate the pre-refactor eager global capture.

    The historical path captured references for every block of both
    endpoints exactly once, before any structural modification, and reused
    that frozen snapshot for every later step. This reproduces that behavior
    with the still-present lazy-capture machinery: the first call captures
    the full universe of blocks/inputs for both endpoints and caches it; every
    later call (one per structural step) reuses the cached snapshot instead of
    recapturing a narrow, per-step subset.
    """
    del endpoints, block_indices, input_indices
    cache = getattr(self, "_eager_reference_cache", None)
    if cache is not None:
        self.reference_inputs = cache
        return
    all_blocks = tuple(range(len(reference_models["base"].visual.transformer.resblocks)))
    all_inputs = all_blocks + ("final",)
    _ORIGINAL_CAPTURE_PER_WEIGHT_SUBSET(
        self,
        reference_models,
        endpoints=("base", "ft"),
        block_indices=all_blocks,
        input_indices=all_inputs,
        loader=loader,
        n_batches=n_batches,
    )
    self._eager_reference_cache = self.reference_inputs


def _run_per_weight(monkeypatch, *, eager: bool, strategy: str, lmc_mode: str, target_layers_total: int, loader) -> tuple[nn.Module, nn.Module]:
    base, ft = _make_per_weight_models()
    if eager:
        monkeypatch.setattr(BlockExtender, "_capture_per_weight_reference_subset", _eager_reference_capture)
    cfg = BlockExtensionConfig(
        target_layers_total=target_layers_total,
        insertion_order="bottom-top",
        extension_density="spread",
        extension_strategy=strategy,
        dampening_factor=1.0,
        n_batches_act=2,
        skip_correction=False,
        skip_final_ln=False,
        ridge_identity=1.0,
        ridge_weight=1e-6,
        lmc_mode=lmc_mode,
        verbose=False,
        show_progress=False,
    )
    run_block_extension(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=loader,
        target_layers_total=target_layers_total,
        config=cfg,
        device="cpu",
    )
    if eager:
        monkeypatch.undo()
    return base, ft


def _assert_state_dicts_match(model_a: nn.Module, model_b: nn.Module) -> None:
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.allclose(state_a[key], state_b[key], atol=1e-5, rtol=1e-5), f"mismatch in {key}"


@pytest.mark.parametrize("lmc_mode", ["independent", "shared"])
def test_lazy_reference_capture_matches_eager_capture_extend(lmc_mode, monkeypatch) -> None:
    """Regression test for the lazy per-step reference-capture rewrite (extend).

    `_capture_per_weight_reference_subset`'s docstring claims the lazy
    per-structural-step capture it introduced is "mathematically equivalent"
    to the eager global capture it replaced, but nothing verified that claim
    -- and a paper appendix comparing two nominally-identical configs run
    before/after this rewrite showed up to a 3.5-point accuracy swing on the
    B/16->L/14 extension direction. This drives extend_and_calibrate once
    through the current lazy path and once through a forced emulation of the
    eager path it replaced, and asserts the resulting corrected weights are
    identical.
    """
    loader = _make_loader(n_samples=16, batch_size=4)

    lazy_base, lazy_ft = _run_per_weight(
        monkeypatch, eager=False, strategy="duplicate_per_weight", lmc_mode=lmc_mode,
        target_layers_total=5, loader=loader,
    )
    eager_base, eager_ft = _run_per_weight(
        monkeypatch, eager=True, strategy="duplicate_per_weight", lmc_mode=lmc_mode,
        target_layers_total=5, loader=loader,
    )

    _assert_state_dicts_match(lazy_base, eager_base)
    _assert_state_dicts_match(lazy_ft, eager_ft)


@pytest.mark.parametrize(
    ("strategy", "replacement"),
    [("interpolate", "interpolate_per_weight"), ("duplicate", "duplicate_per_weight")],
)
def test_legacy_vision_strategies_raise_migration_error(strategy: str, replacement: str) -> None:
    cfg = BlockExtensionConfig(
        blocks_to_add=1,
        extension_strategy=strategy,
        skip_correction=True,
        verbose=False,
        show_progress=False,
    )
    with pytest.raises(ValueError, match=replacement):
        run_block_extension(
            source_base_model=_TinyModel(),
            source_ft_model=_TinyModel(),
            calibration_loader=_make_loader(),
            target_layers_total=None,
            config=cfg,
            device="cpu",
        )


@pytest.mark.parametrize("lmc_mode", ["independent", "shared"])
def test_lazy_reference_capture_matches_eager_capture_shrink(lmc_mode, monkeypatch) -> None:
    """Regression test for the lazy per-step reference-capture rewrite (shrink).

    Same rationale as the extend variant above, but exercised on the
    L/14->B/16 shrink direction (`interpolate_per_weight`), which is the
    other strategy shown to diverge in the paper appendix.
    """
    loader = _make_loader(n_samples=16, batch_size=4)

    lazy_base, lazy_ft = _run_per_weight(
        monkeypatch, eager=False, strategy="interpolate_per_weight", lmc_mode=lmc_mode,
        target_layers_total=2, loader=loader,
    )
    eager_base, eager_ft = _run_per_weight(
        monkeypatch, eager=True, strategy="interpolate_per_weight", lmc_mode=lmc_mode,
        target_layers_total=2, loader=loader,
    )

    _assert_state_dicts_match(lazy_base, eager_base)
    _assert_state_dicts_match(lazy_ft, eager_ft)


def _run_per_weight_via_config(*, reference_capture: str, strategy: str, lmc_mode: str,
                               target_layers_total: int, loader) -> tuple[nn.Module, nn.Module]:
    """Drive extend_and_calibrate through the real `reference_capture` knob.

    The two tests above emulate the eager schedule with a monkeypatch. This
    helper instead exercises the shipped config switch end to end, so the
    diagnostic users can actually select from a config file is covered too.
    """
    torch.manual_seed(0)
    base, ft = _make_per_weight_models()
    cfg = BlockExtensionConfig(
        target_layers_total=target_layers_total,
        insertion_order="bottom-top",
        extension_density="spread",
        extension_strategy=strategy,
        dampening_factor=1.0,
        n_batches_act=2,
        skip_correction=False,
        skip_final_ln=False,
        ridge_identity=100.0,
        ridge_weight=1e-6,
        lmc_mode=lmc_mode,
        reference_capture=reference_capture,
        verbose=False,
        show_progress=False,
    )
    run_block_extension(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=loader,
        target_layers_total=target_layers_total,
        config=cfg,
        device="cpu",
    )
    return base, ft


def _assert_state_dicts_bit_identical(model_a: nn.Module, model_b: nn.Module) -> None:
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"bitwise mismatch in {key}"


@pytest.mark.parametrize("lmc_mode", ["independent", "shared"])
@pytest.mark.parametrize(
    ("strategy", "target_layers_total"),
    [("duplicate_per_weight", 5), ("interpolate_per_weight", 2)],
)
def test_reference_capture_switch_is_bit_identical(lmc_mode, strategy, target_layers_total) -> None:
    """`reference_capture=eager` must not change a single bit of the result.

    The 2026-09-11 transport-swap reproducibility investigation named the
    eager->lazy reference-capture rewrite as its leading hypothesis for a
    ~7-point accuracy gap against a historical table row. This asserts the
    stronger claim the sibling tests only check to atol=1e-5: on CPU the two
    capture schedules agree exactly, so any residual gap must come from
    somewhere else (device nondeterminism, or the artifact/transport path).
    """
    loader = _make_loader(n_samples=16, batch_size=4)

    lazy_base, lazy_ft = _run_per_weight_via_config(
        reference_capture="lazy", strategy=strategy, lmc_mode=lmc_mode,
        target_layers_total=target_layers_total, loader=loader,
    )
    eager_base, eager_ft = _run_per_weight_via_config(
        reference_capture="eager", strategy=strategy, lmc_mode=lmc_mode,
        target_layers_total=target_layers_total, loader=loader,
    )

    _assert_state_dicts_bit_identical(lazy_base, eager_base)
    _assert_state_dicts_bit_identical(lazy_ft, eager_ft)


def test_resolve_block_extension_config_rejects_unknown_reference_capture() -> None:
    with pytest.raises(ValueError):
        resolve_block_extension_config(
            {
                "block_extension_enabled": True,
                "block_extension_params": {"reference_capture": "legacy_eager"},
            }
        )


def test_vision8_shared_configs_differ_only_by_lmc_direction() -> None:
    root = Path(__file__).resolve().parents[1]
    shared_path = root / "configs/vision8_theseus_blockmode_interpolate_per_weight.json"
    reverse_path = root / "configs/vision8_theseus_blockmode_interpolate_per_weight_shared_ft.json"
    shared = json.loads(shared_path.read_text())
    reverse = json.loads(reverse_path.read_text())

    assert shared["block_extension_params"]["lmc_mode"] == "shared"
    assert reverse["block_extension_params"]["lmc_mode"] == "shared_ft"
    shared["block_extension_params"]["lmc_mode"] = "shared_ft"
    assert shared == reverse



# --- Faithful CLIP-style block, for exercising the real correction cascade ---
# The _TinyModel above is a structural stub (no fused in_proj, no c_fc), fine for
# tests that monkeypatch the cascade. Running the cascade for real needs the
# module names and shapes it actually reaches into.


class _ClipLikeAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.randn(3 * dim, dim) * 0.2)
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, query, key=None, value=None, **kwargs):
        del key, value, kwargs
        qkv = nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        w = torch.softmax(q @ k.transpose(-2, -1) / (self.dim**0.5), dim=-1)
        return self.out_proj(w @ v)


class _ClipLikeMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, dim * 2)
        self.c_proj = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(nn.functional.gelu(self.c_fc(x)))


class _ClipLikeBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = _ClipLikeAttn(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = _ClipLikeMLP(dim)

    def forward(self, x: torch.Tensor, attn_mask=None, **kwargs):
        del attn_mask, kwargs
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _ClipLikeVisual(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_ClipLikeBlock(width) for _ in range(depth)])
        self.ln_post = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, 4, 1)
        x = self.input_proj(x)
        for block in self.transformer.resblocks:
            x = block(x)
        return self.ln_post(x).mean(dim=1)


class _ClipLikeModel(nn.Module):
    def __init__(self, in_dim: int = 6, width: int = 8, depth: int = 3):
        super().__init__()
        self.visual = _ClipLikeVisual(in_dim=in_dim, width=width, depth=depth)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _inserted_block_placement(mode: str) -> tuple[float, float]:
    """Extend by one block and measure where the inserted copy sends the stream.

    Returns (err_to_next, err_to_src): distance from the inserted block's output
    to its source block's OUTPUT x_(s+1), and to its source block's INPUT x_s.
    A correction that reproduces the source block lands nearer x_(s+1); one that
    consumes the input gap twice inverts the block and lands on x_s.
    """
    torch.manual_seed(0)
    source_base = _ClipLikeModel(depth=3)
    source_ft = _ClipLikeModel(depth=3)
    loader = _make_loader(n_samples=64, batch_size=8)

    cfg = BlockExtensionConfig(
        blocks_to_add=1,
        insertion_order="bottom-top",
        extension_density="spread",
        extension_strategy="duplicate_per_weight",
        n_batches_act=4,
        skip_correction=False,
        ridge_identity=0.0,
        lmc_mode="independent",
        insertion_target_mode=mode,
        verbose=False,
        show_progress=False,
    )

    src_block = source_base.visual.transformer.resblocks[0]
    run_block_extension(
        source_base_model=source_base,
        source_ft_model=source_ft,
        calibration_loader=loader,
        target_layers_total=None,
        config=cfg,
        device="cpu",
    )
    blocks = source_base.visual.transformer.resblocks
    assert len(blocks) == 4
    inserted = blocks[1]

    x = torch.randn(8, 4, 8)
    with torch.no_grad():
        x_next = src_block(x)       # x_(s+1): what the source block produces
        out = inserted(x_next)      # the inserted copy consumes x_(s+1)
    return (out - x_next).norm().item(), (out - x).norm().item()


@pytest.mark.parametrize("mode", ["direct", "residual"])
def test_inserted_block_does_not_invert_its_source(mode: str) -> None:
    """An inserted block must advance the residual stream, never reverse it.

    This is the invariant the LLM cascade violates: subtracting the block input
    at both the attention step and the MLP step consumes the input gap twice and
    sends the stream back to x_s. Both vision insertion-target modes must hold it.
    """
    err_next, err_src = _inserted_block_placement(mode)
    assert err_next < err_src, (
        f"insertion_target_mode={mode!r}: inserted block landed nearer its source's "
        f"INPUT ({err_src:.4f}) than its OUTPUT ({err_next:.4f}) -- it inverted the block"
    )


def test_insertion_target_mode_is_validated() -> None:
    cfg = BlockExtensionConfig(
        blocks_to_add=1,
        extension_strategy="duplicate_per_weight",
        n_batches_act=1,
        skip_correction=True,
        insertion_target_mode="resiudal",  # typo must not resolve to a silent default
    )
    with pytest.raises(ValueError, match="insertion_target_mode"):
        run_block_extension(
            source_base_model=_ClipLikeModel(depth=3),
            source_ft_model=_ClipLikeModel(depth=3),
            calibration_loader=_make_loader(),
            target_layers_total=None,
            config=cfg,
            device="cpu",
        )
