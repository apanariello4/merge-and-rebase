from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.block_extension import (
    BlockExtender,
    BlockExtensionConfig,
    InputAlignedBlock,
    InputAlignedFinalLayer,
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
    assert cfg.extension_strategy == "interpolate"
    assert cfg.insertion_order == "bottom-top"
    assert cfg.ridge_weight == 1e-6


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


def test_reference_capture_requires_requested_batch_count() -> None:
    extender = BlockExtender(_TinyModel(), _TinyModel(), "cpu", verbose=False, show_progress=False)
    with pytest.raises(ValueError, match="exhausted after 1 batches; requested 2"):
        extender.capture_reference_inputs(_make_loader(n_samples=4), n_batches=2)


def test_select_loader_split_precedence() -> None:
    train = object()
    test = object()
    val = object()

    assert select_loader("train", train, test, val) is train
    assert select_loader("val", train, test, val) is val
    assert select_loader("val", train, test, None) is test
    assert select_loader("test", train, test, val) is test


def test_run_block_extension_increases_depth_and_wraps_modules() -> None:
    source_base = _TinyModel(depth=3)
    source_ft = _TinyModel(depth=3)
    loader = _make_loader()

    cfg = BlockExtensionConfig(
        blocks_to_add=2,
        insertion_order="bottom-top",
        extension_density="spread",
        extension_strategy="duplicate",
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
    assert isinstance(source_base.visual.transformer.resblocks[0], InputAlignedBlock)
    assert isinstance(source_base.visual.ln_post, InputAlignedFinalLayer)


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


def test_skip_correction_interpolate_matches_interpolate_per_weight() -> None:
    """With skip_correction=True, ``interpolate`` and ``interpolate_per_weight``
    should compute the same model.

    ``interpolate`` wraps every block (and the final LN) in an
    identity-initialized `InputAlignedBlock`/`InputAlignedFinalLayer`, then
    only fits those aligners when correction is active
    (`extend_and_calibrate`'s per-step ``if skip_correction: continue``).
    ``interpolate_per_weight`` never wraps blocks at all. When correction is
    skipped in both, the aligners in the former are permanent identity
    passthroughs, so the two strategies should be functionally identical --
    verified here by comparing forward outputs rather than state_dicts, since
    the wrapped and unwrapped models have different module structures.
    """
    torch.manual_seed(0)
    base0 = _TinyModel(depth=3)
    ft0 = _TinyModel(depth=3)
    with torch.no_grad():
        for p in ft0.parameters():
            p.add_(0.05 * torch.randn_like(p))
    loader = _make_loader(n_samples=16, batch_size=4)

    def _run(strategy: str):
        import copy

        base = copy.deepcopy(base0)
        ft = copy.deepcopy(ft0)
        cfg = BlockExtensionConfig(
            blocks_to_add=2,
            insertion_order="bottom-top",
            extension_density="spread",
            extension_strategy=strategy,
            dampening_factor=1.0,
            n_batches_act=2,
            skip_correction=True,
            skip_final_ln=False,
            verbose=False,
            show_progress=False,
        )
        run_block_extension(
            source_base_model=base,
            source_ft_model=ft,
            calibration_loader=loader,
            target_layers_total=None,
            config=cfg,
            device="cpu",
        )
        return base, ft

    base_a, ft_a = _run("interpolate")
    base_b, ft_b = _run("interpolate_per_weight")

    torch.manual_seed(123)
    x = torch.randn(5, 6)
    with torch.no_grad():
        assert torch.allclose(base_a.encode_image(x), base_b.encode_image(x), atol=1e-6, rtol=1e-6)
        assert torch.allclose(ft_a.encode_image(x), ft_b.encode_image(x), atol=1e-6, rtol=1e-6)


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
