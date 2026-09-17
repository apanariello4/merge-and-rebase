from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.block_extension import (
    BlockExtender,
    BlockExtensionConfig,
    InputAlignedBlock,
    InputAlignedFinalLayer,
    _deterministic_calibration_loader,
    resolve_block_extension_config,
    run_block_extension,
    select_loader,
)


def _assert_shared_lmc_direction(monkeypatch, mode: str) -> None:
    source_base = _TinyModel(depth=2)
    source_ft = _TinyModel(depth=2)
    extender = BlockExtender(source_base, source_ft, "cpu", verbose=False, show_progress=False)
    correction_calls: list[str] = []
    apply_calls: list[torch.nn.Module] = []

    monkeypatch.setattr(extender, "capture_reference_inputs", lambda loader, n_batches: None)
    monkeypatch.setattr(extender, "_capture_component_references", lambda loader, n_batches: None)

    def fake_correct(model_name, model, insert_pos, src_idx, loader, n_batches, **kwargs):
        correction_calls.append(model_name)
        kwargs["lmc_store"]["sentinel"] = (torch.eye(8), torch.zeros(8))

    def fake_apply(model, insert_pos, corrections):
        assert set(corrections) == {"sentinel"}
        apply_calls.append(model)

    monkeypatch.setattr(extender, "_correct_block_weights_cascade", fake_correct)
    monkeypatch.setattr(extender, "_apply_block_corrections", fake_apply)

    extender._extend_per_weight(
        loader=object(),
        n_batches=1,
        dampening_factor=1.0,
        blocks_to_add=1,
        target_layers_total=None,
        insertion_order="bottom-top",
        extension_density="spread",
        per_weight_mode="duplicate",
        skip_correction=False,
        lmc_mode=mode,
    )

    expected_source = "base" if mode == "shared" else "ft"
    expected_target = source_ft if mode == "shared" else source_base
    assert correction_calls == [expected_source]
    assert apply_calls == [expected_target]


class _TinyAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(x)


class _TinyMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(torch.relu(self.fc(x)))


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


def test_resolve_block_extension_config_accepts_task_independent_dataset() -> None:
    enabled, cfg = resolve_block_extension_config(
        {
            "block_extension_params": {
                "calibration_dataset": {
                    "path": "zh-plus/tiny-imagenet",
                    "split": "train",
                    "max_samples": 8,
                }
            }
        }
    )

    assert enabled
    assert cfg.calibration_dataset == {
        "path": "zh-plus/tiny-imagenet",
        "split": "train",
        "max_samples": 8,
    }


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


def test_run_block_extension_increases_depth() -> None:
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
    # The per-weight cascade corrects block weights in place; it does not wrap
    # blocks in aligner modules the way the retired non-per-weight path did.
    assert not isinstance(source_base.visual.transformer.resblocks[0], InputAlignedBlock)
    assert not isinstance(source_base.visual.ln_post, InputAlignedFinalLayer)


@pytest.mark.parametrize("strategy", ["interpolate", "duplicate"])
def test_run_block_extension_rejects_non_per_weight_strategies(strategy: str) -> None:
    """The bare strategies were deprecated; only the per-weight cascade is supported."""
    cfg = BlockExtensionConfig(
        blocks_to_add=2,
        extension_strategy=strategy,
        n_batches_act=1,
        skip_correction=True,
    )

    with pytest.raises(ValueError, match="non per-weight"):
        run_block_extension(
            source_base_model=_TinyModel(depth=3),
            source_ft_model=_TinyModel(depth=3),
            calibration_loader=_make_loader(),
            target_layers_total=None,
            config=cfg,
            device="cpu",
        )


def test_calibration_loader_replays_the_same_randomized_window():
    values = torch.arange(12, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(values, torch.zeros(len(values), dtype=torch.long))
    loader = DataLoader(dataset, batch_size=3, shuffle=True)

    frozen = _deterministic_calibration_loader(loader, n_batches=2)
    first = torch.cat([batch[0].flatten() for batch in frozen]).tolist()
    second = torch.cat([batch[0].flatten() for batch in frozen]).tolist()

    assert first == second
    assert len(first) == 6


def test_shared_lmc_fits_base_and_applies_to_ft(monkeypatch) -> None:
    _assert_shared_lmc_direction(monkeypatch, "shared")


def test_shared_reverse_lmc_fits_ft_and_applies_to_base(monkeypatch) -> None:
    _assert_shared_lmc_direction(monkeypatch, "shared_reverse")


def test_shared_reverse_lmc_shrink_fits_ft_and_applies_to_base(monkeypatch) -> None:
    source_base = _TinyModel(depth=2)
    source_ft = _TinyModel(depth=2)
    extender = BlockExtender(source_base, source_ft, "cpu", verbose=False, show_progress=False)
    correction_calls: list[str] = []
    apply_calls: list[torch.nn.Module] = []

    monkeypatch.setattr(extender, "capture_reference_inputs", lambda loader, n_batches: None)
    monkeypatch.setattr(extender, "_capture_component_references", lambda loader, n_batches: None)

    def fake_correct(model_name, model, *args, **kwargs):
        correction_calls.append(model_name)
        kwargs["lmc_store"]["sentinel"] = (torch.eye(8), torch.zeros(8))

    def fake_apply(model, collapse_pos, corrections):
        assert set(corrections) == {"sentinel"}
        apply_calls.append(model)

    monkeypatch.setattr(extender, "_correct_collapsed_block_weights_cascade", fake_correct)
    monkeypatch.setattr(extender, "_apply_block_corrections", fake_apply)

    extender._shrink_per_weight(
        loader=object(),
        n_batches=1,
        dampening_factor=1.0,
        blocks_to_add=-1,
        target_layers_total=None,
        insertion_order="bottom-top",
        extension_density="spread",
        per_weight_mode="duplicate",
        skip_correction=False,
        lmc_mode="shared_reverse",
    )

    assert correction_calls == ["ft"]
    assert apply_calls == [source_base]


def test_resolve_block_extension_config_accepts_shared_reverse() -> None:
    enabled, cfg = resolve_block_extension_config(
        {"block_extension_params": {"lmc_mode": "shared_reverse"}}
    )
    assert enabled
    assert cfg.lmc_mode == "shared_reverse"


def test_vision8_shared_configs_differ_only_by_lmc_direction() -> None:
    root = Path(__file__).resolve().parents[1]
    shared_path = root / "configs/vision8_theseus_blockmode_interpolate_per_weight.json"
    reverse_path = root / "configs/vision8_theseus_blockmode_interpolate_per_weight_shared_reverse.json"
    shared = json.loads(shared_path.read_text())
    reverse = json.loads(reverse_path.read_text())

    assert shared["block_extension_params"]["lmc_mode"] == "shared"
    assert reverse["block_extension_params"]["lmc_mode"] == "shared_reverse"
    shared["block_extension_params"]["lmc_mode"] = "shared_reverse"
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
