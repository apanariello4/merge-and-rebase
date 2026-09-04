from __future__ import annotations

import json
from pathlib import Path

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
