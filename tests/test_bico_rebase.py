from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.block_extension import BlockExtensionConfig, run_block_extension
from merge_and_rebase.rebase.methods import bico as bico_method
from merge_and_rebase.rebase.methods.bico import collect_bilinear_statistics
from merge_and_rebase.rebase.registry import get_method, list_methods


class _TinyVisual(nn.Module):
    def __init__(self, in_dim: int = 6, hid_dim: int = 8, out_dim: int = 5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln(x)
        return self.fc2(x)


class _TinyModel(nn.Module):
    def __init__(self, in_dim: int = 6, hid_dim: int = 8, out_dim: int = 5) -> None:
        super().__init__()
        self.visual = _TinyVisual(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)
        self.logit_scale = nn.Parameter(torch.ones(1))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


class _TinyAttentionBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads=2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_1(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        return x + attended


class _TinyAttentionVisual(nn.Module):
    def __init__(self, depth: int, in_dim: int = 6, width: int = 4, out_dim: int = 5) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(
            [_TinyAttentionBlock(width) for _ in range(depth)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Linear(width, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x).unsqueeze(1).repeat(1, 3, 1)
        for block in self.transformer.resblocks:
            x = block(x)
        return self.proj(self.ln_post(x).mean(dim=1))


class _TinyAttentionModel(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.visual = _TinyAttentionVisual(depth)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _simple_recipe(model, batch):
    images, labels = batch
    outputs = model.encode_image(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return loss, trainable


def _make_loader(n_samples: int = 16, in_dim: int = 6, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n_samples, in_dim)
    y = torch.randint(0, 5, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


class _TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 6)
        self.model = nn.Linear(6, 8)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return self.model(self.embed(input_ids))


class _TinyTextFamily:
    def transport_scope(self, model):
        return model.model

    def extract_calibration_batch(self, batch):
        return {key: batch[key] for key in ("input_ids", "attention_mask", "labels") if key in batch}


def _text_recipe(model, batch):
    output = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
    return output.square().mean(), []


def test_bico_collects_standard_text_batches() -> None:
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "labels": torch.tensor([[1, 2, 3], [4, 5, -100]]),
    }
    stats = collect_bilinear_statistics(
        _TinyTextModel(),
        _TinyTextModel(),
        [batch],
        [batch],
        _text_recipe,
        _text_recipe,
        device="cpu",
        seq_align="mean",
        n_batches=1,
        family_adapter=_TinyTextFamily(),
    )
    assert stats


def test_bico_registered() -> None:
    assert "bico" in list_methods()
    assert get_method("bico").name == "bico"


def test_bico_rejects_misaligned_calibration_labels() -> None:
    source_model = _TinyModel()
    target_model = _TinyModel()
    x = torch.randn(4, 6)
    source_loader = DataLoader(TensorDataset(x, torch.zeros(4, dtype=torch.long)), batch_size=4)
    target_loader = DataLoader(TensorDataset(x, torch.ones(4, dtype=torch.long)), batch_size=4)

    with pytest.raises(ValueError, match="not label-aligned"):
        collect_bilinear_statistics(
            source_model,
            target_model,
            source_loader,
            target_loader,
            _simple_recipe,
            _simple_recipe,
            device="cpu",
            seq_align="mean",
            n_batches=1,
        )


def test_bico_transport_smoke() -> None:
    source_model = _TinyModel(in_dim=6, hid_dim=8, out_dim=5)
    target_model = _TinyModel(in_dim=6, hid_dim=7, out_dim=5)

    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    target_base = {k: v.detach().clone() for k, v in target_model.state_dict().items()}

    delta = {
        key: torch.randn_like(tensor)
        for key, tensor in source_base.items()
        if key.startswith("visual.") and tensor.is_floating_point()
    }

    loader = _make_loader(in_dim=6)
    recipe = _simple_recipe
    method = get_method("bico")

    transported = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        source_recipe=recipe,
        target_recipe=recipe,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        strict=True,
    )

    assert transported
    assert set(transported.keys()) == set(delta.keys())
    for key, tensor in transported.items():
        assert tensor.shape == target_base[key].shape
        assert tensor.dtype == target_base[key].dtype


def test_bico_split_qkv_after_per_weight_extension_has_full_transport_coverage() -> None:
    torch.manual_seed(7)
    source_base_model = _TinyAttentionModel(depth=1)
    source_ft_model = _TinyAttentionModel(depth=1)
    target_model = _TinyAttentionModel(depth=2)
    loader = _make_loader(n_samples=8, batch_size=4)

    run_block_extension(
        source_base_model=source_base_model,
        source_ft_model=source_ft_model,
        calibration_loader=loader,
        target_layers_total=2,
        config=BlockExtensionConfig(
            extension_strategy="interpolate_per_weight",
            skip_correction=True,
            n_batches_act=1,
            verbose=False,
            show_progress=False,
        ),
        device="cpu",
    )

    source_base = {
        key: value.detach().clone() for key, value in source_base_model.state_dict().items()
    }
    source_ft = source_ft_model.state_dict()
    target_base = {
        key: value.detach().clone() for key, value in target_model.state_dict().items()
    }
    delta = {
        key: source_ft[key].detach().clone() - value
        for key, value in source_base.items()
        if key.startswith("visual.") and value.is_floating_point()
    }

    method = bico_method.BiCoRebase()
    prepared = method.prepare(
        source_model=source_ft_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        source_recipe=_simple_recipe,
        target_recipe=_simple_recipe,
        target_base=target_base,
        delta=delta,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        verbose=False,
        show_progress=False,
    )

    diagnostics = prepared["precompute_diagnostics"]
    assert diagnostics["incomplete"] == 0
    assert diagnostics["unsupported"] == 0
    assert diagnostics["usable"] == diagnostics["slots"]

    transported = method.apply(
        prepared,
        target_base=target_base,
        delta=delta,
        strict=True,
        verbose=False,
        show_progress=False,
    )
    assert set(transported) == set(delta)


def test_bico_split_qkv_apply_unpacks_transform_diagnostics(monkeypatch) -> None:
    target_base = {
        "visual.block.attn.in_proj_weight": torch.zeros(6, 2),
    }
    delta = {
        "visual.block.attn.in_proj_weight": torch.ones(6, 2),
    }

    def fake_apply_transforms(**kwargs):
        return kwargs["visual_delta"], object()

    monkeypatch.setattr(
        bico_method._t,
        "_apply_transforms_to_visual_delta",
        fake_apply_transforms,
    )

    transported = bico_method.BiCoRebase().apply(
        {
            "transforms_by_key": {},
            "split_fused_qkv": True,
            "compute_device": torch.device("cpu"),
        },
        target_base=target_base,
        delta=delta,
        strict=True,
        verbose=False,
        show_progress=False,
    )

    assert set(transported) == set(delta)
    assert torch.equal(transported["visual.block.attn.in_proj_weight"], delta["visual.block.attn.in_proj_weight"])


def test_bico_deterministic() -> None:
    torch.manual_seed(42)
    source_model = _TinyModel(in_dim=6, hid_dim=8, out_dim=5)
    target_model = _TinyModel(in_dim=6, hid_dim=7, out_dim=5)

    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    target_base = {k: v.detach().clone() for k, v in target_model.state_dict().items()}

    delta = {
        key: torch.randn_like(tensor)
        for key, tensor in source_base.items()
        if key.startswith("visual.") and tensor.is_floating_point()
    }

    loader = _make_loader(in_dim=6)
    recipe = _simple_recipe
    method = get_method("bico")

    result_a = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        source_recipe=recipe,
        target_recipe=recipe,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        seed=123,
        strict=True,
    )

    result_b = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        source_recipe=recipe,
        target_recipe=recipe,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        seed=123,
        strict=True,
    )

    assert set(result_a.keys()) == set(result_b.keys())
    for key in result_a:
        assert torch.allclose(result_a[key], result_b[key]), f"Mismatch for key {key}"


# ── bico_gradin tests ─────────────────────────────────────────────────────


def test_bico_gradin_registered() -> None:
    assert "bico_gradin" in list_methods()
    assert get_method("bico_gradin").name == "bico_gradin"


def test_bico_gradin_transport_smoke() -> None:
    source_model = _TinyModel(in_dim=6, hid_dim=8, out_dim=5)
    target_model = _TinyModel(in_dim=6, hid_dim=7, out_dim=5)

    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    target_base = {k: v.detach().clone() for k, v in target_model.state_dict().items()}

    delta = {
        key: torch.randn_like(tensor)
        for key, tensor in source_base.items()
        if key.startswith("visual.") and tensor.is_floating_point()
    }

    loader = _make_loader(in_dim=6)
    recipe = _simple_recipe
    method = get_method("bico_gradin")

    transported = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        source_recipe=recipe,
        target_recipe=recipe,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        strict=True,
    )

    assert transported
    assert set(transported.keys()) == set(delta.keys())
    for key, tensor in transported.items():
        assert tensor.shape == target_base[key].shape
        assert tensor.dtype == target_base[key].dtype


def test_bico_gradin_deterministic() -> None:
    torch.manual_seed(42)
    source_model = _TinyModel(in_dim=6, hid_dim=8, out_dim=5)
    target_model = _TinyModel(in_dim=6, hid_dim=7, out_dim=5)

    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    target_base = {k: v.detach().clone() for k, v in target_model.state_dict().items()}

    delta = {
        key: torch.randn_like(tensor)
        for key, tensor in source_base.items()
        if key.startswith("visual.") and tensor.is_floating_point()
    }

    loader = _make_loader(in_dim=6)
    recipe = _simple_recipe
    method = get_method("bico_gradin")

    result_a = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        source_recipe=recipe,
        target_recipe=recipe,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        seed=123,
        strict=True,
    )

    result_b = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        source_recipe=recipe,
        target_recipe=recipe,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        seed=123,
        strict=True,
    )

    assert set(result_a.keys()) == set(result_b.keys())
    for key in result_a:
        assert torch.allclose(result_a[key], result_b[key]), f"Mismatch for key {key}"
