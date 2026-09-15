from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.registry import get_method, list_methods
from merge_and_rebase.rebase.methods.bico import collect_bilinear_statistics


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
    # A transport that assigns no transform still returns correctly shaped
    # zeros, which every other assertion here accepts.
    assert any(float(t.float().abs().sum()) > 0.0 for t in transported.values())


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
    # A transport that assigns no transform still returns correctly shaped
    # zeros, which every other assertion here accepts.
    assert any(float(t.float().abs().sum()) > 0.0 for t in transported.values())


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
