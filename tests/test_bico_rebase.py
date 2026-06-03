from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
