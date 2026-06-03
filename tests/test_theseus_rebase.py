from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.methods import theseus as theseus_mod
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

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _make_loader(n_samples: int = 16, in_dim: int = 6, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n_samples, in_dim)
    y = torch.zeros(n_samples, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_theseus_registered() -> None:
    assert "theseus" in list_methods()
    assert get_method("theseus").name == "theseus"


def test_theseus_transport_smoke() -> None:
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
    method = get_method("theseus")

    transported = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
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


def test_theseus_transform_granularity_param_matches_default() -> None:
    torch.manual_seed(123)
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
    method = get_method("theseus")

    transported_default = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        strict=True,
    )

    transported_param = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        strict=True,
        transform_granularity="param",
    )

    assert set(transported_default.keys()) == set(transported_param.keys())
    for key in transported_default:
        assert torch.allclose(transported_default[key], transported_param[key])


def test_theseus_transform_granularity_block_smoke() -> None:
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
    method = get_method("theseus")

    transported = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        strict=True,
        transform_granularity="block",
    )

    assert transported
    assert set(transported.keys()) == set(delta.keys())
    for key, tensor in transported.items():
        assert tensor.shape == target_base[key].shape
        assert tensor.dtype == target_base[key].dtype


def test_theseus_transform_granularity_global_smoke() -> None:
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
    method = get_method("theseus")

    transported = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        strict=True,
        transform_granularity="global",
    )

    assert transported
    assert set(transported.keys()) == set(delta.keys())
    for key, tensor in transported.items():
        assert tensor.shape == target_base[key].shape
        assert tensor.dtype == target_base[key].dtype


def test_theseus_transform_granularity_module_type_smoke() -> None:
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
    method = get_method("theseus")

    transported = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        strict=True,
        transform_granularity="module_type",
    )

    assert transported
    assert set(transported.keys()) == set(delta.keys())
    for key, tensor in transported.items():
        assert tensor.shape == target_base[key].shape
        assert tensor.dtype == target_base[key].dtype


def test_theseus_transform_granularity_invalid_raises() -> None:
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
    method = get_method("theseus")

    try:
        method.transport(
            source_base=source_base,
            target_base=target_base,
            delta=delta,
            source_model=source_model,
            target_model=target_model,
            source_dataloader=loader,
            target_dataloader=loader,
            device="cpu",
            seq_align="mean",
            num_batches=1,
            strict=True,
            transform_granularity="not_a_mode",
        )
    except ValueError as exc:
        assert "transform_granularity" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid transform_granularity")


def test_fused_qkv_split_merge_roundtrip() -> None:
    w = torch.randn(12, 4)
    b = torch.randn(12)
    sd = {
        "transformer.resblocks.0.attn.in_proj_weight": w,
        "transformer.resblocks.0.attn.in_proj_bias": b,
        "transformer.resblocks.0.attn.out_proj.weight": torch.randn(4, 4),
    }

    split = theseus_mod._split_fused_qkv_state(sd)
    assert "transformer.resblocks.0.attn.q_proj.weight" in split
    assert "transformer.resblocks.0.attn.k_proj.weight" in split
    assert "transformer.resblocks.0.attn.v_proj.weight" in split
    assert "transformer.resblocks.0.attn.in_proj_weight" not in split

    merged = theseus_mod._merge_split_qkv_state(split, reference=sd)
    assert "transformer.resblocks.0.attn.in_proj_weight" in merged
    assert "transformer.resblocks.0.attn.in_proj_bias" in merged
    assert torch.allclose(merged["transformer.resblocks.0.attn.in_proj_weight"], w)
    assert torch.allclose(merged["transformer.resblocks.0.attn.in_proj_bias"], b)


def test_random_dataset_subsampling_uses_randperm_seed() -> None:
    x = torch.arange(20, dtype=torch.float32).unsqueeze(1)
    y = torch.zeros(20, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    iterator = theseus_mod._iter_random_dataset_batches(
        loader,
        loader,
        n_batches=3,
        seed=123,
        batch_size=4,
    )
    assert iterator is not None

    seen: list[int] = []
    for source_batch, _ in iterator:
        inputs = source_batch[0]
        seen.extend(int(v) for v in inputs.squeeze(1).tolist())

    g = torch.Generator(device="cpu")
    g.manual_seed(123)
    expected = torch.randperm(20, generator=g)[:12].tolist()
    assert seen == expected

    g2 = torch.Generator(device="cpu")
    g2.manual_seed(124)
    expected_other_seed = torch.randperm(20, generator=g2)[:12].tolist()
    assert seen != expected_other_seed


def test_transport_weight_proj_suffix_uses_proj_formula() -> None:
    delta = torch.randn(7, 5)
    t_in = torch.randn(7, 9)
    t_out = torch.randn(5, 11)

    proj_plain = theseus_mod._transport_weight(delta, t_in, t_out, key="proj")
    proj_prefixed = theseus_mod._transport_weight(delta, t_in, t_out, key="visual.proj")

    assert torch.allclose(proj_plain, proj_prefixed)
