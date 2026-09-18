from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.methods import theseus as theseus_mod
from merge_and_rebase.rebase.registry import get_method, list_methods
from merge_and_rebase.rebase.runtime import format_rebase_method_label


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


def test_theseus_strict_apply_rejects_missing_transform() -> None:
    with pytest.raises(RuntimeError, match="missing_transform_zero=1"):
        theseus_mod._apply_transforms_to_visual_delta(
            target_visual_base={"weight": torch.zeros(2, 2)},
            visual_delta={"weight": torch.ones(2, 2)},
            transforms_by_key={},
            show_progress=False,
            method_name="theseus",
            device="cpu",
            strict=True,
        )


def test_transport_diagnostics_partition_zero_and_active_paths(caplog, capsys) -> None:
    identity = torch.eye(2)
    target = {
        "active": torch.zeros(2, 2),
        "class_embedding": torch.zeros(2),
        "missing": torch.zeros(2, 2),
        "unsupported": torch.zeros(1, 1, 1),
        "failure": torch.zeros(2, 2),
        "wrong_shape": torch.zeros(3, 3),
    }
    delta = {key: torch.ones_like(value) for key, value in target.items()}
    delta["wrong_shape"] = torch.ones(2, 2)
    transforms = {
        "active": theseus_mod._LayerTransform(kind="weight", t_in=identity, t_out=identity),
        "class_embedding": theseus_mod._LayerTransform(kind="zero"),
        "unsupported": theseus_mod._LayerTransform(kind="unsupported"),
        "failure": theseus_mod._LayerTransform(kind="weight", t_in=torch.eye(3), t_out=identity),
        "wrong_shape": theseus_mod._LayerTransform(kind="weight", t_in=identity, t_out=identity),
    }

    with caplog.at_level("WARNING"):
        aligned, diagnostics = theseus_mod._apply_transforms_to_visual_delta(
            target_visual_base=target,
            visual_delta=delta,
            transforms_by_key=transforms,
            show_progress=False,
            method_name="theseus",
            device="cpu",
            out_of_scope_keys=("out_scope",),
            skipped_not_in_target_keys=("not_in_target",),
        )

    assert diagnostics.actively_transported == 1
    assert diagnostics.intentional_zero == 1
    assert diagnostics.missing_transform_zero == 1
    assert diagnostics.unsupported_zero == 1
    assert diagnostics.transport_failure_zero == 1
    assert diagnostics.wrong_shape_zero == 1
    assert diagnostics.out_of_scope_zero == 1
    assert diagnostics.skipped_not_in_target == 1
    assert set(aligned) == set(target)
    assert any("theseus transport diagnostics" in record.message for record in caplog.records)

    theseus_mod._report_apply_diagnostics(
        method_name="theseus", diagnostics=diagnostics, verbose=True
    )
    report = capsys.readouterr().out
    assert "active=1 matrices=1 vectors=0" in report
    assert "missing_transform_zero=1" in report
    assert "intentional_zero=1" in report


def test_strict_allows_intentional_and_out_of_scope_zero() -> None:
    _, diagnostics = theseus_mod._apply_transforms_to_visual_delta(
        target_visual_base={"class_embedding": torch.zeros(2)},
        visual_delta={"class_embedding": torch.ones(2)},
        transforms_by_key={"class_embedding": theseus_mod._LayerTransform(kind="zero")},
        show_progress=False,
        method_name="theseus",
        device="cpu",
        strict=True,
        out_of_scope_keys=("text_projection",),
    )
    assert diagnostics.intentional_zero == 1
    assert diagnostics.out_of_scope_zero == 1


def test_data_free_precompute_marks_every_structural_exclusion_as_intentional_zero() -> None:
    tensors = {
        "class_embedding": torch.ones(2),
        "positional_embedding": torch.ones(3, 2),
        "conv1.weight": torch.ones(2, 2, 1, 1),
    }
    transforms = theseus_mod._precompute_transforms_data_free(
        source_visual_base=tensors,
        target_visual_base=tensors,
        visual_delta=tensors,
        whiten_power=0.0,
        whiten_eps=1e-6,
        show_progress=False,
        method_name="theseus",
    )
    assert set(transforms) == set(tensors)
    assert all(transform.kind == "zero" for transform in transforms.values())


def test_theseus_rejects_misaligned_calibration_labels() -> None:
    source_model = _TinyModel()
    target_model = _TinyModel()
    x = torch.randn(4, 6)
    source_loader = DataLoader(TensorDataset(x, torch.zeros(4, dtype=torch.long)), batch_size=4)
    target_loader = DataLoader(TensorDataset(x, torch.ones(4, dtype=torch.long)), batch_size=4)

    with pytest.raises(ValueError, match="not label-aligned"):
        theseus_mod.collect_activations(
            source_model,
            target_model,
            source_loader,
            target_loader,
            device="cpu",
            seq_align="mean",
            n_batches=1,
        )


@pytest.mark.parametrize(
    ("whiten_power", "whiten_eps"),
    [(-0.01, 1e-5), (0.51, 1e-5), (0.0, 0.0)],
)
def test_theseus_rejects_invalid_whitening_parameters(whiten_power, whiten_eps):
    method = theseus_mod.TheseusRebase()
    with pytest.raises(ValueError):
        method.prepare(
            source_model=nn.Linear(2, 2),
            target_model=nn.Linear(2, 2),
            source_dataloader=[],
            target_dataloader=[],
            device="cpu",
            patch_qkv=False,
            whiten_power=whiten_power,
            whiten_eps=whiten_eps,
            show_progress=False,
            verbose=False,
        )


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
    # A transport that assigns no transform still returns correctly shaped
    # zeros, which every other assertion here accepts.
    assert any(float(t.float().abs().sum()) > 0.0 for t in transported.values())


def test_partial_whitening_changes_alignment_map() -> None:
    store = theseus_mod.ActivationStore(store_a_gram=True, store_b_gram=True)
    source_rows = torch.tensor([[3.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target_rows = torch.tensor([[1.0, 2.0], [2.0, 0.0], [0.0, 1.0]])
    store.update(source_rows, target_rows)

    raw_map = theseus_mod._compute_alignment_map(
        store,
        center=False,
        whiten_power=0.0,
        whiten_eps=1e-6,
    )
    whitened_map = theseus_mod._compute_alignment_map(
        store,
        center=False,
        whiten_power=0.5,
        whiten_eps=1e-6,
    )

    assert raw_map is not None
    assert whitened_map is not None
    assert raw_map.shape == whitened_map.shape == (2, 2)
    assert not torch.allclose(raw_map, whitened_map)


def test_theseus_transport_with_partial_whitening_smoke() -> None:
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
        whiten_power=0.25,
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


def test_theseus_data_free_transport_smoke_without_dataloaders() -> None:
    source_model = _TinyModel(in_dim=6, hid_dim=8, out_dim=5)
    target_model = _TinyModel(in_dim=6, hid_dim=7, out_dim=5)

    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    target_base = {k: v.detach().clone() for k, v in target_model.state_dict().items()}

    delta = {
        key: torch.randn_like(tensor)
        for key, tensor in source_base.items()
        if key.startswith("visual.") and tensor.is_floating_point()
    }

    method = get_method("theseus")
    transported = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        device="cpu",
        covariance_mode="data_free",
        whiten_power=0.25,
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


def test_data_free_transforms_handle_biases_and_zero_keys() -> None:
    source_weight = torch.eye(4)
    target_weight = torch.eye(6)
    transforms = theseus_mod._precompute_transforms_data_free(
        source_visual_base={
            "class_embedding": torch.ones(4),
            "transformer.resblocks.0.mlp.c_fc.weight": source_weight,
            "transformer.resblocks.0.mlp.c_fc.bias": torch.ones(4),
        },
        target_visual_base={
            "class_embedding": torch.ones(6),
            "transformer.resblocks.0.mlp.c_fc.weight": target_weight,
            "transformer.resblocks.0.mlp.c_fc.bias": torch.ones(6),
        },
        visual_delta={
            "class_embedding": torch.ones(4),
            "transformer.resblocks.0.mlp.c_fc.weight": torch.ones_like(source_weight),
            "transformer.resblocks.0.mlp.c_fc.bias": torch.ones(4),
        },
        whiten_power=0.0,
        whiten_eps=1e-6,
        show_progress=False,
        method_name="theseus",
    )

    assert transforms["class_embedding"].kind == "zero"
    bias_transform = transforms["transformer.resblocks.0.mlp.c_fc.bias"]
    assert bias_transform.kind == "bias"
    assert bias_transform.t_out is not None


def test_prepare_collects_grams_when_whitening_is_enabled(monkeypatch) -> None:
    source_model = _TinyModel()
    target_model = _TinyModel()
    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    delta = {"visual.fc1.weight": torch.ones_like(source_base["visual.fc1.weight"])}
    seen: dict[str, object] = {}

    def fake_collect(*args, **kwargs):
        seen["store_a_gram"] = kwargs["store_a_gram"]
        seen["store_b_gram"] = kwargs["store_b_gram"]
        return {}

    monkeypatch.setattr(theseus_mod, "collect_activations", fake_collect)
    theseus_mod.TheseusRebase().prepare(
        source_model=source_model,
        target_model=target_model,
        source_dataloader=[],
        target_dataloader=[],
        target_base={k: v.detach().clone() for k, v in target_model.state_dict().items()},
        delta=delta,
        device="cpu",
        patch_qkv=False,
        whiten_power=0.25,
        verbose=False,
        show_progress=False,
    )

    assert seen == {"store_a_gram": True, "store_b_gram": True}


def test_data_free_covariance_map_uses_weight_proxies() -> None:
    source = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    target = source @ rotation

    t_in = theseus_mod._compute_alignment_map_from_matrix_proxies(
        source,
        target,
        side="input",
        whiten_power=0.0,
        whiten_eps=1e-6,
    )

    assert t_in.shape == (2, 2)
    source_cov = source.T @ source
    target_cov = target.T @ target
    aligned_cov = t_in.T @ source_cov @ t_in
    assert torch.allclose(aligned_cov, target_cov, atol=1e-5, rtol=1e-5)


def test_theseus_runtime_label_includes_covariance_mode() -> None:
    label = format_rebase_method_label(
        "theseus",
        {"num_batches": 3, "seq_align": "mean", "covariance_mode": "data_free", "whiten_power": 0.25},
    )
    assert label == "theseus(batches=3, align=mean, cov=data_free, whiten=0.25)"


def test_data_free_proj_transform_uses_swapped_axes() -> None:
    source_ref = torch.randn(8, 5)
    target_ref = torch.randn(7, 6)
    visual_delta = {"proj": torch.randn_like(source_ref)}
    transforms = theseus_mod._precompute_transforms_data_free(
        source_visual_base={"proj": source_ref},
        target_visual_base={"proj": target_ref},
        visual_delta=visual_delta,
        whiten_power=0.0,
        whiten_eps=1e-6,
        show_progress=False,
        method_name="theseus",
    )

    transform = transforms["proj"]
    assert transform.t_in is not None
    assert transform.t_out is not None
    assert transform.t_in.shape == (8, 7)
    assert transform.t_out.shape == (5, 6)


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


def test_content_row_mask_and_padding_drop():
    """Padded calibration positions must not reach the cross-covariance.

    Text calibration batches are padded to a fixed length, so on short prompts
    most positions are pad tokens. Folding them into the Procrustes covariance
    lets padding dominate the fitted alignment.
    """
    import torch

    from merge_and_rebase.rebase.methods.theseus import _content_row_mask, _drop_padding_rows

    # batch=2, tokens=4, with 3 real tokens then 1 pad in each row.
    attn = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    mask = _content_row_mask(attn, attn)
    assert mask is not None
    assert mask.tolist() == [True, True, True, False, True, True, True, False]

    src = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    tgt = torch.arange(8 * 5, dtype=torch.float32).reshape(8, 5)
    src_kept, tgt_kept = _drop_padding_rows(src, tgt, mask)
    assert src_kept.shape == (6, 3)
    assert tgt_kept.shape == (6, 5)
    assert torch.equal(src_kept, src[mask])
    assert torch.equal(tgt_kept, tgt[mask])

    # No mask, an all-real mask, and an all-pad mask all leave rows untouched.
    assert _content_row_mask(None, None) is None
    assert _content_row_mask(torch.ones(2, 4, dtype=torch.long), None) is None
    assert _content_row_mask(torch.zeros(2, 4, dtype=torch.long), None) is None

    # Source and target tokenized to different lengths -> rows no longer
    # correspond one-to-one, so don't guess.
    assert _content_row_mask(attn, torch.ones(2, 6, dtype=torch.long)) is None

    # Activations that aren't one row per input token (pooled/head-split) are
    # passed through rather than mis-sliced.
    pooled_src = torch.zeros(2, 3)
    pooled_tgt = torch.zeros(2, 5)
    out_src, out_tgt = _drop_padding_rows(pooled_src, pooled_tgt, mask)
    assert out_src.shape == (2, 3)
    assert out_tgt.shape == (2, 5)
