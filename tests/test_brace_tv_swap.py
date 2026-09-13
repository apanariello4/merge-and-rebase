from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import merge_and_rebase.eval.vision_brace_tv_swap as tv_swap
from merge_and_rebase.eval.vision_brace_tv_swap import (
    assert_endpoint_reconstruction,
    consensus_base,
    reconstruct_endpoint,
    state_dict_sha256,
    task_vector_from_endpoints,
    validate_artifact_bank,
)


def _base(value: float, counter: int = 1) -> dict[str, torch.Tensor]:
    return {
        "visual.weight": torch.tensor([value, value + 1], dtype=torch.float32),
        "visual.counter": torch.tensor(counter, dtype=torch.int64),
    }


def test_task_vector_exactly_reconstructs_endpoint() -> None:
    base = _base(1.0)
    ft = _base(1.0)
    ft["visual.weight"] = torch.tensor([3.0, -2.0])
    tv = task_vector_from_endpoints(base, ft)
    errors = assert_endpoint_reconstruction(base, ft, tv)
    assert errors["max_abs_error"] <= 1e-7
    assert torch.equal(reconstruct_endpoint(base, tv)["visual.weight"], ft["visual.weight"])


def test_consensus_averages_float_tensors_and_preserves_identical_buffers() -> None:
    consensus, distances = consensus_base({"EuroSAT": _base(1.0), "GTSRB": _base(3.0)})
    assert torch.equal(consensus["visual.weight"], torch.tensor([2.0, 3.0]))
    assert consensus["visual.counter"].item() == 1
    assert distances["EuroSAT"] == pytest.approx(distances["GTSRB"])


def test_consensus_rejects_different_nonfloating_buffers() -> None:
    with pytest.raises(ValueError, match="Non-floating"):
        consensus_base({"EuroSAT": _base(1.0, counter=1), "GTSRB": _base(3.0, counter=2)})


def test_validate_artifact_bank_rejects_tampered_task_vector(tmp_path) -> None:
    root = tmp_path / "capture"
    task_dir = root / "tasks" / "EuroSAT"
    task_dir.mkdir(parents=True)
    base = _base(1.0)
    ft = _base(1.0)
    ft["visual.weight"] = torch.tensor([2.0, 3.0])
    tv = task_vector_from_endpoints(base, ft)
    for name, state in (("base", base), ("ft", ft), ("tv", tv)):
        torch.save(state, task_dir / f"{name}.pt")
    (task_dir / "metadata.json").write_text(
        json.dumps({
            "base_sha256": state_dict_sha256(base),
            "ft_sha256": state_dict_sha256(ft),
            "tv_sha256": state_dict_sha256(tv),
        })
    )
    validate_artifact_bank(root, ["EuroSAT"])
    torch.save({"visual.weight": torch.zeros(2)}, task_dir / "tv.pt")
    with pytest.raises(ValueError, match="does not numerically reconstruct"):
        validate_artifact_bank(root, ["EuroSAT"])


def test_reconstruction_accepts_normal_fp32_subtraction_roundoff() -> None:
    base = {"visual.weight": torch.tensor([1.234567, -0.7654321], dtype=torch.float32)}
    ft = {"visual.weight": torch.tensor([1.234568, -0.7654319], dtype=torch.float32)}
    tv = task_vector_from_endpoints(base, ft)
    errors = assert_endpoint_reconstruction(base, ft, tv)
    assert errors["max_abs_error"] <= 1e-7


def test_expanded_template_calls_keyword_only_block_extension(monkeypatch) -> None:
    calls = []
    def fake_extension(*, source_base_model, source_ft_model, **kwargs):
        calls.append({"source_base_model": source_base_model, "source_ft_model": source_ft_model, **kwargs})
    monkeypatch.setattr(tv_swap, "run_block_extension", fake_extension)
    template = tv_swap._expanded_template(SimpleNamespace(model=nn.Linear(2, 2)), 2, "cpu")
    assert isinstance(template, nn.Module)
    assert calls and {"source_base_model", "source_ft_model", "calibration_loader", "target_layers_total", "config", "device"} <= set(calls[0])


def test_capture_writes_reconstruction_metadata_and_complete(monkeypatch, tmp_path) -> None:
    """Exercise capture through artifact finalization without loading CLIP or data."""

    class FakeVisual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.register_buffer("counter", torch.tensor(1, dtype=torch.int64))
            self.transformer = nn.Module()
            self.transformer.resblocks = nn.ModuleList([nn.Identity()])

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = FakeVisual()

    fake_classifier = SimpleNamespace(model=FakeModel(), preprocess=None)
    fake_ft_state = {"visual.weight": torch.tensor([3.0, 5.0])}

    class FakeOpenClipClassifier:
        @staticmethod
        def build(_build_cfg):
            return fake_classifier

    def fake_load_into_model(model, state, strict=False):
        del strict
        model.visual.weight.data.copy_(state["visual.weight"])

    monkeypatch.setattr(tv_swap, "OpenClipClassifier", FakeOpenClipClassifier)
    monkeypatch.setattr(
        tv_swap,
        "_task_loader_context",
        lambda **_kwargs: (SimpleNamespace(train=(), test=(), val=()), [], None),
    )
    monkeypatch.setattr(tv_swap, "resolve_ckpt_path", lambda path: tmp_path / path)
    monkeypatch.setattr(tv_swap, "load_ckpt", lambda _path: fake_ft_state)
    monkeypatch.setattr(tv_swap, "align_to_base_keys", lambda state, _base: state)
    monkeypatch.setattr(tv_swap, "load_into_model", fake_load_into_model)
    monkeypatch.setattr(
        tv_swap,
        "run_block_extension",
        lambda **_kwargs: 2,
    )

    artifact_root = tmp_path / "capture"
    manifest = tv_swap.capture(
        {
            "suite": "vision8",
            "tasks": "Cars",
            "source_clip_model": "fake",
            "source_clip_pretrained": "fake",
            "device": "cpu",
            "target_layers_total": 2,
            "tuned_ckpts": {"Cars": "cars.pt"},
            "block_extension_params": {"n_batches_act": 1},
        },
        condition="independent",
        artifact_root=artifact_root,
        tasks=["Cars"],
    )

    metadata = json.loads((artifact_root / "tasks" / "Cars" / "metadata.json").read_text())
    assert (artifact_root / "COMPLETE").is_file()
    assert manifest["condition"] == "independent"
    assert metadata["fp32_endpoint_reconstruction"]["max_abs_error"] == pytest.approx(0.0)
    assert torch.equal(
        torch.load(artifact_root / "tasks" / "Cars" / "tv.pt", weights_only=True)["visual.weight"],
        torch.tensor([2.0, 3.0]),
    )
