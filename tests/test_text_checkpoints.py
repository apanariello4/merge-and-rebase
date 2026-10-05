from __future__ import annotations

import pytest
import torch

from merge_and_rebase.io.text_checkpoints import (
    HeadCheckpointError,
    is_adapter_reference,
    resolve_checkpoint_reference,
)


def test_is_adapter_reference_local_file() -> None:
    assert not is_adapter_reference("model.pt")
    assert not is_adapter_reference("checkpoint.bin")
    assert not is_adapter_reference("weights.safetensors")


def test_is_adapter_reference_hf_hub() -> None:
    assert is_adapter_reference("username/my-lora-adapter")


def test_is_adapter_reference_local_dir(tmp_path) -> None:
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    assert is_adapter_reference(str(d))


def test_head_checkpoint_rejected(tmp_path) -> None:
    p = tmp_path / "head_only.pt"
    torch.save({"format": "head", "head": {"score.weight": torch.zeros(3, 4)}}, p)

    with pytest.raises(HeadCheckpointError):
        resolve_checkpoint_reference(str(p))




def test_full_checkpoint_accepted(tmp_path) -> None:
    p = tmp_path / "full.pt"
    torch.save({"format": "full", "state_dict": {"w": torch.zeros(3, 4)}}, p)
    result = resolve_checkpoint_reference(str(p))
    assert result == str(p)


def test_unrecognized_file_returns_ref(tmp_path) -> None:
    p = tmp_path / "unknown.pt"
    torch.save({"some_key": "value"}, p)
    result = resolve_checkpoint_reference(str(p))
    assert result == str(p)
