"""Saved transported task-vector load helpers and legacy visual key mapping."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from ..rebase.methods.ariadne.fit import _task_vector_sha256


def _load_saved_sequential_tv(directory, task, target_base_sd, config):
    """Load a write-once sequential DR vector and verify its fit provenance."""
    root = Path(directory)
    path = root / f"{task}_direct_residual_transported_native.pt"
    meta_path = root / f"{task}_direct_residual_transported_native.json"
    meta = json.loads(meta_path.read_text())
    expected = {
        "task": task,
        "endpoint_construction": config.endpoint_construction,
        "target_base_sha256": _state_dict_sha256(target_base_sd),
        "calibration_seed": config.seed,
        "num_batches": config.num_batches,
        "direct_residual_config": json.loads(json.dumps(asdict(config))),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"saved DR vector {path}: {key}={meta.get(key)!r}, expected {value!r}")
    vector = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(vector, dict) or not vector:
        raise ValueError(f"saved DR vector {path} is empty or invalid")
    for key, tensor in vector.items():
        if (
            key not in target_base_sd
            or tensor.shape != target_base_sd[key].shape
            or not torch.isfinite(tensor).all()
            or ".mlp.c_proj." not in key
        ):
            raise ValueError(f"saved DR vector {path} has invalid tensor {key}")
    if _state_dict_sha256(vector) != meta.get("vector_sha256"):
        raise ValueError(f"saved DR vector {path} failed its tensor hash check")
    return vector, {**meta, "path": str(path), "metadata_path": str(meta_path)}


def _legacy_visual_key(key: str) -> str | None:
    if not key.startswith("visual."):
        return None
    out = key[len("visual.") :]
    replacements = (
        (".attn.q_proj.", ".attn.q."),
        (".attn.k_proj.", ".attn.k."),
        (".attn.v_proj.", ".attn.v."),
        (".attn.out_proj.", ".attn.proj."),
        (".mlp.c_fc.", ".mlp.fc1."),
        (".mlp.c_proj.", ".mlp.fc2."),
    )
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


def _legacy_visual_delta(delta: dict[str, torch.Tensor], *, drop_conv1: bool = False) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in delta.items():
        legacy_key = _legacy_visual_key(key)
        if legacy_key is None:
            continue
        if drop_conv1 and legacy_key == "conv1.weight":
            continue
        out[legacy_key] = value.detach().to(device="cpu", dtype=torch.float32)
    return out


# Stable CPU hash used to prove that the native target base was not mutated.
# Same algorithm (sorted keys, dtype, shape, raw bytes) as the Ariadne task-vector hash.
_state_dict_sha256 = _task_vector_sha256
