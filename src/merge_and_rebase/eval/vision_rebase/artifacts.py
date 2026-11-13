"""Saved transported task-vector load helpers and legacy visual key mapping."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...rebase.config_schema import canonicalize
from ...rebase.methods._ariadne.config import direct_residual_config_dict, normalize_direct_residual_config_dict
from ...rebase.methods._ariadne.fit import _task_vector_sha256
from ...rebase.registry import canonical_method_name
from ..utils import to_cpu_fp32


def _load_saved_sequential_tv(directory, task, target_base_sd, config):
    """Load a write-once sequential DR vector and verify its fit provenance."""
    root = Path(directory)
    # The saver names files ``{task}_{method.name}_transported_native.{pt,json}`` where ``method.name`` is the
    # spelling the run used: the canonical name or the legacy alias. Accept either, never both.
    candidates = []
    for method_name in dict.fromkeys((canonical_method_name("direct_residual"), "direct_residual")):
        candidate_path = root / f"{task}_{method_name}_transported_native.pt"
        candidate_meta = root / f"{task}_{method_name}_transported_native.json"
        if candidate_path.exists() or candidate_meta.exists():
            candidates.append((candidate_path, candidate_meta))
    if len(candidates) > 1:
        raise ValueError(
            f"ambiguous saved DR vectors for task {task!r} in {root}: found both "
            + " and ".join(str(c[0]) for c in candidates)
        )
    if not candidates:
        names = list(dict.fromkeys((canonical_method_name("direct_residual"), "direct_residual")))
        raise FileNotFoundError(
            f"no saved DR vector for task {task!r} in {root}: expected "
            + " or ".join(f"{task}_{n}_transported_native.json" for n in names)
        )
    path, meta_path = candidates[0]
    meta = json.loads(meta_path.read_text())
    expected = {
        "task": task,
        "endpoint_construction": config.endpoint_construction,
        "target_base_sha256": _state_dict_sha256(target_base_sd),
        "calibration_seed": config.seed,
        "num_batches": config.num_batches,
        "direct_residual_config": json.loads(json.dumps(direct_residual_config_dict(config))),
    }
    for key, value in expected.items():
        actual = meta.get(key)
        if key == "direct_residual_config" and isinstance(actual, Mapping):
            actual = normalize_direct_residual_config_dict(actual)
        if actual != value:
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


_SOURCE_SUMMARY_KEYS = (
    "target_clip_model",
    "target_clip_pretrained",
    "source_clip_model",
    "source_clip_pretrained",
)
_C_PROJ_WEIGHT = re.compile(r"(?:^|\.)resblocks\.(\d+)\.mlp\.c_proj\.weight$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def check_precomputed_source_summary(summary_path, cfg) -> None:
    """The run summary that produced the vectors must name the same source/target CLIP models as this run."""
    data = json.loads(Path(summary_path).read_text())
    resolved = data["run_logging"]["metadata"]["resolved_config"]
    old = canonicalize(resolved)
    new = canonicalize(cfg)
    for key in _SOURCE_SUMMARY_KEYS:
        if old.get(key) != new.get(key):
            raise ValueError(
                f"precomputed_tv_source_summary {summary_path}: {key}={old.get(key)!r}, this run has {new.get(key)!r}"
            )


def _load_precomputed_tv(directory, task, target_base_sd):
    """Load one task's saved transported delta (no sidecar): validate keys/shapes/finiteness, return (vector, meta)."""
    root = Path(directory)
    names = list(dict.fromkeys((canonical_method_name("direct_residual"), "direct_residual")))
    found = [root / f"{task}_{n}_transported_native.pt" for n in names]
    found = [p for p in found if p.exists()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"precomputed_tv_dir {root}: expected exactly one of "
            + " or ".join(f"{task}_{n}_transported_native.pt" for n in names)
            + f", found {len(found)}"
        )
    path = found[0]
    vector = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(vector, dict) or not vector:
        raise ValueError(f"precomputed task vector {path} is empty or invalid")
    for key, tensor in vector.items():
        if key not in target_base_sd or tensor.shape != target_base_sd[key].shape:
            raise ValueError(f"precomputed task vector {path}: key {key!r} missing from the target or shape mismatch")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"precomputed task vector {path}: non-finite values in {key!r}")
    return vector, {"path": str(path), "file_sha256": _file_sha256(path)}


def truncate_tv_rank(vector, *, energy: float, blocks: str, depth: int):
    """Rank-truncate the ``mlp.c_proj.weight`` deltas of the selected blocks to the smallest k holding ``energy``.

    Returns ``(new_vector, record)``; ``energy == 1.0`` returns the tensors untouched (no SVD). Biases and unselected
    blocks are never modified. The record lists, per truncated key, k, min_dim, kept energy fraction, Frobenius norms
    before/after, plus the sha256 of every final tensor.
    """
    if not 0.0 < energy <= 1.0:
        raise ValueError(f"tv_rank_truncation.energy must be in (0, 1], got {energy}")
    if blocks not in ("all", "last_third"):
        raise ValueError(f"tv_rank_truncation.blocks must be 'all' or 'last_third', got {blocks!r}")
    first = 0 if blocks == "all" else depth - depth // 3
    out = dict(vector)
    keys: dict[str, Any] = {}
    for key, tensor in vector.items():
        match = _C_PROJ_WEIGHT.search(key)
        if match is None or int(match.group(1)) < first:
            continue
        if energy < 1.0:
            mat = tensor.detach().to(torch.float64)
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
            cum = torch.cumsum(s * s, dim=0)
            total = cum[-1]
            k = int(torch.searchsorted(cum, energy * total).item()) + 1 if total > 0 else 0
            k = min(k, s.numel())
            kept = float((cum[k - 1] / total).item()) if k > 0 else 1.0
            recon = (u[:, :k] * s[:k]) @ vh[:k]
            out[key] = recon.to(device=tensor.device, dtype=tensor.dtype)
            min_dim = int(s.numel())
        else:
            k, min_dim, kept = min(tensor.shape), min(tensor.shape), 1.0
        keys[key] = {
            "k": k,
            "min_dim": min_dim,
            "kept_energy_fraction": kept,
            "frobenius_before": float(torch.linalg.norm(tensor.double()).item()),
            "frobenius_after": float(torch.linalg.norm(out[key].double()).item()),
            "sha256": _tensor_sha256(out[key]),
        }
    return out, {"energy": energy, "blocks": blocks, "depth": depth, "keys": keys}


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


_SAVE_POLICIES = ("if_dir_given", "auto")
_SEQUENTIAL_ENDPOINT_CONSTRUCTIONS = {"sequential_source_endpoints", "sequential_delta_on_synthesized_base"}


@dataclass(frozen=True)
class TransportedTvSaveSpec:
    """How a run saves transported task vectors (config keys ``save_transported_*``).

    ``save_policy="if_dir_given"`` (the default) is the historical behaviour: vectors are saved only when
    ``save_transported_tvs_dir`` is given. ``"auto"`` derives ``<summary_dir>/transported_tvs`` when no
    directory is given, and also saves the merged single-transport delta under an additive name.
    """

    directory: str | None
    artifacts: bool
    legacy: bool
    method_name: str
    save_policy: str = "if_dir_given"
    #: ``True`` only when the config named ``save_transported_tvs`` (then, and only then, the summary records it).
    policy_explicit: bool = False
    #: The Ariadne config when its sequential endpoint construction makes the vectors write-once with a sidecar.
    sequential_config: Any | None = None

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        method_name: str,
        direct_fit: bool,
        ariadne_cfg: Any,
        summary_dir: str | os.PathLike[str] | None,
    ) -> TransportedTvSaveSpec:
        policy_explicit = "save_transported_tvs" in cfg
        policy = cfg.get("save_transported_tvs", "if_dir_given")
        if policy not in _SAVE_POLICIES:
            raise ValueError(f"save_transported_tvs must be one of {list(_SAVE_POLICIES)}; got {policy!r}.")
        directory = cfg.get("save_transported_tvs_dir", None)
        if policy == "auto" and not directory and summary_dir is not None:
            directory = os.path.join(os.fspath(summary_dir), "transported_tvs")
        default_artifacts = bool(directory)
        sequential = direct_fit and ariadne_cfg.endpoint_construction in _SEQUENTIAL_ENDPOINT_CONSTRUCTIONS
        return cls(
            directory=directory,
            artifacts=bool(cfg.get("save_transported_artifacts", default_artifacts)),
            legacy=bool(cfg.get("save_transported_tvs_legacy", False)),
            method_name=method_name,
            save_policy=str(policy),
            policy_explicit=policy_explicit,
            sequential_config=ariadne_cfg if sequential else None,
        )

    def summary_record(self) -> dict[str, Any] | None:
        """The additive ``save_policy`` summary entry; ``None`` (key omitted) unless the config named the policy."""
        if not self.policy_explicit:
            return None
        return {"save_transported_tvs": self.save_policy, "transported_tvs_dir": self.directory}


def save_transported_task_vector(
    spec: TransportedTvSaveSpec,
    name: str,
    delta: dict[str, torch.Tensor],
    *,
    target_base_sd: Mapping[str, torch.Tensor],
    merged: bool = False,
) -> list[str] | None:
    """Save one transported vector; returns the written paths (``None`` when saving is off).

    Per-task vectors keep the historical file names (``{task}_{method}_transported_native.pt`` and the optional
    legacy visual variants), the sequential-Ariadne sidecar JSON and its write-once rule. ``merged=True`` is the
    ``save_transported_tvs="auto"`` extension: the merged single-transport delta, named by ``name`` (no sidecar,
    also write-once).
    """
    if spec.artifacts and not spec.directory:
        raise ValueError("save_transported_artifacts=true requires save_transported_tvs_dir.")
    if not (spec.artifacts and spec.directory):
        return None
    os.makedirs(spec.directory, exist_ok=True)
    native_path = os.path.join(spec.directory, f"{name}_{spec.method_name}_transported_native.pt")
    sequential_cfg = None if merged else spec.sequential_config
    if sequential_cfg is not None:
        if os.path.exists(native_path) or os.path.exists(os.path.splitext(native_path)[0] + ".json"):
            raise FileExistsError(f"refusing to overwrite sequential DR vector: {native_path}")
    elif merged and os.path.exists(native_path):
        raise FileExistsError(f"refusing to overwrite merged transported vector: {native_path}")
    torch.save(to_cpu_fp32(delta), native_path)
    if sequential_cfg is not None:
        meta_path = os.path.splitext(native_path)[0] + ".json"
        metadata = {
            "task": name,
            "endpoint_construction": sequential_cfg.endpoint_construction,
            "target_base_sha256": _state_dict_sha256(target_base_sd),
            "vector_sha256": _state_dict_sha256(delta),
            "calibration_seed": sequential_cfg.seed,
            "num_batches": sequential_cfg.num_batches,
            "direct_residual_config": direct_residual_config_dict(sequential_cfg),
        }
        Path(meta_path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"  {name}: saved {'merged ' if merged else ''}transported TV -> {native_path}")
    paths = [native_path]
    if spec.legacy and not merged:
        legacy_path = os.path.join(spec.directory, f"{name}_{spec.method_name}_transported_legacy_visual.pt")
        legacy_no_conv1_path = os.path.join(
            spec.directory, f"{name}_{spec.method_name}_transported_legacy_visual_no_conv1.pt"
        )
        torch.save(_legacy_visual_delta(delta), legacy_path)
        torch.save(_legacy_visual_delta(delta, drop_conv1=True), legacy_no_conv1_path)
        print(f"  {name}: saved legacy visual TV -> {legacy_path}")
        print(f"  {name}: saved legacy visual TV without conv1 -> {legacy_no_conv1_path}")
        paths += [legacy_path, legacy_no_conv1_path]
    return paths


class TransportedTvSaver:
    """Run-level saver: the one call site of :func:`save_transported_task_vector`; records the written paths."""

    def __init__(self, spec: TransportedTvSaveSpec, env: Any) -> None:
        self.spec = spec
        self.env = env
        #: ``name -> written paths`` (the summary's ``transported_artifacts``).
        self.artifacts: dict[str, list[str]] = {}

    def __call__(self, name: str, delta: dict[str, torch.Tensor], *, merged: bool = False) -> None:
        # ``env.target_base_sd`` is read per call: TransFusion's once-only prepare swaps it.
        paths = save_transported_task_vector(
            self.spec, name, delta, target_base_sd=self.env.target_base_sd, merged=merged
        )
        if paths is not None:
            self.artifacts[name] = paths

    @property
    def merged_single_transport_enabled(self) -> bool:
        """``save_transported_tvs="auto"``: also save the merged single-transport delta."""
        return self.spec.save_policy == "auto"
