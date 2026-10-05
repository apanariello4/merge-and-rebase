"""Self-contained canonical hashing for the release golden-hash suite.

Deliberately imports nothing from ``merge_and_rebase``: the repo's own hashing
helpers (``target_informed_runtime._task_vector_sha256``,
``vision_rebase._state_dict_sha256``, ...) will move during the refactors this
suite guards, and a golden hash that moves with its own hasher pins nothing.

The digest of a ``dict[str, Tensor]`` covers, for every key in sorted order: the
key name, the dtype, the shape and the raw little-endian bytes of the detached,
contiguous CPU tensor. Every field is length-prefixed, so no two distinct
dictionaries can serialize to the same byte stream.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import struct
from collections.abc import Iterator, Mapping
from typing import Any

import torch


def _frame(h: hashlib._Hash, payload: bytes) -> None:
    h.update(struct.pack("<Q", len(payload)))
    h.update(payload)


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Raw bytes of ``tensor`` (bfloat16 is viewed as int16, numpy has no such dtype)."""
    t = tensor.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def hash_tensor_dict(tensors: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 over sorted keys: name, dtype, shape and raw bytes of each tensor."""
    if not tensors:
        # An empty result hashes stably too, which would pin a silently broken path.
        raise ValueError("hash_tensor_dict: refusing to hash an empty tensor dict")
    h = hashlib.sha256()
    for key in sorted(tensors):
        t = tensors[key]
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"hash_tensor_dict: {key!r} is {type(t).__name__}, expected Tensor")
        _frame(h, key.encode("utf-8"))
        _frame(h, str(t.dtype).encode("ascii"))
        _frame(h, repr(tuple(t.shape)).encode("ascii"))
        _frame(h, tensor_bytes(t))
    return h.hexdigest()


def flatten_tensors(obj: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten nested mapping/list/tuple containers of tensors into ``{path: tensor}``."""
    out: dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj
    elif isinstance(obj, Mapping):
        for key in obj:
            out.update(flatten_tensors(obj[key], f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            out.update(flatten_tensors(item, f"{prefix}/{i}" if prefix else str(i)))
    else:
        raise TypeError(f"flatten_tensors: unsupported leaf {type(obj).__name__} at {prefix!r}")
    return out


# Whole word-tokens (keys are split on non-alphanumerics) that mark a summary field as
# varying between otherwise identical runs: wall clock, memory peaks, filesystem
# locations, VCS fingerprints. Token-based so that e.g. ``direction`` is NOT dropped by ``dir``.
VOLATILE_KEY_TOKENS = frozenset(
    {
        "time",
        "times",
        "timing",
        "timings",
        "timestamp",
        "elapsed",
        "seconds",
        "duration",
        "peak",
        "memory",
        "rss",
        "path",
        "paths",
        "dir",
        "git",
        "commit",
        "dirty",
        "host",
        "hostname",
        "pid",
    }
)


# Exact keys that embed a process-specific value (``dataset_identity`` is built from ``id(dataset)``).
VOLATILE_KEYS = frozenset({"dataset_identity"})


def is_volatile_key(key: Any) -> bool:
    if str(key) in VOLATILE_KEYS:
        return True
    return any(tok in VOLATILE_KEY_TOKENS for tok in re.split(r"[^a-z0-9]+", str(key).lower()))


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"__tensor__": hash_tensor_dict({"t": value})}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [_jsonable(v) for v in items]
    if isinstance(value, float):
        # repr round-trips a double exactly; json would too, but nan/inf are made explicit.
        return repr(value)
    return value


def strip_volatile(obj: Any) -> Any:
    """Recursively drop mapping keys containing a volatile substring (see VOLATILE_KEY_TOKENS)."""
    if isinstance(obj, Mapping):
        return {k: strip_volatile(v) for k, v in obj.items() if not is_volatile_key(k)}
    if isinstance(obj, (list, tuple)):
        return [strip_volatile(v) for v in obj]
    return obj


def hash_json(obj: Any) -> str:
    """SHA-256 of ``json.dumps(sort_keys=True)`` over a volatile-stripped, tensor-hashed object."""
    payload = json.dumps(_jsonable(strip_volatile(obj)), sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@contextlib.contextmanager
def deterministic_cpu(seed: int = 0) -> Iterator[None]:
    """Single-thread, deterministic-algorithms, freshly seeded; previous state restored on exit."""
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn_only)
        torch.set_num_threads(prev_threads)
