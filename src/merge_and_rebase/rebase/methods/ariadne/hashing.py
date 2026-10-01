"""Stable hashing of task-vector-shaped tensor mappings."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch


def _task_vector_sha256(sd: Mapping[str, torch.Tensor]) -> str:
    """Stable CPU hash of a task-vector-shaped tensor mapping.

    Algorithm: sorted keys, dtype, shape, raw bytes. This is the single
    implementation; ``vision_rebase._state_dict_sha256`` is an alias of it, so any
    caller hashing the same dict through either name gets the same digest.
    """
    digest = hashlib.sha256()
    for key in sorted(sd):
        value = sd[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(memoryview(value.numpy()))
    return digest.hexdigest()
