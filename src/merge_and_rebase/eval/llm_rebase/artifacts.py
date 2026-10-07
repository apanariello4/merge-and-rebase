"""Saving the best-alpha rebased state."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ...merge.runtime import apply_delta, to_cpu_fp32
from ...rebase.methods._ariadne.fit import _task_vector_sha256
from ...run_logging import _code_fingerprint

#: Config keys naming the checkpoints a transported vector was built from (recorded in its sidecar).
_REF_KEYS = (
    "source_model_name_or_path",
    "target_model_name_or_path",
    "source_base_ckpt",
    "target_base_ckpt",
    "tuned_bodies",
)


def save_merged_state(
    path: Any,
    merged_delta: dict[str, torch.Tensor],
    best_alpha: float,
    target_base_sd: dict[str, torch.Tensor],
    *,
    message: str,
) -> None:
    scaled = {k: v * best_alpha for k, v in merged_delta.items()}
    best_sd = apply_delta(target_base_sd, scaled)
    outp = Path(str(path))
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(to_cpu_fp32(best_sd), str(outp))
    print(f"{message} {outp}")


def save_transported_task_vector(
    cfg: Mapping[str, Any],
    merged_delta: dict[str, torch.Tensor],
    *,
    method_name: str,
    best_alpha: float,
    alpha_curve: Any,
) -> dict[str, Any] | None:
    """Save the unscaled transported (merged) task vector and a JSON sidecar; ``None`` when saving is off.

    Mirrors the vision ``save_transported_task_vector``: config keys ``save_transported_tvs_dir`` and
    ``save_transported_artifacts`` (default off, no behaviour change), the vector in fp32 on CPU, write-once.
    The sidecar carries the payload sha256 (sorted keys, dtype, shape, raw bytes: the Ariadne task-vector hash),
    the selected alpha, the full alpha score curve, the method and the model references. Reconstruct the
    rebased state as ``target_base + best_alpha * tau``. Returns the summary record (path, sha256, sidecar).
    """
    directory = cfg.get("save_transported_tvs_dir", None)
    if cfg.get("save_transported_artifacts", False) and not directory:
        raise ValueError("save_transported_artifacts=true requires save_transported_tvs_dir.")
    if not directory:
        return None
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"merged_{method_name}_transported_native.pt")
    meta_path = os.path.splitext(path)[0] + ".json"
    if os.path.exists(path) or os.path.exists(meta_path):
        raise FileExistsError(f"refusing to overwrite transported vector: {path}")
    tau = to_cpu_fp32(merged_delta)
    sha256 = _task_vector_sha256(tau)
    torch.save(tau, path)
    metadata = {
        "method": method_name,
        "vector_sha256": sha256,
        "best_alpha": float(best_alpha),
        "alpha_curve": alpha_curve,
        "refs": {k: cfg.get(k) for k in _REF_KEYS},
        **_code_fingerprint(),
    }
    Path(meta_path).write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
    print(f"Saved transported TV -> {path}")
    return {"path": path, "sha256": sha256, "sidecar": meta_path}
