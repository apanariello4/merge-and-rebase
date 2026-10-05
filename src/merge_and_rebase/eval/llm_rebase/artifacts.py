"""Saving the best-alpha rebased state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...merge.runtime import apply_delta, to_cpu_fp32


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
