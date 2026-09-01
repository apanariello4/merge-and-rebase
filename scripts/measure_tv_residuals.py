"""Measure shared-versus-independent transported task-vector residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(directory: Path, task: str) -> dict[str, torch.Tensor]:
    return torch.load(directory / f"{task}_theseus_transported_native.pt", map_location="cpu", weights_only=True)


def _metrics(shared: dict[str, torch.Tensor], independent: dict[str, torch.Tensor]) -> dict[str, float | int]:
    keys = sorted(set(shared).intersection(independent))
    if not keys:
        raise ValueError("No common tensor keys.")
    shared_sq = independent_sq = residual_sq = dot = 0.0
    for key in keys:
        if shared[key].shape != independent[key].shape:
            raise ValueError(f"Shape mismatch for {key}: {tuple(shared[key].shape)} != {tuple(independent[key].shape)}")
        s = shared[key].float()
        i = independent[key].float()
        shared_sq += float(s.square().sum())
        independent_sq += float(i.square().sum())
        residual_sq += float((i - s).square().sum())
        dot += float((s * i).sum())
    shared_norm = shared_sq**0.5
    independent_norm = independent_sq**0.5
    residual_norm = residual_sq**0.5
    return {
        "key_count": len(keys),
        "shared_norm": shared_norm,
        "independent_norm": independent_norm,
        "residual_norm": residual_norm,
        "relative_residual_norm": residual_norm / shared_norm if shared_norm else float("inf"),
        "cosine_similarity": dot / (shared_norm * independent_norm) if shared_norm and independent_norm else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--independent-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    per_task = {}
    merged_shared: dict[str, torch.Tensor] = {}
    merged_independent: dict[str, torch.Tensor] = {}
    for task in args.tasks:
        shared = _load(args.shared_dir, task)
        independent = _load(args.independent_dir, task)
        per_task[task] = _metrics(shared, independent)
        for key in set(shared).intersection(independent):
            merged_shared[key] = merged_shared.get(key, 0) + shared[key].float()
            merged_independent[key] = merged_independent.get(key, 0) + independent[key].float()
    output = {
        "definition": "r_i = delta_i_independent - delta_i_shared; metrics are over the common transported target-delta keys.",
        "per_task": per_task,
        "equal_weight_task_arithmetic_merge": _metrics(merged_shared, merged_independent),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
