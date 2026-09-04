#!/usr/bin/env python3
"""Aggregate completed ViT-H/14 Vision8 fine-tuning summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/leonardo_scratch/large/userexternal/frinaldi/finetuning/finetune"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    task_root = args.root / "ViT-H-14" / "laion2b_s32b_b79k"
    rows = []
    missing = []
    for task in TASKS:
        summary_path = task_root / task / "full.json"
        checkpoint_path = task_root / task / "full_best_ep.pt"
        if not summary_path.exists() or not checkpoint_path.exists():
            missing.append(task)
            continue
        payload = json.loads(summary_path.read_text())
        metrics = payload.get("metrics", {})
        test_top1 = metrics.get("test_top1")
        if test_top1 is None:
            raise ValueError(f"Missing metrics.test_top1 in {summary_path}")
        rows.append({"task": task, "test_top1": float(test_top1), "summary": str(summary_path), "checkpoint": str(checkpoint_path)})

    if missing:
        raise SystemExit(f"Missing completed task artifacts: {', '.join(missing)}")

    result = {
        "model": "ViT-H-14",
        "pretrained": "laion2b_s32b_b79k",
        "suite": "vision8",
        "tasks": rows,
        "macro_test_top1": sum(row["test_top1"] for row in rows) / len(rows),
    }
    output = args.output or task_root / "vision8_aggregate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved aggregate: {output}")


if __name__ == "__main__":
    main()
