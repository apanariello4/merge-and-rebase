#!/usr/bin/env python3
"""
Pre-download models + datasets needed for the MATH (Hendrycks/Minerva) eval.

Leonardo compute nodes have no network access, so everything an offline
`sbatch` job will touch (model weights, tokenizer, and the lm-eval task
dataset) must be fetched from a login node first, into an HF cache dir that
the compute job then reads with HF_HUB_OFFLINE=1.

Run this on a login node (has internet):

    python scripts/download_math_eval_assets.py \
        --cache-dir /leonardo_scratch/large/userexternal/$USER/.hf_cache

Then point HF_HOME / HF_HUB_CACHE at the same --cache-dir in the sbatch job.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Math-1.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen3-1.7B",
]

# Both hendrycks_math500 and minerva_math500 (lm-eval task defs) pull test
# examples from this single dataset repo; minerva's few-shot exemplars are
# hardcoded in lm-eval's own code, so no extra dataset is needed for that.
DEFAULT_DATASETS = [
    "HuggingFaceH4/MATH-500",
]


def download_models(models: list[str], cache_dir: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    for repo_id in models:
        print(f"\n=== Downloading model: {repo_id} ===")
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir / "hub"),
            token=token,
            # Skip duplicate non-safetensors weight formats to save bandwidth/disk.
            ignore_patterns=["*.bin", "*.bin.index.json", "*.msgpack", "*.h5", "*.pth"],
        )
        print(f"Done: {repo_id}")


def download_datasets(dataset_ids: list[str], cache_dir: Path, token: str | None) -> None:
    import datasets

    for repo_id in dataset_ids:
        print(f"\n=== Downloading dataset: {repo_id} ===")
        ds = datasets.load_dataset(repo_id, cache_dir=str(cache_dir / "datasets"), token=token)
        for split, d in ds.items():
            print(f"  split={split} rows={len(d)}")
        print(f"Done: {repo_id}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("HF_HOME", str(Path.home() / ".hf_cache")),
        help="HF_HOME-style cache root. Model weights go to <cache-dir>/hub, "
        "datasets go to <cache-dir>/datasets. Point the sbatch job's HF_HOME "
        "at the same directory.",
    )
    p.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--skip-datasets", action="store_true")
    p.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HF token, only needed for gated/private repos (none of the defaults are gated).",
    )
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"HF cache root: {cache_dir.resolve()}")

    if not args.skip_models:
        download_models(args.models, cache_dir, args.token)
    if not args.skip_datasets:
        download_datasets(args.datasets, cache_dir, args.token)

    print("\nAll downloads complete.")
    print(f"For the offline sbatch job, export:\n  HF_HOME={cache_dir}\n  HF_HUB_OFFLINE=1")


if __name__ == "__main__":
    main()
