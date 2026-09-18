#!/bin/bash
# Download HF model snapshots with curl into plain local directories (no
# huggingface_hub cache format, no symlinks) so they can be passed directly
# as `model_name_or_path` to configs and loaded fully offline on compute
# nodes (which have no network access -- see scripts/download_math_eval_assets.py
# for the huggingface_hub-based equivalent used for the qwen2.5/qwen3 math run).
#
# Run this on a LOGIN node (has internet):
#   bash scripts/download_models_curl.sh [dest_root]
#
# dest_root defaults to $CACHE_ROOT/models_local (CACHE_ROOT defaults to
# /leonardo_scratch/large/userexternal/$USER/.hf_cache).

set -euo pipefail

CACHE_ROOT="${CACHE_ROOT:-/leonardo_scratch/large/userexternal/$USER/.hf_cache}"
DEST_ROOT="${1:-$CACHE_ROOT/models_local}"

declare -A REPO_FILES
REPO_FILES["Qwen/Qwen2-1.5B"]="config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json model.safetensors"
REPO_FILES["Qwen/Qwen2-Math-1.5B"]="config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json model.safetensors"
REPO_FILES["Qwen/Qwen2.5-3B"]="config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors"
REPO_FILES["Qwen/Qwen2-Math-1.5B-Instruct"]="config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json model.safetensors"
REPO_FILES["Qwen/Qwen2-1.5B-Instruct"]="config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json model.safetensors"
REPO_FILES["Qwen/Qwen2-Math-7B"]="config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json model.safetensors.index.json model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors"

for repo in "${!REPO_FILES[@]}"; do
  name="$(basename "$repo")"
  dest="$DEST_ROOT/$name"
  mkdir -p "$dest"
  echo "=== $repo -> $dest ==="
  for f in ${REPO_FILES[$repo]}; do
    out="$dest/$f"
    if [ -s "$out" ]; then
      echo "  skip (exists): $f"
      continue
    fi
    url="https://huggingface.co/$repo/resolve/main/$f"
    echo "  fetching: $f"
    curl -fL --retry 5 --retry-delay 5 -o "$out.part" "$url"
    mv "$out.part" "$out"
  done
done

echo
echo "Done. Point configs at the local dirs, e.g.:"
for repo in "${!REPO_FILES[@]}"; do
  name="$(basename "$repo")"
  echo "  $repo -> $DEST_ROOT/$name"
done
