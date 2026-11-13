#!/usr/bin/env bash
# Build the two task-data-free calibration sources for all Vision8 tasks of ONE direction on ONE GPU.
#   usage: scripts/calibsrc_build_all.sh <extend|shrink> <gpu> <inversion_iters> <ariadne_run_config.json>
# Environment (optional): CALIBSRC_OUT (default results/calibsrc), CALIBSRC_CACHE_DIR (pool-embedding cache,
#   default calibsrc_cache), PYTHON (default python). Run from the repository root.
# Outputs: $CALIBSRC_OUT/<direction>/{retrieval,inversion}/<Task>.{pt,manifest.json}
# Existing outputs are skipped. Retrieval first (cheap), then inversion.
set -u
if [ $# -ne 4 ]; then sed -n 3,5p "$0"; exit 2; fi
DIR=$1; GPU=$2; ITERS=$3; CFG=$4
OUT=${CALIBSRC_OUT:-results/calibsrc}
PY=${PYTHON:-python}
export PYTHONPATH=$PWD/src HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT/logs"
for SRC in retrieval inversion; do
  for T in DTD EuroSAT MNIST SVHN GTSRB RESISC45 Cars SUN397; do
    if [ -f "$OUT/$DIR/$SRC/$T.manifest.json" ]; then echo "skip $DIR $SRC $T"; continue; fi
    /usr/bin/time -v "$PY" -m merge_and_rebase.eval.vision_rebase.build_calibration_images \
      --config "$CFG" --source $SRC --task $T --iters "$ITERS" --out-dir "$OUT/$DIR/$SRC" \
      > "$OUT/logs/${SRC}_${DIR}_$T.log" 2>&1
    rc=$?
    echo "$(date -Is) $DIR $SRC $T rc=$rc"
  done
done
