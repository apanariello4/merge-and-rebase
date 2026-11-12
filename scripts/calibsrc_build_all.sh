#!/usr/bin/env bash
# Build the two task-data-free calibration sources for all Vision8 tasks of ONE direction on ONE GPU.
#   usage: scripts_calibsrc_build_all.sh <extend|shrink> <gpu> <inversion_iters>
# Outputs: /data/merge_and_rebase_results/calibsrc/<direction>/{retrieval,inversion}/<Task>.{pt,manifest.json}
# Existing outputs are skipped. Retrieval first (cheap), then inversion.
set -u
DIR=$1; GPU=$2; ITERS=$3
WT=/data/merge_and_rebase_wt/ariadne_release_calibsrc
OUT=/data/merge_and_rebase_results/calibsrc
CFG=/home/fillo/merge-and-rebase/configs/final/rebuttal_cal54_20261107/ariadne_${DIR}_cal54.json
export PYTHONPATH=$WT/src HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$GPU
PY=/home/fillo/merge-and-rebase/.venv/bin/python
mkdir -p $OUT/logs
for SRC in retrieval inversion; do
  for T in DTD EuroSAT MNIST SVHN GTSRB RESISC45 Cars SUN397; do
    if [ -f $OUT/$DIR/$SRC/$T.manifest.json ]; then echo "skip $DIR $SRC $T"; continue; fi
    /usr/bin/time -v $PY -m merge_and_rebase.eval.vision_rebase.build_calibration_images \
      --config $CFG --source $SRC --task $T --iters $ITERS --out-dir $OUT/$DIR/$SRC \
      > $OUT/logs/${SRC}_${DIR}_$T.log 2>&1
    rc=$?
    echo "$(date -Is) $DIR $SRC $T rc=$rc"
  done
done
