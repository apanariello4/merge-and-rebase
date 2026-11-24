#!/bin/bash
# GPU 1 worker (2026-11-24): experiment 5 with calibration_streaming (code snapshot 20261124_streaming, reviewed).
# 1) Equivalence check at 40 prompts: IFS_kernO2_claude40s vs IFS_kernO2_claude40 and IFS_ariadneO2_claude40s vs
#    IF_ariadneO2_claude (same configs + calibration_streaming true). Gate: global relative L2 difference of the
#    transported vectors < 1e-3 (kernel) / 1e-2 (ARIADNE, Procrustes route, reviewer: may exceed rounding).
# 2) If a method passes: its 160s and 320s fits, then IFEval split evals (same args as before) of all passing runs.
# Lane 1 (vision) is held with STOP_GPU1 (its running item finishes first) and relaunched at the end.
R=/home/fillo/merge-and-rebase
E=/data/merge_and_rebase_results/metodoK_ifeval_synth_20261119
K=$R/results/metodoK_kernel_20261118
SNAP=$R/results/_code_snapshots/20261124_streaming
LOG=$E/logs/gpu1_streaming.log
cd $R; source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       HF_HUB_CACHE=/data/merge_and_rebase_ckpts/hf_hub PYTHONPATH=$SNAP/src MALLOC_ARENA_MAX=2 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
touch $K/STOP_GPU1; echo "STOP_GPU1 set by gpu1_streaming.sh $(date)" >> $K/logs/queue_gpu1.log
echo "start $(date): waiting for vision lane 1 to exit" >> $LOG
while pgrep -f "queue.sh 1 " >/dev/null; do sleep 120; done
echo "lane 1 exited $(date)" >> $LOG
fit() {  # fit <name>
  [ -e $E/STOP ] && return 1
  while [ $(avail) -lt 60 ]; do sleep 120; done
  echo "start $1 fit $(date) avail=$(avail)" >> $LOG
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 /usr/bin/time -v -o $E/logs/$1.time \
    python -m merge_and_rebase.eval.llm_rebase --config $E/configs/$1.json > $E/logs/$1.log 2>&1
  local rc=$?; echo "end $1 fit rc=$rc $(date) peakRSS_GB=$(awk '/Maximum resident/{print int($6/1048576)}' $E/logs/$1.time)" >> $LOG
  return $rc
}
cmp() {  # cmp <new> <ref> <threshold>  -> 0 if pass
  local a=$(ls $E/tvs/$2/*.pt 2>/dev/null | head -1) b=$(ls $E/tvs/$1/*.pt 2>/dev/null | head -1)
  [ -n "$a" ] && [ -n "$b" ] || { echo "cmp $1: missing tv" >> $LOG; return 1; }
  local o=$(python $E/compare_tvs.py $a $b $E/logs/cmp_$1.json); echo "cmp $1 vs $2: $o" >> $LOG
  python -c "import json,sys;sys.exit(0 if json.load(open('$E/logs/cmp_$1.json'))['global_rel']<$3 else 1)"
}
PASS=""
fit IFS_kernO2_claude40s && cmp IFS_kernO2_claude40s IFS_kernO2_claude40 1e-3 && PASS="$PASS kernO2"
fit IFS_ariadneO2_claude40s && cmp IFS_ariadneO2_claude40s IF_ariadneO2_claude 1e-2 && PASS="$PASS ariadneO2"
echo "passed:$PASS $(date)" >> $LOG
RUNS=""
for m in $PASS; do for s in 160 320; do fit IFS_${m}_claude${s}s && RUNS="$RUNS IFS_${m}_claude${s}s"; done; done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
IFE="--task ifeval --fewshot 0 --n-calib 40 --n-sel 150 --batch 8 --select ifeval_avg_strict"
IFG="--grid 0,0.2,0.4,0.6,0.8,1.0,1.25,1.5,2.0,3.0 --test-alphas 1.0"
for n in $RUNS; do
  t=$(ls $E/tvs/$n/*.pt 2>/dev/null | head -1); [ -n "$t" ] || continue
  while [ $(avail) -lt 15 ]; do sleep 120; done
  echo "start $n split $(date)" >> $LOG
  python $SNAP/llm_alpha_split_eval.py $E/logs/split_$n.json --tv $t --target Qwen/Qwen2.5-1.5B $IFE $IFG > $E/logs/split_$n.log 2>&1
  rc=$?; echo "end $n split rc=$rc $(date)" >> $LOG
done
rm -f $K/STOP_GPU1; echo "STOP_GPU1 removed by gpu1_streaming.sh $(date)" >> $K/logs/queue_gpu1.log
# relaunch the remaining vision lane-1 items (queue.sh skips completed ones) without the LLM cache variable
cd $K; env -u HF_HUB_CACHE -u PYTHONPATH -u PYTORCH_CUDA_ALLOC_CONF SNAP=$R/results/_code_snapshots/20261118b_kernel nohup bash $K/queue.sh 1 "v:KRT_track_O2 v:KRT_track_O2_nofilt v:KRT_track_O2_alltok v:KRT_track_O2_generic v:KRT_indep_O2" > $K/logs/lane1f.out 2>&1 &
echo "lane 1 relaunched; gpu1_streaming done $(date)" >> $LOG
