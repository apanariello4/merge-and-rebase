#!/bin/bash
# GPU 0 eval worker (2026-11-19): every alpha-split evaluation of the chain_gpu1.sh fits, in order, plus the IFEval
# references. Eval-only (no fit), so it shares the machine with the GPU 1 fits; RAM guard 15 GB.
# Starts after lane 0 (its conservation evals on GPU 0) has exited.
R=/home/fillo/merge-and-rebase
E=/data/merge_and_rebase_results/metodoK_ifeval_synth_20261119
L=/data/merge_and_rebase_results/metodoK_llm_rt_20261118
SNAP=$R/results/_code_snapshots/20261119e_svdfix
LOG=$E/logs/evals_gpu0.log
CH=$E/logs/chain_gpu1.log
cd $R; source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       HF_HUB_CACHE=/data/merge_and_rebase_ckpts/hf_hub PYTHONPATH=$SNAP/src
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
echo "evals gpu0 waiting for lane 0 $(date)" >> $LOG
while kill -0 1866180 2>/dev/null; do sleep 60; done
echo "evals gpu0 start $(date)" >> $LOG
GSM="--target meta-llama/Llama-3.2-3B --grid 0,0.2,0.4,0.6,0.8,0.9,1.0,1.1,1.2,1.4,1.6,2.0 --test-alphas 1.0"
IFE="--task ifeval --fewshot 0 --n-calib 40 --n-sel 150 --batch 8 --select ifeval_avg_strict"
IFG="--grid 0,0.2,0.4,0.6,0.8,1.0,1.25,1.5,2.0,3.0 --test-alphas 1.0"
ev() {  # ev <out name> <args...>
  [ -e $E/STOP ] && { echo "STOP $(date)" >> $LOG; exit 0; }
  [ -e $E/logs/split_$1.json ] && return
  while [ $(avail) -lt 15 ]; do sleep 120; done
  local n=$1; shift
  echo "start $n split $(date) avail=$(avail)" >> $LOG
  python $SNAP/llm_alpha_split_eval.py $E/logs/split_$n.json "$@" > $E/logs/split_$n.log 2>&1
  rc=$?; echo "end $n split rc=$rc $(date)" >> $LOG
}
wait_fit() {  # wait_fit <name>: 0 if the fit ended rc=0, 1 otherwise
  until grep -q "end $1 fit rc=" $CH 2>/dev/null; do
    grep -qE "STOP|fits done" $CH 2>/dev/null && ! grep -q "start $1 fit" $CH && return 1
    sleep 120; done
  grep -q "end $1 fit rc=0" $CH
}
tv() { ls $1/*.pt 2>/dev/null | head -1; }
# IFEval references on the same 150 / 351 prompts: target base (alpha 0 of every run) and the native Instruct model
ev ref_qwen15_base --target Qwen/Qwen2.5-1.5B --grid 0 $IFE
ev ref_qwen15_instruct --target Qwen/Qwen2.5-1.5B-Instruct --grid 0 $IFE
if wait_fit LRT_kern65k_indep_gsm8k; then t=$(tv $L/tvs/LRT_kern65k_indep_gsm8k); [ -n "$t" ] && ev LRT_kern65k_indep_gsm8k --tv $t $GSM; fi
if wait_fit LRT_kern65k_seq_gsm8k_e; then t=$(tv $L/tvs/LRT_kern65k_seq_gsm8k); [ -n "$t" ] && ev LRT_kern65k_seq_gsm8k_e --tv $t $GSM; fi
for arm in real claude self wiki; do for m in kern ariadne ariadneO2; do n=IF_${m}_$arm
  if wait_fit $n; then t=$(tv $E/tvs/$n); [ -n "$t" ] && ev $n --tv $t --target Qwen/Qwen2.5-1.5B $IFE $IFG; fi
done; done
if wait_fit LRT_kern65k_seq_mathprompt_e; then t=$(tv $L/tvs/LRT_kern65k_seq_mathprompt); [ -n "$t" ] && ev LRT_kern65k_seq_mathprompt_e --tv $t $GSM; fi
if wait_fit LRT_kern65k_indep_gsm8k_e; then t=$(tv $L/tvs/LRT_kern65k_indep_gsm8k); [ -n "$t" ] && ev LRT_kern65k_indep_gsm8k_e --tv $t $GSM; fi
echo "evals done $(date)" >> $LOG
