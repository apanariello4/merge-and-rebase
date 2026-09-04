#!/usr/bin/env bash
#SBATCH --job-name=vith14_v8_ft
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrB_MOSAIC
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=450G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7
#SBATCH --output=slurm_logs/vith14_v8_ft.%A_%a.out
#SBATCH --error=slurm_logs/vith14_v8_ft.%A_%a.err

set -euo pipefail

REPO="/leonardo/home/userexternal/frinaldi/merge-and-rebase"
PYTHON="${REPO}/.venv/bin/python"
CONFIG="${REPO}/src/merge_and_rebase/finetune/configs/vision-ft-vith14-laion.yaml"
HF_HOME="/leonardo_scratch/large/userexternal/frinaldi/.hf_cache"
TASKS=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)

TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this script as an eight-task array}"
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#TASKS[@]} )); then
  echo "Invalid task index: ${TASK_INDEX}" >&2
  exit 2
fi
TASK="${TASKS[$TASK_INDEX]}"

cd "${REPO}"
mkdir -p slurm_logs

export HF_HOME
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -d "${HF_DATASETS_CACHE}" ]]; then
  echo "Missing dataset cache: ${HF_DATASETS_CACHE}" >&2
  exit 3
fi

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
MANIFEST_DIR="${REPO}/results/finetune/vith14_laion2b_vision8/${RUN_DATE}"
MANIFEST_PATH="${MANIFEST_DIR}/task_${TASK}.json"
CHECKPOINT="/leonardo_scratch/large/userexternal/frinaldi/finetuning/finetune/ViT-H-14/laion2b_s32b_b79k/${TASK}/full_best_ep.pt"
SUMMARY="/leonardo_scratch/large/userexternal/frinaldi/finetuning/finetune/ViT-H-14/laion2b_s32b_b79k/${TASK}/full.json"
LOG_PATH="${REPO}/slurm_logs/vith14_v8_ft.${SLURM_ARRAY_JOB_ID}_${TASK_INDEX}.out"

export TASK CONFIG CHECKPOINT SUMMARY LOG_PATH

mkdir -p "${MANIFEST_DIR}"
"${PYTHON}" - "${MANIFEST_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest_path.write_text(json.dumps({
    "job_id": os.environ["SLURM_JOB_ID"],
    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "array_task_id": int(os.environ["SLURM_ARRAY_TASK_ID"]),
    "task": os.environ["TASK"],
    "config": os.environ["CONFIG"],
    "checkpoint": os.environ["CHECKPOINT"],
    "summary": os.environ["SUMMARY"],
    "log": os.environ["LOG_PATH"],
    "status": "started",
}, indent=2) + "\n")
PY

echo "Running ${TASK} with ${CONFIG}"
"${PYTHON}" scripts/cache_vith14_laion.py --verify
"${PYTHON}" -m merge_and_rebase.finetune.train_vision \
  --vision-config "${CONFIG}" \
  --datasets "${TASK}" \
  --device cuda \
  --skip-existing-task-vectors

"${PYTHON}" - "${MANIFEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = json.loads(manifest_path.read_text())
payload["status"] = "success"
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
