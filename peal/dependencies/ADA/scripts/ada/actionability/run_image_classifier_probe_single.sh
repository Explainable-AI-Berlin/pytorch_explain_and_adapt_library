#!/bin/bash
#SBATCH --job-name=ADA_imgclf_probe
#SBATCH --partition=gpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
CONFIG="${CONFIG:?Set CONFIG to an image-classifier probe config}"
MANIFEST_INDEX="${MANIFEST_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

cd "${REPO}"
echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "[JOB] MANIFEST_INDEX=${MANIFEST_INDEX}"
echo "[JOB] CONFIG=${CONFIG}"
echo "============================================================"

apptainer exec --nv \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export PYTHONPATH=src
    export PYTHONDONTWRITEBYTECODE=1
    python3 -m ada.actionability.cli.train_image_classifier \
      --config '${CONFIG}' \
      --manifest-index '${MANIFEST_INDEX}'
  "

echo "[DONE] DATE=$(date)"
