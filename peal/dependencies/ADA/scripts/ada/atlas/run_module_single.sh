#!/bin/bash
#SBATCH --job-name=ADA_atlas_module
#SBATCH --partition=gpu-2h
#SBATCH --constraint=40gb|80gb|h100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
MODULE="${MODULE:?MODULE is required, for example ada.atlas.cli.score_knn}"
MODULE_ARGS="${MODULE_ARGS:-}"

mkdir -p "${REPO}/logs"
cd "${REPO}"

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] MODULE=${MODULE}"
echo "[JOB] MODULE_ARGS=${MODULE_ARGS}"
echo "============================================================"

nvidia-smi -L || true

apptainer exec --nv \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export PYTHONPATH=src
    export PYTHONDONTWRITEBYTECODE=1
    export TF_ENABLE_ONEDNN_OPTS=0
    python3 -m '${MODULE}' ${MODULE_ARGS}
  "

echo "[DONE] DATE=$(date)"
