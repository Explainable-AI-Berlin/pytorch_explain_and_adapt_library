#!/bin/bash
#SBATCH --job-name=ADA_siglip2_assets
#SBATCH --partition=cpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
MODEL_ID="${MODEL_ID:-google/siglip2-base-patch16-256}"
REVISION="${REVISION:-3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/external/models/google__siglip2-base-patch16-256}"

cd "${REPO}"
echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] MODEL_ID=${MODEL_ID}"
echo "[JOB] REVISION=${REVISION}"
echo "[JOB] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "============================================================"

apptainer exec \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export PYTHONPATH=src
    export PYTHONDONTWRITEBYTECODE=1
    python3 scripts/ada/interpretability/download_siglip2_shared_assets.py \
      --model-id '${MODEL_ID}' \
      --revision '${REVISION}' \
      --output-root '${OUTPUT_ROOT}'
  "

echo "[DONE] DATE=$(date)"
