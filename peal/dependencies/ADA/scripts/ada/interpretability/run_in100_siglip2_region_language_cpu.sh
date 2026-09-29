#!/bin/bash
#SBATCH --job-name=ADA_siglip_lang_cpu
#SBATCH --partition=cpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
CONFIG="${CONFIG:-${REPO}/configs/ada/interpretability/in100_siglip2_region_language.yaml}"

cd "${REPO}"
echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] CONFIG=${CONFIG}"
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
    python3 -m ada.interpretability.cli.build_phrase_bank --config '${CONFIG}' --overwrite
    python3 -m ada.interpretability.cli.build_region_interpretability_manifest --config '${CONFIG}' --overwrite
    python3 -m ada.interpretability.cli.cache_vlm_shared_features --config '${CONFIG}' --device cpu --overwrite
    python3 -m ada.interpretability.cli.build_region_language_cards --config '${CONFIG}' --overwrite
  "

echo "[DONE] DATE=$(date)"
