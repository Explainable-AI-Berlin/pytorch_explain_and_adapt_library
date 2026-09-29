#!/bin/bash
#SBATCH --job-name=ADA_in1k_xin100_artifacts
#SBATCH --partition=cpu-2d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
SCRIPT="${REPO}/scripts/ada/actionability/build_in1k_k10_confirmatory_exclude_in100_artifacts.sh"

mkdir -p "${REPO}/logs"
cd "${REPO}"

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] SCRIPT=${SCRIPT}"
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
    export OMP_NUM_THREADS='${SLURM_CPUS_PER_TASK:-32}'
    export OPENBLAS_NUM_THREADS='${SLURM_CPUS_PER_TASK:-32}'
    export MKL_NUM_THREADS='${SLURM_CPUS_PER_TASK:-32}'
    bash '${SCRIPT}'
  "

echo "[DONE] DATE=$(date)"
