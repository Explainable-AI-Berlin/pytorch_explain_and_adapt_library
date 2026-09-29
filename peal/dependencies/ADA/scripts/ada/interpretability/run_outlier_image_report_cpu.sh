#!/bin/bash
#SBATCH --job-name=ADA_outlier_desc
#SBATCH --partition=cpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
CONFIG="${CONFIG:-${REPO}/configs/ada/interpretability/in100_outlier_image_descriptions.yaml}"

cd "${REPO}"
apptainer exec \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export PYTHONPATH=src
    export PYTHONDONTWRITEBYTECODE=1
    python3 -m ada.interpretability.cli.build_outlier_image_report --config '${CONFIG}' --overwrite
  "
