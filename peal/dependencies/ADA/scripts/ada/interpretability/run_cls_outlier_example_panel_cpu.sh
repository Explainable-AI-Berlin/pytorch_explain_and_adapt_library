#!/bin/bash
#SBATCH --job-name=ADA_cls_outlier_plot
#SBATCH --partition=cpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"

OUTLIER_CSV="${OUTLIER_CSV:-${REPO}/artifacts/ada/interpretability/i0c_in100_outlier_image_descriptions/outlier_image_descriptions.csv}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO}/artifacts/ada/interpretability/manifests/in100_eight_causal_regions}"
DELETION_EVALUATION_CSV="${DELETION_EVALUATION_CSV:-${REPO}/artifacts/ada/actionability/evaluation/e4a_in100_dinov2_cls_k10_deletion_probe_pilot_full_seed0/deletion_evaluation.csv}"
TRAIN_IMAGE_ROOT="${TRAIN_IMAGE_ROOT:-/home/space/datasets/imagenet_torchvision/imagenet100/train}"
VALIDATION_IMAGE_ROOT="${VALIDATION_IMAGE_ROOT:-/home/space/datasets/imagenet_torchvision/imagenet100/val}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/artifacts/ada/interpretability/i0d_in100_cls_outlier_example_panel}"
MAX_IMAGES="${MAX_IMAGES:-12}"
AVERAGE_MAX_IMAGES="${AVERAGE_MAX_IMAGES:-96}"
BASELINE_SEED="${BASELINE_SEED:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-176}"

mkdir -p "${REPO}/logs"
cd "${REPO}"

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] OUTPUT_DIR=${OUTPUT_DIR}"
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
    python3 -m ada.interpretability.cli.plot_cls_outlier_examples \
      --outlier-csv '${OUTLIER_CSV}' \
      --manifest-dir '${MANIFEST_DIR}' \
      --deletion-evaluation-csv '${DELETION_EVALUATION_CSV}' \
      --train-image-root '${TRAIN_IMAGE_ROOT}' \
      --validation-image-root '${VALIDATION_IMAGE_ROOT}' \
      --output-dir '${OUTPUT_DIR}' \
      --max-images '${MAX_IMAGES}' \
      --average-max-images '${AVERAGE_MAX_IMAGES}' \
      --baseline-seed '${BASELINE_SEED}' \
      --image-size '${IMAGE_SIZE}' \
      --overwrite
  "

echo "[DONE] DATE=$(date)"
