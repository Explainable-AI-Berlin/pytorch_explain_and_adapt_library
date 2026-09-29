#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/atlas/run_module_single.sh"
PRED_ROOT="${PRED_ROOT:-${REPO}/artifacts/ada/atlas/predictions/e0_in100}"
SUPPORT_CSV="${SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/scores/e0_in100_dinov2_val_vs_train/support_scores.csv}"
OUTDIR="${OUTDIR:-${REPO}/artifacts/ada/atlas/reports/e0_in100_prediction_support}"

args="--support-csv ${SUPPORT_CSV} --prediction-csv ${PRED_ROOT}/dino_knn/predictions.csv --prediction-csv ${PRED_ROOT}/dino_linear_probe_seed0/predictions.csv --prediction-csv ${PRED_ROOT}/resnet18_imagenet1k_restricted100/predictions.csv --output-dir ${OUTDIR} --bins 10"

job=$(sbatch --parsable \
  --job-name="ADA_in100_pred_summary" \
  --export=ALL,REPO="${REPO}",MODULE="ada.atlas.cli.summarize_prediction_support",MODULE_ARGS="${args}" \
  "${RUNNER}")

echo "prediction_summary ${job}"
