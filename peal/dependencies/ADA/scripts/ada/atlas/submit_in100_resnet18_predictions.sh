#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/atlas/run_module_single.sh"
VAL_CACHE="${VAL_CACHE:-${REPO}/artifacts/ada/atlas/embeddings/imagenet100/val/facebook__dinov2-with-registers-base/cls/cache-6ec1e8a1b614f414}"
PRED_ROOT="${PRED_ROOT:-${REPO}/artifacts/ada/atlas/predictions/e0_in100}"
IMAGENET_META="${IMAGENET_META:-/home/space/datasets/imagenet_torchvision/data/meta.bin}"

args="--query-cache ${VAL_CACHE} --output-dir ${PRED_ROOT}/resnet18_imagenet1k_restricted100 --model-name resnet18 --imagenet-meta ${IMAGENET_META} --batch-size 128"

job=$(sbatch --parsable \
  --job-name="ADA_in100_resnet18_pred" \
  --export=ALL,REPO="${REPO}",MODULE="ada.atlas.cli.predict_torchvision_supervised",MODULE_ARGS="${args}" \
  "${RUNNER}")

echo "resnet18_predictions ${job}"
