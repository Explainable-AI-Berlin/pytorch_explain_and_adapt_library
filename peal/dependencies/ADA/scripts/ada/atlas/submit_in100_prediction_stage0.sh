#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/atlas/run_module_single.sh"
TRAIN_CACHE="${TRAIN_CACHE:-${REPO}/artifacts/ada/atlas/embeddings/imagenet100/train/facebook__dinov2-with-registers-base/cls/combined_train_full_4shards}"
VAL_CACHE="${VAL_CACHE:-${REPO}/artifacts/ada/atlas/embeddings/imagenet100/val/facebook__dinov2-with-registers-base/cls/cache-6ec1e8a1b614f414}"
PRED_ROOT="${PRED_ROOT:-${REPO}/artifacts/ada/atlas/predictions/e0_in100}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip2-base-patch16-256}"
IMAGENET_META="${IMAGENET_META:-/home/space/datasets/imagenet_torchvision/data/meta.bin}"

mkdir -p "${PRED_ROOT}"

submit_module() {
  local name="$1"
  local module="$2"
  local module_args="$3"
  sbatch --parsable \
    --job-name="${name}" \
    --export=ALL,REPO="${REPO}",MODULE="${module}",MODULE_ARGS="${module_args}" \
    "${RUNNER}"
}

knn_args="--query-cache ${VAL_CACHE} --reference-cache ${TRAIN_CACHE} --output-csv ${PRED_ROOT}/dino_knn/predictions.csv --k 1:5:10 --batch-size 256"
knn_job=$(submit_module "ADA_in100_knn_pred" "ada.atlas.cli.predict_knn" "${knn_args}")
echo "knn_predictions ${knn_job}"

linear_args="--train-cache ${TRAIN_CACHE} --query-cache ${VAL_CACHE} --output-dir ${PRED_ROOT}/dino_linear_probe_seed0 --epochs 40 --batch-size 2048 --lr 0.01 --weight-decay 0.0001 --calibration-fraction 0.1 --seed 0"
linear_job=$(submit_module "ADA_in100_dino_linear" "ada.atlas.cli.train_dino_linear_probe" "${linear_args}")
echo "dino_linear_probe ${linear_job}"

siglip_args="--query-cache ${VAL_CACHE} --output-dir ${PRED_ROOT}/siglip2_zeroshot_smoke --model-name ${SIGLIP_MODEL} --imagenet-meta ${IMAGENET_META} --batch-size 32 --max-samples 128"
siglip_job=$(submit_module "ADA_in100_siglip_smoke" "ada.atlas.cli.predict_siglip_zeroshot" "${siglip_args}")
echo "siglip2_zeroshot_smoke ${siglip_job}"
