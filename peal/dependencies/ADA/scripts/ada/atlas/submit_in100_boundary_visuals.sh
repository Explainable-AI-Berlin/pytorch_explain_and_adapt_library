#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/atlas/run_module_single.sh"
ENC_ROOT="${REPO}/artifacts/ada/atlas/embeddings/imagenet100"
VAL_CACHE="${VAL_CACHE:-${ENC_ROOT}/val/facebook__dinov2-with-registers-base/cls/cache-6ec1e8a1b614f414}"
TRAIN_CACHE="${TRAIN_CACHE:-${ENC_ROOT}/train/facebook__dinov2-with-registers-base/cls/combined_train_full_4shards}"
PRED_ROOT="${PRED_ROOT:-${REPO}/artifacts/ada/atlas/predictions/e0_in100}"
OUT_ROOT="${OUT_ROOT:-${REPO}/artifacts/ada/atlas/reports/e0_6_in100_boundary_support}"

DINO_PRED="${DINO_PRED:-${PRED_ROOT}/dino_linear_probe_seed0/predictions.csv}"
RESNET_PRED="${RESNET_PRED:-${PRED_ROOT}/resnet18_imagenet1k_restricted100/predictions.csv}"

mkdir -p "${REPO}/logs" "${OUT_ROOT}"

if [[ ! -d "${VAL_CACHE}" ]]; then
  echo "[error] missing VAL_CACHE: ${VAL_CACHE}" >&2
  exit 1
fi
if [[ ! -d "${TRAIN_CACHE}" ]]; then
  echo "[error] missing TRAIN_CACHE: ${TRAIN_CACHE}" >&2
  exit 1
fi
if [[ ! -f "${DINO_PRED}" ]]; then
  echo "[error] missing DINO_PRED: ${DINO_PRED}" >&2
  exit 1
fi
if [[ ! -f "${RESNET_PRED}" ]]; then
  echo "[error] missing RESNET_PRED: ${RESNET_PRED}" >&2
  exit 1
fi

boundary_args="--query-cache ${VAL_CACHE} --reference-cache ${TRAIN_CACHE} --output-dir ${OUT_ROOT} --k 1:5:10:50 --entropy-k 10:50 --global-search-k 512 --batch-size 256 --reference-shard-size 50000 --device auto --mmap --class-key class_name --reference-percentile-cache ${OUT_ROOT}/reference_class_percentiles.npz --prediction-csv ${DINO_PRED} --prediction-csv ${RESNET_PRED}"
boundary_job="$(
  sbatch --parsable \
    --job-name=ADA_in100_boundary \
    --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.score_boundary_metrics,MODULE_ARGS="${boundary_args}" \
    "${RUNNER}"
)"
echo "boundary ${boundary_job}"

visual_args="--query-cache ${VAL_CACHE} --metrics-csv ${OUT_ROOT}/boundary_metrics.csv --output-dir ${OUT_ROOT}/cls_space --max-points 5000 --title 'ImageNet-100 CLS Boundary Atlas' --prediction-csv ${DINO_PRED} --prediction-csv ${RESNET_PRED}"
visual_job="$(
  sbatch --parsable \
    --dependency="afterok:${boundary_job}" \
    --job-name=ADA_in100_cls_view \
    --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.visualize_cls_space,MODULE_ARGS="${visual_args}" \
    "${RUNNER}"
)"
echo "visualize ${visual_job}"
