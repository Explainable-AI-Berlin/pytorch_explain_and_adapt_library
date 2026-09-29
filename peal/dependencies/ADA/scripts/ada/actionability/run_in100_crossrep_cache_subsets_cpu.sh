#!/bin/bash
#SBATCH --job-name=ADA_crossrep_cache
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
OVERWRITE="${OVERWRITE:-0}"

TRAIN_IN100_DINO2="${REPO}/artifacts/ada/atlas/embeddings/imagenet100/train/facebook__dinov2-with-registers-base/cls/combined_train_full_4shards"
VAL_IN100_DINO2="${REPO}/artifacts/ada/atlas/embeddings/imagenet100/val/facebook__dinov2-with-registers-base/cls/cache-6ec1e8a1b614f414"

TRAIN_IN1K_DINOV3="${REPO}/artifacts/ada/atlas/embeddings/imagenet1k/train/facebook__dinov3-vitb16-pretrain-lvd1689m/cls/combined_train_full_8shards"
VAL_IN1K_DINOV3="${REPO}/artifacts/ada/atlas/embeddings/imagenet1k/val/facebook__dinov3-vitb16-pretrain-lvd1689m/cls/cache-63bb2429628246b8"
TRAIN_IN100_DINOV3="${REPO}/artifacts/ada/atlas/embeddings/imagenet100/train/facebook__dinov3-vitb16-pretrain-lvd1689m/cls/aligned_from_imagenet1k_full"
VAL_IN100_DINOV3="${REPO}/artifacts/ada/atlas/embeddings/imagenet100/val/facebook__dinov3-vitb16-pretrain-lvd1689m/cls/aligned_from_imagenet1k_full"

TRAIN_IN1K_SIGLIP2="${REPO}/artifacts/ada/atlas/embeddings/imagenet1k/train/google__siglip2-base-patch16-256/pooler/combined_train_full_8shards"
VAL_IN1K_SIGLIP2="${REPO}/artifacts/ada/atlas/embeddings/imagenet1k/val/google__siglip2-base-patch16-256/pooler/cache-eb6c52ac9aeb54ce"
TRAIN_IN100_SIGLIP2="${REPO}/artifacts/ada/atlas/embeddings/imagenet100/train/google__siglip2-base-patch16-256/pooler/aligned_from_imagenet1k_full"
VAL_IN100_SIGLIP2="${REPO}/artifacts/ada/atlas/embeddings/imagenet100/val/google__siglip2-base-patch16-256/pooler/aligned_from_imagenet1k_full"

OVERWRITE_FLAG=""
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

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
    python3 -m ada.atlas.cli.subset_cache_by_manifest --source-cache '${TRAIN_IN1K_DINOV3}' --target-manifest-cache '${TRAIN_IN100_DINO2}' --output-dir '${TRAIN_IN100_DINOV3}' ${OVERWRITE_FLAG}
    python3 -m ada.atlas.cli.subset_cache_by_manifest --source-cache '${VAL_IN1K_DINOV3}' --target-manifest-cache '${VAL_IN100_DINO2}' --output-dir '${VAL_IN100_DINOV3}' ${OVERWRITE_FLAG}
    python3 -m ada.atlas.cli.subset_cache_by_manifest --source-cache '${TRAIN_IN1K_SIGLIP2}' --target-manifest-cache '${TRAIN_IN100_DINO2}' --output-dir '${TRAIN_IN100_SIGLIP2}' ${OVERWRITE_FLAG}
    python3 -m ada.atlas.cli.subset_cache_by_manifest --source-cache '${VAL_IN1K_SIGLIP2}' --target-manifest-cache '${VAL_IN100_DINO2}' --output-dir '${VAL_IN100_SIGLIP2}' ${OVERWRITE_FLAG}
  "
