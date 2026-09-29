#!/bin/bash
#SBATCH --job-name=ADA_atlas_cls
#SBATCH --partition=gpu-2h
#SBATCH --constraint=40gb|80gb|h100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
DATASET="${DATASET:-imagenet100}"
SPLIT="${SPLIT:-val}"
ROOT="${ROOT:-/home/space/datasets/imagenet_torchvision/imagenet100/val}"
ENCODER="${ENCODER:-facebook/dinov2-with-registers-base}"
FEATURE="${FEATURE:-cls}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/artifacts/ada/atlas/embeddings}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
ENCODER_INPUT_SIZE="${ENCODER_INPUT_SIZE:-224}"
PRECISION="${PRECISION:-bf16}"
MAX_SAMPLES="${MAX_SAMPLES-128}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"
SHARD_INDEX="${SHARD_INDEX:-}"
NUM_SHARDS="${NUM_SHARDS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${REPO}/logs"
cd "${REPO}"

if [[ ! -f "${CONTAINER}" ]]; then
  echo "[error] missing container: ${CONTAINER}" >&2
  exit 1
fi
if [[ ! -d "${ROOT}" ]]; then
  echo "[error] missing dataset root: ${ROOT}" >&2
  exit 1
fi

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] DATASET=${DATASET}"
echo "[JOB] SPLIT=${SPLIT}"
echo "[JOB] ROOT=${ROOT}"
echo "[JOB] ENCODER=${ENCODER}"
echo "[JOB] MAX_SAMPLES=${MAX_SAMPLES}"
echo "[JOB] START_INDEX=${START_INDEX}"
echo "[JOB] END_INDEX=${END_INDEX}"
echo "[JOB] SHARD_INDEX=${SHARD_INDEX}"
echo "[JOB] NUM_SHARDS=${NUM_SHARDS}"
echo "[JOB] BATCH_SIZE=${BATCH_SIZE}"
echo "[JOB] NUM_WORKERS=${NUM_WORKERS}"
echo "[JOB] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "============================================================"

nvidia-smi -L || true

max_args=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  max_args=(--max-samples "${MAX_SAMPLES}")
fi
range_args=(--start-index "${START_INDEX}")
if [[ -n "${END_INDEX}" ]]; then
  range_args+=(--end-index "${END_INDEX}")
fi
shard_args=()
if [[ -n "${SHARD_INDEX}" ]] || [[ -n "${NUM_SHARDS}" ]]; then
  shard_args=(--shard-index "${SHARD_INDEX}" --num-shards "${NUM_SHARDS}")
fi

apptainer exec --nv \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export PYTHONPATH=src
    export PYTHONDONTWRITEBYTECODE=1
    export TF_ENABLE_ONEDNN_OPTS=0
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    python3 -m ada.atlas.cli.cache_embeddings \
      --dataset '${DATASET}' \
      --split '${SPLIT}' \
      --root '${ROOT}' \
      --encoder '${ENCODER}' \
      --feature '${FEATURE}' \
      --output-root '${OUTPUT_ROOT}' \
      --batch-size '${BATCH_SIZE}' \
      --num-workers '${NUM_WORKERS}' \
      --image-size '${IMAGE_SIZE}' \
      --encoder-input-size '${ENCODER_INPUT_SIZE}' \
      --precision '${PRECISION}' \
      ${max_args[*]} \
      ${range_args[*]} \
      ${shard_args[*]} \
      ${EXTRA_ARGS}
  "

echo "[DONE] DATE=$(date)"
