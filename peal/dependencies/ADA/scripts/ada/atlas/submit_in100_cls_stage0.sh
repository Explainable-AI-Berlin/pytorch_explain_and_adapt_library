#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
WORKER="${REPO}/scripts/ada/atlas/run_cache_dinov2_cls_single.sh"
NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "${REPO}/logs"

submit_one() {
  local dataset="$1"
  local split="$2"
  local root="$3"
  local max_samples="$4"
  local shard_index="$5"
  local num_shards="$6"

  local job_name="ADA_atlas_${dataset}_${split}"
  if [[ -n "${shard_index}" ]]; then
    job_name="${job_name}_s${shard_index}of${num_shards}"
  fi

  sbatch --parsable \
    --job-name="${job_name}" \
    --export=ALL,REPO="${REPO}",DATASET="${dataset}",SPLIT="${split}",ROOT="${root}",MAX_SAMPLES="${max_samples}",SHARD_INDEX="${shard_index}",NUM_SHARDS="${num_shards}",BATCH_SIZE="${BATCH_SIZE}",NUM_WORKERS="${NUM_WORKERS}" \
    "${WORKER}"
}

val_job="$(submit_one imagenet100 val /home/space/datasets/imagenet_torchvision/imagenet100/val "" "" "")"
echo "val ${val_job}"

for shard in $(seq 0 $((NUM_TRAIN_SHARDS - 1))); do
  train_job="$(submit_one imagenet100 train /home/space/datasets/imagenet_torchvision/imagenet100/train "" "${shard}" "${NUM_TRAIN_SHARDS}")"
  echo "train_shard_${shard} ${train_job}"
done
