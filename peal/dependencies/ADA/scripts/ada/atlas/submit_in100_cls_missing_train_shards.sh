#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
WORKER="${REPO}/scripts/ada/atlas/run_cache_dinov2_cls_single.sh"
NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHARDS="${SHARDS:-1 2 3}"

mkdir -p "${REPO}/logs"

for shard in ${SHARDS}; do
  job_id="$(
    sbatch --parsable \
      --job-name="ADA_atlas_imagenet100_train_s${shard}of${NUM_TRAIN_SHARDS}" \
      --export=ALL,REPO="${REPO}",DATASET=imagenet100,SPLIT=train,ROOT=/home/space/datasets/imagenet_torchvision/imagenet100/train,MAX_SAMPLES="",SHARD_INDEX="${shard}",NUM_SHARDS="${NUM_TRAIN_SHARDS}",BATCH_SIZE="${BATCH_SIZE}",NUM_WORKERS="${NUM_WORKERS}" \
      "${WORKER}"
  )"
  echo "train_shard_${shard} ${job_id}"
done
