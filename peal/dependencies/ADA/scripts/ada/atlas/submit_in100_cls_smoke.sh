#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
WORKER="${REPO}/scripts/ada/atlas/run_cache_dinov2_cls_single.sh"

mkdir -p "${REPO}/logs"

job_id="$(
  sbatch --parsable \
    --export=ALL,REPO="${REPO}",DATASET=imagenet100,SPLIT=val,ROOT=/home/space/datasets/imagenet_torchvision/imagenet100/val,MAX_SAMPLES=128,BATCH_SIZE=32,NUM_WORKERS=8 \
    "${WORKER}"
)"

echo "${job_id}"
