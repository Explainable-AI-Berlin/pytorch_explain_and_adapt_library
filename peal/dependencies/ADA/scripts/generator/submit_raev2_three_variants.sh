#!/usr/bin/env bash
set -euo pipefail

ADA_REPO="${ADA_REPO:-$(git rev-parse --show-toplevel)}"
if [[ -f "${ADA_REPO}/.env" ]]; then
  set -a
  source "${ADA_REPO}/.env"
  set +a
fi

RAEV2_DIR="${RAEV2_DIR:-${ADA_REPO}/third_party/RAEv2}"
ADA_CONTAINER="${ADA_CONTAINER:?Set ADA_CONTAINER}"
ADA_STAGE2_CACHE="${ADA_STAGE2_CACHE:?Set ADA_STAGE2_CACHE}"
RESULTS_DIR="${RESULTS_DIR:-${ADA_CHECKPOINT_ROOT:?Set ADA_CHECKPOINT_ROOT}}"
WORKER="${ADA_REPO}/scripts/generator/run_raev2_stage2_train.sh"
CONFIG_ROOT="${RAEV2_DIR}/configs/stage2/training"

variants=(cls_only cls_class class_only)
configs=(
  imagenet-dinov2l-k1-patch-cls-only-cache-premix64-80ep-4gpu-accum32.yaml
  imagenet-dinov2l-k1-patch-cls-cache-premix64-80ep-4gpu-accum32.yaml
  imagenet-dinov2l-k1-patch-class-only-cache-premix64-80ep-4gpu-accum32.yaml
)
experiments=(
  in1k_raev2_dinov2l_k1_patch_cls_only_premix64_80ep_4gpu_b1024
  in1k_raev2_dinov2l_k1_patch_cls_class_premix64_80ep_4gpu_b1024
  in1k_raev2_dinov2l_k1_patch_class_only_premix64_80ep_4gpu_b1024
)

mkdir -p "${ADA_REPO}/logs" "${RESULTS_DIR}"
for index in "${!variants[@]}"; do
  config="${CONFIG_ROOT}/${configs[index]}"
  [[ -f "${config}" ]] || { echo "Missing config: ${config}" >&2; exit 2; }
  job_id=$(sbatch --parsable \
    --job-name="ada_raev2_${variants[index]}" \
    --partition=gpu-7d \
    --constraint=40gb \
    --gpus-per-node=4 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --mem=160G \
    --time=7-00:00:00 \
    --output="${ADA_REPO}/logs/%x-%j.out" \
    --error="${ADA_REPO}/logs/%x-%j.err" \
    --export=ALL,ADA_REPO="${ADA_REPO}",RAEV2_DIR="${RAEV2_DIR}",ADA_CONTAINER="${ADA_CONTAINER}",ADA_STAGE2_CACHE="${ADA_STAGE2_CACHE}",CONFIG="${config}",RESULTS_DIR="${RESULTS_DIR}",EXPERIMENT_NAME="${experiments[index]}",PRECISION=bf16,NPROC=4,COMPILE=0 \
    "${WORKER}")
  echo "${variants[index]}=${job_id}"
done
