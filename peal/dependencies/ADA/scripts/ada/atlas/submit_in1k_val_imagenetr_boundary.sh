#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CACHE_WORKER="${REPO}/scripts/ada/atlas/run_cache_dinov2_cls_single.sh"
RUNNER="${REPO}/scripts/ada/atlas/run_module_single.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/artifacts/ada/atlas/embeddings}"
REPORT_ROOT="${REPORT_ROOT:-${REPO}/artifacts/ada/atlas/reports/e1_in1k_boundary_support}"
PRED_ROOT="${PRED_ROOT:-${REPO}/artifacts/ada/atlas/predictions/e1_in1k_boundary_support}"

IMAGENET_TRAIN_ROOT="${IMAGENET_TRAIN_ROOT:-/home/space/datasets/imagenet_torchvision/data/train}"
IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT:-/home/space/datasets/imagenet_torchvision/data/val}"
IMAGENET_R_ROOT="${IMAGENET_R_ROOT:-/home/space/datasets/imagenet-r}"
IMAGENET_META="${IMAGENET_META:-/home/space/datasets/imagenet_torchvision/data/meta.bin}"

ENCODER="${ENCODER:-facebook/dinov2-with-registers-base}"
FEATURE="${FEATURE:-cls}"
NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BOUNDARY_BATCH_SIZE="${BOUNDARY_BATCH_SIZE:-256}"
REFERENCE_SHARD_SIZE="${REFERENCE_SHARD_SIZE:-100000}"
GLOBAL_SEARCH_K="${GLOBAL_SEARCH_K:-1024}"

mkdir -p "${REPO}/logs" "${REPORT_ROOT}" "${PRED_ROOT}"

if [[ ! -d "${IMAGENET_TRAIN_ROOT}" ]]; then
  echo "[error] missing IMAGENET_TRAIN_ROOT: ${IMAGENET_TRAIN_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${IMAGENET_VAL_ROOT}" ]]; then
  echo "[error] missing IMAGENET_VAL_ROOT: ${IMAGENET_VAL_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${IMAGENET_R_ROOT}" ]]; then
  echo "[error] missing IMAGENET_R_ROOT: ${IMAGENET_R_ROOT}" >&2
  exit 1
fi

plan_cache_dir() {
  local dataset="$1"
  local split="$2"
  local root="$3"
  local shard_index="${4:-}"
  local num_shards="${5:-}"
  local args=(
    -m ada.atlas.cli.cache_embeddings
    --dataset "${dataset}"
    --split "${split}"
    --root "${root}"
    --encoder "${ENCODER}"
    --feature "${FEATURE}"
    --output-root "${OUTPUT_ROOT}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --image-size 256
    --encoder-input-size 224
    --precision bf16
    --dry-run
  )
  if [[ -n "${shard_index}" ]]; then
    args+=(--shard-index "${shard_index}" --num-shards "${num_shards}")
  fi
  PYTHONPATH="${REPO}/src" python3 "${args[@]}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["output_dir"])'
}

submit_cache() {
  local dataset="$1"
  local split="$2"
  local root="$3"
  local shard_index="${4:-}"
  local num_shards="${5:-}"
  local job_name="ADA_atlas_${dataset}_${split}"
  if [[ -n "${shard_index}" ]]; then
    job_name="${job_name}_s${shard_index}of${num_shards}"
  fi
  sbatch --parsable \
    --job-name="${job_name}" \
    --export=ALL,REPO="${REPO}",DATASET="${dataset}",SPLIT="${split}",ROOT="${root}",OUTPUT_ROOT="${OUTPUT_ROOT}",ENCODER="${ENCODER}",FEATURE="${FEATURE}",MAX_SAMPLES="",SHARD_INDEX="${shard_index}",NUM_SHARDS="${num_shards}",BATCH_SIZE="${BATCH_SIZE}",NUM_WORKERS="${NUM_WORKERS}",EXTRA_ARGS="--reuse-existing" \
    "${CACHE_WORKER}"
}

dependency_arg() {
  if [[ "$#" -eq 0 ]]; then
    echo ""
    return
  fi
  local IFS=":"
  echo "--dependency=afterok:$*"
}

train_jobs=()
for shard in $(seq 0 $((NUM_TRAIN_SHARDS - 1))); do
  job="$(submit_cache imagenet1k train "${IMAGENET_TRAIN_ROOT}" "${shard}" "${NUM_TRAIN_SHARDS}")"
  train_jobs+=("${job}")
  echo "train_shard_${shard} ${job}"
done

val_cache="$(plan_cache_dir imagenet1k val "${IMAGENET_VAL_ROOT}")"
val_job="$(submit_cache imagenet1k val "${IMAGENET_VAL_ROOT}")"
echo "val_cache ${val_job} ${val_cache}"

imagenetr_cache="$(plan_cache_dir imagenet-r query "${IMAGENET_R_ROOT}")"
imagenetr_job="$(submit_cache imagenet-r query "${IMAGENET_R_ROOT}")"
echo "imagenetr_cache ${imagenetr_job} ${imagenetr_cache}"

TRAIN_COMBINED="${OUTPUT_ROOT}/imagenet1k/train/$(echo "${ENCODER}" | sed 's#/#__#g; s#:#_#g; s# #_#g')/${FEATURE}/combined_train_full_${NUM_TRAIN_SHARDS}shards"
combine_args="--cache-root ${OUTPUT_ROOT} --dataset imagenet1k --split train --encoder ${ENCODER} --feature ${FEATURE} --num-shards ${NUM_TRAIN_SHARDS} --output-dir ${TRAIN_COMBINED} --overwrite"
combine_job="$(
  sbatch --parsable \
    $(dependency_arg "${train_jobs[@]}") \
    --job-name=ADA_in1k_train_combine \
    --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.combine_planned_shards,MODULE_ARGS="${combine_args}" \
    "${RUNNER}"
)"
echo "combine_train ${combine_job} ${TRAIN_COMBINED}"

val_pred_args="--query-cache ${val_cache} --output-dir ${PRED_ROOT}/resnet18_imagenet1k_val --model-name resnet18 --imagenet-meta ${IMAGENET_META} --batch-size 128"
val_pred_job="$(
  sbatch --parsable \
    --dependency="afterok:${val_job}" \
    --job-name=ADA_in1k_val_resnet18 \
    --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.predict_torchvision_supervised,MODULE_ARGS="${val_pred_args}" \
    "${RUNNER}"
)"
echo "val_resnet18 ${val_pred_job}"

imagenetr_pred_args="--query-cache ${imagenetr_cache} --output-dir ${PRED_ROOT}/resnet18_imagenetr --model-name resnet18 --imagenet-meta ${IMAGENET_META} --batch-size 128"
imagenetr_pred_job="$(
  sbatch --parsable \
    --dependency="afterok:${imagenetr_job}" \
    --job-name=ADA_imagenetr_resnet18 \
    --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.predict_torchvision_supervised,MODULE_ARGS="${imagenetr_pred_args}" \
    "${RUNNER}"
)"
echo "imagenetr_resnet18 ${imagenetr_pred_job}"

PERCENTILE_CACHE="${REPORT_ROOT}/in1k_reference_class_percentiles.npz"

val_boundary_args="--query-cache ${val_cache} --reference-cache ${TRAIN_COMBINED} --output-dir ${REPORT_ROOT}/imagenet1k_val --k 1:5:10:50 --entropy-k 10:50 --global-search-k ${GLOBAL_SEARCH_K} --batch-size ${BOUNDARY_BATCH_SIZE} --reference-shard-size ${REFERENCE_SHARD_SIZE} --device auto --mmap --class-key class_name --reference-percentile-cache ${PERCENTILE_CACHE} --prediction-csv ${PRED_ROOT}/resnet18_imagenet1k_val/predictions.csv"
val_boundary_job="$(
  sbatch --parsable \
    --dependency="afterok:${combine_job}:${val_job}:${val_pred_job}" \
    --job-name=ADA_in1k_val_boundary \
    --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.score_boundary_metrics,MODULE_ARGS="${val_boundary_args}" \
    "${RUNNER}"
)"
echo "val_boundary ${val_boundary_job}"

imagenetr_boundary_args="--query-cache ${imagenetr_cache} --reference-cache ${TRAIN_COMBINED} --output-dir ${REPORT_ROOT}/imagenet-r --k 1:5:10:50 --entropy-k 10:50 --global-search-k ${GLOBAL_SEARCH_K} --batch-size ${BOUNDARY_BATCH_SIZE} --reference-shard-size ${REFERENCE_SHARD_SIZE} --device auto --mmap --class-key class_name --reference-percentile-cache ${PERCENTILE_CACHE} --prediction-csv ${PRED_ROOT}/resnet18_imagenetr/predictions.csv"
imagenetr_boundary_job="$(
  sbatch --parsable \
    --dependency="afterok:${combine_job}:${imagenetr_job}:${imagenetr_pred_job}" \
    --job-name=ADA_imagenetr_boundary \
    --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.score_boundary_metrics,MODULE_ARGS="${imagenetr_boundary_args}" \
    "${RUNNER}"
)"
echo "imagenetr_boundary ${imagenetr_boundary_job}"
