#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CACHE_WORKER="${REPO}/scripts/ada/atlas/run_cache_dinov2_cls_single.sh"
RUNNER="${REPO}/scripts/ada/atlas/run_module_single.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/artifacts/ada/atlas/embeddings}"
REPORT_ROOT="${REPORT_ROOT:-${REPO}/artifacts/ada/atlas/reports/e1_in1k_reviewer_baselines}"
PRED_ROOT="${PRED_ROOT:-${REPO}/artifacts/ada/atlas/predictions/e1_in1k_boundary_support}"

IMAGENET_TRAIN_ROOT="${IMAGENET_TRAIN_ROOT:-/home/space/datasets/imagenet_torchvision/data/train}"
IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT:-/home/space/datasets/imagenet_torchvision/data/val}"
IMAGENET_R_ROOT="${IMAGENET_R_ROOT:-/home/space/datasets/imagenet-r}"

NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-256}"
REFERENCE_SHARD_SIZE="${REFERENCE_SHARD_SIZE:-100000}"
TRUST_SEARCH_K="${TRUST_SEARCH_K:-4096}"
TRUST_ALPHA="${TRUST_ALPHA:-0.10}"
MAHALANOBIS_SHRINKAGE="${MAHALANOBIS_SHRINKAGE:-0.10}"
KDE_BANDWIDTH_SCALE="${KDE_BANDWIDTH_SCALE:-1.0}"
KDE_MIN_BANDWIDTH="${KDE_MIN_BANDWIDTH:-0.01}"
RUN_DINO2="${RUN_DINO2:-1}"
RUN_DINOV3="${RUN_DINOV3:-1}"
RUN_SIGLIP2="${RUN_SIGLIP2:-1}"

DINO2_TRAIN_CACHE="${DINO2_TRAIN_CACHE:-${OUTPUT_ROOT}/imagenet1k/train/facebook__dinov2-with-registers-base/cls/combined_train_full_8shards}"
DINO2_VAL_CACHE="${DINO2_VAL_CACHE:-${OUTPUT_ROOT}/imagenet1k/val/facebook__dinov2-with-registers-base/cls/cache-4e4e26580544477f}"
DINO2_IMAGENETR_CACHE="${DINO2_IMAGENETR_CACHE:-${OUTPUT_ROOT}/imagenet-r/query/facebook__dinov2-with-registers-base/cls/cache-e425727051f435e3}"

VAL_PRED_CSV="${VAL_PRED_CSV:-${PRED_ROOT}/resnet18_imagenet1k_val/predictions.csv}"
IMAGENETR_PRED_CSV="${IMAGENETR_PRED_CSV:-${PRED_ROOT}/resnet18_imagenetr/predictions.csv}"

mkdir -p "${REPO}/logs" "${REPORT_ROOT}"

for path in "${IMAGENET_TRAIN_ROOT}" "${IMAGENET_VAL_ROOT}" "${IMAGENET_R_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "[error] missing dataset root: ${path}" >&2
    exit 1
  fi
done
for path in "${VAL_PRED_CSV}" "${IMAGENETR_PRED_CSV}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing prediction csv: ${path}" >&2
    exit 1
  fi
done

safe_encoder_name() {
  echo "$1" | sed 's#/#__#g; s#:#_#g; s# #_#g'
}

dependency_arg() {
  if [[ "$#" -eq 0 ]]; then
    echo ""
    return
  fi
  local IFS=":"
  echo "--dependency=afterok:$*"
}

plan_cache_dir() {
  local dataset="$1"
  local split="$2"
  local root="$3"
  local encoder="$4"
  local feature="$5"
  local encoder_input_size="$6"
  local shard_index="${7:-}"
  local num_shards="${8:-}"
  local args=(
    -m ada.atlas.cli.cache_embeddings
    --dataset "${dataset}"
    --split "${split}"
    --root "${root}"
    --encoder "${encoder}"
    --feature "${feature}"
    --output-root "${OUTPUT_ROOT}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --image-size 256
    --encoder-input-size "${encoder_input_size}"
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
  local encoder="$4"
  local feature="$5"
  local encoder_input_size="$6"
  local shard_index="${7:-}"
  local num_shards="${8:-}"
  local encoder_tag
  encoder_tag="$(safe_encoder_name "${encoder}")"
  local job_name="ADA_${dataset}_${split}_${feature}_${encoder_tag}"
  if [[ -n "${shard_index}" ]]; then
    job_name="${job_name}_s${shard_index}of${num_shards}"
  fi
  sbatch --parsable \
    --job-name="${job_name}" \
    --export=ALL,REPO="${REPO}",DATASET="${dataset}",SPLIT="${split}",ROOT="${root}",OUTPUT_ROOT="${OUTPUT_ROOT}",ENCODER="${encoder}",FEATURE="${feature}",IMAGE_SIZE=256,ENCODER_INPUT_SIZE="${encoder_input_size}",MAX_SAMPLES="",SHARD_INDEX="${shard_index}",NUM_SHARDS="${num_shards}",BATCH_SIZE="${BATCH_SIZE}",NUM_WORKERS="${NUM_WORKERS}",EXTRA_ARGS="--reuse-existing" \
    "${CACHE_WORKER}"
}

submit_baseline() {
  local tag="$1"
  local query_cache="$2"
  local reference_cache="$3"
  local output_dir="$4"
  local prediction_csv="$5"
  shift 5
  local deps=("$@")
  local args="--query-cache ${query_cache} --reference-cache ${reference_cache} --output-dir ${output_dir} --prediction-csv ${prediction_csv} --class-key class_name --procal-k 10 --trust-k 10 --trust-alpha ${TRUST_ALPHA} --trust-search-k ${TRUST_SEARCH_K} --kde-bandwidth-scale ${KDE_BANDWIDTH_SCALE} --kde-min-bandwidth ${KDE_MIN_BANDWIDTH} --batch-size ${BASELINE_BATCH_SIZE} --reference-shard-size ${REFERENCE_SHARD_SIZE} --device auto --mmap --mahalanobis-shrinkage ${MAHALANOBIS_SHRINKAGE}"
  sbatch --parsable \
    $(dependency_arg "${deps[@]}") \
    --job-name="ADA_baselines_${tag}" \
    --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.score_reviewer_baselines,MODULE_ARGS="${args}" \
    "${RUNNER}"
}

if [[ "${RUN_DINO2}" == "1" && -d "${DINO2_TRAIN_CACHE}" && -d "${DINO2_VAL_CACHE}" && -d "${DINO2_IMAGENETR_CACHE}" ]]; then
  dino2_val_job="$(submit_baseline dino2_in1k_val "${DINO2_VAL_CACHE}" "${DINO2_TRAIN_CACHE}" "${REPORT_ROOT}/dino2_cls/imagenet1k_val" "${VAL_PRED_CSV}")"
  echo "dino2_val_baselines ${dino2_val_job}"
  dino2_r_job="$(submit_baseline dino2_imagenetr "${DINO2_IMAGENETR_CACHE}" "${DINO2_TRAIN_CACHE}" "${REPORT_ROOT}/dino2_cls/imagenet-r" "${IMAGENETR_PRED_CSV}")"
  echo "dino2_imagenetr_baselines ${dino2_r_job}"
elif [[ "${RUN_DINO2}" == "1" ]]; then
  echo "[warn] skipping DINOv2 baseline jobs because one or more existing caches are missing" >&2
else
  echo "[skip] RUN_DINO2=${RUN_DINO2}"
fi

ENCODER_SPECS=()
if [[ "${RUN_DINOV3}" == "1" ]]; then
  ENCODER_SPECS+=("dinov3_cls|facebook/dinov3-vitb16-pretrain-lvd1689m|cls|256")
fi
if [[ "${RUN_SIGLIP2}" == "1" ]]; then
  ENCODER_SPECS+=("siglip2_pooler|google/siglip2-base-patch16-256|pooler|256")
fi

for spec in "${ENCODER_SPECS[@]}"; do
  IFS='|' read -r tag encoder feature encoder_input_size <<< "${spec}"
  echo "============================================================"
  echo "[submit] ${tag}: encoder=${encoder} feature=${feature}"
  echo "============================================================"

  train_jobs=()
  for shard in $(seq 0 $((NUM_TRAIN_SHARDS - 1))); do
    job="$(submit_cache imagenet1k train "${IMAGENET_TRAIN_ROOT}" "${encoder}" "${feature}" "${encoder_input_size}" "${shard}" "${NUM_TRAIN_SHARDS}")"
    train_jobs+=("${job}")
    echo "${tag}_train_shard_${shard} ${job}"
  done

  val_cache="$(plan_cache_dir imagenet1k val "${IMAGENET_VAL_ROOT}" "${encoder}" "${feature}" "${encoder_input_size}")"
  val_job="$(submit_cache imagenet1k val "${IMAGENET_VAL_ROOT}" "${encoder}" "${feature}" "${encoder_input_size}")"
  echo "${tag}_val_cache ${val_job} ${val_cache}"

  imagenetr_cache="$(plan_cache_dir imagenet-r query "${IMAGENET_R_ROOT}" "${encoder}" "${feature}" "${encoder_input_size}")"
  imagenetr_job="$(submit_cache imagenet-r query "${IMAGENET_R_ROOT}" "${encoder}" "${feature}" "${encoder_input_size}")"
  echo "${tag}_imagenetr_cache ${imagenetr_job} ${imagenetr_cache}"

  train_combined="${OUTPUT_ROOT}/imagenet1k/train/$(safe_encoder_name "${encoder}")/${feature}/combined_train_full_${NUM_TRAIN_SHARDS}shards"
  combine_args="--cache-root ${OUTPUT_ROOT} --dataset imagenet1k --split train --encoder ${encoder} --feature ${feature} --num-shards ${NUM_TRAIN_SHARDS} --output-dir ${train_combined} --overwrite"
  combine_job="$(
    sbatch --parsable \
      $(dependency_arg "${train_jobs[@]}") \
      --job-name="ADA_combine_${tag}" \
      --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.combine_planned_shards,MODULE_ARGS="${combine_args}" \
      "${RUNNER}"
  )"
  echo "${tag}_combine_train ${combine_job} ${train_combined}"

  val_base_job="$(submit_baseline "${tag}_in1k_val" "${val_cache}" "${train_combined}" "${REPORT_ROOT}/${tag}/imagenet1k_val" "${VAL_PRED_CSV}" "${combine_job}" "${val_job}")"
  echo "${tag}_val_baselines ${val_base_job}"

  imagenetr_base_job="$(submit_baseline "${tag}_imagenetr" "${imagenetr_cache}" "${train_combined}" "${REPORT_ROOT}/${tag}/imagenet-r" "${IMAGENETR_PRED_CSV}" "${combine_job}" "${imagenetr_job}")"
  echo "${tag}_imagenetr_baselines ${imagenetr_base_job}"
done
