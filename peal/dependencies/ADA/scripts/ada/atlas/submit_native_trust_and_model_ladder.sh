#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/atlas/run_module_single.sh"
CACHE_WORKER="${REPO}/scripts/ada/atlas/run_cache_dinov2_cls_single.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/artifacts/ada/atlas/embeddings}"
REPORT_ROOT="${REPORT_ROOT:-${REPO}/artifacts/ada/atlas/reports/e1_in1k_native_trust_model_ladder}"
PRED_ROOT="${PRED_ROOT:-${REPO}/artifacts/ada/atlas/predictions/e1_in1k_model_ladder}"
OLD_PRED_ROOT="${OLD_PRED_ROOT:-${REPO}/artifacts/ada/atlas/predictions/e1_in1k_boundary_support}"

IMAGENET_TRAIN_ROOT="${IMAGENET_TRAIN_ROOT:-/home/space/datasets/imagenet_torchvision/data/train}"
IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT:-/home/space/datasets/imagenet_torchvision/data/val}"
IMAGENET_R_ROOT="${IMAGENET_R_ROOT:-/home/space/datasets/imagenet-r}"
IMAGENET_META="${IMAGENET_META:-/home/space/datasets/imagenet_torchvision/data/meta.bin}"

NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-8}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-128}"
HF_BATCH_SIZE="${HF_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-256}"
REFERENCE_SHARD_SIZE="${REFERENCE_SHARD_SIZE:-100000}"
TRUST_SEARCH_K="${TRUST_SEARCH_K:-4096}"
TRUST_ALPHA="${TRUST_ALPHA:-0.10}"
MAHALANOBIS_SHRINKAGE="${MAHALANOBIS_SHRINKAGE:-0.10}"
KDE_BANDWIDTH_SCALE="${KDE_BANDWIDTH_SCALE:-1.0}"
KDE_MIN_BANDWIDTH="${KDE_MIN_BANDWIDTH:-0.01}"
PROBE_EPOCHS="${PROBE_EPOCHS:-40}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-4096}"

RUN_RESNET18_NATIVE="${RUN_RESNET18_NATIVE:-1}"
RUN_RESNET50="${RUN_RESNET50:-1}"
RUN_EXTERNAL_RERUN="${RUN_EXTERNAL_RERUN:-1}"
RUN_DINO2_PROBE="${RUN_DINO2_PROBE:-1}"
RUN_MAE="${RUN_MAE:-1}"

DINO2_TRAIN_CACHE="${DINO2_TRAIN_CACHE:-${OUTPUT_ROOT}/imagenet1k/train/facebook__dinov2-with-registers-base/cls/combined_train_full_8shards}"
DINO2_VAL_CACHE="${DINO2_VAL_CACHE:-${OUTPUT_ROOT}/imagenet1k/val/facebook__dinov2-with-registers-base/cls/cache-4e4e26580544477f}"
DINO2_IMAGENETR_CACHE="${DINO2_IMAGENETR_CACHE:-${OUTPUT_ROOT}/imagenet-r/query/facebook__dinov2-with-registers-base/cls/cache-e425727051f435e3}"
DINOV3_TRAIN_CACHE="${DINOV3_TRAIN_CACHE:-${OUTPUT_ROOT}/imagenet1k/train/facebook__dinov3-vitb16-pretrain-lvd1689m/cls/combined_train_full_8shards}"
DINOV3_VAL_CACHE="${DINOV3_VAL_CACHE:-${OUTPUT_ROOT}/imagenet1k/val/facebook__dinov3-vitb16-pretrain-lvd1689m/cls/cache-63bb2429628246b8}"
DINOV3_IMAGENETR_CACHE="${DINOV3_IMAGENETR_CACHE:-${OUTPUT_ROOT}/imagenet-r/query/facebook__dinov3-vitb16-pretrain-lvd1689m/cls/cache-c01693c4ec5fafb7}"
SIGLIP2_TRAIN_CACHE="${SIGLIP2_TRAIN_CACHE:-${OUTPUT_ROOT}/imagenet1k/train/google__siglip2-base-patch16-256/pooler/combined_train_full_8shards}"
SIGLIP2_VAL_CACHE="${SIGLIP2_VAL_CACHE:-${OUTPUT_ROOT}/imagenet1k/val/google__siglip2-base-patch16-256/pooler/cache-eb6c52ac9aeb54ce}"
SIGLIP2_IMAGENETR_CACHE="${SIGLIP2_IMAGENETR_CACHE:-${OUTPUT_ROOT}/imagenet-r/query/google__siglip2-base-patch16-256/pooler/cache-6826d99ad1e1c07a}"

RESNET18_VAL_PRED="${RESNET18_VAL_PRED:-${OLD_PRED_ROOT}/resnet18_imagenet1k_val/predictions.csv}"
RESNET18_R_PRED="${RESNET18_R_PRED:-${OLD_PRED_ROOT}/resnet18_imagenetr/predictions.csv}"

mkdir -p "${REPO}/logs" "${REPORT_ROOT}" "${PRED_ROOT}"

for path in "${IMAGENET_TRAIN_ROOT}" "${IMAGENET_VAL_ROOT}" "${IMAGENET_R_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "[error] missing dataset root: ${path}" >&2
    exit 1
  fi
done
for path in "${RESNET18_VAL_PRED}" "${RESNET18_R_PRED}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing existing ResNet18 prediction csv: ${path}" >&2
    exit 1
  fi
done

safe_name() {
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

submit_module() {
  local job_name="$1"
  local module="$2"
  local args="$3"
  shift 3
  local deps=()
  for dep in "$@"; do
    if [[ -n "${dep}" ]]; then
      deps+=("${dep}")
    fi
  done
  sbatch --parsable \
    $(dependency_arg "${deps[@]}") \
    --job-name="${job_name}" \
    --export=ALL,REPO="${REPO}",MODULE="${module}",MODULE_ARGS="${args}" \
    "${RUNNER}"
}

plan_torchvision_cache() {
  local dataset="$1"
  local split="$2"
  local root="$3"
  local model="$4"
  local shard_index="${5:-}"
  local num_shards="${6:-}"
  local args=(
    -m ada.atlas.cli.cache_torchvision_features
    --dataset "${dataset}"
    --split "${split}"
    --root "${root}"
    --model-name "${model}"
    --feature penultimate
    --output-root "${OUTPUT_ROOT}"
    --batch-size "${FEATURE_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --dry-run
  )
  if [[ -n "${shard_index}" ]]; then
    args+=(--shard-index "${shard_index}" --num-shards "${num_shards}")
  fi
  PYTHONPATH="${REPO}/src" python3 "${args[@]}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["output_dir"])'
}

submit_torchvision_cache() {
  local dataset="$1"
  local split="$2"
  local root="$3"
  local model="$4"
  local shard_index="${5:-}"
  local num_shards="${6:-}"
  local job_name="ADA_${dataset}_${split}_${model}_penult"
  if [[ -n "${shard_index}" ]]; then
    job_name="${job_name}_s${shard_index}of${num_shards}"
  fi
  local args="--dataset ${dataset} --split ${split} --root ${root} --model-name ${model} --feature penultimate --output-root ${OUTPUT_ROOT} --batch-size ${FEATURE_BATCH_SIZE} --num-workers ${NUM_WORKERS} --reuse-existing"
  if [[ -n "${shard_index}" ]]; then
    args="${args} --shard-index ${shard_index} --num-shards ${num_shards}"
  fi
  submit_module "${job_name}" ada.atlas.cli.cache_torchvision_features "${args}"
}

plan_hf_cache() {
  local dataset="$1"
  local split="$2"
  local root="$3"
  local encoder="$4"
  local feature="$5"
  local input_size="$6"
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
    --batch-size "${HF_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --image-size 256
    --encoder-input-size "${input_size}"
    --precision bf16
    --dry-run
  )
  if [[ -n "${shard_index}" ]]; then
    args+=(--shard-index "${shard_index}" --num-shards "${num_shards}")
  fi
  PYTHONPATH="${REPO}/src" python3 "${args[@]}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["output_dir"])'
}

submit_hf_cache() {
  local dataset="$1"
  local split="$2"
  local root="$3"
  local encoder="$4"
  local feature="$5"
  local input_size="$6"
  local shard_index="${7:-}"
  local num_shards="${8:-}"
  local tag
  tag="$(safe_name "${encoder}")"
  local job_name="ADA_${dataset}_${split}_${feature}_${tag}"
  if [[ -n "${shard_index}" ]]; then
    job_name="${job_name}_s${shard_index}of${num_shards}"
  fi
  sbatch --parsable \
    --job-name="${job_name}" \
    --export=ALL,REPO="${REPO}",DATASET="${dataset}",SPLIT="${split}",ROOT="${root}",OUTPUT_ROOT="${OUTPUT_ROOT}",ENCODER="${encoder}",FEATURE="${feature}",IMAGE_SIZE=256,ENCODER_INPUT_SIZE="${input_size}",MAX_SAMPLES="",SHARD_INDEX="${shard_index}",NUM_SHARDS="${num_shards}",BATCH_SIZE="${HF_BATCH_SIZE}",NUM_WORKERS="${NUM_WORKERS}",EXTRA_ARGS="--reuse-existing" \
    "${CACHE_WORKER}"
}

submit_reviewer_baselines() {
  local tag="$1"
  local query_cache="$2"
  local reference_cache="$3"
  local output_dir="$4"
  shift 4
  local pred_csvs=()
  local deps=()
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --pred)
        pred_csvs+=("$2")
        shift 2
        ;;
      --dep)
        deps+=("$2")
        shift 2
        ;;
      *)
        echo "[error] unknown submit_reviewer_baselines arg: $1" >&2
        exit 1
        ;;
    esac
  done
  local pred_args=""
  for csv in "${pred_csvs[@]}"; do
    pred_args="${pred_args} --prediction-csv ${csv}"
  done
  local args="--query-cache ${query_cache} --reference-cache ${reference_cache} --output-dir ${output_dir}${pred_args} --class-key class_name --procal-k 10 --trust-k 10 --trust-alpha ${TRUST_ALPHA} --trust-search-k ${TRUST_SEARCH_K} --kde-bandwidth-scale ${KDE_BANDWIDTH_SCALE} --kde-min-bandwidth ${KDE_MIN_BANDWIDTH} --batch-size ${BASELINE_BATCH_SIZE} --reference-shard-size ${REFERENCE_SHARD_SIZE} --device auto --mmap --mahalanobis-shrinkage ${MAHALANOBIS_SHRINKAGE}"
  submit_module "ADA_${tag}_reviewer" ada.atlas.cli.score_reviewer_baselines "${args}" "${deps[@]}"
}

submit_torchvision_predictions() {
  local model="$1"
  local query_cache="$2"
  local output_dir="$3"
  local dep="$4"
  local args="--query-cache ${query_cache} --output-dir ${output_dir} --model-name ${model} --imagenet-meta ${IMAGENET_META} --batch-size 128"
  if [[ -n "${dep}" ]]; then
    submit_module "ADA_${model}_pred_$(basename "${output_dir}")" ada.atlas.cli.predict_torchvision_supervised "${args}" "${dep}"
  else
    submit_module "ADA_${model}_pred_$(basename "${output_dir}")" ada.atlas.cli.predict_torchvision_supervised "${args}"
  fi
}

submit_probe() {
  local model_id="$1"
  local train_cache="$2"
  local query_cache="$3"
  local output_dir="$4"
  shift 4
  local deps=("$@")
  local args="--train-cache ${train_cache} --query-cache ${query_cache} --output-dir ${output_dir} --epochs ${PROBE_EPOCHS} --batch-size ${PROBE_BATCH_SIZE} --model-id ${model_id} --device auto"
  submit_module "ADA_probe_${model_id}" ada.atlas.cli.train_dino_linear_probe "${args}" "${deps[@]}"
}

submit_native_family() {
  local model="$1"
  local val_pred="$2"
  local r_pred="$3"
  local pred_val_dep="${4:-}"
  local pred_r_dep="${5:-}"

  echo "============================================================"
  echo "[submit] native ${model} penultimate Trust/GDA/ProCal"
  echo "============================================================"

  train_jobs=()
  for shard in $(seq 0 $((NUM_TRAIN_SHARDS - 1))); do
    job="$(submit_torchvision_cache imagenet1k train "${IMAGENET_TRAIN_ROOT}" "${model}" "${shard}" "${NUM_TRAIN_SHARDS}")"
    train_jobs+=("${job}")
    echo "${model}_native_train_shard_${shard} ${job}"
  done
  val_cache="$(plan_torchvision_cache imagenet1k val "${IMAGENET_VAL_ROOT}" "${model}")"
  val_job="$(submit_torchvision_cache imagenet1k val "${IMAGENET_VAL_ROOT}" "${model}")"
  echo "${model}_native_val_cache ${val_job} ${val_cache}"
  r_cache="$(plan_torchvision_cache imagenet-r query "${IMAGENET_R_ROOT}" "${model}")"
  r_job="$(submit_torchvision_cache imagenet-r query "${IMAGENET_R_ROOT}" "${model}")"
  echo "${model}_native_imagenetr_cache ${r_job} ${r_cache}"

  encoder="torchvision/${model}-imagenet1k-v1"
  if [[ "${model}" == "resnet50" ]]; then
    encoder="torchvision/resnet50-imagenet1k-v2"
  fi
  train_combined="${OUTPUT_ROOT}/imagenet1k/train/$(safe_name "${encoder}")/penultimate/combined_train_full_${NUM_TRAIN_SHARDS}shards"
  combine_args="--cache-root ${OUTPUT_ROOT} --dataset imagenet1k --split train --encoder ${encoder} --feature penultimate --num-shards ${NUM_TRAIN_SHARDS} --output-dir ${train_combined} --overwrite"
  combine_job="$(submit_module "ADA_combine_${model}_native" ada.atlas.cli.combine_planned_shards "${combine_args}" "${train_jobs[@]}")"
  echo "${model}_native_combine ${combine_job} ${train_combined}"

  deps_val=("${combine_job}" "${val_job}")
  deps_r=("${combine_job}" "${r_job}")
  if [[ -n "${pred_val_dep}" ]]; then
    deps_val+=("${pred_val_dep}")
  fi
  if [[ -n "${pred_r_dep}" ]]; then
    deps_r+=("${pred_r_dep}")
  fi
  dep_args_val=()
  for dep in "${deps_val[@]}"; do dep_args_val+=(--dep "${dep}"); done
  dep_args_r=()
  for dep in "${deps_r[@]}"; do dep_args_r+=(--dep "${dep}"); done
  val_base="$(submit_reviewer_baselines "native_${model}_in1k_val" "${val_cache}" "${train_combined}" "${REPORT_ROOT}/native_${model}/imagenet1k_val" --pred "${val_pred}" "${dep_args_val[@]}")"
  echo "${model}_native_val_reviewer ${val_base}"
  r_base="$(submit_reviewer_baselines "native_${model}_imagenetr" "${r_cache}" "${train_combined}" "${REPORT_ROOT}/native_${model}/imagenet-r" --pred "${r_pred}" "${dep_args_r[@]}")"
  echo "${model}_native_imagenetr_reviewer ${r_base}"
}

# ResNet50 predictions are needed both for native and external-atlas scoring.
RESNET50_VAL_PRED="${PRED_ROOT}/resnet50_imagenet1k_val/predictions.csv"
RESNET50_R_PRED="${PRED_ROOT}/resnet50_imagenetr/predictions.csv"
resnet50_val_job=""
resnet50_r_job=""
if [[ "${RUN_RESNET50}" == "1" ]]; then
  resnet50_val_job="$(submit_torchvision_predictions resnet50 "${DINO2_VAL_CACHE}" "${PRED_ROOT}/resnet50_imagenet1k_val" "")"
  echo "resnet50_val_predictions ${resnet50_val_job}"
  resnet50_r_job="$(submit_torchvision_predictions resnet50 "${DINO2_IMAGENETR_CACHE}" "${PRED_ROOT}/resnet50_imagenetr" "")"
  echo "resnet50_imagenetr_predictions ${resnet50_r_job}"
fi

if [[ "${RUN_RESNET18_NATIVE}" == "1" ]]; then
  submit_native_family resnet18 "${RESNET18_VAL_PRED}" "${RESNET18_R_PRED}"
fi
if [[ "${RUN_RESNET50}" == "1" ]]; then
  submit_native_family resnet50 "${RESNET50_VAL_PRED}" "${RESNET50_R_PRED}" "${resnet50_val_job}" "${resnet50_r_job}"
fi

dino2_probe_pred=""
dino2_probe_job=""
if [[ "${RUN_DINO2_PROBE}" == "1" ]]; then
  dino2_probe_pred="${PRED_ROOT}/dino2_cls_linear_probe_imagenet1k_val/predictions.csv"
  dino2_probe_job="$(submit_probe dino2_cls_linear_probe "${DINO2_TRAIN_CACHE}" "${DINO2_VAL_CACHE}" "${PRED_ROOT}/dino2_cls_linear_probe_imagenet1k_val")"
  echo "dino2_cls_linear_probe_val ${dino2_probe_job}"
fi

if [[ "${RUN_EXTERNAL_RERUN}" == "1" ]]; then
  val_preds=("${RESNET18_VAL_PRED}")
  r_preds=("${RESNET18_R_PRED}")
  val_deps=()
  r_deps=()
  if [[ "${RUN_RESNET50}" == "1" ]]; then
    val_preds+=("${RESNET50_VAL_PRED}")
    r_preds+=("${RESNET50_R_PRED}")
    val_deps+=("${resnet50_val_job}")
    r_deps+=("${resnet50_r_job}")
  fi
  if [[ "${RUN_DINO2_PROBE}" == "1" ]]; then
    val_preds+=("${dino2_probe_pred}")
    val_deps+=("${dino2_probe_job}")
  fi
  pred_args_val=()
  for pred in "${val_preds[@]}"; do pred_args_val+=(--pred "${pred}"); done
  pred_args_r=()
  for pred in "${r_preds[@]}"; do pred_args_r+=(--pred "${pred}"); done
  dep_args_val=()
  for dep in "${val_deps[@]}"; do dep_args_val+=(--dep "${dep}"); done
  dep_args_r=()
  for dep in "${r_deps[@]}"; do dep_args_r+=(--dep "${dep}"); done

  submit_reviewer_baselines "external_dino2_in1k_val" "${DINO2_VAL_CACHE}" "${DINO2_TRAIN_CACHE}" "${REPORT_ROOT}/external_dino2_cls/imagenet1k_val" "${pred_args_val[@]}" "${dep_args_val[@]}" | awk '{print "external_dino2_val_reviewer " $0}'
  submit_reviewer_baselines "external_dino2_imagenetr" "${DINO2_IMAGENETR_CACHE}" "${DINO2_TRAIN_CACHE}" "${REPORT_ROOT}/external_dino2_cls/imagenet-r" "${pred_args_r[@]}" "${dep_args_r[@]}" | awk '{print "external_dino2_imagenetr_reviewer " $0}'
  submit_reviewer_baselines "external_dinov3_in1k_val" "${DINOV3_VAL_CACHE}" "${DINOV3_TRAIN_CACHE}" "${REPORT_ROOT}/external_dinov3_cls/imagenet1k_val" "${pred_args_val[@]}" "${dep_args_val[@]}" | awk '{print "external_dinov3_val_reviewer " $0}'
  submit_reviewer_baselines "external_dinov3_imagenetr" "${DINOV3_IMAGENETR_CACHE}" "${DINOV3_TRAIN_CACHE}" "${REPORT_ROOT}/external_dinov3_cls/imagenet-r" "${pred_args_r[@]}" "${dep_args_r[@]}" | awk '{print "external_dinov3_imagenetr_reviewer " $0}'
  submit_reviewer_baselines "external_siglip2_in1k_val" "${SIGLIP2_VAL_CACHE}" "${SIGLIP2_TRAIN_CACHE}" "${REPORT_ROOT}/external_siglip2_pooler/imagenet1k_val" "${pred_args_val[@]}" "${dep_args_val[@]}" | awk '{print "external_siglip2_val_reviewer " $0}'
  submit_reviewer_baselines "external_siglip2_imagenetr" "${SIGLIP2_IMAGENETR_CACHE}" "${SIGLIP2_TRAIN_CACHE}" "${REPORT_ROOT}/external_siglip2_pooler/imagenet-r" "${pred_args_r[@]}" "${dep_args_r[@]}" | awk '{print "external_siglip2_imagenetr_reviewer " $0}'
fi

if [[ "${RUN_MAE}" == "1" ]]; then
  echo "============================================================"
  echo "[submit] MAE pooler atlas + ImageNet-1K linear probe"
  echo "============================================================"
  MAE_ENCODER="${MAE_ENCODER:-facebook/vit-mae-base}"
  MAE_FEATURE="${MAE_FEATURE:-pooler}"
  MAE_INPUT_SIZE="${MAE_INPUT_SIZE:-224}"
  mae_train_jobs=()
  for shard in $(seq 0 $((NUM_TRAIN_SHARDS - 1))); do
    job="$(submit_hf_cache imagenet1k train "${IMAGENET_TRAIN_ROOT}" "${MAE_ENCODER}" "${MAE_FEATURE}" "${MAE_INPUT_SIZE}" "${shard}" "${NUM_TRAIN_SHARDS}")"
    mae_train_jobs+=("${job}")
    echo "mae_train_shard_${shard} ${job}"
  done
  mae_val_cache="$(plan_hf_cache imagenet1k val "${IMAGENET_VAL_ROOT}" "${MAE_ENCODER}" "${MAE_FEATURE}" "${MAE_INPUT_SIZE}")"
  mae_val_job="$(submit_hf_cache imagenet1k val "${IMAGENET_VAL_ROOT}" "${MAE_ENCODER}" "${MAE_FEATURE}" "${MAE_INPUT_SIZE}")"
  echo "mae_val_cache ${mae_val_job} ${mae_val_cache}"
  mae_r_cache="$(plan_hf_cache imagenet-r query "${IMAGENET_R_ROOT}" "${MAE_ENCODER}" "${MAE_FEATURE}" "${MAE_INPUT_SIZE}")"
  mae_r_job="$(submit_hf_cache imagenet-r query "${IMAGENET_R_ROOT}" "${MAE_ENCODER}" "${MAE_FEATURE}" "${MAE_INPUT_SIZE}")"
  echo "mae_imagenetr_cache ${mae_r_job} ${mae_r_cache}"
  mae_train_combined="${OUTPUT_ROOT}/imagenet1k/train/$(safe_name "${MAE_ENCODER}")/${MAE_FEATURE}/combined_train_full_${NUM_TRAIN_SHARDS}shards"
  mae_combine_args="--cache-root ${OUTPUT_ROOT} --dataset imagenet1k --split train --encoder ${MAE_ENCODER} --feature ${MAE_FEATURE} --num-shards ${NUM_TRAIN_SHARDS} --output-dir ${mae_train_combined} --overwrite"
  mae_combine_job="$(submit_module "ADA_combine_mae_pooler" ada.atlas.cli.combine_planned_shards "${mae_combine_args}" "${mae_train_jobs[@]}")"
  echo "mae_combine ${mae_combine_job} ${mae_train_combined}"
  mae_probe_pred="${PRED_ROOT}/mae_pooler_linear_probe_imagenet1k_val/predictions.csv"
  mae_probe_job="$(submit_probe mae_pooler_linear_probe "${mae_train_combined}" "${mae_val_cache}" "${PRED_ROOT}/mae_pooler_linear_probe_imagenet1k_val" "${mae_combine_job}" "${mae_val_job}")"
  echo "mae_pooler_linear_probe_val ${mae_probe_job}"
  mae_val_preds=("${RESNET18_VAL_PRED}")
  mae_r_preds=("${RESNET18_R_PRED}")
  mae_val_deps=("${mae_combine_job}" "${mae_val_job}")
  mae_r_deps=("${mae_combine_job}" "${mae_r_job}")
  if [[ "${RUN_RESNET50}" == "1" ]]; then
    mae_val_preds+=("${RESNET50_VAL_PRED}")
    mae_r_preds+=("${RESNET50_R_PRED}")
    mae_val_deps+=("${resnet50_val_job}")
    mae_r_deps+=("${resnet50_r_job}")
  fi
  mae_val_preds+=("${mae_probe_pred}")
  mae_val_deps+=("${mae_probe_job}")
  mae_pred_args_val=()
  for pred in "${mae_val_preds[@]}"; do mae_pred_args_val+=(--pred "${pred}"); done
  mae_pred_args_r=()
  for pred in "${mae_r_preds[@]}"; do mae_pred_args_r+=(--pred "${pred}"); done
  mae_dep_args_val=()
  for dep in "${mae_val_deps[@]}"; do mae_dep_args_val+=(--dep "${dep}"); done
  mae_dep_args_r=()
  for dep in "${mae_r_deps[@]}"; do mae_dep_args_r+=(--dep "${dep}"); done
  submit_reviewer_baselines "mae_pooler_in1k_val" "${mae_val_cache}" "${mae_train_combined}" "${REPORT_ROOT}/mae_pooler/imagenet1k_val" "${mae_pred_args_val[@]}" "${mae_dep_args_val[@]}" | awk '{print "mae_val_reviewer " $0}'
  submit_reviewer_baselines "mae_pooler_imagenetr" "${mae_r_cache}" "${mae_train_combined}" "${REPORT_ROOT}/mae_pooler/imagenet-r" "${mae_pred_args_r[@]}" "${mae_dep_args_r[@]}" | awk '{print "mae_imagenetr_reviewer " $0}'
fi
