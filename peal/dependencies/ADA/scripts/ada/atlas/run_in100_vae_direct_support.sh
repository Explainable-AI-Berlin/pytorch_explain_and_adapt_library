#!/bin/bash
#SBATCH --job-name=ADA_in100_vae_direct
#SBATCH --partition=gpu-2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
DATASET="${DATASET:-imagenet100}"
VAE_TYPE="${VAE_TYPE:-sdvae-mse}"
VAE_NAME_SAFE="${VAE_TYPE//-/_}"
TRAIN_CACHE="${TRAIN_CACHE:-${REPO}/artifacts/ada/atlas/vae_latents/imagenet100/train/sdvae-mse/posterior_mean_flat/vae-cache-9e394ef835546790}"
VAL_CACHE="${VAL_CACHE:-${REPO}/artifacts/ada/atlas/vae_latents/imagenet100/val/sdvae-mse/posterior_mean_flat/vae-cache-0ced7e0c2660d219}"
PREDICTION_JOIN="${PREDICTION_JOIN:-${REPO}/artifacts/ada/atlas/reports/e0_in100_prediction_support/joined_predictions_support.csv}"
STANDARDIZATION_DIR="${STANDARDIZATION_DIR:-${REPO}/artifacts/ada/atlas/vae_standardization/v0_in100_${VAE_NAME_SAFE}_flat_train_seed0}"
STANDARDIZED_ROOT="${STANDARDIZED_ROOT:-${REPO}/artifacts/ada/atlas/vae_standardized}"
RAW_SUPPORT_DIR="${RAW_SUPPORT_DIR:-${REPO}/artifacts/ada/atlas/support/v0_in100_${VAE_NAME_SAFE}_raw_flat_euclidean}"
STD_SUPPORT_DIR="${STD_SUPPORT_DIR:-${REPO}/artifacts/ada/atlas/support/v0_in100_${VAE_NAME_SAFE}_standardized_flat_euclidean}"
RAW_REPORT_DIR="${RAW_REPORT_DIR:-${REPO}/artifacts/ada/atlas/reports/v0_in100_${VAE_NAME_SAFE}_raw_flat_support_error}"
STD_REPORT_DIR="${STD_REPORT_DIR:-${REPO}/artifacts/ada/atlas/reports/v0_in100_${VAE_NAME_SAFE}_standardized_flat_support_error}"
BATCH_SIZE="${BATCH_SIZE:-64}"
REFERENCE_SHARD_SIZE="${REFERENCE_SHARD_SIZE:-10000}"
STANDARDIZE_BATCH_SIZE="${STANDARDIZE_BATCH_SIZE:-4096}"
RUN_TRAIN_LOO="${RUN_TRAIN_LOO:-1}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${REPO}/logs"
cd "${REPO}"

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] DATASET=${DATASET}"
echo "[JOB] VAE_TYPE=${VAE_TYPE}"
echo "[JOB] TRAIN_CACHE=${TRAIN_CACHE}"
echo "[JOB] VAL_CACHE=${VAL_CACHE}"
echo "[JOB] RUN_TRAIN_LOO=${RUN_TRAIN_LOO}"
echo "============================================================"

nvidia-smi -L || true

apptainer exec --nv \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export PYTHONPATH=src
    export PYTHONDONTWRITEBYTECODE=1

    overwrite=()
    if [[ '${OVERWRITE}' == '1' ]]; then
      overwrite+=(--overwrite)
    fi

    std_train='${STANDARDIZED_ROOT}/${DATASET}/train/${VAE_TYPE}/flat_standardized'
    std_val='${STANDARDIZED_ROOT}/${DATASET}/val/${VAE_TYPE}/flat_standardized'

    python3 -m ada.atlas.cli.fit_vae_standardization \
      --train-cache '${TRAIN_CACHE}' \
      --output-dir '${STANDARDIZATION_DIR}' \
      --batch-size '${STANDARDIZE_BATCH_SIZE}' \
      --mmap \
      \"\${overwrite[@]}\"

    python3 -m ada.atlas.cli.standardize_vae_latents \
      --input-cache '${TRAIN_CACHE}' \
      --standardization-dir '${STANDARDIZATION_DIR}' \
      --output-dir \"\${std_train}\" \
      --batch-size '${STANDARDIZE_BATCH_SIZE}' \
      --storage-dtype float32 \
      --mmap \
      \"\${overwrite[@]}\"

    python3 -m ada.atlas.cli.standardize_vae_latents \
      --input-cache '${VAL_CACHE}' \
      --standardization-dir '${STANDARDIZATION_DIR}' \
      --output-dir \"\${std_val}\" \
      --batch-size '${STANDARDIZE_BATCH_SIZE}' \
      --storage-dtype float32 \
      --mmap \
      \"\${overwrite[@]}\"

    mkdir -p '${RAW_SUPPORT_DIR}' '${STD_SUPPORT_DIR}'
    if [[ '${RUN_TRAIN_LOO}' == '1' ]]; then
      python3 -m ada.atlas.cli.score_euclidean_knn \
        --query-cache '${TRAIN_CACHE}' \
        --reference-cache '${TRAIN_CACHE}' \
        --output-csv '${RAW_SUPPORT_DIR}/train_leave_one_out_support.csv' \
        --k 5,10,50 \
        --batch-size '${BATCH_SIZE}' \
        --device auto \
        --leave-one-out \
        --class-conditional \
        --mmap \
        --reference-shard-size '${REFERENCE_SHARD_SIZE}'

      python3 -m ada.atlas.cli.join_vae_support_metrics \
        --support-csv '${RAW_SUPPORT_DIR}/train_leave_one_out_support.csv' \
        --metrics-csv '${TRAIN_CACHE}/posterior_metrics.csv' \
        --output-csv '${RAW_SUPPORT_DIR}/train_leave_one_out_support_with_vae_metrics.csv' \
        --strict

      python3 -m ada.atlas.cli.score_euclidean_knn \
        --query-cache \"\${std_train}\" \
        --reference-cache \"\${std_train}\" \
        --output-csv '${STD_SUPPORT_DIR}/train_leave_one_out_support.csv' \
        --k 5,10,50 \
        --batch-size '${BATCH_SIZE}' \
        --device auto \
        --leave-one-out \
        --class-conditional \
        --mmap \
        --reference-shard-size '${REFERENCE_SHARD_SIZE}'

      python3 -m ada.atlas.cli.join_vae_support_metrics \
        --support-csv '${STD_SUPPORT_DIR}/train_leave_one_out_support.csv' \
        --metrics-csv '${TRAIN_CACHE}/posterior_metrics.csv' \
        --output-csv '${STD_SUPPORT_DIR}/train_leave_one_out_support_with_vae_metrics.csv' \
        --strict
    fi

    python3 -m ada.atlas.cli.score_euclidean_knn \
      --query-cache '${VAL_CACHE}' \
      --reference-cache '${TRAIN_CACHE}' \
      --output-csv '${RAW_SUPPORT_DIR}/val_support.csv' \
      --k 5,10,50 \
      --batch-size '${BATCH_SIZE}' \
      --device auto \
      --class-conditional \
      --mmap \
      --reference-shard-size '${REFERENCE_SHARD_SIZE}'

    python3 -m ada.atlas.cli.join_vae_support_metrics \
      --support-csv '${RAW_SUPPORT_DIR}/val_support.csv' \
      --metrics-csv '${VAL_CACHE}/posterior_metrics.csv' \
      --output-csv '${RAW_SUPPORT_DIR}/val_support_with_vae_metrics.csv' \
      --strict

    python3 -m ada.atlas.cli.score_euclidean_knn \
      --query-cache \"\${std_val}\" \
      --reference-cache \"\${std_train}\" \
      --output-csv '${STD_SUPPORT_DIR}/val_support.csv' \
      --k 5,10,50 \
      --batch-size '${BATCH_SIZE}' \
      --device auto \
      --class-conditional \
      --mmap \
      --reference-shard-size '${REFERENCE_SHARD_SIZE}'

    python3 -m ada.atlas.cli.join_vae_support_metrics \
      --support-csv '${STD_SUPPORT_DIR}/val_support.csv' \
      --metrics-csv '${VAL_CACHE}/posterior_metrics.csv' \
      --output-csv '${STD_SUPPORT_DIR}/val_support_with_vae_metrics.csv' \
      --strict

    support_columns='support_k5_kth_distance,support_k10_kth_distance,support_k50_kth_distance,class_support_k5_kth_distance,class_support_k10_kth_distance,class_support_k50_kth_distance,posterior_kl_raw,latent_norm_raw,latent_norm_scaled_mean,latent_std_scalar'

    python3 -m ada.atlas.cli.summarize_prediction_support \
      --support-csv '${RAW_SUPPORT_DIR}/val_support_with_vae_metrics.csv' \
      --prediction-csv '${PREDICTION_JOIN}' \
      --output-dir '${RAW_REPORT_DIR}' \
      --support-columns \"\${support_columns}\" \
      --primary-support-column support_k50_kth_distance \
      --primary-class-support-column class_support_k50_kth_distance \
      --bins 10 \
      --skip-crossfit

    python3 -m ada.atlas.cli.summarize_prediction_support \
      --support-csv '${STD_SUPPORT_DIR}/val_support_with_vae_metrics.csv' \
      --prediction-csv '${PREDICTION_JOIN}' \
      --output-dir '${STD_REPORT_DIR}' \
      --support-columns \"\${support_columns}\" \
      --primary-support-column support_k50_kth_distance \
      --primary-class-support-column class_support_k50_kth_distance \
      --bins 10 \
      --skip-crossfit

    echo \"[OUTPUT] raw_support_dir=${RAW_SUPPORT_DIR}\"
    echo \"[OUTPUT] std_support_dir=${STD_SUPPORT_DIR}\"
    echo \"[OUTPUT] raw_report_dir=${RAW_REPORT_DIR}\"
    echo \"[OUTPUT] std_report_dir=${STD_REPORT_DIR}\"
  "

echo "[DONE] DATE=$(date)"
