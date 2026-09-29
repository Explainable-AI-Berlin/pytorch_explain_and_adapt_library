#!/bin/bash
#SBATCH --job-name=ADA_in100_vae_cosine
#SBATCH --partition=gpu-2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
DATASET="${DATASET:-imagenet100}"
VAE_TYPE="${VAE_TYPE:-sdvae-mse}"
VAE_NAME_SAFE="${VAE_TYPE//-/_}"
TRAIN_PROJECTED="${TRAIN_PROJECTED:-${REPO}/artifacts/ada/atlas/vae_projected/imagenet100/train/sdvae-mse/pca512}"
VAL_PROJECTED="${VAL_PROJECTED:-${REPO}/artifacts/ada/atlas/vae_projected/imagenet100/val/sdvae-mse/pca512}"
TRAIN_CACHE="${TRAIN_CACHE:-${REPO}/artifacts/ada/atlas/vae_latents/imagenet100/train/sdvae-mse/posterior_mean_flat/vae-cache-9e394ef835546790}"
VAL_CACHE="${VAL_CACHE:-${REPO}/artifacts/ada/atlas/vae_latents/imagenet100/val/sdvae-mse/posterior_mean_flat/vae-cache-0ced7e0c2660d219}"
PREDICTION_JOIN="${PREDICTION_JOIN:-${REPO}/artifacts/ada/atlas/reports/e0_in100_prediction_support/joined_predictions_support.csv}"
SUPPORT_DIR="${SUPPORT_DIR:-${REPO}/artifacts/ada/atlas/support/v0_in100_${VAE_NAME_SAFE}_pca512_cosine}"
REPORT_DIR="${REPORT_DIR:-${REPO}/artifacts/ada/atlas/reports/v0_in100_${VAE_NAME_SAFE}_pca512_cosine_support_error}"
BATCH_SIZE="${BATCH_SIZE:-256}"
REFERENCE_SHARD_SIZE="${REFERENCE_SHARD_SIZE:-20000}"
RUN_TRAIN_LOO="${RUN_TRAIN_LOO:-1}"

mkdir -p "${REPO}/logs"
cd "${REPO}"

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] DATASET=${DATASET}"
echo "[JOB] VAE_TYPE=${VAE_TYPE}"
echo "[JOB] TRAIN_PROJECTED=${TRAIN_PROJECTED}"
echo "[JOB] VAL_PROJECTED=${VAL_PROJECTED}"
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

    mkdir -p '${SUPPORT_DIR}'
    if [[ '${RUN_TRAIN_LOO}' == '1' ]]; then
      python3 -m ada.atlas.cli.score_knn \
        --query-cache '${TRAIN_PROJECTED}' \
        --reference-cache '${TRAIN_PROJECTED}' \
        --output-csv '${SUPPORT_DIR}/train_leave_one_out_support.csv' \
        --k 5,10,50 \
        --batch-size '${BATCH_SIZE}' \
        --device auto \
        --leave-one-out \
        --class-conditional \
        --mmap \
        --reference-shard-size '${REFERENCE_SHARD_SIZE}'

      python3 -m ada.atlas.cli.join_vae_support_metrics \
        --support-csv '${SUPPORT_DIR}/train_leave_one_out_support.csv' \
        --metrics-csv '${TRAIN_CACHE}/posterior_metrics.csv' \
        --output-csv '${SUPPORT_DIR}/train_leave_one_out_support_with_vae_metrics.csv' \
        --strict
    fi

    python3 -m ada.atlas.cli.score_knn \
      --query-cache '${VAL_PROJECTED}' \
      --reference-cache '${TRAIN_PROJECTED}' \
      --output-csv '${SUPPORT_DIR}/val_support.csv' \
      --k 5,10,50 \
      --batch-size '${BATCH_SIZE}' \
      --device auto \
      --class-conditional \
      --mmap \
      --reference-shard-size '${REFERENCE_SHARD_SIZE}'

    python3 -m ada.atlas.cli.join_vae_support_metrics \
      --support-csv '${SUPPORT_DIR}/val_support.csv' \
      --metrics-csv '${VAL_CACHE}/posterior_metrics.csv' \
      --output-csv '${SUPPORT_DIR}/val_support_with_vae_metrics.csv' \
      --strict

    support_columns='support_k5_kth_distance,support_k10_kth_distance,support_k50_kth_distance,class_support_k5_kth_distance,class_support_k10_kth_distance,class_support_k50_kth_distance,posterior_kl_raw,latent_norm_raw,latent_norm_scaled_mean,latent_std_scalar'
    python3 -m ada.atlas.cli.summarize_prediction_support \
      --support-csv '${SUPPORT_DIR}/val_support_with_vae_metrics.csv' \
      --prediction-csv '${PREDICTION_JOIN}' \
      --output-dir '${REPORT_DIR}' \
      --support-columns \"\${support_columns}\" \
      --primary-support-column support_k50_kth_distance \
      --primary-class-support-column class_support_k50_kth_distance \
      --bins 10 \
      --skip-crossfit

    echo \"[OUTPUT] support_dir=${SUPPORT_DIR}\"
    echo \"[OUTPUT] report_dir=${REPORT_DIR}\"
  "

echo "[DONE] DATE=$(date)"
