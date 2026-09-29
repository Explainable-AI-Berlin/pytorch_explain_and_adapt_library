#!/bin/bash
#SBATCH --job-name=ADA_in100_vae_linear
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
VAE_TYPE="${VAE_TYPE:-sdvae-mse}"
VAE_NAME_SAFE="${VAE_TYPE//-/_}"

RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-${REPO}/artifacts/ada/atlas/vae_latents/imagenet100/train/sdvae-mse/posterior_mean_flat/vae-cache-9e394ef835546790}"
RAW_VAL_CACHE="${RAW_VAL_CACHE:-${REPO}/artifacts/ada/atlas/vae_latents/imagenet100/val/sdvae-mse/posterior_mean_flat/vae-cache-0ced7e0c2660d219}"
STD_TRAIN_CACHE="${STD_TRAIN_CACHE:-${REPO}/artifacts/ada/atlas/vae_standardized/imagenet100/train/sdvae-mse/flat_standardized}"
STD_VAL_CACHE="${STD_VAL_CACHE:-${REPO}/artifacts/ada/atlas/vae_standardized/imagenet100/val/sdvae-mse/flat_standardized}"
PCA_TRAIN_CACHE="${PCA_TRAIN_CACHE:-${REPO}/artifacts/ada/atlas/vae_projected/imagenet100/train/sdvae-mse/pca512}"
PCA_VAL_CACHE="${PCA_VAL_CACHE:-${REPO}/artifacts/ada/atlas/vae_projected/imagenet100/val/sdvae-mse/pca512}"

DINO_SUPPORT_CSV="${DINO_SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/scores/e0_in100_dinov2_val_vs_train/support_scores.csv}"
VAE_PCA_EUCLIDEAN_SUPPORT_CSV="${VAE_PCA_EUCLIDEAN_SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/support/v1_in100_sdvae_mse_pca512_euclidean/val_support_with_vae_metrics.csv}"
VAE_PCA_COSINE_SUPPORT_CSV="${VAE_PCA_COSINE_SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/support/v0_in100_sdvae_mse_pca512_cosine/val_support_with_vae_metrics.csv}"
VAE_RAW_SUPPORT_CSV="${VAE_RAW_SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/support/v0_in100_sdvae_mse_raw_flat_euclidean/val_support_with_vae_metrics.csv}"
VAE_STD_SUPPORT_CSV="${VAE_STD_SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/support/v0_in100_sdvae_mse_standardized_flat_euclidean/val_support_with_vae_metrics.csv}"

PRED_ROOT="${PRED_ROOT:-${REPO}/artifacts/ada/atlas/predictions/v0_in100_vae_linear}"
REPORT_ROOT="${REPORT_ROOT:-${REPO}/artifacts/ada/atlas/reports}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LR="${LR:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
CALIBRATION_FRACTION="${CALIBRATION_FRACTION:-0.1}"
SEED="${SEED:-0}"
REPORT_PREFIX="${REPORT_PREFIX:-v0d_in100_${VAE_NAME_SAFE}_linear_seed${SEED}}"

mkdir -p "${REPO}/logs" "${PRED_ROOT}"
cd "${REPO}"

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] VAE_TYPE=${VAE_TYPE}"
echo "[JOB] EPOCHS=${EPOCHS}"
echo "[JOB] BATCH_SIZE=${BATCH_SIZE}"
echo "[JOB] LR=${LR}"
echo "[JOB] WEIGHT_DECAY=${WEIGHT_DECAY}"
echo "[JOB] RAW_TRAIN_CACHE=${RAW_TRAIN_CACHE}"
echo "[JOB] STD_TRAIN_CACHE=${STD_TRAIN_CACHE}"
echo "[JOB] PCA_TRAIN_CACHE=${PCA_TRAIN_CACHE}"
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

    pca_pred='${PRED_ROOT}/${VAE_NAME_SAFE}_pca512_linear_probe_seed${SEED}/predictions.csv'
    raw_pred='${PRED_ROOT}/${VAE_NAME_SAFE}_raw_flat_linear_probe_seed${SEED}/predictions.csv'
    std_pred='${PRED_ROOT}/${VAE_NAME_SAFE}_standardized_flat_linear_probe_seed${SEED}/predictions.csv'

    python3 -m ada.atlas.cli.train_dino_linear_probe \
      --train-cache '${PCA_TRAIN_CACHE}' \
      --query-cache '${PCA_VAL_CACHE}' \
      --output-dir '${PRED_ROOT}/${VAE_NAME_SAFE}_pca512_linear_probe_seed${SEED}' \
      --epochs '${EPOCHS}' \
      --batch-size '${BATCH_SIZE}' \
      --lr '${LR}' \
      --weight-decay '${WEIGHT_DECAY}' \
      --calibration-fraction '${CALIBRATION_FRACTION}' \
      --seed '${SEED}' \
      --model-id '${VAE_NAME_SAFE}_pca512_linear_probe_seed${SEED}' \
      --device auto

    python3 -m ada.atlas.cli.train_dino_linear_probe \
      --train-cache '${RAW_TRAIN_CACHE}' \
      --query-cache '${RAW_VAL_CACHE}' \
      --output-dir '${PRED_ROOT}/${VAE_NAME_SAFE}_raw_flat_linear_probe_seed${SEED}' \
      --epochs '${EPOCHS}' \
      --batch-size '${BATCH_SIZE}' \
      --lr '${LR}' \
      --weight-decay '${WEIGHT_DECAY}' \
      --calibration-fraction '${CALIBRATION_FRACTION}' \
      --seed '${SEED}' \
      --model-id '${VAE_NAME_SAFE}_raw_flat_linear_probe_seed${SEED}' \
      --device auto

    python3 -m ada.atlas.cli.train_dino_linear_probe \
      --train-cache '${STD_TRAIN_CACHE}' \
      --query-cache '${STD_VAL_CACHE}' \
      --output-dir '${PRED_ROOT}/${VAE_NAME_SAFE}_standardized_flat_linear_probe_seed${SEED}' \
      --epochs '${EPOCHS}' \
      --batch-size '${BATCH_SIZE}' \
      --lr '${LR}' \
      --weight-decay '${WEIGHT_DECAY}' \
      --calibration-fraction '${CALIBRATION_FRACTION}' \
      --seed '${SEED}' \
      --model-id '${VAE_NAME_SAFE}_standardized_flat_linear_probe_seed${SEED}' \
      --device auto

    support_columns='support_k5_kth_distance,support_k10_kth_distance,support_k50_kth_distance,class_support_k5_kth_distance,class_support_k10_kth_distance,class_support_k50_kth_distance,posterior_kl_raw,latent_norm_raw,latent_norm_scaled_mean,latent_std_scalar'

    python3 -m ada.atlas.cli.summarize_prediction_support \
      --support-csv '${DINO_SUPPORT_CSV}' \
      --prediction-csv \"\${pca_pred}\" \
      --prediction-csv \"\${raw_pred}\" \
      --prediction-csv \"\${std_pred}\" \
      --output-dir '${REPORT_ROOT}/${REPORT_PREFIX}_vs_dinov2_cls_support' \
      --bins 10 \
      --skip-crossfit

    python3 -m ada.atlas.cli.summarize_prediction_support \
      --support-csv '${VAE_PCA_EUCLIDEAN_SUPPORT_CSV}' \
      --prediction-csv \"\${pca_pred}\" \
      --prediction-csv \"\${raw_pred}\" \
      --prediction-csv \"\${std_pred}\" \
      --output-dir '${REPORT_ROOT}/${REPORT_PREFIX}_vs_vae_pca512_euclidean_support' \
      --support-columns \"\${support_columns}\" \
      --primary-support-column support_k50_kth_distance \
      --primary-class-support-column class_support_k50_kth_distance \
      --bins 10 \
      --skip-crossfit

    python3 -m ada.atlas.cli.summarize_prediction_support \
      --support-csv '${VAE_PCA_COSINE_SUPPORT_CSV}' \
      --prediction-csv \"\${pca_pred}\" \
      --prediction-csv \"\${raw_pred}\" \
      --prediction-csv \"\${std_pred}\" \
      --output-dir '${REPORT_ROOT}/${REPORT_PREFIX}_vs_vae_pca512_cosine_support' \
      --support-columns \"\${support_columns}\" \
      --primary-support-column support_k50_kth_distance \
      --primary-class-support-column class_support_k50_kth_distance \
      --bins 10 \
      --skip-crossfit

    python3 -m ada.atlas.cli.summarize_prediction_support \
      --support-csv '${VAE_RAW_SUPPORT_CSV}' \
      --prediction-csv \"\${pca_pred}\" \
      --prediction-csv \"\${raw_pred}\" \
      --prediction-csv \"\${std_pred}\" \
      --output-dir '${REPORT_ROOT}/${REPORT_PREFIX}_vs_vae_raw_flat_euclidean_support' \
      --support-columns \"\${support_columns}\" \
      --primary-support-column support_k50_kth_distance \
      --primary-class-support-column class_support_k50_kth_distance \
      --bins 10 \
      --skip-crossfit

    python3 -m ada.atlas.cli.summarize_prediction_support \
      --support-csv '${VAE_STD_SUPPORT_CSV}' \
      --prediction-csv \"\${pca_pred}\" \
      --prediction-csv \"\${raw_pred}\" \
      --prediction-csv \"\${std_pred}\" \
      --output-dir '${REPORT_ROOT}/${REPORT_PREFIX}_vs_vae_standardized_flat_euclidean_support' \
      --support-columns \"\${support_columns}\" \
      --primary-support-column support_k50_kth_distance \
      --primary-class-support-column class_support_k50_kth_distance \
      --bins 10 \
      --skip-crossfit

    echo \"[OUTPUT] pca_pred=\${pca_pred}\"
    echo \"[OUTPUT] raw_pred=\${raw_pred}\"
    echo \"[OUTPUT] std_pred=\${std_pred}\"
    echo \"[OUTPUT] report_root=${REPORT_ROOT}/${REPORT_PREFIX}_vs_*\"
  "

echo "[DONE] DATE=$(date)"
