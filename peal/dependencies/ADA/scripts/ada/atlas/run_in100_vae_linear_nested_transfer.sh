#!/bin/bash
#SBATCH --job-name=ADA_in100_vae_nested
#SBATCH --partition=gpu-2h
#SBATCH --constraint=40gb|80gb|h100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
VAE_TYPE="${VAE_TYPE:-sdvae-mse}"
VAE_NAME_SAFE="${VAE_TYPE//-/_}"
PRED_ROOT="${PRED_ROOT:-${REPO}/artifacts/ada/atlas/predictions/v0_in100_vae_linear}"
REPORT_ROOT="${REPORT_ROOT:-${REPO}/artifacts/ada/atlas/reports}"
DINO_SUPPORT_CSV="${DINO_SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/scores/e0_in100_dinov2_val_vs_train/support_scores.csv}"

PCA_EUCLIDEAN_SUPPORT_CSV="${PCA_EUCLIDEAN_SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/support/v1_in100_sdvae_mse_pca512_euclidean/val_support_with_vae_metrics.csv}"
PCA_COSINE_SUPPORT_CSV="${PCA_COSINE_SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/support/v0_in100_sdvae_mse_pca512_cosine/val_support_with_vae_metrics.csv}"
RAW_EUCLIDEAN_SUPPORT_CSV="${RAW_EUCLIDEAN_SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/support/v0_in100_sdvae_mse_raw_flat_euclidean/val_support_with_vae_metrics.csv}"
STD_EUCLIDEAN_SUPPORT_CSV="${STD_EUCLIDEAN_SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/support/v0_in100_sdvae_mse_standardized_flat_euclidean/val_support_with_vae_metrics.csv}"
STD_COSINE_SUPPORT_CSV="${STD_COSINE_SUPPORT_CSV:-${REPO}/artifacts/ada/atlas/support/v0_in100_sdvae_mse_standardized_flat_cosine/val_support_with_vae_metrics.csv}"

mkdir -p "${REPO}/logs"
cd "${REPO}"

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] VAE_TYPE=${VAE_TYPE}"
echo "[JOB] PRED_ROOT=${PRED_ROOT}"
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

    pred_args=(
      --prediction-csv '${PRED_ROOT}/${VAE_NAME_SAFE}_pca512_linear_probe_seed0/predictions.csv'
      --prediction-csv '${PRED_ROOT}/${VAE_NAME_SAFE}_raw_flat_linear_probe_seed0/predictions.csv'
      --prediction-csv '${PRED_ROOT}/${VAE_NAME_SAFE}_standardized_flat_linear_probe_seed0/predictions.csv'
      --prediction-csv '${PRED_ROOT}/${VAE_NAME_SAFE}_pca512_linear_probe_seed1/predictions.csv'
      --prediction-csv '${PRED_ROOT}/${VAE_NAME_SAFE}_raw_flat_linear_probe_seed1/predictions.csv'
      --prediction-csv '${PRED_ROOT}/${VAE_NAME_SAFE}_standardized_flat_linear_probe_seed1/predictions.csv'
      --prediction-csv '${PRED_ROOT}/${VAE_NAME_SAFE}_pca512_linear_probe_seed2/predictions.csv'
      --prediction-csv '${PRED_ROOT}/${VAE_NAME_SAFE}_raw_flat_linear_probe_seed2/predictions.csv'
      --prediction-csv '${PRED_ROOT}/${VAE_NAME_SAFE}_standardized_flat_linear_probe_seed2/predictions.csv'
    )

    run_compare() {
      local name=\"\$1\"
      local support_csv=\"\$2\"
      python3 -m ada.atlas.cli.compare_support_transfer \
        \"\${pred_args[@]}\" \
        --dino-support-csv '${DINO_SUPPORT_CSV}' \
        --vae-support-csv \"\${support_csv}\" \
        --dino-support-name dinov2_cls \
        --vae-support-name \"\${name}\" \
        --output-dir '${REPORT_ROOT}/v0e_in100_${VAE_NAME_SAFE}_linear_nested_dinov2_vs_'\${name}
    }

    run_compare vae_pca512_euclidean '${PCA_EUCLIDEAN_SUPPORT_CSV}'
    run_compare vae_pca512_cosine '${PCA_COSINE_SUPPORT_CSV}'
    run_compare vae_raw_flat_euclidean '${RAW_EUCLIDEAN_SUPPORT_CSV}'
    run_compare vae_standardized_flat_euclidean '${STD_EUCLIDEAN_SUPPORT_CSV}'
    run_compare vae_standardized_flat_cosine '${STD_COSINE_SUPPORT_CSV}'

    echo \"[OUTPUT] report_root=${REPORT_ROOT}/v0e_in100_${VAE_NAME_SAFE}_linear_nested_dinov2_vs_*\"
  "

echo "[DONE] DATE=$(date)"
