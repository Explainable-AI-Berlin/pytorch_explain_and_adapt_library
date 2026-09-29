#!/bin/bash
#SBATCH --job-name=ADA_in100_vae_mse
#SBATCH --partition=gpu-2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
DATASET="${DATASET:-imagenet100}"
VAE_TYPE="${VAE_TYPE:-sdvae-mse}"
VAE_NAME_SAFE="${VAE_TYPE//-/_}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/space/datasets/imagenet_torchvision/imagenet100/train}"
VAL_ROOT="${VAL_ROOT:-/home/space/datasets/imagenet_torchvision/imagenet100/val}"
RAW_OUTPUT_ROOT="${RAW_OUTPUT_ROOT:-${REPO}/artifacts/ada/atlas/vae_latents}"
PROJECTED_ROOT="${PROJECTED_ROOT:-${REPO}/artifacts/ada/atlas/vae_projected}"
PROJECTION_DIR="${PROJECTION_DIR:-${REPO}/artifacts/ada/atlas/vae_projections/v1_in100_${VAE_NAME_SAFE}_pca512_train_seed0}"
SUPPORT_DIR="${SUPPORT_DIR:-${REPO}/artifacts/ada/atlas/support/v1_in100_${VAE_NAME_SAFE}_pca512_euclidean}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PCA_DIM="${PCA_DIM:-512}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${REPO}/logs"
cd "${REPO}"

echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] DATASET=${DATASET}"
echo "[JOB] VAE_TYPE=${VAE_TYPE}"
echo "[JOB] TRAIN_ROOT=${TRAIN_ROOT}"
echo "[JOB] VAL_ROOT=${VAL_ROOT}"
echo "[JOB] PCA_DIM=${PCA_DIM}"
echo "[JOB] ALLOW_DOWNLOAD=${ALLOW_DOWNLOAD}"
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

    extra=()
    if [[ '${ALLOW_DOWNLOAD}' == '1' ]]; then
      extra+=(--allow-download)
    fi
    cache_overwrite=()
    if [[ '${OVERWRITE}' == '1' ]]; then
      cache_overwrite+=(--overwrite)
    else
      cache_overwrite+=(--reuse-existing)
    fi
    projection_overwrite=()
    if [[ '${OVERWRITE}' == '1' ]]; then
      projection_overwrite+=(--overwrite)
    fi

    train_json='${REPO}/logs/in100_${VAE_NAME_SAFE}_train_cache_${SLURM_JOB_ID:-manual}.json'
    val_json='${REPO}/logs/in100_${VAE_NAME_SAFE}_val_cache_${SLURM_JOB_ID:-manual}.json'

    python3 -m ada.atlas.cli.cache_vae_latents \
      --dataset '${DATASET}' \
      --split train \
      --root '${TRAIN_ROOT}' \
      --vae-type '${VAE_TYPE}' \
      --output-root '${RAW_OUTPUT_ROOT}' \
      --batch-size '${BATCH_SIZE}' \
      --num-workers '${NUM_WORKERS}' \
      --image-size 256 \
      --precision bf16 \
      --storage-dtype float16 \
      \"\${extra[@]}\" \
      \"\${cache_overwrite[@]}\" \
      > \"\${train_json}\"

    python3 -m ada.atlas.cli.cache_vae_latents \
      --dataset '${DATASET}' \
      --split val \
      --root '${VAL_ROOT}' \
      --vae-type '${VAE_TYPE}' \
      --output-root '${RAW_OUTPUT_ROOT}' \
      --batch-size '${BATCH_SIZE}' \
      --num-workers '${NUM_WORKERS}' \
      --image-size 256 \
      --precision bf16 \
      --storage-dtype float16 \
      \"\${extra[@]}\" \
      \"\${cache_overwrite[@]}\" \
      > \"\${val_json}\"

    train_cache=\$(python3 -c \"import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])\" \"\${train_json}\")
    val_cache=\$(python3 -c \"import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])\" \"\${val_json}\")

    python3 -m ada.atlas.cli.fit_vae_projection \
      --train-cache \"\${train_cache}\" \
      --output-dir '${PROJECTION_DIR}' \
      --pca-dim '${PCA_DIM}' \
      --batch-size 4096 \
      --mmap \
      \"\${projection_overwrite[@]}\"

    train_projected='${PROJECTED_ROOT}/${DATASET}/train/${VAE_TYPE}/pca${PCA_DIM}'
    val_projected='${PROJECTED_ROOT}/${DATASET}/val/${VAE_TYPE}/pca${PCA_DIM}'

    python3 -m ada.atlas.cli.project_vae_latents \
      --input-cache \"\${train_cache}\" \
      --projection-dir '${PROJECTION_DIR}' \
      --output-dir \"\${train_projected}\" \
      --batch-size 4096 \
      --mmap \
      \"\${projection_overwrite[@]}\"

    python3 -m ada.atlas.cli.project_vae_latents \
      --input-cache \"\${val_cache}\" \
      --projection-dir '${PROJECTION_DIR}' \
      --output-dir \"\${val_projected}\" \
      --batch-size 4096 \
      --mmap \
      \"\${projection_overwrite[@]}\"

    mkdir -p '${SUPPORT_DIR}'
    python3 -m ada.atlas.cli.score_euclidean_knn \
      --query-cache \"\${train_projected}\" \
      --reference-cache \"\${train_projected}\" \
      --output-csv '${SUPPORT_DIR}/train_leave_one_out_support.csv' \
      --k 5,10,50 \
      --batch-size 256 \
      --device auto \
      --leave-one-out \
      --class-conditional \
      --mmap \
      --reference-shard-size 20000

    python3 -m ada.atlas.cli.join_vae_support_metrics \
      --support-csv '${SUPPORT_DIR}/train_leave_one_out_support.csv' \
      --metrics-csv \"\${train_cache}/posterior_metrics.csv\" \
      --output-csv '${SUPPORT_DIR}/train_leave_one_out_support_with_vae_metrics.csv' \
      --strict

    python3 -m ada.atlas.cli.score_euclidean_knn \
      --query-cache \"\${val_projected}\" \
      --reference-cache \"\${train_projected}\" \
      --output-csv '${SUPPORT_DIR}/val_support.csv' \
      --k 5,10,50 \
      --batch-size 256 \
      --device auto \
      --class-conditional \
      --mmap \
      --reference-shard-size 20000

    python3 -m ada.atlas.cli.join_vae_support_metrics \
      --support-csv '${SUPPORT_DIR}/val_support.csv' \
      --metrics-csv \"\${val_cache}/posterior_metrics.csv\" \
      --output-csv '${SUPPORT_DIR}/val_support_with_vae_metrics.csv' \
      --strict

    echo \"[OUTPUT] train_cache=\${train_cache}\"
    echo \"[OUTPUT] val_cache=\${val_cache}\"
    echo \"[OUTPUT] train_projected=\${train_projected}\"
    echo \"[OUTPUT] val_projected=\${val_projected}\"
    echo \"[OUTPUT] support_dir=${SUPPORT_DIR}\"
  "

echo "[DONE] DATE=$(date)"
