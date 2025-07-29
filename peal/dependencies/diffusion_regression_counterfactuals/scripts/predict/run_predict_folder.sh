#!/bin/bash
#SBATCH --job-name=thesis_predict
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/train
set -x

# FOLDER_PATH=$1
FOLDER_PATH="/data/CelebAMask-HQ/CelebA-HQ-img"
SIZE=$2

if [ -z "$FOLDER_PATH" ] || [ -z "$SIZE" ]; then
    echo "Usage: $0 <folder_path> <size>"
    exit 1
fi

PREDICTOR_PATH="/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs/imdb_clean-256/version_0/checkpoints/last.ckpt"

apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    -B /home/space/datasets-sqfs/CelebAMask-HQ.sqfs:/data/CelebAMask-HQ:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python age_predict_folder.py \
    "$FOLDER_PATH" \
    --size "$SIZE" \
    --predictor_path "$PREDICTOR_PATH"
