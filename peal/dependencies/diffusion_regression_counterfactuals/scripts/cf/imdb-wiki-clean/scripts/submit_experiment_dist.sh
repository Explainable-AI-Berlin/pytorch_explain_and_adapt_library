#!/bin/bash
#SBATCH --job-name=dcf_dist_exp
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/cf/imdb-wiki-clean
set -x

# Define variables for the arguments
GMODEL_PATH="/home/tha/diffae/checkpoints/ffhq256_autoenc/last.ckpt"
RMODEL_PATH="/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs/imdb_clean-256/version_0/checkpoints/last.ckpt"
RORACLE_PATH="/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs/imdb_clean_oracle-256/version_0/checkpoints/last.ckpt"

# Algorithm params
IMAGE_FOLDER="/home/tha/datasets/celebahq_samples"
SIZE=256
NUM_STEPS=100
CONFIDENCE_THRESHOLD=0.05
LR=0.0039
TARGET=0.8
STOP_AT=0.8
OPTIMIZER=adam

# DAE params
FORWARD_T=250
BACKWARD_T=10

# Result Folder
IMAGE_FOLDER_BASENAME=$(basename $IMAGE_FOLDER)
TODAY=$(date '+%Y-%m-%d')

# Run without dist first
RESULT_DIR="/home/tha/thesis_runs/dcf_dae_res=$TODAY/dae_imgs=${IMAGE_FOLDER_BASENAME}_t=${TARGET}_dist=none_lr=${LR}_bt=${BACKWARD_T}_opt=${OPTIMIZER}"
apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_dae_imdb.py \
    --gmodel_path=$GMODEL_PATH \
    --rmodel_path=$RMODEL_PATH \
    --roracle_path=$RORACLE_PATH \
    --num_steps=$NUM_STEPS \
    --lr=$LR \
    --optimizer=$OPTIMIZER \
    --target="$TARGET" \
    --stop_at="$STOP_AT" \
    --confidence_threshold=$CONFIDENCE_THRESHOLD \
    --dist="none" \
    --image_folder="$IMAGE_FOLDER" \
    --size="$SIZE" \
    --result_dir="$RESULT_DIR" \
    --forward_t=$FORWARD_T \
    --backward_t=$BACKWARD_T \
    >logs/job-dist-none.out 2>&1 &

# Run the Python script with the arguments
for DIST in l1 l2; do
    for DIST_TYPE in latent pixel; do
        RESULT_DIR="/home/tha/thesis_runs/dcf_dae_res=$TODAY/dae_imgs=${IMAGE_FOLDER_BASENAME}_t=${TARGET}_dist=${DIST_TYPE}-${DIST}_lr=${LR}_bt=${BACKWARD_T}_opt=${OPTIMIZER}"
        echo "Running with the following parameters:"
        echo "IMAGE_FOLDER=$IMAGE_FOLDER, SIZE=$SIZE, TARGET=$TARGET, STOP_AT=$STOP_AT, LR=$LR, BACKWARD_T=$BACKWARD_T, DIST=$DIST, OPTIMIZER=$OPTIMIZER"
        apptainer run \
            -B /home/space/datasets:/home/space/datasets \
            --nv \
            ~/apptainers/thesis.sif \
            python run_diffeocf_dae_imdb.py \
            --gmodel_path=$GMODEL_PATH \
            --rmodel_path=$RMODEL_PATH \
            --roracle_path=$RORACLE_PATH \
            --num_steps=$NUM_STEPS \
            --lr=$LR \
            --optimizer=$OPTIMIZER \
            --target="$TARGET" \
            --stop_at="$STOP_AT" \
            --confidence_threshold=$CONFIDENCE_THRESHOLD \
            --dist="$DIST" \
            --dist_type="$DIST_TYPE" \
            --image_folder="$IMAGE_FOLDER" \
            --size="$SIZE" \
            --result_dir="$RESULT_DIR" \
            --forward_t=$FORWARD_T \
            --backward_t=$BACKWARD_T
    done
done

wait