#!/bin/bash
#SBATCH --job-name=dcf_imdb
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/cf/imdb-wiki-clean

# Define variables for the arguments
GMODEL_PATH="/home/tha/diffae/checkpoints/ffhq256_autoenc/last.ckpt"
RMODEL_PATH="/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs/imdb_clean-256/version_0/checkpoints/last.ckpt"
RORACLE_PATH="/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs/imdb_clean_oracle-256/version_0/checkpoints/last.ckpt"
NUM_STEPS=100
LR=0.005
CONFIDENCE_THRESHOLD=0.05
IMAGE_FOLDER="/home/tha/diffae/imgs_align"
SIZE=256

# DAE params
FORWARD_T=250
BACKWARD_T=20

export THESIS_DEBUG=true
TARGET=1.0
STOP_AT=0.8
RESULT_DIR="/home/tha/thesis_runs/diffeocf_dae_results/test"

# Run the Python script with the arguments
apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_dae_imdb.py \
    --gmodel_path $GMODEL_PATH \
    --rmodel_path $RMODEL_PATH \
    --roracle_path $RORACLE_PATH \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --target $TARGET \
    --stop_at $STOP_AT \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMAGE_FOLDER \
    --size $SIZE \
    --result_dir $RESULT_DIR \
    --forward_t $FORWARD_T \
    --backward_t $BACKWARD_T \
    --limit_samples 1

TARGET=inf
STOP_AT=0.8
RESULT_DIR="/home/tha/thesis_runs/diffeocf_dae_results/test_untargeted"

# Run the Python script with the arguments
apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_dae_imdb.py \
    --gmodel_path $GMODEL_PATH \
    --rmodel_path $RMODEL_PATH \
    --roracle_path $RORACLE_PATH \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --target $TARGET \
    --stop_at $STOP_AT \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMAGE_FOLDER \
    --size $SIZE \
    --result_dir $RESULT_DIR \
    --forward_t $FORWARD_T \
    --backward_t $BACKWARD_T \
    --limit_samples 1
# RESULT_DIR="/home/tha/thesis_runs/diffeocf_dae_results/test_noema"
# Run the Python script with the arguments
# apptainer run \
#     -B /home/space/datasets:/home/space/datasets \
#     --nv \
#     ~/apptainers/thesis.sif \
#     python run_diffeocf_dae_imdb.py \
#     --gmodel_path $GMODEL_PATH \
#     --rmodel_path $RMODEL_PATH \
#     --roracle_path $RORACLE_PATH \
#     --num_steps $NUM_STEPS \
#     --lr $LR \
#     --target $TARGET \
#     --confidence_threshold $CONFIDENCE_THRESHOLD \
#     --image_folder $IMAGE_FOLDER \
#     --size $SIZE \
#     --result_dir $RESULT_DIR \
#     --forward_t $FORWARD_T \
#     --backward_t $BACKWARD_T \
#     --limit_samples 1 --no_ema
