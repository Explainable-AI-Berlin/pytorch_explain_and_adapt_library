#!/bin/bash

# Define variables for the arguments
GMODEL_PATH="/home/tha/diffeo-cf/models/2022_Counterfactuals_pretrained_models/checkpoints/generative_models/CelebA_pGAN.pth"
GMODEL_TYPE="GAN"
RMODEL_PATH="/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs_old/imdb_clean_m1-64/version_0/checkpoints/epoch=0099-step=287400-val_loss=1.167e-02.ckpt"
DATASET="CelebA"
ATTACK_STYLE="z"
NUM_STEPS=100
LR=0.005
TARGET=0.8
CONFIDENCE_THRESHOLD=0.01
IMAGE_FOLDER="/home/tha/datasets/ffhq_samples"
SIZE=64
RESULT_DIR="/home/tha/thesis_runs/diffeocf_results_default_gan_t=$TARGET"

export THESIS_DEBUG=true
# Run the Python script with the arguments
apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_default.py \
    --gmodel_path $GMODEL_PATH \
    --gmodel_type $GMODEL_TYPE \
    --rmodel_path $RMODEL_PATH \
    --dataset $DATASET \
    --attack_style $ATTACK_STYLE \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --target $TARGET \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMAGE_FOLDER \
    --size $SIZE \
    --result_dir $RESULT_DIR
