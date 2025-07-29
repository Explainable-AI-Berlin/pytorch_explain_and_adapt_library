#!/bin/bash
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi

# Define variables for the arguments
GMODEL_PATH="$DCFIR_HOME/pretrained_models/diffae/last.ckpt"

# Algorithm params
NUM_STEPS=200
CONFIDENCE_THRESHOLD=0.05

# DAE params
FORWARD_T=250
BATCH_SIZE=4

# Check if the positional arguments are provided
if [ "$1" == "-h" ] || [ $# -gt 6 ]; then
    echo "Usage: $0 [lr=${LR}] [backward_t=${BACKWARD_T}] [dist=${DIST}] [dist_type=${DIST_TYPE}] [optimizer=${OPTIMIZER}] [rmodel_path=${RMODEL_PATH}] [roracle_path=${RORACLE_PATH}] [output_path=${OUTPUT_PATH}]"
    exit 0
fi

# Positional arguments
LR=${1:-0.002}
BACKWARD_T=${2:-10}
DIST=${3:-none}
DIST_TYPE=${4:-latent}
OPTIMIZER=${5:-adam}
RMODEL_PATH=${6:-"$DCFIR_OUTPATH/regressors/imdb-clean/version_0/checkpoints/last.ckpt"}
RORACLE_PATH="$DCFIR_OUTPATH/regressors/imdb-clean_oracle/version_0/checkpoints/last.ckpt"
OUTPUT_PATH="$DCFIR_OUTPATH/diffae-re/celebahq"

echo "Running with the following parameters:"
echo "LR=$LR, BACKWARD_T=$BACKWARD_T, DIST=$DIST, DIST_TYPE=$DIST_TYPE, OPTIMIZER=$OPTIMIZER"

IMAGE_FOLDER="$DCFIR_OUTPATH/datasets/CelebAMask-HQ"
NAME="CelebaHQ_lr=$LR-bt=$BACKWARD_T-dist=$DIST-dist_type=$DIST_TYPE-opt=$OPTIMIZER"
RESULT_DIR="$OUTPUT_PATH/$NAME"

# Run the Python script with the arguments. It will terminate early if SIGUSR1 is received.
python $DCFIR_HOME/scripts/cf/celebahq/run_diffeocf_dae_celebahq.py \
    --gmodel_path=$GMODEL_PATH \
    --rmodel_path=$RMODEL_PATH \
    --roracle_path=$RORACLE_PATH \
    --batch_size=$BATCH_SIZE \
    --num_steps=$NUM_STEPS \
    --lr=$LR \
    --optimizer=$OPTIMIZER \
    --confidence_threshold=$CONFIDENCE_THRESHOLD \
    --dist="$DIST" \
    --dist_type="$DIST_TYPE" \
    --image_folder="$IMAGE_FOLDER" \
    --result_dir="$RESULT_DIR" \
    --forward_t=$FORWARD_T \
    --backward_t=$BACKWARD_T
