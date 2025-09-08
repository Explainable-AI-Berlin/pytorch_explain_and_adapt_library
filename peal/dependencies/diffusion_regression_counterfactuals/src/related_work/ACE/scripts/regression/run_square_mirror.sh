#!/bin/bash
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --learn_sigma True --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
MODEL_PATH="$DCFIR_OUTPATH/models/square_ace_ddpm/last.pt"
RMODEL_PATH="$DCFIR_OUTPATH/regressors/square/version_0/checkpoints/last.ckpt"
CONFIDENCE_THRESHOLD="0.05"
IMAGE_SIZE="64"
ATACK_STEP=2.0
ATTACK_ITERATIONS=100
BATCH_SIZE=46

TARGET="mirror"
NAME="squares_lower"
OUTPUT_PATH="$DCFIR_OUTPATH/ac-re/square_mirror_lower"
IMAGE_FOLDER="$DCFIR_OUTPATH/datasets/square_val/squares_lower/"

python main_regression_square.py $MODEL_FLAGS \
    --model_path=$MODEL_PATH \
    --rmodel_path=$RMODEL_PATH \
    --attack_step=$ATACK_STEP \
    --target=$TARGET \
    --batch_size=$BATCH_SIZE \
    --confidence_threshold=$CONFIDENCE_THRESHOLD \
    --image_folder=$IMAGE_FOLDER \
    --image_size=$IMAGE_SIZE \
    --timestep_respacing 50 \
    --sampling_time_fraction 0.1 \
    --sampling_dilation 5 \
    --attack_iterations $ATTACK_ITERATIONS \
    --output_path=$OUTPUT_PATH \
    >logs/$NAME.log 2>&1 &

NAME="squares_upper"
OUTPUT_PATH="$DCFIR_OUTPATH/ac-re/square_mirror_upper"
IMAGE_FOLDER="$DCFIR_OUTPATH/datasets/square_val/squares_upper/"

echo "Runnning $NAME"
# Run the Python script with the arguments
python main_regression_square.py $MODEL_FLAGS \
    --model_path=$MODEL_PATH \
    --rmodel_path=$RMODEL_PATH \
    --attack_step=$ATACK_STEP \
    --target=$TARGET \
    --batch_size=$BATCH_SIZE \
    --confidence_threshold=$CONFIDENCE_THRESHOLD \
    --image_folder=$IMAGE_FOLDER \
    --image_size=$IMAGE_SIZE \
    --timestep_respacing 50 \
    --sampling_time_fraction 0.1 \
    --sampling_dilation 5 \
    --attack_iterations $ATTACK_ITERATIONS \
    --output_path=$OUTPUT_PATH \
    >logs/$NAME.log 2>&1 &

wait
