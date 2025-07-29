#!/bin/bash
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi

GMODEL_PATH="$DCFIR_OUTPATH/models/square_diffae/square64_ddim/last.ckpt"
RMODEL_PATH="$DCFIR_OUTPATH/regressors/square/version_0/checkpoints/last.ckpt"
NUM_STEPS=200
BATCH_SIZE=38
LR=0.002
CONFIDENCE_THRESHOLD=0.05

BACKWARD_T=10

IMG_FOLDER="$DCFIR_OUTPATH/datasets/square_val/squares_lower/"
RESULT_DIR="$DCFIR_OUTPATH/diffae-re/square_mirror_lower"
mkdir -p logs

python $DCFIR_HOME/scripts/cf/squares/run_diffeocf_dae_square.py \
    --gmodel_path $GMODEL_PATH \
    --rmodel_path $RMODEL_PATH \
    --backward_t $BACKWARD_T \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMG_FOLDER \
    --result_dir $RESULT_DIR \

IMG_FOLDER="$DCFIR_OUTPATH/datasets/square_val/squares_upper/"
RESULT_DIR="$DCFIR_OUTPATH/diffae-re/square_mirror_upper"
python $DCFIR_HOME/scripts/cf/squares/run_diffeocf_dae_square.py \
    --gmodel_path $GMODEL_PATH \
    --rmodel_path $RMODEL_PATH \
    --backward_t $BACKWARD_T \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMG_FOLDER \
    --result_dir $RESULT_DIR \

