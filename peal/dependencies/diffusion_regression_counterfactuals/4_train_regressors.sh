#!/bin/bash
set -e
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi

# Will be automatically saved in $DCFIR_OUTPATH/regressors
SQUARE_PATH=$DCFIR_OUTPATH/datasets/square
IMDB_CLEAN_PATH=$DCFIR_OUTPATH/datasets/imdb-clean/imdb-clean-1024-cropped

# Train Square Regressor
echo "Training Square Regressor..."
python scripts/train/train_resnet_square.py --folder_path $SQUARE_PATH --name square
# Results in $DCFIR_OUTPATH/regressors/square/version_0/checkpoints/last.ckpt

echo "Training CelebAHq regressor and oracle..."
# Weights to finetune
WEIGHTS_PATH=$DCFIR_HOME/pretrained_models/decision_densenet/celebamaskhq/checkpoint.tar
# Train CelebAHQ Regressor
python $DCFIR_HOME/scripts/train/train_imdb_clean.py \
    --folder_path $IMDB_CLEAN_PATH \
    --name "imdb-clean" \
    --densenet_weights "$WEIGHTS_PATH" \
    --image_size 256
# Results in $DCFIR_OUTPATH/regressors/imdb-clean/version_0/checkpoints/last.ckpt

# Train CelebAHQ Oracle
python $DCFIR_HOME/scripts/train/train_imdb_clean.py \
    --folder_path $IMDB_CLEAN_PATH \
    --name "imdb-clean_oracle" \
    --densenet_weights "$WEIGHTS_PATH" \
    --image_size 256 \
    $FULL_FINETUNE
# Results in $DCFIR_OUTPATH/regressors/imdb-clean_oracle/version_0/checkpoints/last.ckpt
