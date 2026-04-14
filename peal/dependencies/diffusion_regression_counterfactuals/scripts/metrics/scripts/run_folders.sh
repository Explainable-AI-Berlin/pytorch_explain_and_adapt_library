#!/bin/bash
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <results_folder>"
    exit 1
fi

IMGS_FOLDER="$DCFIR_OUTPATH/datasets/CelebAMask-HQ/CelebA-HQ-img/"
RESULTS_FOLDER="$1"
OUTPUT_FOLDER="$DCFIR_OUTPATH/metrics/celebahq"

# Find all cf folders in the results folder. Assuming we used the same original images for all experiments.
FAKE_FOLDERS=$(find "$RESULTS_FOLDER" -mindepth 1 -maxdepth 1 -type d -o -type l -xtype d -print0 | xargs -0 -I {} echo --fake_folder="{}/cf" | tr '\n' ' ')

if [ -z "$FAKE_FOLDERS" ]; then
    echo "No cf folders found in $RESULTS_FOLDER"
    exit 1
fi

echo "FAKE_FOLDERS: $FAKE_FOLDERS"

FR_CLASSIFIER_PATH="$DCFIR_HOME/pretrained_models/decision_densenet/celebamaskhq/checkpoint.tar"
FVA_CLASSIFIER_PATH="$DCFIR_HOME/pretrained_models/resnet50_ft_weight.pkl"
MNAC_CLASSIFIER_PATH="$DCFIR_HOME/pretrained_models/oracle_attribute/celebamaskhq/checkpoint.tar"

echo "Running reference metrics for $IMGS_FOLDER"
python $DCFIR_HOME/scripts/metrics/run_metrics.py \
    --real_folder="$IMGS_FOLDER" \
    $FAKE_FOLDERS \
    --size=256 \
    --ace_fr_classifier_path=$FR_CLASSIFIER_PATH \
    --ace_fva_classifier_path=$FVA_CLASSIFIER_PATH \
    --ace_mnac_classifier_path=$MNAC_CLASSIFIER_PATH \
    $OUTPUT_FOLDER
