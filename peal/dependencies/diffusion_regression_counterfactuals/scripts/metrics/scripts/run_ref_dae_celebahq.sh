#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <experiment_results_folder> <output_folder>"
    exit 1
fi

IMAGE_FOLDER="/data/CelebAMask-HQ/CelebA-HQ-img"
RESULTS_FOLDER="$1"
OUTPUT_FOLDER="$2"

# Find all cf folders in the results folder. Assuming we used the same original images for all experiments.
FAKE_FOLDERS=$(find "$RESULTS_FOLDER" -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -I {} echo --fake_folder="{}/cf" | tr '\n' ' ')

echo "FAKE_FOLDERS: $FAKE_FOLDERS"

FR_CLASSIFIER_PATH="/home/tha/ACE/pretrained/decision_densenet/celebamaskhq/checkpoint.tar"
FVA_CLASSIFIER_PATH="/home/tha/ACE/pretrained/resnet50_ft_weight.pkl"
MNAC_CLASSIFIER_PATH="/home/tha/ACE/pretrained/oracle/oracle_attribute/celebamaskhq/checkpoint.tar"

apptainer run \
    -B /home/space/datasets-sqfs/CelebAMask-HQ.sqfs:/data/CelebAMask-HQ:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python run_metrics.py \
    --real_folder="$IMAGE_FOLDER" \
    $FAKE_FOLDERS \
    --size=256 \
    --metric_type=reference \
    --ace_fr_classifier_path=$FR_CLASSIFIER_PATH \
    --ace_fva_classifier_path=$FVA_CLASSIFIER_PATH \
    --ace_mnac_classifier_path=$MNAC_CLASSIFIER_PATH \
    $OUTPUT_FOLDER
