#!/bin/bash
#SBATCH --job-name=find_images
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/imgs

# Variables
SAMPLE_FOLDER="ace_samples"
DATASET_FOLDER="/data/CelebAMask-HQ/CelebA-HQ-img"
OUTPUT_FOLDER="found_samples"
SIZE=64

# Run the Python script
apptainer run \
    -B /home/space/datasets-sqfs/CelebAMask-HQ.sqfs:/data/CelebAMask-HQ:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python find_images.py $SAMPLE_FOLDER $DATASET_FOLDER $OUTPUT_FOLDER --size $SIZE
