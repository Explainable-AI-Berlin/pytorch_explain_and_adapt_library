#!/bin/bash

FFHQ="/data/ffhq/images1024x1024"
CELEBAHQ="/data/CelebAMask-HQ/CelebA-HQ-img"
CELEBAHQ_SAMPLES="/home/tha/datasets/celebahq_samples"

apptainer run \
    -B /home/space/datasets-sqfs/CelebAMask-HQ.sqfs:/data/CelebAMask-HQ:image-src=/ \
    -B /home/space/datasets-sqfs/ffhq.sqfs:/data/ffhq:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python run_metrics.py \
    --real_folder="$FFHQ" \
    --fake_folder="$CELEBAHQ" \
    --fake_folder="$CELEBAHQ_SAMPLES" \
    --size=256 \
    --metric_type=distribution \
    --limit 30000 &

apptainer run \
    -B /home/space/datasets-sqfs/CelebAMask-HQ.sqfs:/data/CelebAMask-HQ:image-src=/ \
    -B /home/space/datasets-sqfs/ffhq.sqfs:/data/ffhq:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python run_metrics.py \
    --real_folder="$CELEBAHQ" \
    --fake_folder="$CELEBAHQ_SAMPLES" \
    --size=256 \
    --metric_type=distribution \
    --limit 30000 &

wait
