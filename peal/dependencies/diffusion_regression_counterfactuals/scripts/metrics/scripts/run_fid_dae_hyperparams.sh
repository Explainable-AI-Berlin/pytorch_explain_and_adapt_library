#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <results_folder>"
    exit 1
fi

CELEBAHQ_FOLDER="/data/CelebAMask-HQ/CelebA-HQ-img"
FFHQ_FOLDER="/data/ffhq/images1024x1024"

RESULTS_FOLDER="$1"

FAKE_FOLDERS=$(find "$RESULTS_FOLDER" -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -I {} echo --fake_folder="{}/cf" | tr '\n' ' ')

echo "FAKE_FOLDERS: $FAKE_FOLDERS"

apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    -B /home/space/datasets-sqfs/CelebAMask-HQ.sqfs:/data/CelebAMask-HQ:image-src=/ \
    -B /home/space/datasets-sqfs/ffhq.sqfs:/data/ffhq:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python run_metrics.py \
    --real_folder="$CELEBAHQ_FOLDER" \
    $FAKE_FOLDERS \
    --size=256 \
    --metric_type=distribution \
    --limit 30000 &

apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    -B /home/space/datasets-sqfs/CelebAMask-HQ.sqfs:/data/CelebAMask-HQ:image-src=/ \
    -B /home/space/datasets-sqfs/ffhq.sqfs:/data/ffhq:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python run_metrics.py \
    --real_folder="$FFHQ_FOLDER" \
    $FAKE_FOLDERS \
    --size=256 \
    --metric_type=distribution \
    --limit 30000 &

wait
