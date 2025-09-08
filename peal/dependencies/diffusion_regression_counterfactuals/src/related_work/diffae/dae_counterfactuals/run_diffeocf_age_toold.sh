#!/bin/bash

apptainer run --bind /home/space/datasets:/home/space/datasets --nv ~/apptainers/thesis.sif \
    python /home/tha/diffae/dae_counterfactuals/diffeo_cf.py \
    --image_path=/home/tha/diffae/imgs_align/sandy.png \
    --resize=256 \
    --save_at=0.8 \
    --rmodel_path="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/imdb_clean-256/version_0/checkpoints/last.ckpt" \
    --result_dir="dcf_age_toold"
