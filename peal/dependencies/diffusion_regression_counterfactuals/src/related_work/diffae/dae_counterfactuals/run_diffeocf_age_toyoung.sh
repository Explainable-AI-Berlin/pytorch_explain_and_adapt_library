#!/bin/bash

apptainer run --bind /home/space/datasets:/home/space/datasets --nv ~/apptainers/thesis.sif \
    python /home/tha/diffae/dae_counterfactuals/diffeo_cf.py \
    --image_path=/home/tha/diffae/imgs_interpolate/1_a.png \
    --resize=256 \
    --target=0.0 \
    --maximize=False \
    --save_at=0.1 \
    --rmodel_path="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/imdb_clean-256/version_0/checkpoints/last.ckpt" \
    --result_dir="dcf_age_toyoung"
