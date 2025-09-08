#!/bin/bash

python /home/tha/diffae/dae_counterfactuals/diffeo_cf_diversity.py \
    --image_path=/home/tha/diffae/imgs_align/sandy.png \
    --resize=256 \
    --rmodel_path="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/imdb_clean_256/version_1/checkpoints/epoch=49-step=287350.ckpt" \
    --rmodel_type="resnet152" \
    --zsem_path="/home/tha/diffae/dcf_age_toold/sandy_z.pt" \
    --result_dir="dcf_diversity_age_toold"
