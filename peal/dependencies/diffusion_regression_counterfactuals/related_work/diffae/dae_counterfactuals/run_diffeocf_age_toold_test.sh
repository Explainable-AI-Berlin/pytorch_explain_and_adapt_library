#!/bin/bash

# Needs to be run from top dir
apptainer run --bind /home/space/datasets:/home/space/datasets --nv ~/apptainers/thesis.sif \
    python /home/tha/diffae/dae_counterfactuals/diffeo_cf.py \
    --image_path="/home/space/datasets/imdb-wiki-clean/imdb-clean/data/clean/imdb-clean-1024-cropped/54/nm0848554_rm3904746496_1983-5-14_2004.jpg" \
    --resize=256 \
    --save_at=0.8 \
    --rmodel_path="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/imdb_clean-256/version_0/checkpoints/last.ckpt" \
    --result_dir="dcf_age_toold_test"
