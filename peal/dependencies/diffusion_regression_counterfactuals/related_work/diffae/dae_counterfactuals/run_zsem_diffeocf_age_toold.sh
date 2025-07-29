#!/bin/bash

for i in {0..8}
do
    echo "---------------- DHA: FIXED_ZSEM: $i SEED: $RANDOM ----------------"
    python /home/tha/diffae/dae_counterfactuals/diffeo_cf_fixed_zsem.py \
        --image_path=/home/tha/diffae/imgs_align/sandy.png \
        --resize=256 \
        --save_at=0.6 \
        --rmodel_path="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/imdb_clean_256/version_1/checkpoints/epoch=49-step=287350.ckpt" \
        --rmodel_type="resnet152" \
        --z_sem="/home/tha/diffae/dcf_age_toold/sandy_z.pt" \
        --seed=$RANDOM\
        --result_dir="result_fixed_zsem/zsem_dcf_age_toold_$i"
done
