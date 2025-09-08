#!/bin/bash
# Run Repeatedly
for i in {0..8}; do
    echo "---------------- DHA: REPEATED DCF: $i SEED: $RANDOM ----------------"
    python /home/tha/diffae/dae_counterfactuals/diffeo_cf.py \
        --image_path=/home/tha/diffae/imgs_align/sandy.png \
        --resize=256 \
        --save_at=0.6 \
        --rmodel_path="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/imdb_clean_256/version_1/checkpoints/epoch=49-step=287350.ckpt" \
        --rmodel_type="resnet152" \
        --seed=$RANDOM \
        --result_dir="result_rep/dcf_age_toold_$i"
done
