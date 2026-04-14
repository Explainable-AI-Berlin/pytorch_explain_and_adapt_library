#!/bin/bash

python run_oracle.py \
    --folder "fake_ref/square_cf/ace_default_square_mirror_lower" \
    --folder "fake_ref/square_cf/ace_default_square_mirror_upper" \
    --size 64 \
    --model "/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs/square_m1/version_0/checkpoints/epoch=0099-val_loss=0.00.ckpt" \
    --oracle "/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs/square_m2/version_0/checkpoints/epoch=0099-val_loss=0.00.ckpt" \
    --batch_size 64
