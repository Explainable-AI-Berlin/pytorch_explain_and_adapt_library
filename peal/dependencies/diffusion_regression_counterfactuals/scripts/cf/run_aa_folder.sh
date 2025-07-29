#!/bin/bash

python adversarial_attack_folder.py \
    --input_folder $FOLDER/orig/ \
    --predictor_type resnet18 \
    --predictor_path /home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs/square_resnet18/square_resnet/version_0/checkpoints/epoch=82-step=33200.ckpt \
    --size 64 \
    --target 1
