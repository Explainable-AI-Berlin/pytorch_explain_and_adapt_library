#!/bin/bash

python run_metrics.py \
    --real_folder="real_dist/square" \
    --fake_folder="fake_dist/square/full_copy" \
    --fake_folder="fake_dist/square/subset" \
    --fake_folder="fake_dist/square/just_white" \
    --fake_folder="fake_dist/square/just_black" \
    --limit=100000 \
    --size=64 \
    --batch_size=128 \
    --metric_type=distribution
