#!/bin/bash

python run_metrics.py \
    --real_folder="real_dist/celebA" \
    --fake_folder="real_dist/celebA" \
    --fake_folder="fake_dist/celebA/subset" \
    --limit=100000 \
    --size=128 \
    --batch_size=128 \
    --metric_type=distribution
