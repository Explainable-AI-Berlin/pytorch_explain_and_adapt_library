#!/bin/bash

python run_metrics.py \
    --real_folder="real_ref/dae" \
    --fake_folder="fake_ref/dae_aa/" \
    --fake_folder="fake_ref/dae_dcf/" \
    --limit=100000 \
    --size=256 \
    --batch_size=128 \
    --metric_type=reference \
    "results_metrics/ref_dae"
