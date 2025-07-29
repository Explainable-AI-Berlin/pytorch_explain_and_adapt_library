#!/bin/bash
set -e

echo "Comparing FID of square_mirrored with original square"
python run_metrics.py \
    --real_folder="real_dist/square" \
    --fake_folder="real_dist/square_mirror/square_lower" \
    --fake_folder="real_dist/square_mirror/square_upper" \
    --fake_folder="fake_dist/square_cf/ace_default_square_mirror_lower" \
    --fake_folder="fake_dist/square_cf/ace_default_square_mirror_upper" \
    --limit=100000 \
    --size=64 \
    --batch_size=128 \
    --metric_type=distribution

echo "Processing squares_lower"
python run_metrics.py \
    --real_folder="real_dist/square_mirror/square_lower" \
    --fake_folder="fake_dist/square_cf/ace_default_square_mirror_lower" \
    --limit=100000 \
    --size=64 \
    --batch_size=128 \
    --metric_type=distribution

python run_metrics.py \
    --real_folder="real_ref/ace_default_square_mirror_lower" \
    --fake_folder="fake_ref/square_cf/ace_default_square_mirror_lower" \
    --limit=100000 \
    --size=64 \
    --batch_size=128 \
    --metric_type=reference

echo "Processing squares_upper"
python run_metrics.py \
    --real_folder="real_dist/square_mirror/square_upper" \
    --fake_folder="fake_dist/square_cf/ace_default_square_mirror_upper" \
    --limit=100000 \
    --size=64 \
    --batch_size=128 \
    --metric_type=distribution

python run_metrics.py \
    --real_folder="real_ref/ace_default_square_mirror_upper" \
    --fake_folder="fake_ref/square_cf/ace_default_square_mirror_upper" \
    --limit=100000 \
    --size=64 \
    --batch_size=128 \
    --metric_type=reference
