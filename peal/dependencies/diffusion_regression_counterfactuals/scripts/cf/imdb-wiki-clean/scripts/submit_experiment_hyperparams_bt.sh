#!/bin/bash
set -e

# Positional arguments
IMAGE_FOLDER="/home/tha/datasets/celebahq_samples"
SIZE=256
targets=(0.1 0.8)
LR=0.0039 # First one is from ACE: 1/255
backward_ts=(5 10 20)
OPTIMIZER=adam
DIST=none

get_stop_at() {
    if [ "$1" == "inf" ]; then
        stop_at=0.8
    elif [ "$1" == "-inf" ]; then
        stop_at=0.1
    else
        stop_at="$1"
    fi
    echo "$stop_at"
}

# Submit to SLURM
echo "Submitting jobs to SLURM..."
for target in "${targets[@]}"; do
    for backward_t in "${backward_ts[@]}"; do
        stop_at=$(get_stop_at "$target")
        echo "sbatch --partition=gpu-5h scripts/run_diffeocf_dae_age.sh $IMAGE_FOLDER $SIZE $target $stop_at $LR $backward_t $DIST $OPTIMIZER"
        sbatch --partition=gpu-5h scripts/run_diffeocf_dae_age.sh $IMAGE_FOLDER $SIZE $target $stop_at $LR $backward_t $DIST $OPTIMIZER
    done
done
