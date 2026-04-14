#!/bin/bash
set -e

# Positional arguments
IMAGE_FOLDER="/home/tha/datasets/celebahq_samples"
SIZE=256
targets=(0.1 0.8 inf -inf)
lrs=(0.0039 0.01) # First one is from ACE: 1/255
BACKWARD_T=20
dists=(none l1 l2)
# optimizers=(adam sgd)
optimizers=(adam)

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
    for dist in "${dists[@]}"; do
        for lr in "${lrs[@]}"; do
            for optimizer in "${optimizers[@]}"; do
                stop_at=$(get_stop_at "$target")
                echo "sbatch --partition=gpu-5h scripts/run_diffeocf_dae_age.sh $IMAGE_FOLDER $SIZE $target $stop_at $lr $BACKWARD_T $dist $optimizer"
                sbatch --partition=gpu-5h scripts/run_diffeocf_dae_age.sh $IMAGE_FOLDER $SIZE $target $stop_at $lr $BACKWARD_T $dist $optimizer
            done
        done
    done
done
