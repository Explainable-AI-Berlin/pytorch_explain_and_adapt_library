#!/bin/sh
#SBATCH --job-name=dae_square_latent
#SBATCH --partition=gpu-2d
#SBATCH --constraint=80gb
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/diffae

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

apptainer run \
    -B /home/tha/datasets/squashed/square.sqfs:/data/square:image-src=/ \
    --nv /home/tha/apptainers/thesis.sif \
    python run_square64_latent.py
