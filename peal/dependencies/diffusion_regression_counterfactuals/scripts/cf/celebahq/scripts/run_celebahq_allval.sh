#!/bin/bash
#SBATCH --job-name=dcf_celebahq_allval
#SBATCH --partition=gpu-7d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/cf/celebahq
#SBATCH --signal=SIGUSR1@600

bash scripts/run_diffeocf_dae_celebahq.sh \
    0.002 \
    10 \
    l1=1e-5 \
    latent \
    adam \
    /home/tha/thesis_runs/regressor/imdb_wiki_densenet_linear_only-256/version_0/checkpoints/last.ckpt \
    /home/tha/thesis_runs/regressor/imdb_wiki_densenet_fullft-256/version_0/checkpoints/last.ckpt \
    /home/tha/thesis_runs/dae/celebahq_allval