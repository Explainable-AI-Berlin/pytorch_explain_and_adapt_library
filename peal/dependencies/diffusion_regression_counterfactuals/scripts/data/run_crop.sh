#!/bin/bash
#SBATCH --job-name=crop_imdbwikiclean
#SBATCH --partition=cpu-2d
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/diff_cf_ir

apptainer run -B /home/space/datasets:/home/space/datasets ~/apptainers/thesis.sif python imdb_clean_dataset.py /home/space/datasets/imdb-wiki-clean/imdb-clean /home/space/datasets/imdb-wiki-clean/imdb-clean/data/imdb-clean-1024-cropped
