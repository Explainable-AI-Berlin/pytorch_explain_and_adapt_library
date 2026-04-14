#!/bin/bash
#SBATCH --job-name=thesis_gpu_test
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out
set -x

cd /home/tha/master-thesis-xai/diff_cf_ir/scripts/train

apptainer run -B /home/space/datasets:/home/space/datasets \
	-B /home/tha/datasets/squashed/imdb-clean.sqfs:/data:image-src=/ \
	--nv \
	python train_resnet_imdb_clean.py --folder_path /data/imdb-clean-1024-cropped/ --name imdb_clean_256 --image_size 256

