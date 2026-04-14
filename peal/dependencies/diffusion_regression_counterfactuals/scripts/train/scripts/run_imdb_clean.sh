#!/bin/bash
#SBATCH --job-name=reg_imdb_clean
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/train
set -x

if [ -z "$1" ] || [ -z "$2" ]; then
	echo "Usage: $0 <name> <size> [oracle]"
	exit 1
fi

NAME="$1"
SIZE="$2"
if [ -n "$3" ]; then
	echo "Running for oracle"
	ORACLE="--oracle"
	NAME="${NAME}_oracle"
fi

apptainer run \
	-B /home/space/datasets:/home/space/datasets \
	-B /home/tha/datasets/squashed/imdb-clean.sqfs:/data/imdb-clean:image-src=/ \
	--nv \
	~/apptainers/thesis.sif \
	python train_resnet_imdb_clean.py \
	--folder_path /data/imdb-clean/imdb-clean-1024-cropped \
	--name "$NAME-$SIZE" \
	--image_size "$SIZE" \
	$ORACLE
