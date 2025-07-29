#!/bin/bash
#SBATCH --job-name=reg_imdb_clean
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/train
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
	echo "Usage: $0 <weights_path> <size> [full_finetune=0] [oracle=0]"
	exit 1
fi

NAME="imdb_wiki_densenet_linear_only"
WEIGHTS_PATH="$1"
SIZE="$2"
if [ "$3" == "1" ]; then
	echo "Running full finetune"
	FULL_FINETUNE="--full_finetune"
	NAME="imdb_wiki_densenet_fullft"
fi

if [ "$4" == "1" ]; then
	echo "Running oracle"
	ORACLE="--oracle"
	NAME="${NAME}_oracle"
fi

apptainer run \
	-B /home/space/datasets:/home/space/datasets \
	-B /home/tha/datasets/squashed/imdb-clean.sqfs:/data/imdb-clean:image-src=/ \
	--nv \
	~/apptainers/thesis.sif \
	python train_imdb_clean.py \
	--folder_path /data/imdb-clean/imdb-clean-1024-cropped \
	--name "$NAME-$SIZE" \
	--densenet_weights "$WEIGHTS_PATH" \
	--image_size "$SIZE" \
	$FULL_FINETUNE $ORACLE
