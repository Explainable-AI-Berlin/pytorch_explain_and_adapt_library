#!/bin/bash
#SBATCH --job-name=reg_imdb_test
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/
set -x

if [ -z "$1" ] || [ -z "$2" ]; then
	echo "Usage: $0 <size> <resnet_type>"
	exit 1
fi

SIZE="$1"
RESNET_TYPE="$2"
LR=0.00001
NAME="imdb-test-$RESNET_TYPE-lr=$LR"

apptainer run \
	-B /home/space/datasets:/home/space/datasets \
	-B /home/tha/datasets/squashed/imdb-clean.sqfs:/data/imdb-clean:image-src=/ \
	--nv \
	~/apptainers/thesis.sif \
	python test/train_resnet_imdb_clean_test.py \
	--folder_path /data/imdb-clean/imdb-clean-1024-cropped \
	--name "$NAME-$SIZE" \
	--image_size "$SIZE" \
	--learning_rate $LR \
	--resnet_type "$RESNET_TYPE"
