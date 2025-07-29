#!/bin/bash
set -e
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi
set -x

DATASETS_PATH=$DCFIR_OUTPATH/datasets
mkdir -p $DATASETS_PATH

# Download the imdb-wiki-clean dataset
echo "Downloading the imdb-wiki-clean dataset..."
cd $DATASETS_PATH || exit
git clone git@github.com:yiminglin-ai/imdb-clean.git
cd imdb-clean || exit
bash run_all.sh
# Create the crop
cd $DCFIR_HOME || exit
python diff_cf_ir/imdb_clean_dataset.py $DATASETS_PATH/imdb-clean/imdb-clean-1024 $DATASETS_PATH/imdb-clean/imdb-clean-1024-cropped

# Download the CelebA-HQ dataset
echo "Downloading the CelebA-HQ dataset..."
cd $DATASETS_PATH || exit
# From https://github.com/switchablenorms/CelebAMask-HQ
gdown 1badu11NqxGf6qM3PTTooQDJvQbejgbTv
unzip CelebAMask-HQ.zip
cd CelebAMask-HQ/ || exit
# download CelebA attributes
gdown 0B7EVK8r0v71pblRyaVFSWGxPY0U

# Generate the square dataset
echo "Generating the square dataset..."
cd $DCFIR_HOME || exit
python $DCFIR_HOME/diff_cf_ir/generate_squares.py $DATASETS_PATH/square
python $DCFIR_HOME/diff_cf_ir/generate_squares.py $DATASETS_PATH/square_val --split val
