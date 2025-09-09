#!/bin/bash
export DCFIR_HOME="$PWD"
export DCFIR_OUTPATH="$PWD/diff_cf_ir_results"
set -e
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi
set -x

DATA_PATH=$DCFIR_OUTPATH/datasets/square
OUT_PATH_ACE=$DCFIR_OUTPATH/models/square_ace_ddpm
OUT_PATH_DIFFAE=$DCFIR_OUTPATH/models/square_diffae/square64_ddim

if [ ! -d "$DCFIR_OUTPATH/models/square_diffae/square64_ddim/square64_ddim/last.ckpt" ]; then
    echo "Train Square DiffAE..."
    # Train square DiffAE - DDIM
    python run_square64_ddim.py $OUT_PATH_DIFFAE
    # Results in $DCFIR_OUTPATH/models/square_diffae/square64_ddim/last.ckpt

    # Train square DiffAE - DDIM
    python run_square64_latent.py $OUT_PATH_DIFFAE
    # Results in $DCFIR_OUTPATH/models/square_diffae/square64_ddim/square64_autoenc_latent/last.ckpt

else
    echo "Square DiffAE already exists. Skipping training."
fi
