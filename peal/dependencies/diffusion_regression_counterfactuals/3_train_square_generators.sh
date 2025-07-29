#!/bin/bash
set -e
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi
set -x

DATA_PATH=$DCFIR_OUTPATH/datasets/square
OUT_PATH_ACE=$DCFIR_OUTPATH/models/square_ace_ddpm
OUT_PATH_DIFFAE=$DCFIR_OUTPATH/models/square_diffae/square64_ddim

# Train square DDPM for ACE
python related_work/ACE/square-train-diffusion.py $DATA_PATH $OUT_PATH_ACE
# checkpoint in $DCFIR_OUTPATH/models/square_ace_ddpm/last.pt

# Train square DiffAE - DDIM
python related_work/diffae/run_square64_ddim.py $OUT_PATH_DIFFAE
# Results in $DCFIR_OUTPATH/models/square_diffae/square64_ddim/last.ckpt

# Train square DiffAE - DDIM
python related_work/diffae/run_square64_latent.py $OUT_PATH_DIFFAE
# Results in $DCFIR_OUTPATH/models/square_diffae/square64_ddim/square64_autoenc_latent/last.ckpt