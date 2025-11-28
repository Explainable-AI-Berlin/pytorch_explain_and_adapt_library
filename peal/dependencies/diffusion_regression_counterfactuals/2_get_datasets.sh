#!/bin/bash
export DCFIR_HOME="$PWD"
export DCFIR_OUTPATH="$PEAL_RUNS/square/diff_cf_ir_results"
set -e

if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi

set -x

DATASETS_PATH="$DCFIR_OUTPATH/datasets"
mkdir -p "$DATASETS_PATH"



###############################################
# Generate the square datasets
###############################################
cd "$DCFIR_HOME" || exit

if [ ! -d "$DATASETS_PATH/square" ]; then
    echo "Generating the square dataset..."
    python generate_squares.py "$DATASETS_PATH/square"
else
    echo "Square dataset already exists. Skipping generation."
fi

if [ ! -d "$DATASETS_PATH/square_val" ]; then
    echo "Generating the square_val dataset..."
    python generate_squares.py "$DATASETS_PATH/square_val" --split val
else
    echo "Square_val dataset already exists. Skipping generation."
fi