#!/bin/bash
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi

IMGS_FOLDER="$DCFIR_OUTPATH/datasets/square_val/squares_upper/imgs"
FAKE_FOLDER_ACE="$DCFIR_OUTPATH/ac-re/square_mirror_upper/cf"
FAKE_FOLDER_DAE="$DCFIR_OUTPATH/diffae-re/square_mirror_upper/cf"
OUTPUT_FOLDER="$DCFIR_OUTPATH/metrics/square/square_mirror_upper"

echo "Running reference metrics for $IMGS_FOLDER"
python $DCFIR_HOME/scripts/metrics/run_metrics_square.py \
    --real_folder="$IMGS_FOLDER" \
    --fake_folder $FAKE_FOLDER_ACE \
    --fake_folder $FAKE_FOLDER_DAE \
    $OUTPUT_FOLDER

IMGS_FOLDER="$DCFIR_OUTPATH/datasets/square_val/squares_lower/imgs"
FAKE_FOLDER_ACE="$DCFIR_OUTPATH/ac-re/square_mirror_lower/cf"
FAKE_FOLDER_DAE="$DCFIR_OUTPATH/diffae-re/square_mirror_lower/cf"
OUTPUT_FOLDER="$DCFIR_OUTPATH/metrics/square/square_mirror_lower"

echo "Running reference metrics for $IMGS_FOLDER"
python $DCFIR_HOME/scripts/metrics/run_metrics_square.py \
    --real_folder="$IMGS_FOLDER" \
    --fake_folder $FAKE_FOLDER_ACE \
    --fake_folder $FAKE_FOLDER_DAE \
    $OUTPUT_FOLDER
