#!/bin/bash
set -e
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi

ACRE_CELEBAHQ_RESULTS="$DCFIR_OUTPATH/ac-re/celebahq"
DIFFAERE_CELEBAHQ_RESULTS="$DCFIR_OUTPATH/diffae-re/celebahq"

echo "Running metrics for Squares"
bash $DCFIR_HOME/scripts/metrics/scripts/run_metrics_squares_mirror.sh
# Result in $DCFIR_OUTPATH/metrics/square

echo "Running metrics CelebaHQ..."
bash $DCFIR_HOME/scripts/metrics/scripts/run_folders.sh $ACRE_CELEBAHQ_RESULTS
bash scripts/metrics/scripts/run_folders.sh $DIFFAERE_CELEBAHQ_RESULTS
# Result in $DCFIR_OUTPATH/metrics/celebahq
