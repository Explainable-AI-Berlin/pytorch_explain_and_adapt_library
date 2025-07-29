#!/bin/bash
set -e

export DCFIR_HOME="$PWD"
export DCFIR_OUTPATH="$PWD/diff_cf_ir_results"
source setup_pythonpath.sh

bash 2_get_datasets.sh
bash 3_train_square_generators.sh
bash 4_train_regressors.sh
bash 5_produce_results.sh
bash 6_metrics.sh
