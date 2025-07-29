#!/bin/bash
set -e
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi

# ---------------- AC-RE ----------------
echo "Running AC-RE experiments..."
cd $DCFIR_HOME/related_work/ACE || exit

echo "Running Square experiments..."
bash related_work/ACE/scripts/regression/run_square_mirror.sh

echo "Running CelebAHQ experiments..."
bash $DCFIR_HOME/related_work/ACE/scripts/regression/run_age_celebahq_allval.sh

echo "Running fine-grained reference values..."
bash $DCFIR_HOME/related_work/ACE/scripts/regression/run_age_imdbwiki_multitarget.sh

# ---------------- DIFF-AE-RE ----------------
echo "Running Diff-AE-RE experiments..."
cd $DCFIR_HOME || exit
echo "Running Square experiments..."
bash $DCFIR_HOME/scripts/cf/squares/scripts/run_diffeocf_dae_square_mirror.sh

echo "Running CelebAHQ experiments..."
bash $DCFIR_HOME/scripts/cf/celebahq/scripts/run_diffeocf_dae_celebahq.sh

echo "Running fine-grained reference values..."
bash $DCFIR_HOME/scripts/cf/imdb-wiki-clean/scripts/run_multitarget.sh
