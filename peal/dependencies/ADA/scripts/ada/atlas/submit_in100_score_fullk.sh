#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/atlas/run_module_single.sh"
ENC_ROOT="${REPO}/artifacts/ada/atlas/embeddings/imagenet100"
VAL_CACHE="${ENC_ROOT}/val/facebook__dinov2-with-registers-base/cls/cache-6ec1e8a1b614f414"
TRAIN_COMBINED="${ENC_ROOT}/train/facebook__dinov2-with-registers-base/cls/combined_train_full_4shards"
SCORES_DIR="${REPO}/artifacts/ada/atlas/scores/e0_in100_dinov2_val_vs_train"
SCORES_CSV="${SCORES_DIR}/support_scores.csv"

mkdir -p "${SCORES_DIR}" "${REPO}/logs"

score_args="--query-cache ${VAL_CACHE} --reference-cache ${TRAIN_COMBINED} --output-csv ${SCORES_CSV} --k 5:10:50 --batch-size 256 --class-conditional"
score_job="$(
  sbatch --parsable \
    --job-name=ADA_atlas_in100_score \
    --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.score_knn,MODULE_ARGS="${score_args}" \
    "${RUNNER}"
)"
echo "score ${score_job}"
