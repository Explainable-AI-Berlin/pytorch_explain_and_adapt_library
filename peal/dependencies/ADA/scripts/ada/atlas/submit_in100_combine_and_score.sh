#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/atlas/run_module_single.sh"
ENC_ROOT="${REPO}/artifacts/ada/atlas/embeddings/imagenet100"
TRAIN_ROOT="${ENC_ROOT}/train/facebook__dinov2-with-registers-base/cls"
VAL_CACHE="${ENC_ROOT}/val/facebook__dinov2-with-registers-base/cls/cache-6ec1e8a1b614f414"
TRAIN_COMBINED="${TRAIN_ROOT}/combined_train_full_4shards"
SCORES_DIR="${REPO}/artifacts/ada/atlas/scores/e0_in100_dinov2_val_vs_train"
SCORES_CSV="${SCORES_DIR}/support_scores.csv"

SHARD0="${TRAIN_ROOT}/cache-f0aeaa9a2d12d0d1"
SHARD1="${TRAIN_ROOT}/cache-a7f3aed369506b13"
SHARD2="${TRAIN_ROOT}/cache-3dbe1872c5c2ecbf"
SHARD3="${TRAIN_ROOT}/cache-0c59f42b3cb05ea9"

mkdir -p "${SCORES_DIR}" "${REPO}/logs"

combine_args="--output-dir ${TRAIN_COMBINED} --input-dir ${SHARD0} --input-dir ${SHARD1} --input-dir ${SHARD2} --input-dir ${SHARD3} --overwrite"
combine_job="$(
  sbatch --parsable \
    --job-name=ADA_atlas_in100_combine \
    --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.combine_caches,MODULE_ARGS="${combine_args}" \
    "${RUNNER}"
)"
echo "combine ${combine_job}"

score_args="--query-cache ${VAL_CACHE} --reference-cache ${TRAIN_COMBINED} --output-csv ${SCORES_CSV} --k 5:10:50 --batch-size 256 --class-conditional"
score_job="$(
  sbatch --parsable \
    --dependency="afterok:${combine_job}" \
    --job-name=ADA_atlas_in100_score \
    --export=ALL,REPO="${REPO}",MODULE=ada.atlas.cli.score_knn,MODULE_ARGS="${score_args}" \
    "${RUNNER}"
)"
echo "score ${score_job}"
