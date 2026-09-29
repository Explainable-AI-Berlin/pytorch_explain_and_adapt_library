#!/bin/bash
#SBATCH --job-name=ADA_qwen_preflight
#SBATCH --partition=gpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
SNAPSHOT="${SNAPSHOT:-${REPO}/external/models/Qwen__Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}"
QWEN_RUNTIME_OVERLAY="${QWEN_RUNTIME_OVERLAY:-${REPO}/external/qwen3vl_runtime_4_57_2}"
OUT_DIR="${OUT_DIR:-${REPO}/artifacts/ada/interpretability/qwen3vl_runtime_preflight}"

mkdir -p "${REPO}/logs" "${OUT_DIR}"
cd "${REPO}"

apptainer exec --nv \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export PYTHONPATH='${QWEN_RUNTIME_OVERLAY}:src'
    export PYTHONDONTWRITEBYTECODE=1
    python3 scripts/ada/interpretability/qwen3vl_runtime_preflight.py \
      --snapshot '${SNAPSHOT}' \
      --output-dir '${OUT_DIR}'
  "
