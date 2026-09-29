#!/usr/bin/env bash
#SBATCH --job-name=ada_export_models
#SBATCH --partition=cpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ADA_REPO="${ADA_REPO:-$(git rev-parse --show-toplevel)}"
if [[ -f "${ADA_REPO}/.env" ]]; then
  set -a
  source "${ADA_REPO}/.env"
  set +a
fi

RAEV2_DIR="${RAEV2_DIR:-${ADA_REPO}/third_party/RAEv2}"
ADA_CONTAINER="${ADA_CONTAINER:?Set ADA_CONTAINER}"
ADA_CHECKPOINT_ROOT="${ADA_CHECKPOINT_ROOT:?Set ADA_CHECKPOINT_ROOT}"
ADA_MODEL_BUNDLE_ROOT="${ADA_MODEL_BUNDLE_ROOT:?Set ADA_MODEL_BUNDLE_ROOT}"
SOURCE_REPO_COMMIT="${SOURCE_REPO_COMMIT:-$(git -C "${ADA_REPO}" rev-parse HEAD)}"
ADA_CONFIG_ROOT="${ADA_CONFIG_ROOT:-${RAEV2_DIR}/configs/stage2/training}"

mkdir -p "${ADA_REPO}/logs" "${ADA_MODEL_BUNDLE_ROOT}"
apptainer exec \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${ADA_CONTAINER}" \
  /bin/bash -lc "cd '${ADA_REPO}' && python3 scripts/generator/export_raev2_model_bundle.py --checkpoint-root '${ADA_CHECKPOINT_ROOT}' --config-root '${ADA_CONFIG_ROOT}' --output-root '${ADA_MODEL_BUNDLE_ROOT}' --source-repo-commit '${SOURCE_REPO_COMMIT}' --overwrite"
