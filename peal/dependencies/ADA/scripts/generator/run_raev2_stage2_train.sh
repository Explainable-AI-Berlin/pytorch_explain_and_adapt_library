#!/usr/bin/env bash
#SBATCH --job-name=ada_raev2_train
#SBATCH --partition=gpu-7d
#SBATCH --constraint=40gb
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00
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
ADA_CONTAINER="${ADA_CONTAINER:?Set ADA_CONTAINER to the RAE Apptainer image}"
CONFIG="${CONFIG:?Set CONFIG to a Stage-2 YAML file}"
RESULTS_DIR="${RESULTS_DIR:-${ADA_CHECKPOINT_ROOT:?Set ADA_CHECKPOINT_ROOT}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME}"
RAEV2_DEPS_DIR="${RAEV2_DEPS_DIR:-${RAEV2_DIR}/.deps/gmuon}"
PRECISION="${PRECISION:-bf16}"
NPROC="${NPROC:-4}"
COMPILE="${COMPILE:-0}"

mkdir -p "${ADA_REPO}/logs" "${RESULTS_DIR}"
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
[[ -f "${RAEV2_DIR}/src/train.py" ]] || {
  echo "RAEv2 integration is not applied. Run integrations/raev2/apply_overlay.sh." >&2
  exit 2
}

COMPILE_ARG=()
case "${COMPILE}" in
  1|true|TRUE|yes|YES|on|ON) COMPILE_ARG+=(--compile) ;;
esac

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
export ADA_STAGE2_CACHE
export EXPERIMENT_NAME

apptainer exec --nv \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${ADA_CONTAINER}" \
  /bin/bash -lc "cd '${RAEV2_DIR}' && PYTHONPATH='${RAEV2_DEPS_DIR}:src' torchrun --standalone --nproc_per_node='${NPROC}' src/train.py --config '${CONFIG}' --results-dir '${RESULTS_DIR}' --precision '${PRECISION}' ${COMPILE_ARG[*]}"
