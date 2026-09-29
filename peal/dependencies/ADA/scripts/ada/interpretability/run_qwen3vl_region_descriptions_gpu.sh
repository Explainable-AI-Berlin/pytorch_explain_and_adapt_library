#!/bin/bash
#SBATCH --job-name=ADA_qwen_region
#SBATCH --partition=gpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
CONFIG="${CONFIG:-${REPO}/configs/ada/interpretability/in100_qwen3vl_region_descriptions_4b_canary.yaml}"
RUN_INFERENCE="${RUN_INFERENCE:-0}"
QWEN_RUNTIME_OVERLAY="${QWEN_RUNTIME_OVERLAY:-${REPO}/external/qwen3vl_runtime_4_57_2}"

cd "${REPO}"
echo "============================================================"
echo "[JOB] HOST=$(hostname)"
echo "[JOB] DATE=$(date)"
echo "[JOB] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[JOB] CONFIG=${CONFIG}"
echo "[JOB] RUN_INFERENCE=${RUN_INFERENCE}"
echo "[JOB] QWEN_RUNTIME_OVERLAY=${QWEN_RUNTIME_OVERLAY}"
echo "============================================================"

extra_args=()
if [[ "${RUN_INFERENCE}" == "1" ]]; then
  extra_args+=(--run-inference)
fi

apptainer exec --nv \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    if [[ -d '${QWEN_RUNTIME_OVERLAY}' ]]; then
      export PYTHONPATH='${QWEN_RUNTIME_OVERLAY}:src'
    else
      export PYTHONPATH=src
    fi
    export PYTHONDONTWRITEBYTECODE=1
    python3 - <<'PY'
import sys
try:
    import transformers
    print('[qwen-runtime] transformers_version=' + getattr(transformers, '__version__', ''))
    print('[qwen-runtime] transformers_path=' + getattr(transformers, '__file__', ''))
    print('[qwen-runtime] sys_path0=' + sys.path[0])
except Exception as exc:
    print('[qwen-runtime] import_error=' + repr(exc))
PY
    python3 -m ada.interpretability.cli.build_qwen_region_descriptions --config '${CONFIG}' --overwrite ${extra_args[*]:-}
  "

echo "[DONE] DATE=$(date)"
