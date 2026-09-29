#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
DEPENDENCY="${DEPENDENCY:-}"
SNAPSHOT="${SNAPSHOT:-}"
OUT_DIR="${OUT_DIR:-}"
QWEN_RUNTIME_OVERLAY="${QWEN_RUNTIME_OVERLAY:-}"
RUNNER="${REPO}/scripts/ada/interpretability/run_qwen3vl_preflight_gpu.sh"

args=()
if [[ -n "${DEPENDENCY}" ]]; then
  args+=(--dependency="${DEPENDENCY}")
fi

export_vars="ALL,REPO=${REPO}"
if [[ -n "${SNAPSHOT}" ]]; then
  export_vars+=",SNAPSHOT=${SNAPSHOT}"
fi
if [[ -n "${OUT_DIR}" ]]; then
  export_vars+=",OUT_DIR=${OUT_DIR}"
fi
if [[ -n "${QWEN_RUNTIME_OVERLAY}" ]]; then
  export_vars+=",QWEN_RUNTIME_OVERLAY=${QWEN_RUNTIME_OVERLAY}"
fi

sbatch --parsable "${args[@]}" --export="${export_vars}" "${RUNNER}"
