#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${REPO}/configs/ada/interpretability/in100_qwen3vl_region_descriptions_4b_canary.yaml}"
RUN_INFERENCE="${RUN_INFERENCE:-0}"
DEPENDENCY="${DEPENDENCY:-}"
RUNNER="${REPO}/scripts/ada/interpretability/run_qwen3vl_region_descriptions_gpu.sh"

sbatch_args=(--parsable)
if [[ -n "${DEPENDENCY}" ]]; then
  sbatch_args+=(--dependency="${DEPENDENCY}")
fi

sbatch "${sbatch_args[@]}" --export=ALL,REPO="${REPO}",CONFIG="${CONFIG}",RUN_INFERENCE="${RUN_INFERENCE}" "${RUNNER}"
