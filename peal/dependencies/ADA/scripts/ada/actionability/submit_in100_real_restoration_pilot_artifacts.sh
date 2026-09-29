#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/actionability/run_in100_real_restoration_pilot_artifacts_cpu.sh"
DEPENDENCY="${DEPENDENCY:-}"

args=(--parsable --export=ALL,REPO="${REPO}" "${RUNNER}")
if [[ -n "${DEPENDENCY}" ]]; then
  args=(--parsable --dependency="afterok:${DEPENDENCY}" --export=ALL,REPO="${REPO}" "${RUNNER}")
fi

sbatch "${args[@]}"
