#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/actionability/run_real_restoration_probe_single.sh"
DEPENDENCY="${DEPENDENCY:-}"
ARRAY="${ARRAY:-0-719}"

args=(--parsable --array="${ARRAY}" --export=ALL,REPO="${REPO}" "${RUNNER}")
if [[ -n "${DEPENDENCY}" ]]; then
  args=(--parsable --dependency="afterok:${DEPENDENCY}" --array="${ARRAY}" --export=ALL,REPO="${REPO}" "${RUNNER}")
fi

sbatch "${args[@]}"
