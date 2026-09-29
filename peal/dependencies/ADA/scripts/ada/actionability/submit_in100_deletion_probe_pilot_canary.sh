#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/actionability/run_deletion_probe_pilot_single.sh"
DEPENDENCY="${DEPENDENCY:-}"

args=(--parsable --array="${ARRAY:-0-0}" --export=ALL,REPO="${REPO}" "${RUNNER}")
if [[ -n "${DEPENDENCY}" ]]; then
  args=(--parsable --dependency="afterok:${DEPENDENCY}" --array="${ARRAY:-0-0}" --export=ALL,REPO="${REPO}" "${RUNNER}")
fi

sbatch "${args[@]}"
