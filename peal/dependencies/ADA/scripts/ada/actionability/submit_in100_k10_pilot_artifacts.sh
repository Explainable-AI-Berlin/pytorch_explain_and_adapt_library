#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/actionability/run_in100_k10_pilot_artifacts_cpu.sh"

job_id="$(
  sbatch --parsable \
    --export=ALL,REPO="${REPO}" \
    "${RUNNER}"
)"

echo "${job_id}"
