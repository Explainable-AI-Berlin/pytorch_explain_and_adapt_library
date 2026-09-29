#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
REGION_RUNNER="${REPO}/scripts/ada/actionability/run_in1k_region_build_cpu.sh"
ARTIFACT_RUNNER="${REPO}/scripts/ada/actionability/run_in1k_k10_confirmatory_artifacts_cpu.sh"

REGION_JOB="$(
  sbatch --parsable \
    --export=ALL,REPO="${REPO}" \
    "${REGION_RUNNER}"
)"
ARTIFACT_JOB="$(
  sbatch --parsable \
    --dependency="afterok:${REGION_JOB}" \
    --export=ALL,REPO="${REPO}" \
    "${ARTIFACT_RUNNER}"
)"

echo "region_job=${REGION_JOB}"
echo "artifact_job=${ARTIFACT_JOB}"
