#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
DELETION_CONFIG="${DELETION_CONFIG:?Set DELETION_CONFIG to a deletion-probe config}"
RESTORATION_CONFIG="${RESTORATION_CONFIG:-}"
DELETION_ARRAY="${DELETION_ARRAY:-0-215}"
RESTORATION_ARRAY="${RESTORATION_ARRAY:-0-719}"
DEPENDENCY="${DEPENDENCY:-}"

DELETION_RUNNER="${REPO}/scripts/ada/actionability/run_deletion_probe_pilot_single.sh"
RESTORATION_RUNNER="${REPO}/scripts/ada/actionability/run_real_restoration_probe_single.sh"
DELETION_EVAL_RUNNER="${REPO}/scripts/ada/actionability/run_deletion_probe_eval_cpu.sh"
RESTORATION_EVAL_RUNNER="${REPO}/scripts/ada/actionability/run_real_restoration_probe_eval_cpu.sh"

DEPENDENCY_ARGS=()
if [[ -n "${DEPENDENCY}" ]]; then
  DEPENDENCY_ARGS=(--dependency="afterok:${DEPENDENCY}")
fi

DELETION_JOB="$(
  sbatch --parsable "${DEPENDENCY_ARGS[@]}" --array="${DELETION_ARRAY}" \
    --export=ALL,REPO="${REPO}",CONFIG="${DELETION_CONFIG}" \
    "${DELETION_RUNNER}"
)"
DELETION_EVAL_JOB="$(
  sbatch --parsable --dependency="afterok:${DELETION_JOB}" \
    --export=ALL,REPO="${REPO}",CONFIG="${DELETION_CONFIG}" \
    "${DELETION_EVAL_RUNNER}"
)"

echo "deletion_job=${DELETION_JOB}"
echo "deletion_eval_job=${DELETION_EVAL_JOB}"

if [[ -n "${RESTORATION_CONFIG}" ]]; then
  RESTORATION_JOB="$(
    sbatch --parsable "${DEPENDENCY_ARGS[@]}" --array="${RESTORATION_ARRAY}" \
      --export=ALL,REPO="${REPO}",CONFIG="${RESTORATION_CONFIG}" \
      "${RESTORATION_RUNNER}"
  )"
  RESTORATION_EVAL_JOB="$(
    sbatch --parsable --dependency="afterok:${DELETION_JOB}:${RESTORATION_JOB}" \
      --export=ALL,REPO="${REPO}",CONFIG="${RESTORATION_CONFIG}" \
      "${RESTORATION_EVAL_RUNNER}"
  )"
  echo "restoration_job=${RESTORATION_JOB}"
  echo "restoration_eval_job=${RESTORATION_EVAL_JOB}"
fi
