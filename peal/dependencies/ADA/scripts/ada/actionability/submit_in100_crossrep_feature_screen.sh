#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
FEATURE="${FEATURE:-dinov3_cls}"
CACHE_DEPENDENCY="${CACHE_DEPENDENCY:-}"
DELETION_ARRAY="${DELETION_ARRAY:-0-215}"
RESTORATION_ARRAY="${RESTORATION_ARRAY:-0-719}"

case "${FEATURE}" in
  dinov3_cls)
    DELETION_CONFIG="${REPO}/configs/ada/actionability/in100_crossrep_dinov3_cls_deletion_probe.yaml"
    RESTORATION_CONFIG="${REPO}/configs/ada/actionability/in100_crossrep_dinov3_cls_restoration_probe.yaml"
    ;;
  siglip2_pooler)
    DELETION_CONFIG="${REPO}/configs/ada/actionability/in100_crossrep_siglip2_pooler_deletion_probe.yaml"
    RESTORATION_CONFIG="${REPO}/configs/ada/actionability/in100_crossrep_siglip2_pooler_restoration_probe.yaml"
    ;;
  *)
    echo "Unknown FEATURE=${FEATURE}. Use dinov3_cls or siglip2_pooler." >&2
    exit 2
    ;;
esac

DELETION_RUNNER="${REPO}/scripts/ada/actionability/run_deletion_probe_pilot_single.sh"
RESTORATION_RUNNER="${REPO}/scripts/ada/actionability/run_real_restoration_probe_single.sh"
DELETION_EVAL_RUNNER="${REPO}/scripts/ada/actionability/run_deletion_probe_eval_cpu.sh"
RESTORATION_EVAL_RUNNER="${REPO}/scripts/ada/actionability/run_real_restoration_probe_eval_cpu.sh"

DEPENDENCY_ARGS=()
if [[ -n "${CACHE_DEPENDENCY}" ]]; then
  DEPENDENCY_ARGS=(--dependency="afterok:${CACHE_DEPENDENCY}")
fi

DELETION_JOB=$(
  sbatch --parsable "${DEPENDENCY_ARGS[@]}" --array="${DELETION_ARRAY}" \
    --export=ALL,REPO="${REPO}",CONFIG="${DELETION_CONFIG}" \
    "${DELETION_RUNNER}"
)
RESTORATION_JOB=$(
  sbatch --parsable "${DEPENDENCY_ARGS[@]}" --array="${RESTORATION_ARRAY}" \
    --export=ALL,REPO="${REPO}",CONFIG="${RESTORATION_CONFIG}" \
    "${RESTORATION_RUNNER}"
)
DELETION_EVAL_JOB=$(
  sbatch --parsable --dependency="afterok:${DELETION_JOB}" \
    --export=ALL,REPO="${REPO}",CONFIG="${DELETION_CONFIG}" \
    "${DELETION_EVAL_RUNNER}"
)
RESTORATION_EVAL_JOB=$(
  sbatch --parsable --dependency="afterok:${DELETION_JOB}:${RESTORATION_JOB}" \
    --export=ALL,REPO="${REPO}",CONFIG="${RESTORATION_CONFIG}" \
    "${RESTORATION_EVAL_RUNNER}"
)

echo "feature=${FEATURE}"
echo "deletion_job=${DELETION_JOB}"
echo "restoration_job=${RESTORATION_JOB}"
echo "deletion_eval_job=${DELETION_EVAL_JOB}"
echo "restoration_eval_job=${RESTORATION_EVAL_JOB}"
