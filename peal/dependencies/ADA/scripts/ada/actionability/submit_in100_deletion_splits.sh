#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/atlas/run_module_single.sh"
REGION_CONFIG="${REGION_CONFIG:-${REPO}/configs/ada/actionability/in100_regions.yaml}"
DELETION_CONFIG="${DELETION_CONFIG:-${REPO}/configs/ada/actionability/in100_deletion.yaml}"

mkdir -p "${REPO}/logs"

region_args="--config ${REGION_CONFIG}"
region_job="$(
  sbatch --parsable \
    --job-name=ADA_in100_regions \
    --export=ALL,REPO="${REPO}",MODULE=ada.actionability.cli.build_regions,MODULE_ARGS="${region_args}" \
    "${RUNNER}"
)"
echo "regions ${region_job}"

deletion_args="--config ${DELETION_CONFIG}"
deletion_job="$(
  sbatch --parsable \
    --dependency="afterok:${region_job}" \
    --job-name=ADA_in100_deletion_manifests \
    --export=ALL,REPO="${REPO}",MODULE=ada.actionability.cli.build_deletion_splits,MODULE_ARGS="${deletion_args}" \
    "${RUNNER}"
)"
echo "deletion ${deletion_job}"
