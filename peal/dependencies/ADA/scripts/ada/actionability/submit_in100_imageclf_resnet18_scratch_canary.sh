#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/actionability/run_image_classifier_probe_single.sh"
CONFIG="${CONFIG:-${REPO}/configs/ada/actionability/in100_imageclf_resnet18_scratch_deletion_probe.yaml}"

sbatch --parsable --array="${ARRAY:-0,3}" \
  --export=ALL,REPO="${REPO}",CONFIG="${CONFIG}" \
  "${RUNNER}"
