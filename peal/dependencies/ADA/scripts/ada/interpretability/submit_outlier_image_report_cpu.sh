#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${REPO}/configs/ada/interpretability/in100_outlier_image_descriptions.yaml}"
RUNNER="${REPO}/scripts/ada/interpretability/run_outlier_image_report_cpu.sh"

sbatch --parsable --export=ALL,REPO="${REPO}",CONFIG="${CONFIG}" "${RUNNER}"
