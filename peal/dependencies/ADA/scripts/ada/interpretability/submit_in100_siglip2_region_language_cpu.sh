#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/interpretability/run_in100_siglip2_region_language_cpu.sh"

sbatch --parsable --export=ALL,REPO="${REPO}" "${RUNNER}"
