#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/interpretability/run_interpretability_tests_cpu.sh"

sbatch --parsable --export=ALL,REPO="${REPO}" "${RUNNER}"
