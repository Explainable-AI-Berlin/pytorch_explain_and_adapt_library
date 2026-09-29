#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"

sbatch \
  "${REPO}/scripts/ada/actionability/run_in1k_k10_confirmatory_exclude_in100_artifacts_cpu.sh"
