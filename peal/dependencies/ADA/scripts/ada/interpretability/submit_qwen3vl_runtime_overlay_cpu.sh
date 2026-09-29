#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"

sbatch --parsable "${REPO}/scripts/ada/interpretability/run_qwen3vl_runtime_overlay_cpu.sh"
