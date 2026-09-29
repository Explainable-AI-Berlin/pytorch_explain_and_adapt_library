#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
REVISION="${REVISION:-ebb281ec70b05090aa6165b016eac8ec08e71b17}"
RUNNER="${REPO}/scripts/ada/interpretability/run_qwen3vl_assets_cpu.sh"

sbatch --parsable --export=ALL,REPO="${REPO}",MODEL_ID="${MODEL_ID}",REVISION="${REVISION}" "${RUNNER}"
