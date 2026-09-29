#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/interpretability/run_siglip2_shared_assets_cpu.sh"
MODEL_ID="${MODEL_ID:-google/siglip2-base-patch16-256}"
REVISION="${REVISION:-3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/external/models/google__siglip2-base-patch16-256}"

sbatch --parsable --export=ALL,REPO="${REPO}",MODEL_ID="${MODEL_ID}",REVISION="${REVISION}",OUTPUT_ROOT="${OUTPUT_ROOT}" "${RUNNER}"
