#!/usr/bin/env bash
set -euo pipefail

ADA_REPO="${ADA_REPO:-$(git rev-parse --show-toplevel)}"
VARIANT="${1:-cls_only}"
shift || true

mkdir -p "${ADA_REPO}/logs"
sbatch --parsable \
  --job-name="ada_sample_${VARIANT}" \
  --export=ALL,ADA_REPO="${ADA_REPO}",VARIANT="${VARIANT}" \
  "${ADA_REPO}/scripts/generator/run_raev2_sample.sh" "$@"
