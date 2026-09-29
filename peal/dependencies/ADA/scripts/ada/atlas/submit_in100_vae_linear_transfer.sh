#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"

sbatch --parsable \
  --export=ALL,REPO="${REPO}" \
  "${REPO}/scripts/ada/atlas/run_in100_vae_linear_transfer.sh"
