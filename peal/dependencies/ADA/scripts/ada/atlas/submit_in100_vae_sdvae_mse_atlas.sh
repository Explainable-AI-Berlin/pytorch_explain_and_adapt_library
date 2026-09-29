#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/atlas/run_in100_vae_sdvae_mse_atlas.sh"
DEPENDENCY="${DEPENDENCY:-}"

args=(--parsable --export=ALL,REPO="${REPO}" "${RUNNER}")
if [[ -n "${DEPENDENCY}" ]]; then
  args=(--parsable --dependency="afterok:${DEPENDENCY}" --export=ALL,REPO="${REPO}" "${RUNNER}")
fi

sbatch "${args[@]}"
