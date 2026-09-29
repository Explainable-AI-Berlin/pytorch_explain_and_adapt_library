#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUNNER="${REPO}/scripts/ada/actionability/run_in100_crossrep_cache_subsets_cpu.sh"

sbatch --parsable --export=ALL,REPO="${REPO}",OVERWRITE="${OVERWRITE:-0}" "${RUNNER}"
