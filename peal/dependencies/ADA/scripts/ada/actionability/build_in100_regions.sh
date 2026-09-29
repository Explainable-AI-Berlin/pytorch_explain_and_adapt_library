#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${REPO}/configs/ada/actionability/in100_regions.yaml}"

cd "${REPO}"
export PYTHONPATH=src
python3 -m ada.actionability.cli.build_regions --config "${CONFIG}" "$@"
