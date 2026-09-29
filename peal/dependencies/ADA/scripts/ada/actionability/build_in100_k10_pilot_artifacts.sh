#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
ENRICH_CONFIG="${ENRICH_CONFIG:-${REPO}/configs/ada/actionability/in100_region_enrichment.yaml}"
SELECT_CONFIG="${SELECT_CONFIG:-${REPO}/configs/ada/actionability/in100_pilot_selection.yaml}"
DELETE_CONFIG="${DELETE_CONFIG:-${REPO}/configs/ada/actionability/in100_deletion_pilot.yaml}"

cd "${REPO}"
export PYTHONPATH=src
python3 -m ada.actionability.cli.enrich_regions --config "${ENRICH_CONFIG}" "$@"
python3 -m ada.actionability.cli.select_pilot_regions --config "${SELECT_CONFIG}"
python3 -m ada.actionability.cli.build_deletion_controls --config "${DELETE_CONFIG}"
