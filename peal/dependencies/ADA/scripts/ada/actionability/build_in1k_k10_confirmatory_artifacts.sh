#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
ENRICH_CONFIG="${ENRICH_CONFIG:-${REPO}/configs/ada/actionability/in1k_region_enrichment.yaml}"
SELECT_CONFIG="${SELECT_CONFIG:-${REPO}/configs/ada/actionability/in1k_confirmatory_selection.yaml}"
DELETE_CONFIG="${DELETE_CONFIG:-${REPO}/configs/ada/actionability/in1k_deletion_confirmatory.yaml}"
RESTORE_CONFIG="${RESTORE_CONFIG:-${REPO}/configs/ada/actionability/in1k_real_restoration_confirmatory.yaml}"

cd "${REPO}"
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1

python3 -m ada.actionability.cli.enrich_regions --config "${ENRICH_CONFIG}"
python3 -m ada.actionability.cli.select_pilot_regions --config "${SELECT_CONFIG}"
python3 -m ada.actionability.cli.build_deletion_controls --config "${DELETE_CONFIG}"
python3 -m ada.actionability.cli.build_restoration_manifests --config "${RESTORE_CONFIG}"
