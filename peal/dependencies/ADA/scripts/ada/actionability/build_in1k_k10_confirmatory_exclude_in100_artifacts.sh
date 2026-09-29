#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
EXCLUDE_CONFIG="${EXCLUDE_CONFIG:-${REPO}/configs/ada/actionability/in1k_excluded_development_wnids_in100.yaml}"
SELECT_CONFIG="${SELECT_CONFIG:-${REPO}/configs/ada/actionability/in1k_confirmatory_selection_exclude_in100.yaml}"
DELETE_CONFIG="${DELETE_CONFIG:-${REPO}/configs/ada/actionability/in1k_deletion_confirmatory_exclude_in100.yaml}"
RESTORE_CONFIG="${RESTORE_CONFIG:-${REPO}/configs/ada/actionability/in1k_real_restoration_confirmatory_exclude_in100.yaml}"

cd "${REPO}"
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1

python3 -m ada.actionability.cli.build_development_wnid_exclusion \
  --config "${EXCLUDE_CONFIG}" \
  ${EXCLUDE_OVERWRITE:+--overwrite}

EXCLUDED_SHA256="$(cut -d' ' -f1 "${REPO}/artifacts/ada/actionability/exclusions/e7a_in1k_excluded_development_wnids_in100/excluded_development_wnids.sha256")"

python3 -m ada.actionability.cli.select_pilot_regions \
  --config "${SELECT_CONFIG}" \
  --excluded-wnids-sha256 "${EXCLUDED_SHA256}" \
  ${SELECT_OVERWRITE:+--overwrite}
python3 -m ada.actionability.cli.build_deletion_controls \
  --config "${DELETE_CONFIG}" \
  ${DELETE_OVERWRITE:+--overwrite}
python3 -m ada.actionability.cli.build_restoration_manifests \
  --config "${RESTORE_CONFIG}" \
  ${RESTORE_OVERWRITE:+--overwrite}
