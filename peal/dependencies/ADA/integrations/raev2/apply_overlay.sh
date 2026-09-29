#!/usr/bin/env bash
set -euo pipefail

INTEGRATION_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${INTEGRATION_DIR}/../.." && pwd)
RAEV2_DIR=${1:-"${REPO_ROOT}/third_party/RAEv2"}
EXPECTED_COMMIT=$(tr -d '[:space:]' < "${INTEGRATION_DIR}/UPSTREAM_COMMIT")

if [[ ! -d "${RAEV2_DIR}/.git" && ! -f "${RAEV2_DIR}/.git" ]]; then
  echo "RAEv2 submodule not found at ${RAEV2_DIR}. Run git submodule update --init." >&2
  exit 1
fi

ACTUAL_COMMIT=$(git -C "${RAEV2_DIR}" rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Expected RAEv2 ${EXPECTED_COMMIT}, found ${ACTUAL_COMMIT}." >&2
  exit 1
fi

PATCH=${INTEGRATION_DIR}/patches/raev2-ada-development.patch
if git -C "${RAEV2_DIR}" apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo "RAEv2 tracked patch is already applied."
else
  git -C "${RAEV2_DIR}" apply --check "${PATCH}"
  git -C "${RAEV2_DIR}" apply "${PATCH}"
fi

cp -a "${INTEGRATION_DIR}/overlay/." "${RAEV2_DIR}/"
echo "Applied ADA RAEv2 integration to ${RAEV2_DIR}"
