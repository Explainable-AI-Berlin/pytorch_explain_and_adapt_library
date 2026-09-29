#!/usr/bin/env bash
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
TMP=$(mktemp -d /tmp/ada-raev2-release.XXXXXX)
trap 'rm -rf "${TMP}"' EXIT

cd "${REPO}"
git clone --quiet third_party/RAEv2 "${TMP}"
bash integrations/raev2/apply_overlay.sh "${TMP}"

find scripts integrations -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
python3 -m py_compile \
  scripts/generator/download_raev2_dinov2l_stage1.py \
  scripts/generator/export_raev2_model_bundle.py \
  scripts/generator/sample_raev2_generator.py \
  scripts/generator/validate_raev2_model_bundle.py

for suite in actionability atlas interpretability; do
  PYTHONPATH=src python3 -m unittest discover -s "tests/ada/${suite}" -q
done

PYTHONPATH="${TMP}/src" python3 - <<'PY'
from pathlib import Path
import runpy

paths = (
    Path("integrations/raev2/overlay/tests/test_cache_index_order.py"),
    Path("integrations/raev2/overlay/tests/test_latent_cache_sampler.py"),
)
for path in paths:
    namespace = runpy.run_path(str(path))
    for name, function in namespace.items():
        if name.startswith("test_") and callable(function):
            function()
print("RAEv2 overlay tests: OK")
PY

python3 - <<'PY'
from pathlib import Path
import tomllib

tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
print("pyproject.toml: OK")
PY
