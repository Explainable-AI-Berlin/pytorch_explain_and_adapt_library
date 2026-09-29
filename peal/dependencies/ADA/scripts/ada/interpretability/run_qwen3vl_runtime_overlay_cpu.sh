#!/bin/bash
#SBATCH --job-name=ADA_qwen_runtime
#SBATCH --partition=cpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CONTAINER="${CONTAINER:-${REPO}/container.sif}"
QWEN_RUNTIME_OVERLAY="${QWEN_RUNTIME_OVERLAY:-${REPO}/external/qwen3vl_runtime_4_57_2}"
QWEN_RUNTIME_TMP="${QWEN_RUNTIME_OVERLAY}.build.${SLURM_JOB_ID:-$$}"

mkdir -p "${REPO}/logs"
cd "${REPO}"

apptainer exec \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    rm -rf '${QWEN_RUNTIME_TMP}'
    mkdir -p '${QWEN_RUNTIME_TMP}'
    python3 -m pip install \
      --target '${QWEN_RUNTIME_TMP}' \
      --upgrade \
      --no-deps \
      'transformers==4.57.2' \
      'accelerate>=1,<2' \
      'huggingface-hub>=0.36,<1' \
      'tokenizers==0.22.2' \
      'safetensors>=0.4.3' \
      sentencepiece
    python3 - <<'PY'
from pathlib import Path

path = Path('${QWEN_RUNTIME_TMP}') / 'transformers' / 'tokenization_utils_base.py'
text = path.read_text()
needle = '_config.model_type not in ['
replacement = \"_config.get('model_type') not in [\"
if needle in text:
    path.write_text(text.replace(needle, replacement))
elif replacement not in text:
    raise RuntimeError('could not find local-tokenizer compatibility patch site')
print('[qwen-runtime] applied local-tokenizer compatibility patch')
PY
    PYTHONPATH='${QWEN_RUNTIME_TMP}' python3 - <<'PY'
import json
from pathlib import Path

import accelerate
import sentencepiece
import torch
import transformers

overlay = Path('${QWEN_RUNTIME_TMP}')
torch_path = Path(torch.__file__).resolve()
if str(torch_path).startswith(str(overlay.resolve())):
    raise RuntimeError(f'torch was imported from the overlay: {torch_path}')
if (overlay / 'torch').exists() or any(overlay.glob('torch-*.dist-info')):
    raise RuntimeError('overlay unexpectedly contains torch')

out = overlay / 'runtime_manifest.json'
payload = {
    'transformers_version': transformers.__version__,
    'transformers_path': transformers.__file__,
    'overlay_final_path': '${QWEN_RUNTIME_OVERLAY}',
    'overlay_validation_path': '${QWEN_RUNTIME_TMP}',
    'transformers_local_tokenizer_patch': True,
    'accelerate_version': accelerate.__version__,
    'sentencepiece_version': getattr(sentencepiece, '__version__', ''),
    'torch_version': torch.__version__,
    'torch_path': torch.__file__,
    'overlay_contains_torch': False,
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY
    rm -rf '${QWEN_RUNTIME_OVERLAY}'
    mv '${QWEN_RUNTIME_TMP}' '${QWEN_RUNTIME_OVERLAY}'
  "
