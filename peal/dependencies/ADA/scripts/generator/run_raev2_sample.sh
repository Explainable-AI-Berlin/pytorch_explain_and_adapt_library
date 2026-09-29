#!/usr/bin/env bash
#SBATCH --job-name=ada_raev2_sample
#SBATCH --partition=gpu-2h
#SBATCH --constraint=40gb
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ADA_REPO="${ADA_REPO:-$(git rev-parse --show-toplevel)}"
if [[ -f "${ADA_REPO}/.env" ]]; then
  set -a
  source "${ADA_REPO}/.env"
  set +a
fi

VARIANT="${VARIANT:-cls_only}"
case "${VARIANT}" in
  cls_only|cls_class|class_only|cls_ca8) ;;
  *) echo "VARIANT must be cls_only, cls_class, class_only, or cls_ca8" >&2; exit 2 ;;
esac

RAEV2_DIR="${RAEV2_DIR:-${ADA_REPO}/third_party/RAEv2}"
if [[ ! -f "${RAEV2_DIR}/src/stage2/state_utils.py" ]]; then
  echo "ADA RAEv2 integration is missing under ${RAEV2_DIR}. Run: bash integrations/raev2/apply_overlay.sh" >&2
  exit 2
fi
ADA_CONTAINER="${ADA_CONTAINER:?Set ADA_CONTAINER}"
ADA_MODEL_BUNDLE_ROOT="${ADA_MODEL_BUNDLE_ROOT:?Set ADA_MODEL_BUNDLE_ROOT}"
ADA_CLS_STATS_ROOT="${ADA_CLS_STATS_ROOT:?Set ADA_CLS_STATS_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-${ADA_REPO}/outputs/raev2_samples/${VARIANT}}"
INDICES="${INDICES:-0,1,2,3}"
CLASS_IDS="${CLASS_IDS:-}"
SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-2}"
STEPS="${STEPS:-50}"
SEED="${SEED:-20260817}"
CLS_NOISE_SIGMA="${CLS_NOISE_SIGMA:-0}"
CLS_NOISE_SEED="${CLS_NOISE_SEED:-20260817}"

CHECKPOINT="${ADA_MODEL_BUNDLE_ROOT}/${VARIANT}/ema.pt"
CONFIG="${ADA_MODEL_BUNDLE_ROOT}/${VARIANT}/config.yaml"
CLS_ARRAY="${CLS_ARRAY:-${ADA_CLS_STATS_ROOT}/cls.float32.npy}"
LABEL_ARRAY="${LABEL_ARRAY:-${ADA_CLS_STATS_ROOT}/y.int16.npy}"

for path in "${CHECKPOINT}" "${CONFIG}" "${CLS_ARRAY}" "${LABEL_ARRAY}"; do
  [[ -f "${path}" ]] || { echo "Missing required artifact: ${path}" >&2; exit 2; }
done
mkdir -p "${ADA_REPO}/logs" "${OUTPUT_DIR}"

export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONDONTWRITEBYTECODE=1

apptainer exec --nv \
  -B "${HOME}:${HOME}" \
  -B /home/space:/home/space \
  "${ADA_CONTAINER}" \
  /bin/bash -lc "cd '${RAEV2_DIR}' && PYTHONPATH='src' python3 '${ADA_REPO}/scripts/generator/sample_raev2_generator.py' --config '${CONFIG}' --checkpoint '${CHECKPOINT}' --output-dir '${OUTPUT_DIR}' --cls-array '${CLS_ARRAY}' --label-array '${LABEL_ARRAY}' --indices '${INDICES}' --class-ids '${CLASS_IDS}' --samples-per-condition '${SAMPLES_PER_CONDITION}' --steps ${STEPS} --seed ${SEED} --cls-noise-sigma ${CLS_NOISE_SIGMA} --cls-noise-seed ${CLS_NOISE_SEED} --precision bf16"
