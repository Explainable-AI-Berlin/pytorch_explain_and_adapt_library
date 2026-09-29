#!/bin/bash
# ImageNet Representation-Autoencoder generator with a frozen OpenAI CLIP ViT-L/14
# encoder (reproduction_scripts/reproduce_didae_results.sh, "ImageNet RAE" block).
# Runs the RAEv2 two-stage pipeline on every GPU visible in the current
# allocation, each stage resumable and retried:
#
#   bash reproduction_scripts/run_imagenet_rae_clip.sh stage1   # RAE decoder (ViT-XL) on CLIP patch tokens
#   bash reproduction_scripts/run_imagenet_rae_clip.sh stats    # EMA decoder extraction + latent statistics
#   bash reproduction_scripts/run_imagenet_rae_clip.sh cache    # latent + CLS cache on node-local /tmp
#   bash reproduction_scripts/run_imagenet_rae_clip.sh stage2   # DDT flow-matching model on the latents,
#                                          #   conditioned on the CLIP image embedding
#   bash reproduction_scripts/run_imagenet_rae_clip.sh all      # the stages in order (7-day run)
#
# Launch from inside the interactive allocation, in the apptainer container:
#   nohup bash reproduction_scripts/run_imagenet_rae_clip.sh all > logs/imagenet_rae_clip.out 2>&1 &
# torchrun spawns one process per visible GPU; do not wrap this in srun.
#
# The actual stage logic lives in peal/generators/rae_pipeline.py so that
# `python train_generator.py --config configs/didae_experiments/generators/imagenet_rae_clip.yaml`
# runs exactly the same commands; this script only adds the environment and the
# retry loop. Environment: no W&B key on the nodes -> WANDB_MODE=offline; the
# conda site-packages are read-only -> omegaconf lives in RAEv2/.deps;
# RAEv2 scripts resolve paths against their own tree -> the pipeline cds into it.

set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-all}

export PEAL_RUNS=${PEAL_RUNS:-$PWD/peal_runs}
export PEAL_DATA=${PEAL_DATA:-$PWD/datasets}
export PEAL_BASE=${PEAL_BASE:-$(pwd)}
export WANDB_MODE=offline
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
export HF_HOME=${HF_HOME:-/tmp/hf}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export RAEV2_DIR=${RAEV2_DIR:-$PEAL_BASE/peal/dependencies/ADA/third_party/RAEv2}
export PYTHONPATH=$PEAL_BASE:$RAEV2_DIR/.deps:$RAEV2_DIR/src
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" /tmp/rae_clip_tmp logs

CONFIG=$PEAL_BASE/configs/didae_experiments/generators/imagenet_rae_clip.yaml
NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "config: $CONFIG"; echo "GPUs visible: $NGPU"; echo "mode: $MODE"

for i in $(seq 1 200); do
    if pgrep -f "[p]eal.generators.rae_pipeline" > /dev/null; then
        echo "!! another rae_pipeline.py is already running -- refusing to start"; exit 1
    fi
    echo "=== attempt $i ($MODE) starting at $(date) ==="
    python -u -m peal.generators.rae_pipeline --config "$CONFIG" --stage "$MODE"
    rc=$?
    echo "=== attempt $i exited rc=$rc at $(date) ==="
    [ "$rc" -eq 0 ] && break
    sleep 90
done
