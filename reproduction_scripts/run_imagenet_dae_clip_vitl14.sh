#!/bin/bash
# ImageNet diffusion autoencoder with a frozen OpenAI CLIP ViT-L/14 encoder on
# every GPU visible in the current allocation (reproduction_scripts/
# reproduce_didae_results.sh, "ImageNet DiffAE with a frozen OpenAI CLIP" block).
#
#   bash reproduction_scripts/run_imagenet_dae_clip_vitl14.sh smoke   # ~15 min DDP test on a 10k-image
#                                               #   symlink subset: 200 optimizer
#                                               #   steps, ckpt every 50, infer +
#                                               #   latent stages, run dir *_smoke
#   bash reproduction_scripts/run_imagenet_dae_clip_vitl14.sh full    # the 7-day run, retried on crash
#
# Launch from inside the interactive allocation, in the apptainer container
# (SLURM_JOB_NAME must be "bash"/"interactive" so Lightning ignores the Slurm env
# and spawns one rank per visible GPU itself; never wrap this in srun):
#   nohup bash reproduction_scripts/run_imagenet_dae_clip_vitl14.sh full > logs/imagenet_dae_full.out 2>&1 &
#
# Per attempt this is the top-level call
#   python train_generator.py --config configs/didae_experiments/generators/imagenet_diffusion_autoencoder_openai_clip_vit_l14.yaml
# and, once the run directory exists, every retry is
#   python train_generator.py --config $PEAL_RUNS/imagenet/diffusion_autoencoder_openai_clip_vit_l14/config.yaml --continue_training True
# after promoting the furthest-along last*.ckpt (by global_step, never mtime) to
# last.ckpt, which is what diffae's train() resumes from. experiment.py now
# passes enable_version_counter=False so Lightning overwrites last.ckpt instead
# of writing last-v1.ckpt, last-v2.ckpt, ... after a resume; the promotion step
# stays as a safety net for directories written by the old code.
#
# Environment: no W&B key and no internet on the nodes -> WANDB_MODE=offline and
# HF/torch caches pointed at ~/.cache (CLIP ViT-L/14 and DINOv2-small are there).

set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-full}

export PEAL_RUNS=${PEAL_RUNS:-$PWD/peal_runs}
export PEAL_DATA=${PEAL_DATA:-$PWD/datasets}
export PEAL_BASE=${PEAL_BASE:-$(pwd)}
export WANDB_MODE=offline
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
export PEAL_NUM_WORKERS=${PEAL_NUM_WORKERS:-8}   # loader workers per rank
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-60}  # progress-bar refresh in the log
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" logs

FULL_CONFIG=$PEAL_BASE/configs/didae_experiments/generators/imagenet_diffusion_autoencoder_openai_clip_vit_l14.yaml
DATA_CONFIG=$PEAL_BASE/configs/sce_experiments/data/imagenet_generator.yaml
RUN=$PEAL_RUNS/imagenet/diffusion_autoencoder_openai_clip_vit_l14
MAX_ATTEMPTS=100
SLEEP=120

if [ "$MODE" = "smoke" ]; then
    SMOKE_DIR=${SMOKE_DIR:-/tmp/imagenet_dae_smoke}
    RUN=${RUN}_smoke
    MAX_ATTEMPTS=3
    SLEEP=20
    mkdir -p "$SMOKE_DIR"
    if [ ! -d "$SMOKE_DIR/train_set" ]; then
        python - "$(grep '^dataset_path' "$DATA_CONFIG" | awk '{print $3}')" "$SMOKE_DIR/train_set" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
n = 0
for c in sorted(os.listdir(src)):
    os.makedirs(os.path.join(dst, c), exist_ok=True)
    for f in sorted(os.listdir(os.path.join(src, c)))[:10]:
        os.symlink(os.path.join(src, c, f), os.path.join(dst, c, f)); n += 1
print("smoke subset:", n, "symlinks under", dst)
PY
    fi
    sed -e "s#^dataset_path .*#dataset_path            : $SMOKE_DIR/train_set#" \
        -e "s#^num_samples .*#num_samples             : 10000#" \
        "$DATA_CONFIG" > "$SMOKE_DIR/imagenet_generator_smoke.yaml"
    sed -e "s#^base_path .*#base_path : $RUN#" \
        -e "s#^data: .*#data: $SMOKE_DIR/imagenet_generator_smoke.yaml#" \
        -e "s#^total_samples: .*#total_samples: 12800#" \
        -e "s#^save_every_samples: .*#save_every_samples: 3200#" \
        -e "s#^max_time: .*#max_time: \"00:00:40:00\"#" \
        "$FULL_CONFIG" > "$SMOKE_DIR/imagenet_dae_smoke.yaml"
    CONFIG=$SMOKE_DIR/imagenet_dae_smoke.yaml
else
    CONFIG=$FULL_CONFIG
fi

NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "mode: $MODE"; echo "config: $CONFIG"; echo "run dir: $RUN"; echo "GPUs visible: $NGPU"
echo "workers/rank: $PEAL_NUM_WORKERS"; echo "job: ${SLURM_JOB_ID:-?} on $(hostname), ends $(date -d @${SLURM_JOB_END_TIME:-0} 2>/dev/null)"

promote_last_checkpoint() {
    # For every directory under the run dir holding last*.ckpt files: make the
    # one with the highest global_step last.ckpt and delete the rest.
    python - "$RUN" <<'PY'
import glob, os, sys, torch
run = sys.argv[1]
dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(run, "**", "last*.ckpt"), recursive=True)})
for d in dirs:
    cands = sorted(glob.glob(os.path.join(d, "last*.ckpt")))
    if len(cands) <= 1:
        continue
    steps = {}
    for p in cands:
        try:
            ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
            steps[p] = int(ck["global_step"])
            del ck
        except Exception as e:  # truncated write from a crash
            print(f"[promote] {p}: unreadable ({e!r}), removing")
            os.remove(p)
    if not steps:
        continue
    best = max(steps, key=steps.get)
    target = os.path.join(d, "last.ckpt")
    print(f"[promote] {d}: " + ", ".join(f"{os.path.basename(p)}@{s}" for p, s in sorted(steps.items())))
    if best != target:
        os.replace(best, target)
        print(f"[promote]   -> {os.path.basename(best)} (step {steps[best]}) is now last.ckpt")
    for p in steps:
        if p != best and p != target and os.path.exists(p):
            os.remove(p); print(f"[promote]   removed {os.path.basename(p)} (step {steps[p]})")
PY
}

for i in $(seq 1 $MAX_ATTEMPTS); do
    if pgrep -f "train_generator.py --config .*(imagenet_diffusion_autoencoder_openai_clip_vit_l14|imagenet_dae_smoke)" > /dev/null; then
        echo "!! another train_generator.py for this run is already running -- refusing to start"; exit 1
    fi
    if [ -n "${SLURM_JOB_END_TIME:-}" ] && [ $(( SLURM_JOB_END_TIME - $(date +%s) )) -lt 1800 ]; then
        echo "!! less than 30 min left in the allocation -- not starting another attempt"; exit 2
    fi
    echo "=== attempt $i ($MODE) starting at $(date) ==="
    if [ -f "$RUN/config.yaml" ] && ls "$RUN"/square64_ddim/last*.ckpt > /dev/null 2>&1; then
        promote_last_checkpoint
        echo "resuming: python train_generator.py --config $RUN/config.yaml --continue_training True"
        python train_generator.py --config "$RUN/config.yaml" --continue_training True
    else
        echo "fresh start: python train_generator.py --config $CONFIG"
        python train_generator.py --config "$CONFIG"
    fi
    rc=$?
    echo "=== attempt $i exited rc=$rc at $(date) ==="
    [ "$rc" -eq 0 ] && break
    # stray DDP ranks keep the GPUs busy after a crash of the parent
    pkill -f "train_generator.py --config .*(imagenet_diffusion_autoencoder_openai_clip_vit_l14|imagenet_dae_smoke)" 2>/dev/null
    sleep $SLEEP
done
