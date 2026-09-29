#!/bin/bash
# Keeps an eye on the CelebA RAE training (reproduce_didae_results.sh, "python train_generator.py
# --config .../celeba_rae_clip.yaml", running in another session) and validates Procrustes-CFKD
# on it as soon as a usable stage-2 checkpoint exists:
#   every 30 min : utils/celeba_rae_status.py -> logs/celeba_rae_watch.out
#   stage-2 epoch >= EARLY_EPOCH (default 8 of 24) : pin it (slim EMA extract), check the encoder,
#                   run Procrustes-CFKD with exact DDPM inversion, then the SDEdit(t=0.5) variant
#   stage 2 finished (ep-0000024.pt)             : the same two runs again on the final weights
# One GPU process at a time (16 GiB cgroup). Results: $PEAL_RUNS/celeba1k/Blond_Hair/
# classifier_poisoned098/procrustes_rae_clip_cfkd_ddpm_ep<XX>[_sdedit_t05]/logs (tag "gain").
#   nohup bash utils/celeba_rae_clip/watch_and_validate.sh > logs/celeba_rae_watch.driver 2>&1 &
export WANDB_MODE=offline HF_HUB_OFFLINE=1
cd "$PEAL_BASE"
EARLY_EPOCH=${EARLY_EPOCH:-8}
RUN=$PEAL_RUNS/celeba/rae_clip
CK=$RUN/stage2/ddt_l_cls/checkpoints
LOG=logs/celeba_rae_watch.out
STATE=logs/celeba_rae_watch.state; touch $STATE
CFG=configs/didae_experiments/adaptors/celeba1kx098_resnet18_procrustes_rae_clip_cfkd.yaml
newest_epoch() { ls $CK/ep-*.pt 2>/dev/null | grep -v last | sed 's/.*ep-0*\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }
validate() {  # $1 = tag (epXX | final), $2 = checkpoint name
  echo "=== validate $1 on $2 $(date)" | tee -a $LOG
  python utils/celeba_rae_clip/pin_stage2.py "$2" >> $LOG 2>&1 || { echo "pin failed" | tee -a $LOG; return 1; }
  python utils/celeba_rae_clip/check_encoder.py $RUN/config_pinned.yaml 16 2>&1 | grep check_encoder | tee -a $LOG
  for variant in "" "_sdedit_t05"; do
    out=configs/didae_experiments/adaptors/_generated_procrustes_rae_clip_cfkd_$1$variant.yaml
    sed -e "s#config_pinned.yaml#config_pinned$variant.yaml#" -e "s#procrustes_rae_clip_cfkd_ddpm#procrustes_rae_clip_cfkd_ddpm_$1$variant#" $CFG > $out
    echo "=== run_cfkd $out $(date)" | tee -a $LOG
    python -W ignore run_cfkd.py --config "$out" > logs/celeba_rae_cfkd_$1$variant.log 2>&1
    echo "=== run_cfkd exit $? $(date); gain:" | tee -a $LOG
    grep -n "^gain\|worst_group\|avg_group\|flip_rate" logs/celeba_rae_cfkd_$1$variant.log | tail -4 | tee -a $LOG
  done
}
while true; do
  { echo "===== $(date)"; timeout 300 python utils/celeba_rae_status.py 2>&1 | grep -v "^$"; } >> $LOG
  ep=$(newest_epoch)
  if [ -n "$ep" ] && [ "$ep" -ge "$EARLY_EPOCH" ] && ! grep -q "early_done" $STATE; then
    validate ep$(printf %02d $ep) $(printf ep-%07d.pt $ep) && echo early_done >> $STATE
  fi
  if [ -f $CK/ep-0000024.pt ] && ! grep -q "final_done" $STATE; then
    validate final ep-0000024.pt && echo final_done >> $STATE
  fi
  grep -q "final_done" $STATE && { echo "=== watcher done $(date)" >> $LOG; break; }
  sleep 1800
done
