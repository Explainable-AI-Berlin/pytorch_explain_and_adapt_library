"""Pin the CelebA RAE generator to one stage-2 checkpoint for evaluation on a 16 GiB node.

  python utils/celeba_rae_clip/pin_stage2.py [ep-XXXXXXX.pt | newest]

Writes $PEAL_RUNS/celeba/rae_clip/stage2_slim/ep-XXXXXXX_ema.pt (EMA weights only; a full
stage-2 checkpoint carries model+ema+optimizer, ~20 GB) and two generator configs next to
config.yaml (get_generator(str) resets base_path to the yaml's directory, so they must live
in the run dir):
  config_epXX.yaml          ddpm sampler, exact edit-friendly inversion (the DiffAE setup)
  config_epXX_sdedit_t05.yaml  ddpm, SDEdit from t=0.5 with fresh noise (edit shows more)
and config_pinned.yaml -> a copy of config_epXX.yaml, which the CFKD validation config uses.
"""

import os, sys, glob, re, shutil, torch, yaml

RUN = os.path.join(os.environ["PEAL_RUNS"], "celeba", "rae_clip")
cfg = yaml.safe_load(open(os.path.join(RUN, "config.yaml")))
ckpt_dir = os.path.join(RUN, "stage2", cfg["stage2_experiment"], "checkpoints")
arg = sys.argv[1] if len(sys.argv) > 1 else "newest"
if arg == "newest":
    cands = sorted(
        p for p in glob.glob(os.path.join(ckpt_dir, "ep-*.pt")) if "last" not in p
    )
    if not cands:
        raise SystemExit(f"no stage-2 checkpoint in {ckpt_dir}")
    src = cands[-1]
else:
    src = arg if os.path.isabs(arg) else os.path.join(ckpt_dir, arg)
ep = int(re.search(r"ep-(\d+)", os.path.basename(src)).group(1))
slim_dir = os.path.join(RUN, "stage2_slim")
os.makedirs(slim_dir, exist_ok=True)
slim = os.path.join(slim_dir, f"ep-{ep:07d}_ema.pt")
if not os.path.exists(slim):
    state = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    out = {
        "epoch": state.get("epoch"),
        "step": state.get("step"),
        "ema": {k: v.clone() for k, v in state["ema"].items()},
    }
    torch.save(out, slim)
    print(
        f"[pin] wrote {slim} ({os.path.getsize(slim)/1e9:.1f} GB) from {src}",
        flush=True,
    )
    del state, out
base = {k: v for k, v in cfg.items()}
base["stage2_checkpoint"] = slim
base["stage2_weights"] = "ema"
base["continue_training"] = False
variants = {
    f"config_ep{ep:02d}.yaml": dict(
        sampler={"type": "ddpm", "num_steps": 20},
        inversion_mode="exact",
        inversion_t=1.0,
        render_noise="inverted",
    ),
    f"config_ep{ep:02d}_sdedit_t05.yaml": dict(
        sampler={"type": "ddpm", "num_steps": 20},
        inversion_mode="sdedit",
        inversion_t=0.5,
        render_noise="fresh",
    ),
}
for name, extra in variants.items():
    c = dict(base)
    c.update(extra)
    with open(os.path.join(RUN, name), "w") as f:
        f.write(
            f"# pinned to stage-2 epoch {ep} EMA weights ({slim}); written by utils/celeba_rae_clip/pin_stage2.py\n"
        )
        yaml.safe_dump(c, f, sort_keys=False)
    print("[pin] wrote", os.path.join(RUN, name))
shutil.copy(
    os.path.join(RUN, f"config_ep{ep:02d}.yaml"),
    os.path.join(RUN, "config_pinned.yaml"),
)
shutil.copy(
    os.path.join(RUN, f"config_ep{ep:02d}_sdedit_t05.yaml"),
    os.path.join(RUN, "config_pinned_sdedit_t05.yaml"),
)
print(f"[pin] config_pinned.yaml -> epoch {ep}")
