"""Shadow copy of the CelebA/OpenAI-CLIP generator whose final.ckpt holds only the
UNet weights (model.* and ema_model.* without the frozen encoder.* keys, no optimizer
state). Same weights, same global_step; loads in a fraction of the host RAM so a DiDAE
run fits the 16 GiB Slurm cgroup. Streams tensors one at a time via mmap."""

import os, torch, yaml, gc, datetime

PEAL_RUNS = os.environ["PEAL_RUNS"]
SRC = f"{PEAL_RUNS}/celeba/diffusion_autoencoder_openai_clip_vit_l14"
DST = f"{PEAL_RUNS}/celeba/diffusion_autoencoder_openai_clip_vit_l14_slim"
os.makedirs(f"{DST}/square64_ddim", exist_ok=True)
ck = torch.load(
    f"{SRC}/square64_ddim/final.ckpt", map_location="cpu", weights_only=False, mmap=True
)
sd = ck["state_dict"]
keys = list(sd.keys())
enc_keys = [k for k in keys if ".encoder." in k]
keep = {}
for k in keys:
    if ".encoder." in k:
        continue
    keep[k] = sd[k].clone()
print(
    f"total keys {len(keys)}, encoder keys dropped {len(enc_keys)}, kept {len(keep)}; kept params {sum(v.numel() for v in keep.values())/1e6:.1f} M",
    flush=True,
)
# spot-check 3 encoder tensors: model vs ema copy identical (frozen)
for k in enc_keys[:3] + enc_keys[-3:]:
    if k.startswith("model.encoder."):
        d = (sd[k] - sd["ema_" + k]).abs().max().item()
        print(f"  {k}: max|model - ema| = {d:.2e}", flush=True)
slim = {
    k: v
    for k, v in ck.items()
    if k not in ("state_dict", "optimizer_states", "lr_schedulers", "callbacks")
}
slim["state_dict"] = keep
slim["optimizer_states"] = []
del sd, ck
gc.collect()
torch.save(slim, f"{DST}/square64_ddim/final.ckpt")
cfg = yaml.safe_load(open(f"{SRC}/config.yaml"))
cfg["base_path"] = DST
yaml.safe_dump(cfg, open(f"{DST}/config.yaml", "w"))
open(f"{DST}/README.txt", "w").write(
    f"Shadow of {SRC} made {datetime.date.today()} by utils/celeba_openai_clip_didae/make_slim_generator.py.\n"
    f"square64_ddim/final.ckpt = the SAME weights as {SRC}/square64_ddim/final.ckpt "
    f"(global_step {slim.get('global_step')}) minus the frozen CLIP encoder.* keys (rebuilt from clip.load() by set_encoder()) "
    "and the optimizer state. Loads in far less host RAM; for DiDAE runs inside the 16 GiB cgroup, not for continuing training.\n"
)
print(
    "written",
    DST,
    os.path.getsize(f"{DST}/square64_ddim/final.ckpt") / 1e9,
    "GB; global_step",
    slim.get("global_step"),
    flush=True,
)
