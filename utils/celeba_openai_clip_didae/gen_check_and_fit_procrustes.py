"""Diagnostics for the CelebA / OpenAI-CLIP diffusion autoencoder used by DiDAE.

1. Load the generator exactly the way run_didae.py does (get_generator on the run
   config) and report which checkpoint that picks.
2. Reconstruct 24 validation images the way log_sample() writes samples/ and
   samples_ema/ (DDIM T=20, random xT, cond = encoder(x)), once with the raw
   model and once with the EMA model -- DiDAE only ever decodes with the EMA.
3. Reconstruct + edit with the DDPM edit-friendly inversion DiDAE uses.
4. Fit an OrthogonalProcrustesDictionary (40 CelebA attributes) in this
   generator's z_sem space and save it next to the generator so a DiDAE config
   can point at it.
5. Realised-edit test on the Male direction: MSAE 'homme' (#4898) vs the
   Procrustes Male component, decoded with EMA and with raw weights.
"""

import os, sys, json, time
import torch
from torchvision.utils import save_image, make_grid

sys.path.insert(0, os.environ["PEAL_BASE"])
from peal.global_utils import load_yaml_config, set_random_seed
from peal.generators.generator_factory import get_generator
from peal.sparse_dictionaries.sparse_dictionary_factory import get_sparse_dictionary

PEAL_RUNS = os.environ["PEAL_RUNS"]
PEAL_BASE = os.environ["PEAL_BASE"]
GEN_DIR = f"{PEAL_RUNS}/celeba/diffusion_autoencoder_openai_clip_vit_l14"
OUT = f"{GEN_DIR}/diagnostics_20260911"
os.makedirs(OUT, exist_ok=True)
set_random_seed(0)
dev = "cuda"


def log(*a):
    print("[diag]", *a, flush=True)


# ---------------------------------------------------------------- 1. load
final_path = f"{GEN_DIR}/square64_ddim/final.ckpt"
last_path = f"{GEN_DIR}/square64_ddim/last.ckpt"
picked = final_path if os.path.exists(final_path) else last_path
ck = torch.load(picked, map_location="cpu", weights_only=False, mmap=True)
log(
    f"load_models() will pick {picked}: global_step={ck.get('global_step')} epoch={ck.get('epoch')}"
)
del ck

t0 = time.time()
gen = get_generator(generator=f"{GEN_DIR}/config.yaml", device=dev)
log(
    f"generator loaded in {time.time()-t0:.0f}s; model is None? {gen.model is None}; sample_type={getattr(gen,'sample_type',None)}"
)
lit = gen.model
lit.eval()
for m in (lit.model, lit.ema_model):
    m.eval()

# how different are raw and EMA weights?
with torch.no_grad():
    diffs, norms = [], []
    for (n1, p1), (n2, p2) in zip(
        lit.model.named_parameters(), lit.ema_model.named_parameters()
    ):
        if n1.startswith("encoder."):
            continue
        diffs.append((p1 - p2).norm() ** 2)
        norms.append(p1.norm() ** 2)
    rel = (torch.stack(diffs).sum().sqrt() / torch.stack(norms).sum().sqrt()).item()
log(
    f"relative L2 distance ||theta_raw - theta_ema|| / ||theta_raw|| over UNet params: {rel:.4f}"
)

# ---------------------------------------------------------------- 2. data
ds_val = gen.generator_datasets[1]


def get_x(ds, idxs):
    xs = []
    for i in idxs:
        item = ds[i]
        x = item[0] if isinstance(item, (list, tuple)) else item["x"]
        xs.append(x)
    return torch.stack(xs)


x = get_x(ds_val, list(range(24))).to(dev)
log(f"x batch {tuple(x.shape)} range [{x.min().item():.2f}, {x.max().item():.2f}]")

# ---------------------------------------------------------------- 3. DDIM recon like samples/ and samples_ema/
torch.manual_seed(0)
xT = torch.randn_like(x)
with torch.no_grad():
    rec_raw = lit.eval_sampler.sample(model=lit.model, noise=xT, x_start=x)
    rec_ema = lit.eval_sampler.sample(model=lit.ema_model, noise=xT, x_start=x)


def mse(a, b):
    return torch.mean((a - b) ** 2).item()


log(
    f"DDIM(T=20) recon MSE vs input: raw={mse(rec_raw, x):.4f}  ema={mse(rec_ema, x):.4f}"
)
save_image(
    (make_grid(torch.cat([x, rec_raw, rec_ema]), nrow=24) + 1) / 2,
    f"{OUT}/ddim_recon_real_raw_ema.png",
)
save_image(
    (make_grid(rec_raw, nrow=8) + 1) / 2, f"{OUT}/ddim_recon_raw_like_samples_dir.png"
)
save_image(
    (make_grid(rec_ema, nrow=8) + 1) / 2,
    f"{OUT}/ddim_recon_ema_like_samples_ema_dir.png",
)

# ---------------------------------------------------------------- 4. DDPM edit-friendly inversion (what DiDAE uses)
gen.set_sampler({"type": "ddpm", "num_steps": 20, "spacing": "uniform"})
with torch.no_grad():
    z, xT_inv, zs = gen.encode(x, sample_type="ddpm_inv")
    rec_inv_ema = gen.decode((z, xT_inv, zs), sample_type="ddpm_inv")
    # same with the raw weights
    lit.model, lit.ema_model = lit.ema_model, lit.model
    z_raw, xT_inv_r, zs_r = gen.encode(x, sample_type="ddpm_inv")
    rec_inv_raw = gen.decode((z_raw, xT_inv_r, zs_r), sample_type="ddpm_inv")
    lit.model, lit.ema_model = lit.ema_model, lit.model
log(
    f"DDPM-inversion recon MSE vs input: ema={mse(rec_inv_ema, x):.5f} raw={mse(rec_inv_raw, x):.5f} (identity by construction)"
)
log(
    f"z_sem: encoder is shared -> max|z_ema - z_raw| = {(z - z_raw).abs().max().item():.2e}; ||z|| mean {z.norm(dim=1).mean().item():.3f}"
)
save_image(
    (make_grid(torch.cat([x, rec_inv_ema, rec_inv_raw]), nrow=24) + 1) / 2,
    f"{OUT}/ddpm_inversion_recon_real_ema_raw.png",
)

# ---------------------------------------------------------------- 5. fit Procrustes (40 attrs) in this z_sem space
sd_cfg = load_yaml_config(
    f"{PEAL_BASE}/configs/didae_experiments/sparse_dictionaries/procrustes_sae_celeba_40comps.yaml"
)
sd_cfg.base_path = f"{GEN_DIR}/OrthogonalProcrustesDictionary40Comps"
sd_cfg.weights_path = f"{sd_cfg.base_path}/weights.npz"
if os.path.exists(sd_cfg.weights_path):
    log(f"Procrustes weights already at {sd_cfg.weights_path}, loading")
    gen.config.sparse_dictionary = sd_cfg
    gen.sparse_dictionary = get_sparse_dictionary(sd_cfg)
else:
    t0 = time.time()
    gen.config.sparse_dictionary = sd_cfg
    gen.fit_sparse_dictionary()
    log(f"Procrustes fitted+saved in {time.time()-t0:.0f}s -> {sd_cfg.weights_path}")
proc = gen.sparse_dictionary
W_proc = proc.get_components().to(dev).float()  # [768, 40]
attrs = (
    sd_cfg.task["y_selection"]
    if isinstance(sd_cfg.task, dict)
    else sd_cfg.task.y_selection
)
MALE = attrs.index("Male")
BLOND = attrs.index("Blond_Hair")
log(
    f"Procrustes W {tuple(W_proc.shape)}, Male comp {MALE}, Blond comp {BLOND}; orthonormal check |W^T W - I|max = {(W_proc.T @ W_proc - torch.eye(40, device=dev)).abs().max().item():.2e}"
)

# ---------------------------------------------------------------- 6. MSAE
msae = get_sparse_dictionary(
    f"{PEAL_BASE}/configs/didae_experiments/sparse_dictionaries/msae_decomposition.yaml"
)
W_msae = msae.get_components().to(dev).float()  # [768, 6144]
HOMME = 4898
u_msae = W_msae[:, HOMME]
u_proc = W_proc[:, MALE]
log(
    f"MSAE W {tuple(W_msae.shape)}; ||u_homme||={u_msae.norm().item():.3f}; cos(u_homme, u_procMale)={torch.nn.functional.cosine_similarity(u_msae, u_proc, dim=0).item():.3f}"
)

# empirical bounds for the homme atom from the DiDAE run, Procrustes bounds on the val set
bounds = {}
with open(
    f"{PEAL_RUNS}/celeba1k/Blond_Hair/classifier_poisoned098/didae_msae_openai_clip_ddpm/MSAEDecomposition/c_min_and_maxes.txt"
) as f:
    for line in f:
        i = int(line.split(":")[0].split()[1])
        mn = float(line.split("min=")[1].split(",")[0])
        mx = float(line.split("max=")[1])
        bounds[i] = (mn, mx)
log(f"homme bounds from run: {bounds[HOMME]}")


# ---------------------------------------------------------------- 7. realised-edit test on the Male direction
# Male readout in CLIP space: Procrustes Male projection (F1 0.98 concept in matches.txt is the MSAE one; use both)
def male_scores(zz):
    return {"proc_male": (zz @ u_proc).cpu(), "msae_homme": (zz @ u_msae).cpu()}


x_big = get_x(ds_val, list(range(64))).to(dev)
with torch.no_grad():
    z0, xT0, zs0 = gen.encode(x_big, sample_type="ddpm_inv")
    c_homme = z0 @ u_msae
    c_male = z0 @ u_proc
    c_male_all = get_x(ds_val, list(range(64, 320))).to(dev)
    z_more = gen.encode(c_male_all, only_semantic=True)
    male_min, male_max = (z_more @ u_proc).min().item(), (z_more @ u_proc).max().item()
log(
    f"Procrustes Male projection range on 256 val imgs: [{male_min:.3f}, {male_max:.3f}]; homme projection on 64: [{c_homme.min().item():.3f},{c_homme.max().item():.3f}]"
)

results = []


def run_edit(name, u, c_fact, target_hi, target_lo, decoder):
    """Push every sample's coefficient along u to the far bound of the *other* side, decode, re-encode."""
    u_n2 = u @ u
    # samples currently in the lower half go to target_hi, others to target_lo
    mid = 0.5 * (target_hi + target_lo)
    tgt = torch.where(
        c_fact < mid,
        torch.full_like(c_fact, target_hi),
        torch.full_like(c_fact, target_lo),
    )
    scale = (tgt - c_fact) / u_n2
    delta = scale.unsqueeze(1) * u.unsqueeze(0)
    z_cf = z0 + delta
    with torch.no_grad():
        if decoder == "raw":
            lit.model, lit.ema_model = lit.ema_model, lit.model
        x_cf = gen.decode((z_cf, xT0, zs0), sample_type="ddpm_inv")
        z_re = gen.encode(x_cf, only_semantic=True)
        if decoder == "raw":
            lit.model, lit.ema_model = lit.ema_model, lit.model
    ach = ((z_re - z0) * delta).sum(1) / (delta * delta).sum(1).clamp_min(1e-8)
    req_c = tgt - c_fact
    got_c = (z_re @ u) - c_fact
    img_mse = torch.mean((x_cf - x_big) ** 2, dim=(1, 2, 3))
    r = dict(
        direction=name,
        decoder=decoder,
        realised_ratio_median=ach.median().item(),
        realised_ratio_mean=ach.mean().item(),
        coef_requested_median=req_c.abs().median().item(),
        coef_achieved_median=(got_c / req_c).median().item(),
        frac_side_flipped=((z_re @ u - mid).sign() != (c_fact - mid).sign())
        .float()
        .mean()
        .item(),
        img_mse_median=img_mse.median().item(),
    )
    # cross readout: did the Procrustes Male score move the intended way?
    if name != "proc_male":
        pm0 = z0 @ u_proc
        pm1 = z_re @ u_proc
        r["proc_male_shift_signed_median"] = (
            (torch.sign(req_c) * (pm1 - pm0)).median().item()
        )
    results.append(r)
    log(json.dumps(r))
    save_image(
        (make_grid(torch.cat([x_big[:16], x_cf[:16]]), nrow=16) + 1) / 2,
        f"{OUT}/edit_{name}_{decoder}.png",
    )


for decoder in ("ema", "raw"):
    run_edit("msae_homme", u_msae, c_homme, bounds[HOMME][1], bounds[HOMME][0], decoder)
    run_edit("proc_male", u_proc, c_male, male_max, male_min, decoder)

with open(f"{OUT}/realised_edit_results.json", "w") as f:
    json.dump(
        {
            "picked_ckpt": picked,
            "rel_param_dist_raw_ema": rel,
            "ddim_recon_mse": {"raw": mse(rec_raw, x), "ema": mse(rec_ema, x)},
            "edits": results,
        },
        f,
        indent=1,
    )
log("done; outputs in", OUT)
