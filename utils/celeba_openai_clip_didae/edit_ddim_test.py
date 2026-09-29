"""Same Male-direction edit test as gen_check_and_fit_procrustes.py, but through the
DDIM inversion path (encode_stochastic -> render), with EMA and raw weights, and at
several step multipliers. Answers: does the edit reach the image at all when the
noise maps do not pin the image down?"""

import os, sys, json, torch
from torchvision.utils import save_image, make_grid

sys.path.insert(0, os.environ["PEAL_BASE"])
from peal.global_utils import set_random_seed
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


gen = get_generator(generator=f"{GEN_DIR}/config.yaml", device=dev)
lit = gen.model
lit.eval()
lit.model.eval()
lit.ema_model.eval()
ds_val = gen.generator_datasets[1]


def get_x(idxs):
    xs = []
    for i in idxs:
        it = ds_val[i]
        xs.append(it[0] if isinstance(it, (list, tuple)) else it["x"])
    return torch.stack(xs)


x = get_x(list(range(64))).to(dev)

proc = get_sparse_dictionary(
    f"{GEN_DIR}/OrthogonalProcrustesDictionary40Comps/config.yaml"
)
W_proc = proc.get_components().to(dev).float()
u_proc = W_proc[:, 20]
msae = get_sparse_dictionary(
    f"{PEAL_BASE}/configs/didae_experiments/sparse_dictionaries/msae_decomposition.yaml"
)
u_msae = msae.get_components().to(dev).float()[:, 4898]
HOMME_B = (-0.9332, 4.1705)
with torch.no_grad():
    z_more = gen.encode(get_x(list(range(64, 320))).to(dev), only_semantic=True)
MALE_B = ((z_more @ u_proc).min().item(), (z_more @ u_proc).max().item())

results = []


def swap():
    lit.model, lit.ema_model = lit.ema_model, lit.model


for decoder in ("ema", "raw"):
    if decoder == "raw":
        swap()
    with torch.no_grad():
        z0, xT = gen.encode(x, sample_type="ddim_inv")
        x_rec = gen.decode((z0, xT), sample_type="ddim_inv")
    rec_mse = torch.mean((x_rec - x) ** 2).item()
    log(f"[{decoder}] DDIM inversion recon MSE = {rec_mse:.4f}")
    save_image(
        (make_grid(torch.cat([x[:16], x_rec[:16]]), nrow=16) + 1) / 2,
        f"{OUT}/ddim_inv_recon_{decoder}.png",
    )
    for name, u, (lo, hi) in (
        ("msae_homme", u_msae, HOMME_B),
        ("proc_male", u_proc, MALE_B),
    ):
        c = z0 @ u
        mid = 0.5 * (lo + hi)
        for mult in (1.0, 2.0):
            tgt = torch.where(c < mid, torch.full_like(c, hi), torch.full_like(c, lo))
            scale = mult * (tgt - c) / (u @ u)
            delta = scale.unsqueeze(1) * u.unsqueeze(0)
            z_cf = z0 + delta
            with torch.no_grad():
                x_cf = gen.decode((z_cf, xT), sample_type="ddim_inv")
                z_re = gen.encode(x_cf, only_semantic=True)
            ach = ((z_re - z0) * delta).sum(1) / (delta * delta).sum(1)
            r = dict(
                path="ddim_inv",
                decoder=decoder,
                direction=name,
                step_mult=mult,
                realised_ratio_median=ach.median().item(),
                frac_side_flipped=((z_re @ u - mid).sign() != (c - mid).sign())
                .float()
                .mean()
                .item(),
                proc_male_side_flipped=(
                    (z_re @ u_proc - 0.5 * sum(MALE_B)).sign()
                    != (z0 @ u_proc - 0.5 * sum(MALE_B)).sign()
                )
                .float()
                .mean()
                .item(),
                img_mse_vs_recon_median=torch.mean((x_cf - x_rec) ** 2, dim=(1, 2, 3))
                .median()
                .item(),
            )
            results.append(r)
            log(json.dumps(r))
            save_image(
                (make_grid(torch.cat([x[:16], x_rec[:16], x_cf[:16]]), nrow=16) + 1)
                / 2,
                f"{OUT}/ddim_edit_{name}_{decoder}_x{mult:g}.png",
            )
    if decoder == "raw":
        swap()
json.dump(results, open(f"{OUT}/ddim_edit_results.json", "w"), indent=1)
log("done")
