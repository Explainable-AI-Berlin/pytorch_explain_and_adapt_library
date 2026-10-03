"""Fit a BatchTopK SAE on a generator's semantic latent and match its atoms to labels.

Optional route next to the existing dictionaries; it changes no existing file or config.
It encodes the generator's own dataset once with the same encoder that the generator's
edit() and fit_sparse_dictionary() use, trains the SAE on those activations, writes
c_min_and_maxes.txt for every atom (so linesearch "dynamic" works), and matches atoms to
the labels [tumor, hospital]. The suggested component_indices go into the explainer
config (attempt k steps along atom component_indices[k]).

Camelyon note: the poisoned datasets keep only hospitals 0 and 1 (dataset_utils skips
confounder >= 2), so the hospital atom is matched on hospital 1 vs 0.

Usage:
  python fit_sae_components.py --generator_config <generator config.yaml or CFKD run config.yaml> \
      --sd_config <BatchTopKSAE yaml> [--max_images 200000] [--num_workers 8]

Outputs in the SAE base_path: activations.pt (cache), sae_weights.npz, config.yaml,
c_min_and_maxes.txt, atom_stats.npz, atom_matching.txt, suggested_component_indices.yaml.
"""
import argparse
import copy
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from peal.generators.generator_factory import get_generator
from peal.global_utils import load_yaml_config, save_yaml_config, set_random_seed
from peal.sparse_dictionaries.batch_topk import BatchTopKSAE, BatchTopKSAEConfig
from peal.sparse_dictionaries.matryoshka_sparse_autoencoder import ActivationStore


def load_generator_dict(path):
    """A generator config, or the generator section of a CFKD run config."""
    with open(path) as f:
        d = yaml.safe_load(f)
    if isinstance(d, dict) and "adaptor_type" in d and isinstance(d.get("generator"), dict):
        d = d["generator"]
    d = copy.deepcopy(d)
    d["sparse_dictionary"] = None  # only the encoder is needed; never fit or load the old dictionary
    d["is_loaded"] = True  # never take a training or "move existing base_path" branch
    return d


def encoder_and_datasets(gen):
    """Same latent and data as the generator's own fit_sparse_dictionary()."""
    name = type(gen).__name__
    if name == "DiffusionAutoencoder":
        try:
            fe = gen.model.ema_model.encoder
        except AttributeError:
            fe = gen.encoder
        return fe, [gen.generator_datasets[0], gen.generator_datasets[1]]
    if name == "PathldmAutoencoder":
        return gen.encoder, [gen.generator_datasets[1]]
    raise ValueError(f"unsupported generator type {name}")


@torch.no_grad()
def encode_dataset(fe, datasets, max_images, batch_size, num_workers, device):
    X, Y, n = [], [], 0
    for ds in datasets:
        buf = getattr(ds, "task_config", None)
        ds.task_config = None  # raw labels: [tumor, hospital, ...]
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        for batch in loader:
            x, y = batch[0], batch[1]
            X.append(fe(x.to(device)).float().cpu())
            Y.append(torch.as_tensor(y).float().reshape(len(x), -1).cpu())
            n += len(x)
            if n % (batch_size * 100) < batch_size:
                print(f"encoded {n} images", flush=True)
            if max_images and n >= max_images:
                break
        ds.task_config = buf
        if max_images and n >= max_images:
            break
    X, Y = torch.cat(X)[:max_images or None], torch.cat(Y)[:max_images or None]
    return X, Y


def train_sae(sae_model, X, log_every=500):
    """The training loop of BatchTopKSAE.fit_with_evaluation, without its evaluation."""
    sae_model.mu = X.mean(0)
    sae = sae_model.sae.to(sae_model.config.device)
    cfg = sae.config
    store = ActivationStore(cfg, X - sae_model.mu)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    n_steps = int(cfg["num_tokens"] // cfg["batch_size"])
    print(f"training SAE: {n_steps} steps, batch {cfg['batch_size']}, dict {cfg['dict_size']}, top_k {cfg['top_k']}")
    for i in range(n_steps):
        out = sae(store.next_batch())
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        opt.step()
        opt.zero_grad()
        if i % log_every == 0 or i == n_steps - 1:
            print(f"step {i}: loss {out['loss'].item():.4f} l0 {float(out['l0_norm']):.2f} "
                  f"l2 {float(out['l2_loss']):.4f}", flush=True)


def corr_columns(A, v):
    """Pearson correlation of every column of A [N, K] with v [N]."""
    A = A - A.mean(0)
    v = v - v.mean()
    denom = A.norm(dim=0) * v.norm() + 1e-12
    return (A * v[:, None]).sum(0) / denom


@torch.no_grad()
def atom_statistics(sae_model, X, Y, device, chunk=4096):
    acts = torch.cat([sae_model.encode(X[i:i + chunk].to(device)).float().cpu()
                      for i in range(0, len(X), chunk)])
    tumor, hospital = Y[:, 0], Y[:, 1]
    sub = (hospital == 0) | (hospital == 1)
    stats = {
        "frequency": (acts > 0).float().mean(0),
        "mean_activation": acts.mean(0),
        "corr_tumor": corr_columns(acts[sub], tumor[sub]),
        "corr_hospital1_vs_0": corr_columns(acts[sub], (hospital[sub] == 1).float()),
    }
    for h in range(int(hospital.max().item()) + 1):
        stats[f"corr_hospital{h}_vs_rest"] = corr_columns(acts, (hospital == h).float())
    return {k: v.numpy() for k, v in stats.items()}, int(sub.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generator_config", required=True)
    p.add_argument("--sd_config", required=True)
    p.add_argument("--max_images", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--top_n", type=int, default=10)
    args = p.parse_args()
    device = "cuda"

    sd = load_yaml_config(args.sd_config, BatchTopKSAEConfig)
    out = Path(sd.base_path)
    out.mkdir(parents=True, exist_ok=True)
    set_random_seed(sd.seed)

    cache = out / "activations.pt"
    if cache.exists():
        c = torch.load(cache)
        X, Y = c["X"], c["Y"]
        print(f"loaded cached activations {tuple(X.shape)} from {cache}")
    else:
        gen = get_generator(load_generator_dict(args.generator_config), device=device)
        fe, datasets = encoder_and_datasets(gen)
        fe.eval()
        print(f"encoding with {type(gen).__name__} encoder; datasets {[len(d) for d in datasets]}")
        X, Y = encode_dataset(fe, datasets, args.max_images, args.batch_size, args.num_workers, device)
        torch.save({"X": X, "Y": Y, "generator_config": args.generator_config}, cache)
        print(f"cached activations {tuple(X.shape)} labels {tuple(Y.shape)} to {cache}")
        del gen, fe
        torch.cuda.empty_cache()
    assert Y.shape[1] >= 2, f"need labels [tumor, hospital], got shape {tuple(Y.shape)}"

    sd.act_size = X.shape[1]
    sd.weights_path = sd.weights_path or str(out / sd.weights_name)
    sae_model = BatchTopKSAE(sd)
    train_sae(sae_model, X)
    sae_model.save_on_disk(sd.weights_path)
    save_yaml_config(sd, str(out / "config.yaml"))
    print(f"saved SAE weights to {sd.weights_path}")

    # bounds for linesearch "dynamic": c = z @ W over the data, same units as the generators use
    W = sae_model.get_components().detach().to(device)
    cmin = torch.full((W.shape[1],), float("inf"), device=device)
    cmax = torch.full((W.shape[1],), float("-inf"), device=device)
    for i in range(0, len(X), 4096):
        c = X[i:i + 4096].to(device) @ W
        cmin, cmax = torch.minimum(cmin, c.min(0).values), torch.maximum(cmax, c.max(0).values)
    with open(out / "c_min_and_maxes.txt", "w") as f:
        for i in range(W.shape[1]):
            f.write(f"Component {i}: min={cmin[i].item():.4f}, max={cmax[i].item():.4f}\n")

    stats, n_sub = atom_statistics(sae_model, X, Y, device)
    np.savez(out / "atom_stats.npz", **stats)
    ct, ch, fr = stats["corr_tumor"], stats["corr_hospital1_vs_0"], stats["frequency"]
    alive = fr > 0
    tumor_score = np.where(alive, np.abs(ct) - np.abs(ch), -np.inf)
    hosp_score = np.where(alive, np.abs(ch) - np.abs(ct), -np.inf)
    lines = [f"SAE {sd.base_path}: {W.shape[1]} atoms on {tuple(X.shape)} activations; "
             f"dead atoms {int((~alive).sum())}; matching on {n_sub} samples with hospital in {{0,1}}",
             "", "atom   freq    corr_tumor  corr_h1vs0   (top by |corr_tumor| - |corr_h1vs0|)"]
    for i in np.argsort(-tumor_score)[:args.top_n]:
        lines.append(f"{i:5d}  {fr[i]:.3f}   {ct[i]:+.3f}      {ch[i]:+.3f}")
    lines += ["", "atom   freq    corr_tumor  corr_h1vs0   (top by |corr_h1vs0| - |corr_tumor|)"]
    for i in np.argsort(-hosp_score)[:args.top_n]:
        lines.append(f"{i:5d}  {fr[i]:.3f}   {ct[i]:+.3f}      {ch[i]:+.3f}")
    best_t, best_h = int(np.argmax(tumor_score)), int(np.argmax(hosp_score))
    lines += ["", f"suggested component_indices: [{best_t}, {best_h}]  # [tumor atom, hospital atom]"]
    report = "\n".join(lines)
    print(report)
    (out / "atom_matching.txt").write_text(report + "\n")
    (out / "suggested_component_indices.yaml").write_text(
        f"component_indices: [{best_t}, {best_h}]  # [tumor atom, hospital atom]\n")


if __name__ == "__main__":
    main()
