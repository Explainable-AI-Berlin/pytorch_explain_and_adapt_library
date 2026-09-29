"""Fit a sparse-dictionary config in the CelebA RAE's z_sem space (OpenAI CLIP ViT-L/14 run at
256 px by RAEv2 -- cosine 0.95 to the DiffAE's 224-px embedding, so the DiffAE dictionaries do
not transfer) and write its c_min_and_maxes.txt. Needs only the frozen encoder, so it runs
before stage 1/2 are trained.
  python utils/celeba_rae_clip/fit_dictionary.py <sd_config.yaml> <name>
-> $PEAL_RUNS/celeba/rae_clip/<name>/{config.yaml,weights.npz,c_min_and_maxes.txt}"""

import os, sys, time, torch

sys.path.insert(0, os.environ["PEAL_BASE"])
from peal.global_utils import load_yaml_config, set_random_seed
from peal.generators.generator_factory import get_generator
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig

RUNS, BASE = os.environ["PEAL_RUNS"], os.environ["PEAL_BASE"]
GEN = f"{RUNS}/celeba/rae_clip"
sd_yaml, name = sys.argv[1], sys.argv[2]
set_random_seed(0)
gen = get_generator(generator=f"{GEN}/config.yaml", device="cuda")
cfg = load_yaml_config(sd_yaml)
cfg.base_path = f"{GEN}/{name}"
cfg.weights_path = f"{cfg.base_path}/weights.npz"
gen.config.sparse_dictionary = cfg
t0 = time.time()
gen.fit_sparse_dictionary()
W = gen.sparse_dictionary.get_components().to("cuda").float()
print(
    f"[fit] {name}: W {tuple(W.shape)} saved to {cfg.weights_path} in {time.time()-t0:.0f}s; fitted_on={gen.config.sparse_dictionary.fitted_on_encoder}",
    flush=True,
)
ds = get_datasets(
    load_yaml_config(
        f"{BASE}/configs/didae_experiments/data/celeba_unpoisoned_generator.yaml",
        DataConfig,
    )
)[0]
dl = torch.utils.data.DataLoader(ds, batch_size=200, num_workers=0)
cmin = cmax = None
with torch.no_grad():
    for i, b in enumerate(dl):
        if i >= 30:
            break
        x = (b[0] if isinstance(b, (list, tuple)) else b["x"]).to("cuda")
        c = gen.encode(x, only_semantic=True) @ W
        cmin = c.min(0).values if cmin is None else torch.minimum(cmin, c.min(0).values)
        cmax = c.max(0).values if cmax is None else torch.maximum(cmax, c.max(0).values)
with open(f"{cfg.base_path}/c_min_and_maxes.txt", "w") as f:
    for k in range(W.shape[1]):
        f.write(f"Component {k}: min={cmin[k].item():.4f}, max={cmax[k].item():.4f}\n")
print("[bounds]", open(f"{cfg.base_path}/c_min_and_maxes.txt").read(), flush=True)
