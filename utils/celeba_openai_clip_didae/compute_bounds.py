"""Write c_min_and_maxes.txt (raw z_sem @ W projections) for a dictionary dir, mirroring
DiDAE._ensure_component_bounds but over the group-balanced 6400-sample CelebA split.
Usage: compute_bounds.py <dictionary_dir>"""

import os, sys, torch

sys.path.insert(0, os.environ["PEAL_BASE"])
from peal.global_utils import load_yaml_config
from peal.generators.generator_factory import get_generator
from peal.sparse_dictionaries.sparse_dictionary_factory import get_sparse_dictionary
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig

RUNS = os.environ["PEAL_RUNS"]
BASE = os.environ["PEAL_BASE"]
d = sys.argv[1]
gen = get_generator(
    generator=f"{RUNS}/celeba/diffusion_autoencoder_openai_clip_vit_l14_slim/config.yaml",
    device="cuda",
)
sd = get_sparse_dictionary(f"{d}/config.yaml")
W = sd.get_components().to("cuda").float()
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
with open(f"{d}/c_min_and_maxes.txt", "w") as f:
    for k in range(W.shape[1]):
        f.write(f"Component {k}: min={cmin[k].item():.4f}, max={cmax[k].item():.4f}\n")
print("[bounds]", open(f"{d}/c_min_and_maxes.txt").read(), flush=True)
