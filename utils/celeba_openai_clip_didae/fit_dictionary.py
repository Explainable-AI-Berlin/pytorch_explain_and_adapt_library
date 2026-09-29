"""Fit a sparse dictionary config in the CelebA OpenAI-CLIP generator's z_sem space and
save it under the (real) generator dir:  fit_dictionary.py <sd_config.yaml> <dir_name>"""

import os, sys, time

sys.path.insert(0, os.environ["PEAL_BASE"])
from peal.global_utils import load_yaml_config, set_random_seed
from peal.generators.generator_factory import get_generator

PEAL_RUNS = os.environ["PEAL_RUNS"]
GEN = f"{PEAL_RUNS}/celeba/diffusion_autoencoder_openai_clip_vit_l14"
SLIM = GEN + "_slim"
sd_yaml, name = sys.argv[1], sys.argv[2]
set_random_seed(0)
gen = get_generator(generator=f"{SLIM}/config.yaml", device="cuda")
cfg = load_yaml_config(sd_yaml)
cfg.base_path = f"{GEN}/{name}"
cfg.weights_path = f"{cfg.base_path}/weights.npz"
gen.config.sparse_dictionary = cfg
t0 = time.time()
gen.fit_sparse_dictionary()
W = gen.sparse_dictionary.get_components()
print(
    f"[fit] {name}: W {tuple(W.shape)} saved to {cfg.weights_path} in {time.time()-t0:.0f}s; fitted_on={gen.config.sparse_dictionary.fitted_on_encoder}",
    flush=True,
)
