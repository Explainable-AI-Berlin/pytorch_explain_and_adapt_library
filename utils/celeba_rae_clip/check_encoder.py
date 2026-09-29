"""Is the RAE's z_sem the DiffAE's z_sem? Both are meant to be clip.encode_image() of OpenAI
CLIP ViT-L/14, and the CelebA Procrustes dictionaries + c_min_and_maxes.txt were fitted on the
DiffAE's. Encodes N CelebA images through the RAE wrapper and through the DiffAE's own CLIP
preprocessing (bicubic 224, CLIP mean/std) and prints max |diff| and cosine.
  python utils/celeba_rae_clip/check_encoder.py <generator config.yaml> [N]
"""

import os, sys, torch, clip

sys.path.insert(0, os.environ["PEAL_BASE"])
from peal.generators.generator_factory import get_generator

gen_cfg = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
dev = "cuda"
gen = get_generator(generator=gen_cfg, device=dev)
ds = gen.generator_datasets[1]
xs = torch.stack(
    [(ds[i][0] if isinstance(ds[i], (list, tuple)) else ds[i]["x"]) for i in range(N)]
).to(dev)
x01 = gen.generator_dataset.project_to_pytorch_default(xs).clamp(0, 1)
with torch.no_grad():
    z_rae = gen.encode(x01, only_semantic=True).float()
    model, _ = clip.load("ViT-L/14", device=dev)
    MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (
        0.26862954,
        0.26130258,
        0.27577711,
    )
    xr = torch.nn.functional.interpolate(
        x01, size=(224, 224), mode="bicubic", align_corners=False, antialias=True
    ).clamp(0, 1)
    m = torch.tensor(MEAN, device=dev).view(1, -1, 1, 1)
    s = torch.tensor(STD, device=dev).view(1, -1, 1, 1)
    z_clip = model.encode_image((xr - m) / s).float()
cos = torch.nn.functional.cosine_similarity(z_rae, z_clip, dim=-1)
print(
    f"[check_encoder] N={N} |z_rae|={z_rae.norm(dim=-1).mean():.3f} |z_clip|={z_clip.norm(dim=-1).mean():.3f} "
    f"max|diff|={(z_rae - z_clip).abs().max():.4f} cos min={cos.min():.4f} mean={cos.mean():.4f}",
    flush=True,
)
print(
    "[check_encoder] SAME as the DiffAE z_sem"
    if cos.min() > 0.999
    else "[check_encoder] DIFFERENT from the DiffAE z_sem (RAEv2 runs CLIP at 256 px; measured cos 0.95 mean) -- use dictionaries fitted with utils/celeba_rae_clip/fit_dictionary.py, never the DiffAE ones"
)
