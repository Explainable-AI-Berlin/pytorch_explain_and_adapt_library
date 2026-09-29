"""Group accuracy by confounder presence, before and after the repair.

Overall accuracy hides a shortcut: these probes sit at 99.6 %, so a confounder
that only bites when it is inconsistent with the label cannot move the average.
This splits the real val+test images by whether the confounder concept is active
(the SAE atom's code on the image's CLIP embedding) and reports accuracy per
group, plus the individual images the repair fixed or broke.

  python reproduction_scripts/overnight_group_eval.py <pair> <adaptor config>
"""

import json
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.environ["PEAL_BASE"])
from peal.architectures.predictors import get_predictor
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig
from peal.global_utils import load_yaml_config
from peal.sparse_dictionaries.msae_decomposition import (
    MSAEDecomposition,
    MSAEDecompositionConfig,
)

PAIR, CFG = sys.argv[1], sys.argv[2]
RUNS, BASE = os.environ["PEAL_RUNS"], os.environ["PEAL_BASE"]
RUN = str(yaml.safe_load(open(CFG))["base_dir"]).replace("$PEAL_RUNS", RUNS)
S = json.load(open(RUN + "/overnight_summary.json"))
P = f"{RUNS}/imagenet_probes/{PAIR}_dinov3_linear"
FIX = RUN + "/cfkd_overnight/1/finetuned_model/model.cpl"
atoms = S["confounder_atoms"]
cn = S["class_names"]
DEV = "cuda"

orig, _ = get_predictor(P + "/model.cpl", device=DEV)
orig.eval()
fine, _ = get_predictor(FIX, device=DEV)
fine.eval()
sd = MSAEDecomposition(
    load_yaml_config(
        f"{BASE}/configs/didae_experiments/sparse_dictionaries/msae_decomposition.yaml",
        MSAEDecompositionConfig,
    )
)
import clip

clip_model, _ = clip.load("ViT-L/14", device=DEV)
clip_model.eval()
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).view(1, 3, 1, 1)

data_cfg = load_yaml_config(
    yaml.safe_load(open(RUN + "/config.yaml"))["data"], DataConfig
)
_, val, test = get_datasets(data_cfg)
rows = []
for name, dset in (("val", val), ("test", test)):
    for xb, yb in DataLoader(dset, batch_size=32, shuffle=False, num_workers=2):
        yb = yb[0] if isinstance(yb, (tuple, list)) else yb
        xb = xb.to(DEV)
        with torch.no_grad():
            p0 = torch.softmax(orig(xb), -1).cpu().numpy()
            p1 = torch.softmax(fine(xb), -1).cpu().numpy()
            z = clip_model.encode_image(
                (
                    torch.nn.functional.interpolate(xb, 224, mode="bicubic").clamp(0, 1)
                    - MEAN
                )
                / STD
            ).float()
            code = sd.encode(z).detach().cpu().numpy()
        for j in range(xb.shape[0]):
            rows.append(
                dict(
                    split=name,
                    y=int(yb[j]),
                    a0=int(np.argmax(p0[j])),
                    a1=int(np.argmax(p1[j])),
                    conf=float(max(code[j][a] for a in atoms)),
                    p0=p0[j].tolist(),
                    p1=p1[j].tolist(),
                    idx=len(rows),
                )
            )
print(f"{len(rows)} real val+test images; confounder atom(s) {atoms}", flush=True)

present = [r for r in rows if r["conf"] > 0]
absent = [r for r in rows if r["conf"] <= 0]
out = {"pair": PAIR, "atoms": atoms, "n": len(rows), "n_conf_present": len(present)}
for label, grp in (
    ("confounder present", present),
    ("confounder absent", absent),
    ("all", rows),
):
    if not grp:
        continue
    b = np.mean([r["a0"] == r["y"] for r in grp])
    a = np.mean([r["a1"] == r["y"] for r in grp])
    print(
        f"  {label:20s} n={len(grp):4d}  acc {b:.4f} -> {a:.4f}  ({a-b:+.4f})",
        flush=True,
    )
    out[label.replace(" ", "_")] = dict(
        n=len(grp), before=float(b), after=float(a), gain=float(a - b)
    )
# per (class, confounder) group, the classic worst-group table
worst = []
for y in (0, 1):
    for c, lab in ((True, "conf"), (False, "noconf")):
        grp = [r for r in rows if r["y"] == y and (r["conf"] > 0) == c]
        if not grp:
            continue
        b = np.mean([r["a0"] == r["y"] for r in grp])
        a = np.mean([r["a1"] == r["y"] for r in grp])
        print(
            f"  {cn[y]:14s} {lab:7s} n={len(grp):4d}  acc {b:.4f} -> {a:.4f}  ({a-b:+.4f})",
            flush=True,
        )
        worst.append(
            dict(
                cls=cn[y],
                group=lab,
                n=len(grp),
                before=float(b),
                after=float(a),
                gain=float(a - b),
            )
        )
out["groups"] = worst
out["worst_group_before"] = min(g["before"] for g in worst) if worst else None
out["worst_group_after"] = min(g["after"] for g in worst) if worst else None
fixed = [r for r in rows if r["a0"] != r["y"] and r["a1"] == r["y"]]
broke = [r for r in rows if r["a0"] == r["y"] and r["a1"] != r["y"]]
out["fixed"] = len(fixed)
out["broken"] = len(broke)
out["fixed_with_confounder"] = sum(1 for r in fixed if r["conf"] > 0)
print(
    f"  repaired {len(fixed)} images ({out['fixed_with_confounder']} of them with the confounder present), broke {len(broke)}",
    flush=True,
)
json.dump(out, open(RUN + "/overnight_group_eval.json", "w"), indent=1)

# save the repaired images for the figure's (d) panel
if fixed:
    ASSETS = f"{BASE}/docs/paper_figures/fig1_assets_{PAIR}_{os.path.basename(RUN)}/fix_samples"
    os.makedirs(ASSETS, exist_ok=True)
    keep = sorted(fixed, key=lambda r: -(max(r["p1"]) - max(r["p0"])))[:2]
    idxs = {r["idx"] for r in keep}
    seen, saved = 0, []
    for name, dset in (("val", val), ("test", test)):
        for xb, yb in DataLoader(dset, batch_size=32, shuffle=False, num_workers=2):
            for j in range(xb.shape[0]):
                if seen in idxs:
                    r = next(q for q in keep if q["idx"] == seen)
                    ip = f"{ASSETS}/{len(saved)}_natural.png"
                    Image.fromarray(
                        (np.transpose(xb[j].numpy(), (1, 2, 0)) * 255)
                        .clip(0, 255)
                        .astype(np.uint8)
                    ).save(ip)
                    saved.append(
                        dict(
                            image=ip,
                            true=r["y"],
                            p_before=r["p0"],
                            p_after=r["p1"],
                            caption=f"held-out {cn[r['y']]}",
                            natural=True,
                        )
                    )
                seen += 1
    if saved:
        json.dump(saved, open(f"{ASSETS}/samples.json", "w"), indent=1)
        print("  wrote natural fix samples ->", f"{ASSETS}/samples.json", flush=True)
print("GROUPEVAL " + json.dumps(out), flush=True)
