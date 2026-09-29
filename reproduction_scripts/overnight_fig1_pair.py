"""Everything after the DiDAE render, for one ImageNet pair, unattended.

  python reproduction_scripts/overnight_fig1_pair.py <pair> <adaptor config>

1. read the finished run's sweep_results and verified flips
2. judge every direction with the LLM-as-human teacher -> confounder atoms
   (falls back to the first atom of sweep_atom_subset if the teacher is unusable)
3. build CFKD datasets from the confounder counterfactuals, holding out 20 %
4. repair the probe with DFR (last layer), PEAL's own 50:50 validation weighting
5. evaluate: test accuracy before/after, flip rate on the held-out counterfactuals,
   and -- the interesting one -- natural validation images that the original probe
   got wrong and the repaired probe gets right
6. write the figure spec and render it

Writes <run_dir>/overnight_summary.json with everything needed to compare pairs.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.environ["PEAL_BASE"])
from peal.architectures.predictors import get_predictor
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig
from peal.global_utils import load_yaml_config

PAIR, CFG_PATH = sys.argv[1], sys.argv[2]
RUNS = os.environ["PEAL_RUNS"]
BASE = os.environ["PEAL_BASE"]
P = f"{RUNS}/imagenet_probes/{PAIR}_dinov3_linear"
HERE = f"{BASE}/docs/paper_figures"
DEV = "cuda"

CONCEPTS = {
    "hummingbird_vs_bee_eater": (
        "the bird species: hummingbird vs bee eater (beak, plumage, body shape)",
        "a man-made feeder, perch or other object next to the bird, and the background vegetation",
    ),
    "fireboat_vs_lifeboat": (
        "the type of boat: fireboat vs lifeboat (hull shape, deck equipment, crew)",
        "a water jet / spray of water being pumped into the air",
    ),
    "fireboat_vs_lifeboat_curated5717": (
        "the type of boat: fireboat vs lifeboat (hull shape, deck equipment, crew)",
        "a water jet / spray of water being pumped into the air",
    ),
    "koala_vs_wombat": (
        "the animal: koala vs wombat (ears, nose, body)",
        "the tree, branch or foliage it sits on and the surrounding vegetation",
    ),
    "koala_vs_wombat_curated1752": (
        "the animal: koala vs wombat (ears, nose, body)",
        "the tree, branch or foliage it sits on and the surrounding vegetation",
    ),
    "cheeseburger_vs_hotdog": (
        "the food: cheeseburger vs hotdog (bun shape, filling)",
        "the plate, wrapper, table and other surroundings of the food",
    ),
    "dogsled_vs_horse_cart": (
        "the vehicle and its animal team: dogsled vs horse cart",
        "the snow and winter landscape versus grass, road and summer scenery",
    ),
    "freight_car_vs_passenger_car": (
        "the type of rail car: freight car vs passenger car",
        "graffiti on the car and the rails and railway infrastructure around it",
    ),
}
CLASS_NAMES = {
    "hummingbird_vs_bee_eater": ["hummingbird", "bee eater"],
    "fireboat_vs_lifeboat": ["fireboat", "lifeboat"],
    "fireboat_vs_lifeboat_curated5717": ["fireboat", "lifeboat"],
    "koala_vs_wombat": ["koala", "wombat"],
    "koala_vs_wombat_curated1752": ["koala", "wombat"],
    "cheeseburger_vs_hotdog": ["cheeseburger", "hotdog"],
    "dogsled_vs_horse_cart": ["dogsled", "horse cart"],
    "freight_car_vs_passenger_car": ["freight car", "passenger car"],
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


raw = yaml.safe_load(open(CFG_PATH))
RUN = str(raw["base_dir"]).replace("$PEAL_RUNS", RUNS)
VARIANT = os.path.basename(RUN)
cn = CLASS_NAMES[PAIR]
target_name, conf_name = CONCEPTS[PAIR]
summary = {"pair": PAIR, "run": RUN, "variant": VARIANT, "class_names": cn}
log("run dir", RUN)


def l224(im):
    a = (
        np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC)).astype(
            np.float32
        )
        / 255.0
    )
    return torch.from_numpy(a).permute(2, 0, 1)


def crop(pair_png, panel):
    im = Image.open(pair_png).convert("RGB")
    w = im.size[0]
    b = (w - 768) // 4
    x0 = b + panel * (256 + b)
    return im.crop((x0, 2, x0 + 256, 258))


# ---------------------------------------------------------------- 1. the render's directions
res = torch.load(RUN + "/sweep_results.pt", map_location="cpu", weights_only=False)
res = [r for r in res if (r.get("success_count") or 0) > 0]
res.sort(
    key=lambda r: (
        -(r["success_count"] or 0),
        -(r["ambient_flip_count"] or 0),
        -(r["latent_flip_count"] or 0),
    )
)


def atom_of(r):
    m = re.search(r"(?:SAE[ _]#?)(\d+)", str(r.get("dimension_name", "")))
    return int(m.group(1)) if m else int(r.get("direction_idx", -1))


log(
    "directions with verified flips:",
    [(atom_of(r), r["success_count"]) for r in res[:6]],
)
summary["directions"] = [
    {
        "atom": atom_of(r),
        "name": str(r.get("dimension_name", "")),
        "verified": int(r["success_count"] or 0),
        "ambient": int(r["ambient_flip_count"] or 0),
        "latent": int(r["latent_flip_count"] or 0),
    }
    for r in res[:6]
]

flip_dirs = sorted(glob.glob(RUN + "/successful_flips/*"))
by_atom = {}
for d in flip_dirs:
    ids = [int(x) for x in re.findall(r"(?:SAE_|SAE #)(\d+)", os.path.basename(d))]
    cfs = sorted(glob.glob(d + "/*_cf*_conf*.png"))
    for a in ids:
        by_atom.setdefault(a, []).extend(
            [
                (c, c.replace("_cf", "_pair", 1))
                for c in cfs
                if os.path.exists(c.replace("_cf", "_pair", 1))
            ]
        )
log("verified flips per atom:", {a: len(v) for a, v in by_atom.items()})

probe, _ = get_predictor(P + "/model.cpl", device=DEV)
probe.eval()

# ---------------------------------------------------------------- 2. judge the directions
judge_path = RUN + "/overnight_judge.json"
if os.path.exists(judge_path):
    judged = json.load(open(judge_path))
    log("judging: reusing", judge_path)
else:
    judged = {}
    try:
        from peal.adaptors.didae import DiDAE, _unwrap_dataloader
        from peal.teachers.llm2model_teacher import LLM2ModelTeacher

        stage = f"/tmp/overnight_judge_{PAIR}"
        os.makedirs(stage, exist_ok=True)
        d_obj = DiDAE(adaptor_config=CFG_PATH)
        ds_j = _unwrap_dataloader(d_obj.train_dataloader).dataset
        teacher = LLM2ModelTeacher(
            dataset=ds_j,
            target_name=target_name,
            confounder_name=conf_name,
            batch_size=16,
            stage_dir=stage,
        )
        for a, items in by_atom.items():
            colls, srcs, confs = [], [], []
            for k, (cf, pr) in enumerate(items[:12]):
                o = crop(pr, 0)
                with torch.no_grad():
                    p = torch.softmax(
                        probe(torch.stack([l224(o), l224(Image.open(cf))]).to(DEV)), -1
                    ).cpu()
                src = int(p[0].argmax())
                col = Image.new("RGB", (516, 260), "white")
                col.paste(o, (0, 2))
                col.paste(crop(pr, 1), (258, 2))
                cp = f"{stage}/{a}_{k:03d}.png"
                col.save(cp)
                colls.append(cp)
                srcs.append(src)
                confs.append(float(p[1][1 - src]))
            v = teacher.get_feedback(
                collage_path_list=colls,
                y_target_end_confidence_list=confs,
                y_source_list=srcs,
                y_list=srcs,
            )
            share = sum(1 for x in v if str(x).lower() == "false") / max(len(v), 1)
            judged[str(a)] = {
                "n": len(v),
                "false_share": share,
                "verdicts": [str(x) for x in v],
            }
            log(f"  atom {a}: false share {share:.2f} of {len(v)}")
    except Exception as e:
        log("LLM judging unavailable:", repr(e)[:200])
    json.dump(judged, open(judge_path, "w"), indent=1)

subset = raw.get("sweep_atom_subset") or []
conf_atoms = [int(a) for a, v in judged.items() if v.get("false_share", 0) >= 0.5]
if not conf_atoms and subset:
    conf_atoms = [int(subset[0])]
    log("falling back to the hypothesised confounder atom", conf_atoms)
conf_atoms = [a for a in conf_atoms if a in by_atom]
summary["confounder_atoms"] = conf_atoms
summary["judge"] = judged
log("confounder atoms:", conf_atoms)
if not conf_atoms:
    json.dump(summary, open(RUN + "/overnight_summary.json", "w"), indent=1)
    sys.exit("no confounder direction with verified flips -- nothing to repair")

# ---------------------------------------------------------------- 3. CFKD datasets (80 % train, 20 % held out)
items = []
for a in conf_atoms:
    for cf, pr in by_atom[a]:
        items.append({"atom": a, "cf": cf, "pair": pr})
with torch.no_grad():
    for it in items:
        o = crop(it["pair"], 0)
        p = torch.softmax(
            probe(torch.stack([l224(o), l224(Image.open(it["cf"]))]).to(DEV)), -1
        ).cpu()
        it["src"] = int(p[0].argmax())
        it["p_orig"] = p[0].tolist()
        it["p_cf"] = p[1].tolist()
items = [
    it for it in items if int(np.argmax(it["p_orig"])) != int(np.argmax(it["p_cf"]))
]
rng = np.random.RandomState(0)
idx = rng.permutation(len(items))
# A counterfactual validation split is not optional: model selection scores the
# repair against it (50:50 with the original val data), so with an empty one no
# epoch ever beats the untouched probe and nothing is saved. Hold out >= 1 and
# still leave >= 5 to train on, which needs at least 6 counterfactuals.
n_hold = max(1, int(0.2 * len(items)))
hold = [items[i] for i in idx[:n_hold]]
train = [items[i] for i in idx[n_hold:]]
log(
    f"counterfactuals: {len(items)} usable, {len(train)} train / {len(hold)} held out",
    "| source classes",
    Counter(i["src"] for i in items),
)
summary["n_counterfactuals"] = len(items)
if len(items) < 6 or len(train) < 5:
    summary["skipped"] = (
        f"only {len(items)} counterfactual(s) for the confounder direction(s) -- too few to repair"
    )
    log(summary["skipped"])
    json.dump(summary, open(RUN + "/overnight_summary.json", "w"), indent=1)
    print(
        "SUMMARY "
        + json.dumps(
            {k: v for k, v in summary.items() if k not in ("judge", "directions")}
        ),
        flush=True,
    )
    sys.exit(0)

NEW = RUN + "/cfkd_overnight"
if os.path.exists(NEW):
    shutil.rmtree(NEW)
os.makedirs(NEW + "/1")
cfgp = f"/tmp/cfkd_{PAIR}_overnight.yaml"
subprocess.run(
    [
        sys.executable,
        "utils/make_cfkd_on_directions_config.py",
        "--run",
        RUN,
        "--directions",
        "0",
        "--target_name",
        target_name,
        "--confounder_name",
        conf_name,
        "--subdir",
        "cfkd_overnight",
        "--out",
        cfgp,
    ],
    check=True,
    capture_output=True,
)
c = yaml.safe_load(open(cfgp))
c["base_dir"] = NEW
c["current_iteration"] = 0
c["mixing_ratio"] = 0.5
c["continuous_learning"] = "deep_feature_reweighting"
c["model_path"] = NEW + "/1/finetuned_model"
c["training"].update(
    dict(
        optimizer="adamw",
        learning_rate=1e-3,
        max_epochs=8,
        steps_per_epoch=100,
        dropout=0.0,
        train_batch_size=32,
        val_batch_size=4,
        test_batch_size=32,
        class_balanced=False,
    )
)
yaml.safe_dump(c, open(NEW + "/config.yaml", "w"), sort_keys=False, allow_unicode=True)

from peal.adaptors.didae import DiDAE, _unwrap_dataloader

ds = _unwrap_dataloader(DiDAE(adaptor_config=CFG_PATH).train_dataloader).dataset
for split, sel in (("train", train), ("validation", hold)):
    ds.serialize_dataset(
        output_dir=f"{NEW}/1/{split}_dataset",
        x_list=[l224(Image.open(i["cf"])) for i in sel],
        y_list=[i["src"] for i in sel],
        sample_names=[f"false_{i['src']}_{k}" for k, i in enumerate(sel)],
        classifier=probe,
    )

# ---------------------------------------------------------------- 4. DFR repair (library 50:50 validation)
from torch.utils.tensorboard import SummaryWriter
from peal.adaptors.counterfactual_knowledge_distillation import CFKD
from peal.training.trainers import calculate_test_accuracy

cfkd = CFKD(adaptor_config=NEW + "/config.yaml")
acc0 = calculate_test_accuracy(
    cfkd.student, cfkd.test_dataloader, cfkd.device, False, None, tracking_level=1
)
log("test accuracy before:", round(float(acc0), 4))
t0 = time.time()
cfkd.finetune_student(
    finetune_iteration=1,
    dataset_path=os.path.join(NEW, "1", "train_dataset"),
    writer=SummaryWriter(os.path.join(NEW, "logs")),
)
acc1 = calculate_test_accuracy(
    cfkd.student, cfkd.test_dataloader, cfkd.device, False, None, tracking_level=1
)
log(f"test accuracy after: {float(acc1):.4f}  ({time.time()-t0:.0f}s)")
summary["test_acc_before"], summary["test_acc_after"] = float(acc0), float(acc1)
summary["test_acc_gain"] = float(acc1) - float(acc0)

# ---------------------------------------------------------------- 5. evaluation
fine_path = NEW + "/1/finetuned_model/model.cpl"
if not os.path.exists(fine_path):
    summary["skipped"] = (
        "the repair never beat the untouched probe on the weighted validation set, "
        "so no checkpoint was written"
    )
    log(summary["skipped"])
    json.dump(summary, open(RUN + "/overnight_summary.json", "w"), indent=1)
    print(
        "SUMMARY "
        + json.dumps(
            {k: v for k, v in summary.items() if k not in ("judge", "directions")}
        ),
        flush=True,
    )
    sys.exit(0)
fine, _ = get_predictor(fine_path, device=DEV)
fine.eval()
with torch.no_grad():
    fb = fa = 0
    for it in hold:
        o = crop(it["pair"], 0)
        x = torch.stack([l224(o), l224(Image.open(it["cf"]))]).to(DEV)
        pb = torch.softmax(probe(x), -1).cpu().numpy()
        pa = torch.softmax(fine(x), -1).cpu().numpy()
        fb += int(np.argmax(pb[0]) != np.argmax(pb[1]))
        fa += int(np.argmax(pa[0]) != np.argmax(pa[1]))
        it["pa_cf"] = pa[1].tolist()
        it["pa_orig"] = pa[0].tolist()
log(
    f"held-out counterfactuals: confounder edit flips {fb}/{len(hold)} before, {fa}/{len(hold)} after"
)
(
    summary["heldout_flips_before"],
    summary["heldout_flips_after"],
    summary["n_heldout"],
) = (fb, fa, len(hold))

# natural validation images the original probe gets wrong and the repaired one gets right
data_cfg = load_yaml_config(
    yaml.safe_load(open(RUN + "/config.yaml"))["data"], DataConfig
)
_, val, test = get_datasets(data_cfg)
fixed, broke = [], 0
from torch.utils.data import DataLoader

for split_name, dset in (("val", val), ("test", test)):
    dl = DataLoader(dset, batch_size=32, shuffle=False, num_workers=2)
    with torch.no_grad():
        for xb, yb in dl:
            yb = yb[0] if isinstance(yb, (tuple, list)) else yb
            xb = xb.to(DEV)
            p0 = torch.softmax(probe(xb), -1).cpu().numpy()
            p1 = torch.softmax(fine(xb), -1).cpu().numpy()
            for j in range(xb.shape[0]):
                yt = int(yb[j])
                a0, a1 = int(np.argmax(p0[j])), int(np.argmax(p1[j]))
                if a0 != yt and a1 == yt:
                    fixed.append(
                        dict(
                            split=split_name,
                            true=yt,
                            p_before=p0[j].tolist(),
                            p_after=p1[j].tolist(),
                            x=xb[j].cpu().numpy(),
                        )
                    )
                elif a0 == yt and a1 != yt:
                    broke += 1
log(f"natural images repaired: {len(fixed)} (and {broke} newly broken)")
summary["natural_fixed"], summary["natural_broken"] = len(fixed), broke

ASSETS = f"{HERE}/fig1_assets_{PAIR}_{VARIANT}"
os.makedirs(ASSETS + "/fix_samples", exist_ok=True)
samples = []
for k, f in enumerate(sorted(fixed, key=lambda r: -max(r["p_after"]))[:2]):
    ip = f"{ASSETS}/fix_samples/{k}_natural.png"
    Image.fromarray(
        (np.transpose(f["x"], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
    ).save(ip)
    samples.append(
        dict(
            image=ip,
            true=f["true"],
            p_before=f["p_before"],
            p_after=f["p_after"],
            caption=f"held-out {cn[f['true']]}",
            natural=True,
        )
    )
if len(samples) < 2:  # fall back to counterfactual evidence
    for k, it in enumerate(
        [h for h in hold if int(np.argmax(h["p_orig"])) != int(np.argmax(h["p_cf"]))][
            : 2 - len(samples)
        ]
    ):
        ip = f"{ASSETS}/fix_samples/cf{k}.png"
        shutil.copyfile(it["cf"], ip)
        nm = next(
            (d["name"] for d in summary["directions"] if d["atom"] == it["atom"]),
            str(it["atom"]),
        )
        samples.append(
            dict(
                image=ip,
                cf_image=ip,
                true=it["src"],
                p_before=it["p_cf"],
                p_after=it["pa_cf"],
                caption=re.sub(r"\s*\(SAE #\d+\)", "", nm) or f"atom {it['atom']}",
                natural=False,
            )
        )
json.dump(samples, open(f"{ASSETS}/fix_samples/samples.json", "w"), indent=1)
summary["fix_panel_natural"] = bool(samples and samples[0].get("natural"))
json.dump(summary, open(RUN + "/overnight_summary.json", "w"), indent=1)
log("summary ->", RUN + "/overnight_summary.json")
print(
    "SUMMARY "
    + json.dumps(
        {k: v for k, v in summary.items() if k not in ("judge", "directions")}
    ),
    flush=True,
)
