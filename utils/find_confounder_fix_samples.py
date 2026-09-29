"""
Pick validation images that the ORIGINAL probe misclassifies and the CFKD-finetuned
probe classifies correctly -- the qualitative "fix" evidence for Figure 1 when no
confounder labels exist (ImageNet pairs).  Restrict with --true_class to the class
that carries the shortcut (fireboat vs lifeboat: lifeboats with a water jet are
called fireboat, so --true_class 1).

usage (inside appruns, GPU):
  python utils/find_confounder_fix_samples.py \
      --data_config $PEAL_RUNS/imagenet_probes/fireboat_vs_lifeboat_dinov3_linear/openai_clip_rae_guided_g2_didae_msae/config.yaml \
      --original  $PEAL_RUNS/imagenet_probes/fireboat_vs_lifeboat_dinov3_linear/model.cpl \
      --finetuned $PEAL_RUNS/imagenet_probes/fireboat_vs_lifeboat_dinov3_linear/openai_clip_rae_guided_g2_didae_msae/cfkd_finetuning_llmteacher/1/finetuned_model/model.cpl \
      --true_class 1 --n 6 \
      --out docs/paper_figures/fig1_assets_fireboat_g2/fix_samples
Writes <out>/<rank>_<key>.png (the original validation image, 224 px) and <out>/samples.json
with p(class) before / after for the figure script (SPEC["fix_samples"]).
"""

import argparse, json, os, shutil
import torch, yaml
from torch.utils.data import DataLoader

from peal.data.dataset_factory import get_datasets
from peal.global_utils import load_yaml_config
from peal.architectures.predictors import get_predictor
from peal.data.interfaces import DataConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_config",
        required=True,
        help="DiDAE/CFKD config.yaml (its `data` block is used) or a data yaml",
    )
    ap.add_argument("--original", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--true_class", type=int, default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--class_names", nargs="+", default=["fireboat", "lifeboat"])
    a = ap.parse_args()

    raw = yaml.safe_load(open(a.data_config))
    data_cfg = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    data_cfg = load_yaml_config(data_cfg, DataConfig)
    train, val, test = get_datasets(data_cfg)
    ds = val if a.split == "val" else test
    ds.enable_idx()
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    orig, _ = get_predictor(a.original, device=dev)
    orig.eval()
    fine, _ = get_predictor(a.finetuned, device=dev)
    fine.eval()

    hits = []
    with torch.no_grad():
        for x, (y, idx) in loader:
            x = x.to(dev)
            p0 = torch.softmax(orig(x), dim=1).cpu()
            p1 = torch.softmax(fine(x), dim=1).cpu()
            for j in range(x.shape[0]):
                yt = int(y[j])
                if a.true_class is not None and yt != a.true_class:
                    continue
                pred0, pred1 = int(p0[j].argmax()), int(p1[j].argmax())
                if pred0 != yt and pred1 == yt:
                    hits.append(
                        dict(
                            index=int(idx[j]),
                            key=ds.keys[int(idx[j])],
                            true=yt,
                            p_before=[round(float(v), 3) for v in p0[j]],
                            p_after=[round(float(v), 3) for v in p1[j]],
                        )
                    )
    # strongest fixes first: most confident wrong before, most confident right after
    hits.sort(key=lambda h: (h["p_before"][h["true"]], -h["p_after"][h["true"]]))
    hits = hits[: a.n]
    os.makedirs(a.out, exist_ok=True)
    img_dir = os.path.join(ds.root_dir, ds.config.x_selection)
    for r, h in enumerate(hits):
        src = os.path.join(img_dir, ds.image_name(h["key"]))
        dst = os.path.join(
            a.out, f"{r}_{os.path.basename(h['key']).rsplit('.', 1)[0]}.png"
        )
        try:
            from PIL import Image

            Image.open(src).convert("RGB").resize((224, 224)).save(dst)
        except Exception as exc:  # fall back to a raw copy
            print("resize failed", exc)
            shutil.copy(src, dst)
        h["image"] = dst
        h["label_before"] = a.class_names[int(torch.tensor(h["p_before"]).argmax())]
        h["label_after"] = a.class_names[int(torch.tensor(h["p_after"]).argmax())]
        print(
            r,
            h["key"],
            "before",
            h["label_before"],
            h["p_before"],
            "after",
            h["label_after"],
            h["p_after"],
        )
    json.dump(hits, open(os.path.join(a.out, "samples.json"), "w"), indent=1)
    print(
        f"{len(hits)} samples written to {a.out}/samples.json  (of {len(ds)} {a.split} images)"
    )


if __name__ == "__main__":
    main()
