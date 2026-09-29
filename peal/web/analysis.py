"""
Per-direction analysis shown on the feedback page of a web-demo job:

    python -m peal.web.analysis <job_dir>
    python -m peal.web.analysis --accuracy-only <job_dir>   # accuracy.json only

Writes ``<job>/analysis.json`` with

* the headline accuracy of the uploaded classifier on all uploaded images;
* one entry per ranked direction (``run/sweep_results.pt``) with its latent,
  ambient and verified flip counts, the verified factual/counterfactual pairs
  written to ``run/successful_flips`` (see its ``index.json``) and four group
  accuracies: class x whether the direction's SAE concept is active on the real
  image, plus their mean (average group accuracy) and minimum (worst group
  accuracy).

The concept split follows reproduction_scripts/overnight_group_eval.py: an image
has the concept when the MSAE code of the direction's atom (any of its atoms for
a concept-replacement pair) on the image's OpenAI CLIP ViT-L/14 embedding is
positive. Unlike that script, which evaluates a trained probe on its held-out
val+test split, this runs on every uploaded image: the uploaded classifier was
not trained on them by PEAL, and the upload is small.
"""

import json
import os
import re
import sys

#: CLIP's input normalization (openai/CLIP, ViT-L/14).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def direction_atoms(name):
    """SAE atom indices in a direction name, e.g. ``"squirting (SAE #5717)"``."""
    return [int(a) for a in re.findall(r"SAE #(\d+)", str(name))]


def group_accuracies(labels, correct, present):
    """Accuracy of the four (class, concept present) groups.

    Parameters
    ----------
    labels, correct, present : sequence
        Per image: true class (0 or 1), whether the prediction was right and
        whether the concept is active.

    Returns
    -------
    dict
        ``groups`` (list of ``{"class", "concept", "n", "accuracy"}``, empty
        groups with accuracy ``None``), ``average`` (mean over the non-empty
        groups) and ``worst`` (their minimum), both ``None`` without data.
    """
    groups = []
    for y in (0, 1):
        for has in (True, False):
            hits = [
                c
                for lab, c, p in zip(labels, correct, present)
                if lab == y and p == has
            ]
            groups.append(
                {
                    "class": y,
                    "concept": has,
                    "n": len(hits),
                    "accuracy": (sum(hits) / len(hits)) if hits else None,
                }
            )
    accs = [g["accuracy"] for g in groups if g["accuracy"] is not None]
    return {
        "groups": groups,
        "average": sum(accs) / len(accs) if accs else None,
        "worst": min(accs) if accs else None,
    }


def _flip_index(run_dir, job_dir):
    """``{direction_idx: {"n_verified", "pairs": [...]}}`` from successful_flips."""
    root = os.path.join(run_dir, "successful_flips")
    path = os.path.join(root, "index.json")
    if not os.path.isfile(path):
        return {}
    out = {}
    for entry in json.load(open(path)):
        folder = os.path.join(root, entry["folder"])
        out[int(entry["direction_idx"])] = {
            "n_verified": entry.get("n_verified"),
            "pairs": [
                {
                    **p,
                    "image": os.path.relpath(os.path.join(folder, p["file"]), job_dir),
                }
                for p in entry["pairs"]
            ],
        }
    return out


def _predictions_and_codes(job_dir, atoms, device, after_model=None):
    """Labels, predictions and MSAE codes of ``atoms`` for every uploaded image.

    With ``atoms=None`` only labels and predictions are computed (no CLIP, no
    dictionary) and the codes are empty lists. With ``after_model`` (a path)
    its predictions are returned as a fourth list, else ``None``.
    """
    import clip
    import torch
    from torch.utils.data import DataLoader

    from peal.architectures.predictors import get_predictor
    from peal.data.dataset_factory import get_datasets
    from peal.data.interfaces import DataConfig
    from peal.global_utils import load_yaml_config
    from peal.sparse_dictionaries.msae_decomposition import (
        MSAEDecomposition,
        MSAEDecompositionConfig,
    )
    from peal.web.paths import get_project_resource_dir

    model, _ = get_predictor(os.path.join(job_dir, "model.onnx"), device=device)
    if hasattr(model, "eval"):
        model.eval()
    model_after = None
    if after_model is not None:
        model_after, _ = get_predictor(after_model, device=device)
        if hasattr(model_after, "eval"):
            model_after.eval()
    preds_after = [] if model_after is not None else None
    with_codes = atoms is not None
    sd = (
        None
        if not with_codes
        else MSAEDecomposition(
            load_yaml_config(
                os.path.join(
                    get_project_resource_dir(),
                    "configs/didae_experiments/sparse_dictionaries/msae_decomposition.yaml",
                ),
                MSAEDecompositionConfig,
            )
        )
    )
    if with_codes:
        clip_model, _ = clip.load("ViT-L/14", device=device)
        clip_model.eval()
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    data_cfg = load_yaml_config(os.path.join(job_dir, "data.yaml"), DataConfig)
    labels, preds, codes = [], [], []
    for dset in get_datasets(data_cfg):
        if dset is None or len(dset) == 0:
            continue
        for xb, yb in DataLoader(dset, batch_size=64, shuffle=False, num_workers=4):
            yb = yb[0] if isinstance(yb, (tuple, list)) else yb
            yb = yb.view(yb.shape[0], -1)[:, 0] if yb.dim() > 1 else yb
            xb = xb.to(device)
            with torch.no_grad():
                preds.extend(model(xb).argmax(-1).cpu().tolist())
                if model_after is not None:
                    preds_after.extend(model_after(xb).argmax(-1).cpu().tolist())
            labels.extend(int(v) for v in yb.tolist())
            if not with_codes:
                codes.extend([] for _ in range(xb.shape[0]))
                continue
            with torch.no_grad():
                x01 = dset.project_to_pytorch_default(xb).clamp(0, 1)
                x224 = torch.nn.functional.interpolate(
                    x01, size=(224, 224), mode="bicubic", align_corners=False
                ).clamp(0, 1)
                z = clip_model.encode_image((x224 - mean) / std).float()
                c = sd.encode(z).detach().float().cpu()
            codes.extend(c[:, atoms].tolist() if atoms else [[] for _ in range(len(c))])
    return labels, preds, codes, preds_after


def _from_cache(job_dir, atoms, device, after_layer=None):
    """``_predictions_and_codes`` from ``peal.web.cache``, or ``None`` when the
    cache (or a part it would need) is missing. ``after_layer`` is the DFR
    layer ``(W1, b1)`` applied to the cached penultimate features."""
    import numpy as np

    from peal.web import cache

    c = cache.load(job_dir)
    if c is None or (atoms is not None and "clip" not in c):
        return None
    if after_layer is not None and "feats" not in c:
        return None
    labels = c["labels"].astype(int).tolist()
    preds = c["logits"].argmax(1).tolist()
    preds_after = None
    if after_layer is not None:
        W1, b1 = after_layer
        preds_after = (c["feats"] @ W1.T + b1).argmax(1).tolist()
    if atoms is None:
        return labels, preds, [[] for _ in labels], preds_after
    import torch

    from peal.global_utils import load_yaml_config
    from peal.sparse_dictionaries.msae_decomposition import (
        MSAEDecomposition,
        MSAEDecompositionConfig,
    )
    from peal.web.paths import get_project_resource_dir

    sd = MSAEDecomposition(
        load_yaml_config(
            os.path.join(
                get_project_resource_dir(),
                "configs/didae_experiments/sparse_dictionaries/msae_decomposition.yaml",
            ),
            MSAEDecompositionConfig,
        )
    )
    codes = []
    with torch.no_grad():
        for start in range(0, len(labels), 1024):
            z = torch.from_numpy(np.ascontiguousarray(c["clip"][start : start + 1024]))
            code = sd.encode(z.to(device)).detach().float().cpu()
            codes.extend(
                code[:, atoms].tolist() if atoms else [[] for _ in range(len(code))]
            )
    return labels, preds, codes, preds_after


def _dfr_layer(job_dir):
    """``(W1, b1)`` written by ``peal.web.dfr`` next to the corrected model."""
    import numpy as np

    path = os.path.join(job_dir, "run", "dfr_layer.npz")
    if not os.path.isfile(path):
        return None
    with np.load(path) as z:
        return z["weight"], z["bias"]


def quick_accuracy(job_dir, device=None):
    """Write ``<job_dir>/accuracy.json``: the classifier's accuracy on all
    uploaded images, overall and per class, before DiDAE starts. It is the
    first thing the job page shows, so a wrong normalization, input size or
    output order shows up in seconds instead of after the sweep."""
    import torch

    job_dir = os.path.abspath(job_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # One pass that also caches features and CLIP embeddings for the later
    # steps (peal.web.cache); the plain pass is the fallback.
    got = None
    try:
        from peal.web import cache

        cache.build(job_dir, device=device)
        got = _from_cache(job_dir, None, device)
    except Exception as exc:
        print(f"[analysis] cache build failed, plain pass: {type(exc).__name__}: {exc}")
    labels, preds, _, _ = got or _predictions_and_codes(job_dir, None, device)
    per_class = {}
    for y in sorted(set(labels)):
        hits = [p == y for p, lab in zip(preds, labels) if lab == y]
        per_class[str(y)] = {"n": len(hits), "accuracy": sum(hits) / len(hits)}
    result = {
        "n_images": len(labels),
        "accuracy": (
            (sum(p == y for p, y in zip(preds, labels)) / len(labels))
            if labels
            else None
        ),
        "per_class": per_class,
        "predicted_share": {
            str(c): preds.count(c) / len(preds) for c in sorted(set(preds))
        },
    }
    tmp = os.path.join(job_dir, ".accuracy.json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, os.path.join(job_dir, "accuracy.json"))
    return result


def analyse(job_dir, device=None):
    """Write ``<job_dir>/analysis.json`` (see the module docstring) and return it."""
    import torch

    job_dir = os.path.abspath(job_dir)
    run_dir = os.path.join(job_dir, "run")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sweep = torch.load(
        os.path.join(run_dir, "sweep_results.pt"),
        map_location="cpu",
        weights_only=False,
    )
    flips = _flip_index(run_dir, job_dir)
    atoms_by_dir = {
        int(r["direction_idx"]): direction_atoms(r.get("dimension_name", ""))
        for r in sweep
    }
    all_atoms = sorted({a for atoms in atoms_by_dir.values() for a in atoms})
    col = {a: i for i, a in enumerate(all_atoms)}
    corrected = os.path.join(run_dir, "model.onnx")
    after = (
        corrected
        if os.path.isfile(os.path.join(job_dir, "dfr.json"))
        and os.path.isfile(corrected)
        else None
    )
    # The cached path only applies the dictionary to stored CLIP embeddings,
    # seconds on the CPU. It runs while DiDAE waits for verdicts and still
    # holds most of the GPU, so it stays off the GPU.
    got = _from_cache(
        job_dir,
        all_atoms,
        "cpu",
        after_layer=_dfr_layer(job_dir) if after is not None else None,
    )
    if got is None or (after is not None and got[3] is None):
        got = _predictions_and_codes(job_dir, all_atoms, device, after_model=after)
    labels, preds, codes, preds_after = got
    correct = [int(p == y) for p, y in zip(preds, labels)]
    correct_after = (
        [int(p == y) for p, y in zip(preds_after, labels)]
        if preds_after is not None
        else None
    )

    directions = []
    for rank, r in enumerate(sweep):
        idx = int(r["direction_idx"])
        atoms = atoms_by_dir[idx]
        present = [
            any(code[col[a]] > 0 for a in atoms) if atoms else False for code in codes
        ]
        entry = {
            "rank": rank + 1,
            "direction_idx": idx,
            "name": str(r.get("dimension_name", f"direction {idx}")),
            "atoms": atoms,
            "latent_flips": r.get("latent_flip_count"),
            "ambient_flips": r.get("ambient_flip_count"),
            "verified_flips": r.get("success_count"),
            "total_attempted": r.get("total_attempted"),
            "n_concept_present": sum(present),
            "pairs": flips.get(idx, {}).get("pairs", []),
        }
        entry.update(group_accuracies(labels, correct, present))
        if correct_after is not None:
            entry["after"] = group_accuracies(labels, correct_after, present)
        directions.append(entry)

    result = {
        "n_images": len(labels),
        "accuracy": (sum(correct) / len(correct)) if correct else None,
        "accuracy_after": (
            sum(correct_after) / len(correct_after) if correct_after else None
        ),
        "directions": directions,
    }
    tmp = os.path.join(job_dir, ".analysis.json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, os.path.join(job_dir, "analysis.json"))
    return result


if __name__ == "__main__":
    if sys.argv[1] == "--accuracy-only":
        print(json.dumps(quick_accuracy(sys.argv[2])))
        sys.exit(0)
    r = analyse(sys.argv[1])
    print(
        json.dumps(
            {
                "n_images": r["n_images"],
                "accuracy": r["accuracy"],
                "n_directions": len(r["directions"]),
            }
        )
    )
