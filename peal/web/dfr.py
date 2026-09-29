"""
Last-layer correction (DFR, deep feature reweighting) of a web-demo job:

    python -m peal.web.dfr <job_dir>

The network stays frozen; only its final linear layer (found by
``peal.architectures.onnx_predictor.find_final_linear``) is refitted, as a
regularised logistic regression on the penultimate features of

* the uploaded images with their class labels, and
* the verified counterfactuals of the directions the user judged: a "false"
  (spurious) direction's counterfactual keeps its ORIGINAL class, since that
  edit must not change the prediction; a "true" direction's counterfactual
  gets its TARGET class.

Three convex solvers (``solver``), each with its regularisation chosen by
stratified 5-fold cross-validation over the originals and counterfactuals
(scored by the mean of the balanced accuracy on the original images and the
accuracy on the counterfactuals of the held-out fold):

* ``logistic``: L2 logistic regression, as in the DFR paper (Kirichenko et
  al., 2023);
* ``svm``: soft-margin linear SVM (hinge loss);
* ``ridge``: least-squares classifier on +-1 labels, the closed-form option.

The SVM and ridge scores are not probabilities, so a 1-D Platt scaling fitted
on their cross-validated scores turns them into calibrated logits; it is folded
into the same linear layer. Writes ``run/model.onnx`` (the corrected model;
the uncorrected one stays as ``run/model_original.onnx``) and ``dfr.json``
with accuracies and counterfactual consistency before and after.
"""

import json
import os
import re
import shutil
import sys

import numpy as np

#: Candidate inverse regularisation strengths (ridge: alpha = 1 / C).
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SOLVERS = ("logistic", "svm", "ridge")
#: Total weight of the counterfactuals relative to the original images.
CF_WEIGHT = 0.5


def _feedback(run_dir):
    """``{direction_idx: "true" | "false" | ...}`` from ``direction_feedback.txt``."""
    out = {}
    path = os.path.join(run_dir, "direction_feedback.txt")
    if os.path.isfile(path):
        for line in open(path):
            m = re.match(r"direction=(\d+), feedback=(\w+)", line.strip())
            if m:
                out[int(m.group(1))] = m.group(2)
    return out


def _counterfactuals(run_dir, feedback):
    """Paths, labels and kinds of the verified counterfactuals of judged directions."""
    root = os.path.join(run_dir, "successful_flips")
    index = os.path.join(root, "index.json")
    if not os.path.isfile(index):
        return []
    out = []
    for entry in json.load(open(index)):
        verdict = feedback.get(int(entry["direction_idx"]))
        if verdict not in ("true", "false"):
            continue
        for p in entry["pairs"]:
            path = os.path.join(
                root, entry["folder"], p["file"].replace("_pair", "_cf", 1)
            )
            if not os.path.isfile(path):
                continue
            label = p["orig_class"] if verdict == "false" else p["target_class"]
            out.append(
                {
                    "path": path,
                    "label": int(label),
                    "kind": verdict,
                    "direction_idx": int(entry["direction_idx"]),
                }
            )
    return out


def _features(feature_model, batches, device):
    import torch

    feats = []
    with torch.no_grad():
        for xb in batches:
            f = feature_model(xb.to(device))
            feats.append(f.reshape(f.shape[0], -1).float().cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, 0), np.float32)


def correct(job_dir, device=None, solver="logistic"):
    """Refit the final layer of ``<job_dir>/model.onnx``; see the module docstring."""
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import LinearSVC

    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    from torch.utils.data import DataLoader

    from peal.architectures.onnx_predictor import (
        find_final_linear,
        truncate_to_features,
        write_final_linear,
    )
    from peal.architectures.predictors import get_predictor
    from peal.data.dataset_factory import get_datasets
    from peal.data.interfaces import DataConfig
    from peal.global_utils import load_yaml_config
    from peal.web import cache
    from peal.web.cache import final_layer

    job_dir = os.path.abspath(job_dir)
    run_dir = os.path.join(job_dir, "run")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(job_dir, "model.onnx")
    info = find_final_linear(model_path)
    if info is None:
        raise RuntimeError(
            "no final linear layer found in model.onnx; DFR is not available"
        )
    feedback = _feedback(run_dir)
    cfs = _counterfactuals(run_dir, feedback)
    n_false = sum(c["kind"] == "false" for c in cfs)

    # frozen feature extractor: the graph cut at the final layer's input
    feat_path = os.path.join(run_dir, "dfr_features.onnx")
    truncate_to_features(model_path, feat_path, info["feature_tensor"])
    feature_model, _ = get_predictor(feat_path, device=device)
    if hasattr(feature_model, "eval"):
        feature_model.eval()

    # current weights of the two selected classes, [2, F] and [2]
    W0, b0 = final_layer(model_path, info)
    if len(W0) != 2:
        raise RuntimeError(f"DFR expects a two-output model, found {len(W0)} outputs")

    data_cfg = load_yaml_config(os.path.join(job_dir, "data.yaml"), DataConfig)
    splits = {}
    datasets = get_datasets(data_cfg)
    # penultimate features of the uploaded images: from the job's cache
    # (peal.web.cache, built during the accuracy check) when it has them,
    # else one pass through the frozen feature extractor
    cached = cache.load(job_dir)
    if cached is not None and "feats" in cached:
        for sid, name in enumerate(("train", "val", "test")):
            m = cached["split"] == sid
            if m.any():
                splits[name] = (
                    cached["feats"][m],
                    cached["labels"][m].astype(np.int64),
                )
    else:
        for name, dset in zip(("train", "val", "test"), datasets):
            if dset is None or len(dset) == 0:
                continue
            xs, ys = [], []
            for xb, yb in DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=cache.NUM_WORKERS
            ):
                yb = yb[0] if isinstance(yb, (tuple, list)) else yb
                yb = yb.view(yb.shape[0], -1)[:, 0] if yb.dim() > 1 else yb
                xs.append(xb)
                ys.append(yb.long().numpy())
            splits[name] = (_features(feature_model, xs, device), np.concatenate(ys))
    ref = datasets[0]

    # counterfactual images: [0, 1] PNGs at generator resolution -> the
    # classifier's input size and normalization
    h, w = data_cfg.input_size[1], data_cfg.input_size[2]
    cf_batches = []
    for start in range(0, len(cfs), 64):
        xb = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        Image.open(c["path"])
                        .convert("RGB")
                        .resize((w, h), Image.BILINEAR),
                        dtype=np.float32,
                    )
                    / 255.0
                ).permute(2, 0, 1)
                for c in cfs[start : start + 64]
            ]
        )
        cf_batches.append(ref.project_from_pytorch_default(xb))
    Xc = _features(feature_model, cf_batches, device) if cfs else None
    yc = np.array([c["label"] for c in cfs], dtype=np.int64)
    kind = np.array([c["kind"] for c in cfs])

    Xtr, ytr = splits["train"]
    Xva, yva = splits.get("val", (Xtr[:0], ytr[:0]))
    # fit on train + val originals and all counterfactuals; test stays held out
    X_orig = np.concatenate([Xtr, Xva]) if len(yva) else Xtr
    y_orig = np.concatenate([ytr, yva]) if len(yva) else ytr
    mu, sd = X_orig.mean(0), X_orig.std(0) + 1e-6
    n_o = len(y_orig)
    X_all = np.concatenate([X_orig, Xc]) if cfs else X_orig
    X_all = (X_all - mu) / sd
    y_all = np.concatenate([y_orig, yc]) if cfs else y_orig
    is_cf = np.arange(len(y_all)) >= n_o
    # strata: class x (original / counterfactual), so every fold has both
    strata = y_all * 2 + is_cf

    def weights(mask_cf):
        n_cf = int(mask_cf.sum())
        w = np.ones(len(mask_cf))
        if n_cf:
            w[mask_cf] = CF_WEIGHT * (len(mask_cf) - n_cf) / n_cf
        return w

    def fit_raw(C, X, y, sw):
        """Linear score f(x) = coef . x + icpt (standardised features)."""
        if solver == "logistic":
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        elif solver == "svm":
            clf = LinearSVC(C=C, loss="hinge", class_weight="balanced", max_iter=20000)
        else:
            clf = RidgeClassifier(alpha=1.0 / C, class_weight="balanced")
        clf.fit(X, y, sample_weight=sw)
        return clf.coef_.reshape(-1).astype(np.float64), float(
            np.ravel(clf.intercept_)[0]
        )

    def score_of(pred, y, cf_mask):
        o = ~cf_mask
        bal = np.mean(
            [(pred[o & (y == c)] == c).mean() for c in (0, 1) if (o & (y == c)).any()]
        )
        cf = (pred[cf_mask] == y[cf_mask]).mean() if cf_mask.any() else bal
        return float((bal + cf) / 2)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores, oof = [], {}
    for C in C_GRID:
        f_oof = np.zeros(len(y_all))
        for tr, te in folds.split(X_all, strata):
            coef, icpt = fit_raw(C, X_all[tr], y_all[tr], weights(is_cf[tr]))
            f_oof[te] = X_all[te] @ coef + icpt
        oof[C] = f_oof
        pred = (f_oof > 0).astype(int)
        scores.append({"C": C, "cv_score": score_of(pred, y_all, is_cf)})
    best = max(scores, key=lambda s: s["cv_score"])
    # out-of-fold consistency on the counterfactuals: an honest estimate, the
    # "after" metrics below include counterfactuals the fit has seen
    oof_pred = (oof[best["C"]] > 0).astype(int)
    cv_consistency = {
        f"{k}_cf_consistency": float(
            (oof_pred[is_cf][kind == k] == yc[kind == k]).mean()
        )
        for k in ("false", "true")
        if cfs and (kind == k).any()
    }
    coef, icpt = fit_raw(best["C"], X_all, y_all, weights(is_cf))
    # Platt scaling: calibrated logit = a * f + c, fitted on the out-of-fold
    # scores (logistic regression is already calibrated: a = 1, c = 0)
    a, c = 1.0, 0.0
    if solver != "logistic":
        platt = LogisticRegression(C=1e6, max_iter=5000)
        platt.fit(oof[best["C"]].reshape(-1, 1), y_all, sample_weight=weights(is_cf))
        a, c = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    coef_x = a * coef / sd
    icpt_x = a * (icpt - (coef * mu / sd).sum()) + c
    # symmetric two-class logits whose softmax equals the sigmoid of the score
    W1 = np.stack([-coef_x / 2, coef_x / 2]).astype(np.float32)
    b1 = np.array([-icpt_x / 2, icpt_x / 2], dtype=np.float32)

    def predict(Wm, bm, X):
        return (X @ Wm.T + bm).argmax(1)

    def metrics(Wm, bm):
        out = {}
        for name, (X, y) in splits.items():
            p = predict(Wm, bm, X)
            out[f"{name}_accuracy"] = float((p == y).mean())
            out[f"{name}_per_class"] = {
                str(c): float((p[y == c] == c).mean()) for c in (0, 1) if (y == c).any()
            }
        if cfs:
            p = predict(Wm, bm, Xc)
            for k in ("false", "true"):
                m = kind == k
                if m.any():
                    out[f"{k}_cf_consistency"] = float((p[m] == yc[m]).mean())
        return out

    result = {
        "method": "dfr",
        "solver": solver,
        "platt": [a, c],
        "layer": {k: info[k] for k in ("node", "op", "n_features", "selected")},
        "n_counterfactuals": len(cfs),
        "n_false_counterfactuals": int(n_false),
        "false_directions": sorted(d for d, v in feedback.items() if v == "false"),
        "cv": scores,
        "cv_after": cv_consistency,
        "C": best["C"],
        "before": metrics(W0, b0),
        "after": metrics(W1, b1),
    }
    out_dir = run_dir
    original = os.path.join(out_dir, "model_original.onnx")
    if not os.path.isfile(original):
        shutil.copy(model_path, original)
    write_final_linear(model_path, os.path.join(out_dir, "model.onnx"), info, W1, b1)
    # the corrected layer for peal.web.analysis (applied to the cached features)
    np.savez(os.path.join(out_dir, "dfr_layer.npz"), weight=W1, bias=b1)
    os.remove(feat_path)
    with open(os.path.join(job_dir, "dfr.json"), "w") as f:
        json.dump(result, f, indent=1)
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    solver = "logistic"
    if args and args[0].startswith("--solver="):
        solver = args.pop(0).split("=", 1)[1]
    r = correct(args[0], solver=solver)
    print(
        json.dumps(
            {k: r[k] for k in ("solver", "C", "n_counterfactuals", "before", "after")},
            indent=1,
        )
    )
