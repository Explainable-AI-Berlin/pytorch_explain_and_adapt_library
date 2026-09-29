"""
Per-image cache of a web-demo job, built in ONE pass over the upload:

    python -m peal.web.cache <job_dir>

The uploaded classifier is the slow part of a job (an onnx2torch-converted
ViT runs at ~90 img/s on an RTX 3090), and before this cache the accuracy
check, step 1 of DiDAE, the DFR correction and the per-direction analysis
(twice) each pushed every image through it again. Now the accuracy check
builds ``<job>/cache/images.npz`` and the others read from it:

``keys``
    image key (``dataset.keys``) of every image, in the order below;
``split``
    0 / 1 / 2 for the train / val / test split of ``data.yaml``;
``labels``
    class index;
``logits``
    the classifier's two outputs;
``feats``
    its penultimate features (the input of the final linear layer that DFR
    refits), present only when ``find_final_linear`` finds that layer; the
    logits are then computed from them (checked against the full model);
``clip``
    OpenAI CLIP ViT-L/14 image embedding (the space of the MSAE dictionary),
    as ``peal.web.analysis`` computes it.
"""

import os
import sys

import numpy as np

CACHE_REL = os.path.join("cache", "images.npz")
#: CLIP's input normalization (openai/CLIP, ViT-L/14).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
NUM_WORKERS = int(os.environ.get("PEAL_WEB_NUM_WORKERS", "8"))


def cache_path(job_dir):
    """Path of the cache file of ``job_dir``."""
    return os.path.join(os.path.abspath(job_dir), CACHE_REL)


def load(job_dir):
    """The cache of ``job_dir`` as a dict of arrays, or ``None`` if not built."""
    path = cache_path(job_dir)
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def final_layer(model_path, info):
    """Weight ``[2, F]`` and bias ``[2]`` of the final linear layer found by
    ``find_final_linear``, restricted to the model's two selected outputs."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(model_path).graph
    inits = {i.name: i for i in graph.initializer}
    W = numpy_helper.to_array(inits[info["weight"]]).astype(np.float32)
    rows = info["selected"] or list(range(info["n_classes"]))
    W0 = np.stack(
        [W[c] if info["weight_layout"] == "class_major" else W[:, c] for c in rows]
    )
    b0 = (
        numpy_helper.to_array(inits[info["bias"]]).astype(np.float32).reshape(-1)[rows]
        if info["bias"]
        else np.zeros(len(rows), np.float32)
    )
    return W0, b0


def _labels(yb):
    yb = yb[0] if isinstance(yb, (tuple, list)) else yb
    return yb.view(yb.shape[0], -1)[:, 0] if yb.dim() > 1 else yb


def build(job_dir, device=None, with_clip=True, log=print):
    """Compute and write the cache (see the module docstring); returns it."""
    import torch
    from torch.utils.data import DataLoader

    from peal.architectures.onnx_predictor import (
        find_final_linear,
        truncate_to_features,
    )
    from peal.architectures.predictors import get_predictor
    from peal.data.dataset_factory import get_datasets
    from peal.data.interfaces import DataConfig
    from peal.global_utils import load_yaml_config

    job_dir = os.path.abspath(job_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(job_dir, "model.onnx")
    os.makedirs(os.path.join(job_dir, "cache"), exist_ok=True)

    model, _ = get_predictor(model_path, device=device)
    if hasattr(model, "eval"):
        model.eval()
    feature_model, W0, b0 = None, None, None
    try:
        info = find_final_linear(model_path)
        if info is not None:
            feat_path = os.path.join(job_dir, "cache", "features.onnx")
            truncate_to_features(model_path, feat_path, info["feature_tensor"])
            feature_model, _ = get_predictor(feat_path, device=device)
            if hasattr(feature_model, "eval"):
                feature_model.eval()
            W0, b0 = final_layer(model_path, info)
            W0_t = torch.from_numpy(W0).to(device)
            b0_t = torch.from_numpy(b0).to(device)
    except Exception as exc:  # no features: logits from the full model only
        log(f"[cache] no penultimate features ({type(exc).__name__}: {exc})")
        feature_model = None

    clip_model = None
    if with_clip:
        import clip

        clip_model, _ = clip.load("ViT-L/14", device=device)
        clip_model.eval()
        mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    data_cfg = load_yaml_config(os.path.join(job_dir, "data.yaml"), DataConfig)
    out = {k: [] for k in ("keys", "split", "labels", "logits", "feats", "clip")}
    checked = False
    for split, dset in enumerate(get_datasets(data_cfg)):
        if dset is None or len(dset) == 0:
            continue
        keys = list(dset.keys)
        pos = 0
        loader = DataLoader(dset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
        for xb, yb in loader:
            n = xb.shape[0]
            out["keys"].extend(keys[pos : pos + n])
            pos += n
            out["split"].append(np.full(n, split, np.int8))
            out["labels"].append(_labels(yb).long().numpy())
            xb = xb.to(device)
            with torch.no_grad():
                if feature_model is not None:
                    f = feature_model(xb).float()
                    f = f.reshape(n, -1)
                    logits = f @ W0_t.T + b0_t
                    if not checked:
                        # the final layer must really produce the model output
                        full = model(xb).float()
                        ok = full.shape == logits.shape and torch.allclose(
                            full, logits, atol=1e-3, rtol=1e-3
                        )
                        if not ok:
                            log(
                                "[cache] final layer does not reproduce the model "
                                "output; caching logits of the full model only"
                            )
                            feature_model = None
                            logits = full
                        checked = True
                    if feature_model is not None:
                        out["feats"].append(f.cpu().numpy())
                else:
                    logits = model(xb).float()
                out["logits"].append(logits.cpu().numpy())
                if clip_model is not None:
                    x01 = dset.project_to_pytorch_default(xb).clamp(0, 1)
                    x224 = torch.nn.functional.interpolate(
                        x01, size=(224, 224), mode="bicubic", align_corners=False
                    ).clamp(0, 1)
                    z = clip_model.encode_image((x224 - mean) / std).float()
                    out["clip"].append(z.cpu().numpy())
        if pos != len(keys):
            raise RuntimeError(f"split {split}: {pos} images loaded, {len(keys)} keys")

    arrays = {
        "keys": np.array(out["keys"], dtype=str),
        "split": np.concatenate(out["split"]),
        "labels": np.concatenate(out["labels"]),
        "logits": np.concatenate(out["logits"]),
    }
    # features are only kept when every batch produced them
    if feature_model is not None and out["feats"]:
        arrays["feats"] = np.concatenate(out["feats"])
    if out["clip"]:
        arrays["clip"] = np.concatenate(out["clip"])
    path = cache_path(job_dir)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return arrays


if __name__ == "__main__":
    c = build(sys.argv[1])
    print({k: v.shape for k, v in c.items()})
