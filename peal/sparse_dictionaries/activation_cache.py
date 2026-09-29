"""Optional on-disk cache for encoder activations.

Pushing the 182k CelebA images through the semantic encoder takes ~25 minutes
and produces the same 768-d vectors every time, which makes a hyperparameter
sweep over the dictionary mostly a sweep over image decoding. Caching the
activations turns a run into ~2 minutes of dictionary fitting plus evaluation.

Opt in with ``PEAL_ACT_CACHE=1``. It is off by default on purpose: silently
reading a stale checkpoint's numbers has cost this project time before, and a
cache that is on by default is one more way for that to happen. The key below
covers the dataset identity and the encoder weights' mtime and size, so a
retrained encoder or a changed split misses rather than returns stale vectors —
but only an explicit opt-in makes the trade-off visible in the command line.
"""

import hashlib
import json
import os

import torch
from peal.log import get_logger

_log = get_logger(__name__)


def _encoder_fingerprint(generator) -> dict:
    """Whatever identifies the weights the activations came from."""
    out = {}
    base = getattr(getattr(generator, "config", None), "base_path", None)
    if base:
        for rel in ("square64_ddim/final.ckpt", "square64_ddim/last.ckpt"):
            path = os.path.join(base, rel)
            if os.path.exists(path):
                st = os.stat(path)
                out[rel] = [int(st.st_mtime), int(st.st_size)]
    cfg = getattr(generator, "config", None)
    out["model_type"] = getattr(cfg, "model_type", None)
    out["encoder_path"] = getattr(cfg, "encoder_path", None)
    # Two runs over the same dataset with different encoders must not share a
    # cache entry. set_encoder copies `encoder` into model_type, but keying on
    # both means the distinction does not depend on that happening first.
    out["encoder"] = getattr(cfg, "encoder", None)
    out["encoder_input_space"] = getattr(cfg, "encoder_input_space", None)
    out["encoder_dimensions"] = getattr(
        getattr(generator, "config", None), "encoder_dimensions", None
    )
    return out


def cache_path(generator, data_config) -> str:
    """Path of the cache file for this generator/dataset combination.

    The file name is ``acts_<sha1[:16]>.pt`` under
    ``<generator.config.base_path>/activation_cache``; the digest covers the
    encoder fingerprint and every ``data_config`` field that changes which
    images are encoded or how they are preprocessed.

    Parameters
    ----------
    generator : InvertibleGenerator
        Generator whose encoder produces the activations; only its ``config``
        is inspected.
    data_config : DataConfig
        Dataset configuration; missing fields count as ``None`` in the key.

    Returns
    -------
    str
        Absolute or base-path-relative path of the ``.pt`` cache file.
    """
    key = {
        "encoder": _encoder_fingerprint(generator),
        "dataset_class": getattr(data_config, "dataset_class", None),
        "dataset_path": getattr(data_config, "dataset_path", None),
        "label_rel_path": getattr(data_config, "label_rel_path", None),
        "num_samples": getattr(data_config, "num_samples", None),
        "split": getattr(data_config, "split", None),
        "input_size": getattr(data_config, "input_size", None),
        "normalization": getattr(data_config, "normalization", None),
        "x_selection": getattr(data_config, "x_selection", None),
        "set_negative_to_zero": getattr(data_config, "set_negative_to_zero", None),
        "crop_size": getattr(data_config, "crop_size", None),
        "downsize": getattr(data_config, "downsize", None),
    }
    digest = hashlib.sha1(
        json.dumps(key, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return os.path.join(
        generator.config.base_path, "activation_cache", f"acts_{digest}.pt"
    )


def enabled() -> bool:
    """Whether ``PEAL_ACT_CACHE`` is set to anything but ``0``/``false``/empty."""
    return os.environ.get("PEAL_ACT_CACHE", "0") not in ("0", "", "false", "False")


def extract(datasets, feature_extractor, device, batch_size=256, num_workers=4):
    """Encode every dataset, returning one (X, Y) pair per dataset.

    Parameters
    ----------
    datasets : list of torch.utils.data.Dataset
        Datasets to encode, each yielding ``x`` or ``(x, y, ...)`` items.
    feature_extractor : callable
        Maps a batch of inputs on ``device`` to activations ``[B, act_size]``.
    device : str or torch.device
        Device the inputs are moved to before encoding.
    batch_size : int
        DataLoader batch size.
    num_workers : int
        DataLoader worker processes.

    Returns
    -------
    list of tuple
        ``[(X, Y), ...]`` with ``X`` a float CPU tensor ``[N, act_size]`` and
        ``Y`` the concatenated labels, or ``None`` when the dataset yields no
        label. Progress is printed every 100 batches.
    """
    out = []
    for i, dataset in enumerate(datasets):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        xs, ys = [], []
        with torch.no_grad():
            for b, batch in enumerate(loader):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                xs.append(feature_extractor(x.to(device)).detach().float().cpu())
                if isinstance(batch, (list, tuple)) and len(batch) > 1:
                    ys.append(batch[1].detach().cpu())
                if b % 100 == 0:
                    _log.info(
                        "%s", f"  [activations] split {i}: batch {b}/{len(loader)}"
                    )
        out.append((torch.cat(xs), torch.cat(ys) if ys else None))
    return out


def load_or_extract(generator, datasets, data_config, feature_extractor, device, **kw):
    """The cached form of :func:`extract`, when PEAL_ACT_CACHE says so.

    With the cache disabled this is exactly :func:`extract`. Otherwise the
    file from :func:`cache_path` is reused when its stored dataset sizes match
    ``[len(d) for d in datasets]``; on a miss or size mismatch the activations
    are extracted, written to that file and returned.

    Parameters
    ----------
    generator : InvertibleGenerator
        Used only to derive the cache path.
    datasets : list of torch.utils.data.Dataset
        Datasets to encode.
    data_config : DataConfig
        Used only to derive the cache path.
    feature_extractor : callable
        Encoder applied to each batch.
    device : str or torch.device
        Device for the encoder inputs.
    **kw
        Forwarded to :func:`extract` (``batch_size``, ``num_workers``).

    Returns
    -------
    list of tuple
        Same layout as :func:`extract`.
    """
    if not enabled():
        return extract(datasets, feature_extractor, device, **kw)

    path = cache_path(generator, data_config)
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("sizes") == [len(d) for d in datasets]:
            _log.info("%s", f"[activations] reusing {path}")
            return [(x, y) for x, y in blob["data"]]
        _log.info(
            "%s",
            f"[activations] {path} has sizes {blob.get('sizes')}, expected "
            f"{[len(d) for d in datasets]} — re-extracting.",
        )

    data = extract(datasets, feature_extractor, device, **kw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"sizes": [len(d) for d in datasets], "data": data}, path)
    _log.info("%s", f"[activations] wrote {path}")
    return data
