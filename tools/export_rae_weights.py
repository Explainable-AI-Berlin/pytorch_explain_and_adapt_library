"""
Export the weights of a trained RAE generator into a folder that holds nothing
but the weights, ready for `huggingface_hub.upload_folder`:

    decoder.pt      stage-1 EMA decoder (as extracted by rae_pipeline)
    stats.pt        stage-1 latent normalisation statistics
    stage2_ema.pt   stage-2 EMA state dict alone (the run checkpoints carry
                    model + ema + optimizer, ~17 GB; the EMA is ~5 GB fp32)

Usage:
    python tools/export_rae_weights.py --base_path $PEAL_RUNS/imagenet/rae_clip \\
        --stage2_checkpoint $PEAL_RUNS/imagenet/rae_clip/stage2/ddt_xl_cls/checkpoints/ep-0000043.pt \\
        --out /tmp/peal-rae-clip-imagenet [--dtype bf16] [--upload <org>/peal-rae-clip-imagenet --private]

    # from an already extracted EMA file (stage2_slim/ep-*_ema.pt) the same way;
    # the script detects a bare state dict.

A generator config then points at the folder: `weights: hf://<org>/<repo>` or
`weights: /path/to/folder` (see RAEDiffusionAutoencoderConfig.weights).
"""

import argparse
import os
import shutil

import torch


def _human(nbytes):
    """Format a byte count as ``12.3 MB``.

    Parameters
    ----------
    nbytes : int or float
        Size in bytes.

    Returns
    -------
    str
        The size with a binary unit from B to TB.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def extract_stage2_ema(ckpt_path, out_path, weights="ema", dtype="keep"):
    """Save one state dict out of a stage-2 checkpoint as a bare file.

    Handles a full run checkpoint (``model`` / ``ema`` / optimizer keys), a
    checkpoint without the requested key (falls back to ``model``) and an
    already extracted bare state dict. ``module.`` prefixes from DDP are
    stripped.

    Parameters
    ----------
    ckpt_path : str
        The stage-2 ``ep-*.pt`` or a slim ``*_ema.pt``.
    out_path : str
        Where the bare state dict is written with ``torch.save``.
    weights : {"ema", "model"}, optional
        Which state dict to take from a full checkpoint.
    dtype : {"keep", "bf16", "fp16", "fp32"}, optional
        Cast floating-point tensors to this dtype.

    Returns
    -------
    tuple
        ``(n_tensors, meta)`` where ``meta`` holds ``epoch`` / ``step`` when
        the checkpoint carries them.
    """
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError, TypeError):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and weights in state:
        sd = state[weights]
        meta = {k: state[k] for k in ("epoch", "step") if k in state}
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
        meta = {k: state[k] for k in ("epoch", "step") if k in state}
        print(f"[export_rae_weights] no '{weights}' key in {ckpt_path}; using 'model'")
    else:  # bare state dict (stage2_slim/ep-*_ema.pt)
        sd = state
        meta = {}
    cast = {
        "keep": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    out = {}
    for k, v in sd.items():
        k = k[len("module.") :] if k.startswith("module.") else k
        if torch.is_tensor(v):
            v = v.detach().clone()
            if cast is not None and v.is_floating_point():
                v = v.to(cast)
        out[k] = v
    torch.save(out, out_path)
    return len(out), meta


def main():
    """Assemble the weights folder and optionally upload it to the Hub.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If ``stage1_assets/decoder.pt`` or ``stats.pt`` is missing.
    FileExistsError
        If ``--out`` exists and is not empty.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--base_path", required=True, help="generator run dir (holds stage1_assets/)"
    )
    ap.add_argument(
        "--stage2_checkpoint", required=True, help="stage-2 ep-*.pt or a slim *_ema.pt"
    )
    ap.add_argument(
        "--out", required=True, help="output folder (created; must be empty or absent)"
    )
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--dtype", default="keep", choices=["keep", "bf16", "fp16", "fp32"])
    ap.add_argument(
        "--upload", default=None, help="Hugging Face repo id to upload the folder to"
    )
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    assets = os.path.join(args.base_path, "stage1_assets")
    for name in ("decoder.pt", "stats.pt"):
        if not os.path.isfile(os.path.join(assets, name)):
            raise FileNotFoundError(
                f"{assets}/{name} missing; train / extract stage 1 first"
            )
    if os.path.isdir(args.out) and os.listdir(args.out):
        raise FileExistsError(
            f"{args.out} is not empty; the weights folder must hold only the weights"
        )
    os.makedirs(args.out, exist_ok=True)

    for name in ("decoder.pt", "stats.pt"):
        shutil.copy2(os.path.join(assets, name), os.path.join(args.out, name))
    n, meta = extract_stage2_ema(
        args.stage2_checkpoint,
        os.path.join(args.out, "stage2_ema.pt"),
        args.weights,
        args.dtype,
    )
    print(f"[export_rae_weights] stage2_ema.pt: {n} tensors {meta}")
    total = 0
    for name in sorted(os.listdir(args.out)):
        size = os.path.getsize(os.path.join(args.out, name))
        total += size
        print(f"  {name:16s} {_human(size)}")
    print(f"  {'total':16s} {_human(total)}")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            args.upload, private=args.private, exist_ok=True, repo_type="model"
        )
        api.upload_folder(folder_path=args.out, repo_id=args.upload, repo_type="model")
        print(f"[export_rae_weights] uploaded to https://huggingface.co/{args.upload}")


if __name__ == "__main__":
    main()
