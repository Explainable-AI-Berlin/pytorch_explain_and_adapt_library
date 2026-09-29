from __future__ import annotations

from pathlib import Path


def load_imagenet_wnid_to_name(meta_path: str | Path) -> dict[str, str]:
    import torch

    meta = torch.load(Path(meta_path), map_location="cpu", weights_only=False)
    if isinstance(meta, tuple) and meta:
        wnid_to_classes = meta[0]
    elif isinstance(meta, dict) and "wnid_to_classes" in meta:
        wnid_to_classes = meta["wnid_to_classes"]
    else:
        raise ValueError(f"Unsupported ImageNet metadata format in {meta_path}")

    out: dict[str, str] = {}
    for wnid, names in wnid_to_classes.items():
        if isinstance(names, str):
            name = names
        else:
            name = str(names[0])
        out[str(wnid)] = name.replace("_", " ")
    return out


def load_imagenet_wnid_to_index(meta_path: str | Path) -> dict[str, int]:
    import torch

    meta = torch.load(Path(meta_path), map_location="cpu", weights_only=False)
    if isinstance(meta, tuple) and meta:
        wnid_to_classes = meta[0]
        candidate_order = meta[1] if len(meta) >= 2 else None
    elif isinstance(meta, dict) and "val_wnids" in meta:
        wnid_to_classes = meta.get("wnid_to_classes")
        candidate_order = meta["val_wnids"]
    else:
        raise ValueError(f"Unsupported ImageNet metadata format in {meta_path}")
    if wnid_to_classes is None:
        raise ValueError(f"ImageNet metadata lacks wnid_to_classes in {meta_path}")
    if candidate_order is not None and len(candidate_order) == len(wnid_to_classes):
        class_order = list(candidate_order)
    else:
        # Torchvision's ImageNet/ImageFolder class IDs are sorted WNID folder
        # names. `val_wnids` in meta.bin may instead be per-validation-image
        # labels, which is not a 1000-way classifier index order.
        class_order = sorted(wnid_to_classes.keys())
    return {str(wnid): int(idx) for idx, wnid in enumerate(class_order)}


def class_display_names(class_names: list[str], meta_path: str | Path | None) -> list[str]:
    if meta_path is None:
        return [name.replace("_", " ") for name in class_names]
    wnid_to_name = load_imagenet_wnid_to_name(meta_path)
    return [wnid_to_name.get(name, name.replace("_", " ")) for name in class_names]
