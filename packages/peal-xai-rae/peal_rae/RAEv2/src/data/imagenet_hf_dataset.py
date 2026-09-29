"""
ImageNet dataset loader using HuggingFace Arrow format.

This module provides a PyTorch Dataset wrapper for ImageNet data stored in
Apache Arrow format, as preprocessed by the repa-baseline repository.
"""
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
try:
    from datasets import load_from_disk
except ModuleNotFoundError:
    load_from_disk = None
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .imagenet_classes import IMAGENET_CLASSES


class ImageNetHFDataset(Dataset):
    """
    PyTorch Dataset for ImageNet using HuggingFace Arrow format.

    This dataset loads ImageNet images and labels from pre-processed Arrow files,
    which provide efficient memory-mapped access to the data without requiring
    the full dataset to be loaded into memory.

    Supports both label conditioning (returns int) and text conditioning (returns str).

    Args:
        data_dir: Path to directory containing the arrow dataset.
                 Should contain 'imagenet-latents-images' folder.
        split: Dataset split, either "train" or "val". Default: "train".
        transform: Optional transform to apply to images.
        condition_type: Type of conditioning - "label" (int) or "text" (string prompts). Default: "label".
        prompt_template: Template for generating text prompts. Default: "a photo of a {class_name}".
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[object] = None,
        condition_type: str = "label",
        prompt_template: str = "a photo of a {class_name}",
    ):
        """Initialize the ImageNet HF dataset."""
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.condition_type = condition_type
        self.prompt_template = prompt_template
        self.dataset_type = "arrow"

        # Determine the path to the arrow dataset
        arrow_path = self.data_dir / "imagenet-latents-images"
        split_str = "val" if split == "val" else ""
        dataset_path = arrow_path / split_str if split_str else arrow_path
        if dataset_path.exists():
            if load_from_disk is None:
                raise ModuleNotFoundError(
                    "Found ImageNet Arrow dataset at "
                    f"{dataset_path}, but HuggingFace `datasets` is unavailable. "
                    "Install `datasets` in the runtime rather than silently falling back "
                    "to ImageFolder."
                )
            # Load the dataset using HuggingFace datasets
            self.dataset = load_from_disk(str(dataset_path))
        else:
            # Fallback: use a standard ImageFolder tree only when no Arrow dataset exists.
            split_dir = self.data_dir / split
            if split_dir.exists():
                imagefolder_root = split_dir
            elif self.data_dir.exists():
                imagefolder_root = self.data_dir
            else:
                raise FileNotFoundError(
                    f"ImageNet Arrow split '{split}' not found at {dataset_path}, and no fallback "
                    f"ImageFolder directory exists at {split_dir} or {self.data_dir}."
                )
            self.dataset = ImageFolder(str(imagefolder_root))
            self.dataset_type = "imagefolder"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def relative_path(self, idx: int) -> str:
        """Return a stable source locator without changing the training item API."""
        idx = int(idx)
        if self.dataset_type == "imagefolder":
            path = Path(self.dataset.samples[idx][0])
            try:
                return path.relative_to(Path(self.dataset.root)).as_posix()
            except ValueError:
                return path.as_posix()

        # HF image records may retain an original path/filename. Avoid decoding
        # the image a second time when the Arrow schema exposes such a column.
        column_names = set(getattr(self.dataset, "column_names", []))
        for key in ("relative_path", "path", "file_name", "filename"):
            if key in column_names:
                value = self.dataset[idx][key]
                if value:
                    return str(value)
        # Arrow rows are exactly recoverable from the dataset revision and row
        # index recorded in cache metadata.
        return f"arrow:{self.split}:{idx}"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, str]]:
        if self.dataset_type == "arrow":
            sample = self.dataset[idx]
            image = sample["image"]  # PIL Image
            label = sample["label"]  # int32
        else:
            image, label = self.dataset[idx]

        # Convert PIL image to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)

        # Return based on conditioning type
        if self.condition_type == "text":
            class_name = IMAGENET_CLASSES[label]
            return image, self.prompt_template.format(class_name=class_name)
        else:
            return image, label

    @property
    def num_classes(self) -> int:
        """Return the number of classes in ImageNet."""
        return 1000
