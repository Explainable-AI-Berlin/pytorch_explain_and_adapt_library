import os
from typing import Tuple
from matplotlib import cm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import VisionDataset
import torch
import csv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import lightning as L


class SquaresDataset(VisionDataset):
    img_folder = "imgs"
    masks_folder = "masks"

    def __init__(
        self, label="ColorA", mask_mode=False, get_mode="regr", *args, **kwargs
    ):
        """Loads the square data folder. Expected format:

        root
        ├── data.csv
        ├── imgs
        │   ├── 0.png
        │   ├── ...
        ├── masks
        │   ├── 0.png
        │   ├── ...
        """
        super().__init__(*args, **kwargs)

        self.data: list[Tuple[str, float]] = []
        csv_path = os.path.join(self.root, "data.csv")
        with open(csv_path, mode="r") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                self.data.append((row["Name"], float(row[label])))

        self.mask_mode = mask_mode
        self.get_mode = get_mode

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

    def load_image(self, index):
        base_name, label = self.data[index]
        # DHA: Assume preprocessed images
        img_path = os.path.join(self.root, self.img_folder, base_name)

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        label = torch.Tensor([label])
        img = self.transform(img)

        if self.get_mode == "dae":  # Diffusion Autoencoder
            return {"img": img, "index": index, "labels": label}
        elif self.get_mode == "ace":
            return img, {}
        elif self.mask_mode:
            mask_path = os.path.join(self.root, self.masks_folder, base_name)
            with open(mask_path, "rb") as f:
                mask = Image.open(f).convert("RGB")
                mask = transforms.ToTensor()(mask)

            return base_name, img, label, mask
        else:
            return img, label

    def __getitem__(self, index):
        return self.load_image(index)

    def __len__(self):
        return len(self.data)


class SquaresDataModule(L.LightningDataModule):
    def __init__(self, folder_path, transform=None, batch_size=32):
        """Loads the square data folder. Expected format:

        root
        ├── data.csv
        ├── imgs
        │   ├── 0.png
        │   ├── ...
        ├── masks
        │   ├── 0.png
        │   ├── ...
        """
        super(SquaresDataModule, self).__init__()
        self.folder_path = folder_path
        self.csv_path = os.path.join(folder_path, "data.csv")
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        )
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = SquaresDataset(root=self.folder_path, transform=self.transform)
        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset, [0.8, 0.2]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def total_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=4,
        )


def inner_square_color(x: torch.Tensor, mask: torch.Tensor):
    """
    Checks the red values of the inner square, given the hint mask.
    """
    mask = mask.to(torch.bool)
    mask[[1, 2], :, :] = False  # Only select red channel

    if mask.sum().item() != 64:
        print("WARNING: Mask sum is not 64 (square should be 8x8).")

    intensity_foreground = x[mask].mean()
    return intensity_foreground


def background_color(x: torch.Tensor, mask: torch.Tensor):
    """
    Computes the background color of the square image (inverse of the square mask and for all channels).
    """
    mask = mask.to(torch.bool)
    mask = ~mask  # Invert mask

    intensity_background = x[mask].mean()
    return intensity_background


def get_experiment_targets(y_true: torch.Tensor):
    # Mirror experiment: if the values are in the lower half, we should add 0.5
    # (to get them to the other side). Otherwise, we should subtract 0.5.
    y_rgb = (y_true * 255).to(torch.uint8)
    mirrored = torch.where(y_rgb <= 127, y_rgb + 128, y_rgb - 128).to(y_true)
    return mirrored / 255
