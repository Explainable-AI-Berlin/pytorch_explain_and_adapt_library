import os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch
import csv

import lightning as L
from PIL import Image

from diff_cf_ir.img_utils import (
    create_cropped_images_old,
    create_cropped_images_like_celebahq,
)

labels = [
    "filename",
    "age",
    "gender",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "head_roll",
    "head_yaw",
    "head_pitch",
]

FILENAME_COL = "filename"
LABEL_COL = "age"
MAX_AGE = 100.0


def load_labels(csv_path):
    label_dict = {}

    with open(csv_path, mode="r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label_dict[row[FILENAME_COL]] = float(row[LABEL_COL]) / 100

    return label_dict


class ImdbCleanDataset(ImageFolder):

    def __init__(
        self,
        transform=None,
        # normalize=False,
        data_shape=[3, 512, 512],
        split="train",
        *args,
        **kwargs,
    ):
        """Loads the folder for the imdb-clean dataset with the age as the label.

        Split should be "train", "test" or "valid".

        Folder structure:

        root
        ├── imdb_train_new_1024.csv
        ├── imdb_test_new_1024.csv
        ├── imdb_valid_new_1024.csv
        ├── 00
        │   ├── *.jpg
        │   ├── ...
        ├── 04
        │   ├── *.jpg
        │   ├── ...
        """
        assert (
            "cropped" in kwargs["root"]
        ), "Please provide the path to the cropped images."
        super().__init__(*args, **kwargs)

        # Prepare labels
        csv_path = self.root + f"/imdb_{split}_new_1024.csv"
        self.label_dict = load_labels(csv_path)
        self.img_list = list(self.label_dict.keys())
        # self.normalize = normalize
        self.data_shape = data_shape

        if transform is None:
            transforms_list = [
                transforms.ToTensor(),
                transforms.Resize(self.data_shape[1:]),
            ]

            # if normalize:
            #     transforms_list.append(
            #         transforms.Normalize(mean=self.data_mean, std=self.data_std)
            #     )

            self.manual_transforms = transforms.Compose(transforms_list)
        else:
            self.manual_transforms = transform

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img: Image.Image = Image.open(os.path.join(self.root, img_path))

        label = torch.Tensor([self.label_dict[img_path]])

        if self.manual_transforms is not None:
            img = self.manual_transforms(img)

        tuple_with_path = (img, label)
        return tuple_with_path

    def __len__(self):
        return len(self.img_list)


class ImdbCleanDataModule(L.LightningDataModule):
    def __init__(
        self,
        folder_path,
        data_shape=[3, 256, 256],
        transform=None,
        batch_size=32,
        seed=0,
    ):
        """Loads the folder for the Imdb-Wiki-Clean.

        Folder structure:

        root
        ├── imdb_train_new_1024.csv
        ├── imdb_test_new_1024.csv
        ├── imdb_valid_new_1024.csv
        ├── 00
        │   ├── *.jpg
        │   ├── ...
        ├── 04
        │   ├── *.jpg
        │   ├── ...
        """
        super().__init__()

        self.folder_path = folder_path
        self.transform = transform
        self.batch_size = batch_size
        # Set seed
        self.data_shape = data_shape

        self.generator = torch.manual_seed(seed)

    def setup(self, stage=None):
        self.train_set = ImdbCleanDataset(
            root=self.folder_path,
            transform=self.transform,
            data_shape=self.data_shape,
        )
        self.val_set = ImdbCleanDataset(
            root=self.folder_path,
            transform=self.transform,
            data_shape=self.data_shape,
            split="valid",
        )
        print(
            f"IMDB-WIKI dataset length: train={len(self.train_set)} val={len(self.val_set)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            generator=self.generator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=2,
            generator=self.generator,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create cropped images for the imdb-clean dataset."
    )
    parser.add_argument(
        "root",
        type=str,
        help=(
            "Path to the root directory containing the imdb-wiki-clean dataset. "
            "Should contain the 3 split csv files."
        ),
    )
    parser.add_argument(
        "output",
        type=str,
        help=(
            "Path to the output directory where cropped images will be saved. "
            "For example could be in $root/data/imdb-clean-1024-cropped"
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="celebahq",
        choices=["celebahq", "old"],
        help=(
            "Method to use for cropping. "
            "celebahq uses the same method as the CelebA-HQ dataset, "
            "while old uses a custom method."
        ),
    )
    args = parser.parse_args()

    if args.method == "celebahq":
        print("Using celebahq method")
        create_cropped_images_like_celebahq(args.root, args.output)
    elif args.method == "old":
        print("Using old method")
        create_cropped_images_old(args.root, args.output)

    # Example: Run in container
    # apptainer run -B /home/space/datasets:/home/space/datasets  ~/apptainers/thesis.sif python imdb_clean_dataset.py /home/space/datasets/imdb-wiki-clean/imdb-clean /home/space/datasets/imdb-wiki-clean/imdb-clean/data/imdb-clean-1024-cropped
