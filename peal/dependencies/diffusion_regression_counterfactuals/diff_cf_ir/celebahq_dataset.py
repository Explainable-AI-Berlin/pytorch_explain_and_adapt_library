import os
from typing import Union
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import VisionDataset
import torch
import pandas as pd

import lightning as L
from PIL import Image


class CelebAHQDataset(VisionDataset):
    COLUMNS = [
        "filename",
        "5_o_Clock_Shadow",
        "Arched_Eyebrows",
        "Attractive",
        "Bags_Under_Eyes",
        "Bald",
        "Bangs",
        "Big_Lips",
        "Big_Nose",
        "Black_Hair",
        "Blond_Hair",
        "Blurry",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Chubby",
        "Double_Chin",
        "Eyeglasses",
        "Goatee",
        "Gray_Hair",
        "Heavy_Makeup",
        "High_Cheekbones",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "Narrow_Eyes",
        "No_Beard",
        "Oval_Face",
        "Pale_Skin",
        "Pointy_Nose",
        "Receding_Hairline",
        "Rosy_Cheeks",
        "Sideburns",
        "Smiling",
        "Straight_Hair",
        "Wavy_Hair",
        "Wearing_Earrings",
        "Wearing_Hat",
        "Wearing_Lipstick",
        "Wearing_Necklace",
        "Wearing_Necktie",
        "Young",
    ]

    IMG_FOLDER = "CelebA-HQ-img"
    LABEL_COL = "Young"
    YOUNG_VALUE = 0.1
    OLD_VALUE = 0.8

    def load_metadata(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, skiprows=2, header=None, sep=r" +", engine="python")
        df.columns = self.COLUMNS
        return df

    def __init__(
        self,
        root: str,
        transform=None,
        get_mode="train",
        partition_file: Union[str, None] = None,
        partition="test",
    ):
        """Loads the folder for the CelebaHQ dataset with the "Young" as the label.

        Mode should be "train" or "cf". If "train" the dataset will return the label as is.
        If "cf" the dataset will return the label as the target for the counterfactual (Young -> Old and vice versa).

        Folder structure:

        root
        ├── CelebAMask-HQ-mask-anno
        ├── CelebAMask-HQ-attribute-anno.txt
        ├── CelebAMask-HQ-pose-anno.txt
        ├── CelebA-HQ-to-CelebA-mapping.txt
        ├── CelebA-HQ-img
        │   ├── *.jpg
        │   ├── ...
        """
        # Prepare labels
        self.root = root
        csv_path = os.path.join(self.root, "CelebAMask-HQ-attribute-anno.txt")
        self.meta = self.load_metadata(csv_path)
        self.get_mode = get_mode
        self.partition = partition

        self.filter_meta_partition(partition_file)

        if transform is None:
            transforms_list = [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
            ]
            self.transform = transforms.Compose(transforms_list)
        else:
            self.transform = transform

    def filter_meta_partition(self, partition_file):
        if partition_file is not None:
            mapping = pd.read_csv(
                os.path.join(self.root, "CelebA-HQ-to-CelebA-mapping.txt"),
                sep=r" +",
                engine="python",
            )
            mapping["filename"] = mapping["idx"].astype(str) + ".jpg"
            self.meta = self.meta.merge(mapping, on="filename", validate="1:1")

            partition_meta = pd.read_csv(
                partition_file, sep=" ", header=None, names=["orig_file", "partition"]
            )
            partition_mapping = {
                "train": 0,
                "val": 1,
                "test": 2,
            }
            self.meta = self.meta.merge(partition_meta, on="orig_file", validate="1:1")
            self.meta = self.meta[
                self.meta["partition"] == partition_mapping[self.partition]
            ]
            print(f"CelebaHQDataset: {self.partition} set size: {len(self.meta)}")
        else:
            print("CelebaHQDataset: No Partition file provided. Using all data.")

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        img_path = row["filename"]
        img: Image.Image = Image.open(
            os.path.join(self.root, self.IMG_FOLDER, img_path)
        )

        label = torch.Tensor([row[self.LABEL_COL]])

        if self.transform is not None:
            img = self.transform(img)

        if self.get_mode != "cf":
            return img, label
        else:
            # Target is the opposite of the label
            target = self.OLD_VALUE if label == 1 else self.YOUNG_VALUE
            return img_path, img, torch.Tensor([target])

    def __len__(self):
        return len(self.meta)


# class CelebAHQDataModule(L.LightningDataModule):
#     def __init__(
#         self,
#         folder_path,
#         transform=None,
#         batch_size=16,
#     ):
#         """Loads the folder for the CelebaHQ dataset.

#         Folder structure:

#         root
#         ├── CelebAMask-HQ-mask-anno
#         ├── CelebAMask-HQ-pose-anno.txt
#         ├── CelebA-HQ-img
#         │   ├── *.jpg
#         │   ├── ...
#         """
#         super().__init__()

#         self.folder_path = folder_path
#         self.transform = transform
#         self.batch_size = batch_size

#     def setup(self, stage=None):
#         self.train_set = CelebAHQDataset(
#             root=self.folder_path,
#             transform=self.transform,
#         )
#         self.val_set = CelebAHQDataset(
#             root=self.folder_path,
#             transform=self.transform,
#             partition="val",
#         )

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_set,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=4,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_set,
#             batch_size=self.batch_size,
#             num_workers=4,
#         )
