from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import torch
import csv

import lightning as L
from PIL.Image import Image

landmark_labels = "image_id,lefteye_x,lefteye_y,righteye_x,righteye_y,nose_x,nose_y,leftmouth_x,leftmouth_y,rightmouth_x,rightmouth_y".split(
    ","
)


class CelebALandmarksDataset(ImageFolder):

    # Model Metadata
    # # n_bits = 5
    # # temp = 0.7

    # Copied from torch
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]

    def load_labels(self, csv_path):
        label_dict = {}

        with open(csv_path, mode="r") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                label_dict[row["image_id"]] = [
                    (label, float(row[label])) for label in self.labels
                ]
        return label_dict

    def __init__(
        self,
        labels: list[str],
        transform=None,
        normalize=False,
        data_shape=[3, 64, 64],
        *args,
        **kwargs
    ):
        """Loads the folder for the CelebA dataset with landmarks as features.

        root
        ├── list_landmarks_align_celeba.csv
        ├── img_align_celeba
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   ├── ...
        """
        super(CelebALandmarksDataset, self).__init__(*args, **kwargs)

        # Prepare labels
        self.labels = labels
        self.label_dict: dict[str, list[tuple[str, float]]] = {}
        csv_path = self.root + "/list_landmarks_align_celeba.csv"
        self.label_dict = self.load_labels(csv_path)
        self.normalize = normalize
        self.data_shape = data_shape

        if transform is None:
            transforms_list = [
                transforms.ToTensor(),
                transforms.Resize(self.data_shape[1:]),
            ]

            if normalize:
                transforms_list.append(
                    transforms.Normalize(mean=self.data_mean, std=self.data_std)
                )

            self.manual_transforms = transforms.Compose(transforms_list)
        else:
            self.manual_transforms = transform

    def __getitem__(self, index):
        original_tuple = super(CelebALandmarksDataset, self).__getitem__(index)
        img: Image = original_tuple[0]
        path = self.imgs[index][0]
        base_name = path.split("/")[-1]

        def process_label(label: str, value: float) -> float:
            # Rescale the labels to the new image size
            if label.endswith("_x"):
                scaled_px = value * (self.data_shape[1] / img.width)
                return scaled_px / self.data_shape[1]
            if label.endswith("_y"):
                scaled_px = value * (self.data_shape[2] / img.height)
                return scaled_px / self.data_shape[2]

            return value

        label = torch.Tensor(
            [process_label(label, value) for label, value in self.label_dict[base_name]]
        )

        if self.manual_transforms is not None:
            img = self.manual_transforms(img)

        tuple_with_path = (index, img, label)
        return tuple_with_path


# TODO: See https://pytorch.org/vision/master/auto_examples/transforms/plot_transforms_getting_started.html
# class ResizeImageAndLabel(torch.nn.Module):

#     def __init__(self, size: list[int]):
#         super(ResizeImageAndLabel, self).__init__()
#         self.size = size

#     def forward(
#         self, img, bboxes, label
#     ):  # we assume inputs are always structured like this

#         return img, bboxes, label


class CelebALandmarksDataModule(L.LightningDataModule):
    def __init__(
        self,
        folder_path,
        labels,
        data_shape=[3, 64, 64],
        transform=None,
        batch_size=32,
        seed=0,
    ):
        """Loads the folder for the CelebA dataset with landmarks as features.

        root
        ├── list_landmarks_align_celeba.csv
        ├── img_align_celeba
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   ├── ...
        """
        super(CelebALandmarksDataModule, self).__init__()

        self.folder_path = folder_path
        self.transform = transform
        self.batch_size = batch_size
        # Set seed
        self.generator = torch.manual_seed(seed)
        self.labels = labels
        self.data_shape = data_shape

    def setup(self, stage=None):
        self.dataset = CelebALandmarksDataset(
            root=self.folder_path,
            transform=self.transform,
            labels=self.labels,
            data_shape=self.data_shape,
        )
        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset, [0.8, 0.2], generator=self.generator
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            generator=self.generator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=4,
            generator=self.generator,
        )

    def total_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=4,
            generator=self.generator,
        )
