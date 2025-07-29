import torch
from PIL import Image
import os

from torchvision import transforms

from diff_cf_ir.file_utils import is_image_file


def default_transforms(size, ddpm=False):
    image_transforms = [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]

    if ddpm:
        image_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(image_transforms)


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder: str, size: int, transform=None):
        self.folder = folder
        self.image_files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if is_image_file(f):
                    self.image_files.append(os.path.join(root, f))

        self.transform = transform if transform else default_transforms(size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path)
        if img is not None:
            img = self.transform(img)
        return img_path, img


class PairedImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, real_folder, fake_folder, size: int, transform=None):
        self.real_folder = real_folder
        self.fake_folder = fake_folder
        self.transform = transform if transform else default_transforms(size)

        real_files = set(f for f in os.listdir(real_folder) if is_image_file(f))
        fake_files = set(f for f in os.listdir(fake_folder) if is_image_file(f))

        missing_files = fake_files - real_files
        if missing_files:
            raise FileNotFoundError(
                f"The following images are missing in the real folder: {missing_files}"
            )

        self.image_files = list(fake_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        real_img = Image.open(os.path.join(self.real_folder, img_name))
        fake_img = Image.open(os.path.join(self.fake_folder, img_name))

        if self.transform:
            real_img = self.transform(real_img)
            fake_img = self.transform(fake_img)

        return real_img, fake_img, img_name
