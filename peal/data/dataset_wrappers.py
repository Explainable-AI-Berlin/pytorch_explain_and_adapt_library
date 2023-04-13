import torch

from peal.data.dataset_interfaces import PealDataset


class GlowDatasetWrapper(PealDataset):
    def __init__(self, base_dataset, n_bits):
        self.base_dataset = base_dataset
        self.n_bits = n_bits
        self.n_bins = 2.0**n_bits

    def __len__(self):
        return self.base_dataset.__len__()

    def __getitem__(self, idx):
        image, label = self.base_dataset.__getitem__(idx)
        image = image * 255
        image = torch.floor(image / 2 ** (8 - self.n_bits))
        image = image / self.n_bins - 0.5
        image = image + torch.rand_like(image) / self.n_bins
        return image, label

    def project_to_pytorch_default(self, x):
        """
        This function maps processed image back to human visible image
        """
        return x + 0.5


class VAEDatasetWrapper(PealDataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return self.base_dataset.__len__()

    def __getitem__(self, idx):
        x, y = self.base_dataset.__getitem__(idx)
        return x, x
