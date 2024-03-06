import os
import random
import types
import shutil
import copy
import torch
import io
import blobfile as bf

from mpi4py import MPI
from torch import nn
from types import SimpleNamespace

from peal.generators.interfaces import EditCapableGenerator
from peal.global_utils import load_yaml_config
from peal.global_utils import load_yaml_config
from peal.data.dataloaders import get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.data.dataset_interfaces import PealDataset

#from haven import haven_utils as hu
#from haven import haven_wizard as hw
from PIL import Image

from peal.dependencies.dive.src import datasets, wrappers
from peal.dependencies.dive.src.wrappers.tcvae import TCVAE

def DiveTCVAE(EditCapableGenerator):
    def __init__(self, config, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)

        self.tcvae = TCVAE(exp_dict=self.config.__dict__, savedir=self.config["savedir"])
        self.dataset = get_datasets(self.config.data)[0]

    def sample_x(self):
        self.prior = torch.distributions.Normal(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0]))
        z_sample = self.prior.sample()
        return self.decode(z_sample)

    def train_model(
            self,
    ):
        data = iter(
            get_dataloader(
                self.dataset,
                mode="train",
                batch_size=self.config.batch_size,
                training_config={"steps_per_epoch": self.config.max_steps},
            )
        )

