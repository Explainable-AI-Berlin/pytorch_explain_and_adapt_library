import copy
import math
import os
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from peal.dependencies.glow.model import gaussian_log_p, Glow
from peal.generators.interfaces import InvertibleGenerator
from peal.global_utils import load_yaml_config, save_yaml_config
from peal.data.dataloaders import get_dataloader
from peal.data.dataset_factory import get_datasets

from peal.dependencies.glow.train import training

from typing import Union

from peal.generators.interfaces import GeneratorConfig
from peal.data.interfaces import DataConfig
from peal.training.loggers import log_images_to_writer


class GlowGeneratorConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str = "GlowGenerator"
    """
    The config of the data.
    """
    data: DataConfig = DataConfig()
    batch: int = 16
    iter: int = 400000
    n_flow: int = 32
    n_block: int = 4
    no_lu: bool = False
    affine: bool = False
    n_bits: int = 5
    lr: float = 1e-4
    img_size: int = 64
    temp: float = 0.7
    n_sample: int = 20
    base_path: str = "glow_run"
    x_selection: Union[list, type(None)] = None
    full_args: Union[None, dict] = None



class GlowGenerator(InvertibleGenerator):
    def __init__(self, config, model_dir=None, device="cpu", predictor_dataset=None, train=True):
        super().__init__()
        self.config = load_yaml_config(config)
        print(self.config)
        print('initialize generator')
        self.glow = Glow(
            3, self.config.n_flow, self.config.n_block, affine=self.config.affine, conv_lu=not self.config.no_lu
        ).to(device)
        print('initializing generator done!')
        if os.path.exists(os.path.join(self.config.base_path, "final.pt")):
            print('load weights!')
            print('load weights!')
            print('load weights!')
            self.glow.load_state_dict(
                torch.load(os.path.join(self.config.base_path, "final.pt")),
            )
            print('weights loaded!')
            print('weights loaded!')
            print('weights loaded!')

        self.train_dataset, self.val_dataset, _ = get_datasets(self.config.data)
        self.dataset = self.val_dataset

    def encode(self, x, t=1.0):
        out = x
        z_outs = []

        for block in self.glow.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)

        return z_outs

    def decode(self, z, t=1.0, reconstruct=True):
        for i, block in enumerate(self.glow.blocks[::-1]):
            if i == 0:
                x = block.reverse(z[-1], z[-1], reconstruct=reconstruct)

            else:
                x = block.reverse(x, z[-(i + 1)], reconstruct=reconstruct)

        return x

    def sample_z(self, n="auto"):
        if isinstance(n, str) and n == "auto":
            n_sample = self.config.batch

        else:
            n_sample = n

        z_sample = []
        z_shapes = self.calc_z_shapes()
        for z in z_shapes:
            z_new = torch.randn(n_sample, *z) * self.config.temp
            z_sample.append(z_new.to(next(self.parameters()).device))

        return z_sample

    def log_prob_z(self, z):
        log_probs = []

        for it, block in enumerate(self.glow.blocks):
            zero = torch.zeros_like(z[it])
            mean, log_sd = block.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(z[it], mean, log_sd)
            log_probs.append(torch.flatten(log_p, start_dim=1).sum(1))

        log_p_sum = torch.sum(torch.stack(log_probs, dim=0), dim=0)

        n_pixel = np.prod(self.config.data.input_size)
        log_p = log_p_sum - self.config.n_bits * n_pixel
        return -torch.clone(log_p / (math.log(2) * n_pixel)).detach()

    def calc_z_shapes(self):
        z_shapes = []

        n_channel = copy.deepcopy(self.config.data.input_size[0])
        input_size = copy.deepcopy(self.config.data.input_size[1])

        for i in range(self.config.n_block - 1):
            input_size //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))

        return z_shapes

    def train_model(
        self,
    ):
        # write the yaml config on disk
        if not os.path.exists(self.config.base_path):
            Path(self.config.base_path).mkdir(parents=True, exist_ok=True)

        save_yaml_config(self.config, os.path.join(self.config.base_path, "config.yaml"))

        writer = SummaryWriter(os.path.join(self.config.base_path, "logs"))
        train_dataloader = get_dataloader(
            self.train_dataset, mode="train", batch_size=self.config.batch
        )
        if len(self.val_dataset) > 0:
            val_dataloader = get_dataloader(
                self.val_dataset, mode="train", batch_size=self.config.batch
            )

        else:
            val_dataloader = get_dataloader(
                self.train_dataset, mode="train", batch_size=self.config.batch
            )

        if not self.config.x_selection is None:
            self.train_dataset.task_config = SimpleNamespace(
                **{"x_selection": self.config.x_selection}
            )
            self.val_dataset.task_config = SimpleNamespace(
                **{"x_selection": self.config.x_selection}
            )
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print(self.train_dataset.task_config)

        args = types.SimpleNamespace(**self.config.__dict__)
        args.train_dataloader = train_dataloader
        log_images_to_writer(args.train_dataloader, writer, "train")
        args.val_dataloader = val_dataloader
        args.writer = writer
        args.glow = self.glow
        training(args)