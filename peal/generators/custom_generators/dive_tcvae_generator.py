import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

import torchvision.transforms as transforms
from PIL import Image

from peal.generators.interfaces import InvertibleGenerator
from peal.global_utils import load_yaml_config
from peal.data.dataloaders import get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.data.dataset_interfaces import PealDataset

import pandas as pd

from peal.dependencies.dive.src.wrappers.tcvae import TCVAE


class DiveTCVAE(InvertibleGenerator):
    def __init__(self, config, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.config.savedir = self.config.base_path  # hacky
        self.config.crop_size = None
        self.exp_dict = self.config.__dict__
        self.tcvae = TCVAE(exp_dict=self.exp_dict, savedir=self.exp_dict["savedir"])
        self.train_dataset, self.val_dataset, _ = get_datasets(self.config.data)

        self.prior = torch.distributions.Normal(
            torch.tensor([0.0] * self.config.z_dim),
            torch.tensor([1.0] * self.config.z_dim),
        )

        self.fid = torchmetrics.image.fid.FrechetInceptionDistance(
            feature=192, reset_real_features=False
        )
        real_images = [self.train_dataset[i][0] for i in range(500)]
        self.fid = self.fid.to("cuda")
        self.fid.update(
            torch.tensor(255 * torch.stack(real_images, dim=0), dtype=torch.uint8).to(
                "cuda"
            ),
            real=True,
        )

    def sample_z(self, batch_size=1):
        return self.prior.sample((batch_size,))

    def log_prob_z(self, z):
        dist = torch.distributions.Normal(self.mu, torch.exp(0.5 * self.logvar))
        return dist.log_prob(z)

    def sample_x(self, batch_size=1):
        z_sample = self.sample_z(batch_size=batch_size)
        return self.tcvae.model.decode(z_sample.cuda())

    def encode(self, x) -> list[torch.Tensor]:
        mu, logvar = self.tcvae.encode(x)
        z = self.tcvae.model.reparemetrize(mu, logvar)

        return [tensor for tensor in z]

    def decode(self, z):
        return self.decode(z)

    def train_model(
        self,
    ):
        writer = SummaryWriter(os.path.join(self.config.base_path, "logs"))
        train_dataloader = get_dataloader(
            self.train_dataset, mode="train", batch_size=self.config.batch_size
        )

        val_dataloader = get_dataloader(
            self.val_dataset, mode="train", batch_size=self.config.batch_size
        )
        score_list = []

        for epoch in range(self.config.max_epoch):
            os.makedirs(os.path.join(self.config.base_path, str(epoch)), exist_ok=True)
            train_dict = self.tcvae.train_on_loader(epoch, train_dataloader)
            val_dict = self.tcvae.val_on_loader(epoch, val_dataloader, vis_flag=True)

            Image.fromarray(val_dict["val_images"]).save(
                os.path.join(self.config.base_path, str(epoch), "reconstruction.png"),
                "PNG",
            )
            del val_dict["val_images"]

            score_dict = {}
            score_dict.update(self.tcvae.get_lr())
            score_dict.update(train_dict)
            score_dict.update(val_dict)
            for name, value in score_dict.items():
                writer.add_scalar(name, value, global_step=epoch)

            # Add score_dict to score_list
            # score_list += [score_dict]

            # Report
            torch.save(
                self.tcvae.state_dict(),
                os.path.join(self.config.base_path, str(epoch), "model.pt"),
            )
            generated_samples = self.sample_x(batch_size=100)
            for i in range(5):
                image_pil = transforms.ToPILImage()(
                    (generated_samples[random.randint(0, 99), :, :, :] + 1) / 2
                )
                image_pil.save(
                    os.path.join(self.config.base_path, str(epoch), f"sample{i}.png")
                )
            self.fid.update(
                (255 * generated_samples.to("cuda")).to(torch.uint8), real=False
            )
            writer.add_scalar("fid", float(self.fid.compute()), global_step=epoch)

    # def edit(
    #         self,
    #         x_in: torch.Tensor,
    #         target_confidence_goal: float,
    #         source_classes: torch.Tensor,
    #         target_classes: torch.Tensor,
    #         classifier: nn.Module,
    #         explainer_config: ExplainerConfig,
    #         classifier_dataset: PealDataset,
    #         pbar=None,
    #         mode="",
    # ): #-> Tuple[ list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    #
    #     dataset = [
    #         (
    #             torch.zeros([len(x_in)], dtype=torch.long),
    #             x_in,
    #             [source_classes, target_classes],
    #         )
    #     ]
