import os
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

import torchvision.transforms as transforms
from PIL import Image

from peal.generators.interfaces import InvertibleGenerator
from peal.global_utils import load_yaml_config, save_yaml_config, embed_numberstring
from peal.data.dataloaders import get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.data.dataset_interfaces import PealDataset

import pandas as pd

from peal.dependencies.dive.src.wrappers.tcvae import TCVAE


class DiveTCVAE(InvertibleGenerator):
    def __init__(self, config, model_dir=None, device="cpu", classifier_dataset=None, train=True):
        super().__init__()
        self.config = load_yaml_config(config)
        self.config.savedir = self.config.base_path  # hacky
        self.config.crop_size = None
        self.exp_dict = self.config.__dict__
        print(self.config)
        print('initialize generator')
        self.tcvae = TCVAE(exp_dict=self.exp_dict, savedir=self.exp_dict["savedir"])
        print('initializing generator done!')
        if os.path.exists(os.path.join(self.config.base_path, "final.pt")):
            # TODO this is kind of dangerous
            self.tcvae.load_state_dict(
                torch.load(os.path.join(self.config.base_path, "final.pt")), strict=False
            )

        self.train_dataset, self.val_dataset, _ = get_datasets(self.config.data)
        self.dataset = self.val_dataset

        self.prior = torch.distributions.Normal(
            torch.tensor([0.0] * self.config.z_dim),
            torch.tensor([1.0] * self.config.z_dim),
        )

        self.fid = torchmetrics.image.fid.FrechetInceptionDistance(
            feature=192, reset_real_features=False
        )
        real_images = [self.train_dataset[i][0] for i in range(min(len(self.train_dataset), 500))]
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
        # dist = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        # return dist.log_prob(z)
        return torch.sum([z_elem**2 for z_elem in z])

    def sample_x(self, batch_size=1):
        z_sample = self.sample_z(batch_size=batch_size)
        return self.tcvae.model.decode(z_sample.cuda())

    def encode(self, x) -> list[torch.Tensor]:
        mu, logvar = self.tcvae.model.encode(x)
        z = self.tcvae.model.reparameterize(mu, logvar)
        return [z]

    def decode(self, z):
        return self.tcvae.model.decode(z[0])

    def train_model(
        self,
    ):
        # write the yaml config on disk
        if not os.path.exists(self.config.base_path):
            Path(self.config.base_path).mkdir(parents=True, exist_ok=True)

        save_yaml_config(self.config, os.path.join(self.config.base_path, "config.yaml"))

        writer = SummaryWriter(os.path.join(self.config.base_path, "logs"))
        train_dataloader = get_dataloader(
            self.train_dataset, mode="train", batch_size=self.config.batch_size
        )

        val_dataloader = get_dataloader(
            self.val_dataset, mode="train", batch_size=self.config.batch_size
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
        score_list = []

        for epoch in range(self.config.max_epoch):
            os.makedirs(os.path.join(self.config.base_path, str(epoch)), exist_ok=True)
            train_dict = self.tcvae.train_on_loader(epoch, train_dataloader)
            val_dict = self.tcvae.val_on_loader(epoch, val_dataloader, vis_flag=True)
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print(self.train_dataset.task_config)

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
            Path(os.path.join(self.config.base_path, embed_numberstring(epoch))).mkdir(
                parents=True, exist_ok=True
            )
            torch.save(
                self.tcvae.state_dict(),
                os.path.join(self.config.base_path, embed_numberstring(epoch), "model.pt"),
            )
            generated_samples = self.sample_x(batch_size=self.config.batch_size)
            for i in range(5):
                image_pil = transforms.ToPILImage()(
                    (generated_samples[random.randint(0, self.config.batch_size-1), :, :, :] + 1) / 2
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
