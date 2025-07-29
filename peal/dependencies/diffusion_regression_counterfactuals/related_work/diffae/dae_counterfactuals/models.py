import sys
import os




import torch
from templates import *

import matplotlib

matplotlib.use("Agg")
matplotlib.rc("text")

device = "cuda" if torch.cuda.is_available() else "cpu"


class DAEModel:
    def __init__(
        self,
        forward_t: int = 250,
        backward_t: int = 20,
        conf: TrainConfig = ffhq256_autoenc(),
    ):
        self.conf = conf
        self.model = self.load_dae_model()

        self.forward_t = forward_t
        self.backward_t = backward_t

    def load_dae_model(self):
        model = LitModel(self.conf)
        state = torch.load(
            f"checkpoints/{self.conf.name}/last.ckpt", map_location="cpu"
        )
        model.load_state_dict(state["state_dict"], strict=False)
        model.ema_model.eval()
        model.ema_model.to(device)
        return model

    def encode(self, batch: torch.Tensor):
        batch = batch.to(device)
        z_sem: torch.Tensor = self.model.encode(batch)
        xT: torch.Tensor = self.model.encode_stochastic(batch, z_sem, T=self.forward_t)
        return z_sem, xT
    
    def encode_latent_only(self, batch: torch.Tensor):
        batch = batch.to(device)
        z_sem: torch.Tensor = self.model.encode(batch)
        return z_sem
    
    def decode(self, xT, z_sem) -> torch.Tensor:
        pred = self.model.render(xT, z_sem, T=self.backward_t, grads=True)
        return pred

    def generate_more_cf(
        self, z_sem: torch.Tensor, num_cfs: int, batch_size: int
    ) -> torch.Tensor:
        """Generates more counterfactuals by choosing random stochastic codes (xT) as the base noise.

        Parameters
        ----------
        z_sem : torch.Tensor
            The latent tensor to condition the generation on.
        num_cfs : int
            The number of counterfactuals to generate

        Returns
        -------
        torch.Tensor
            The batch of counterfactuals
        """
        with torch.no_grad():
            noises = torch.randn(num_cfs, 3, self.conf.img_size, self.conf.img_size).to(
                z_sem
            )

            preds = []
            # TODO Batch size I guess
            for i in tqdm(range(num_cfs), desc="Generating CFs"):
                xT = noises[i][None]
                pred = self.decode(xT, z_sem)
                preds.append(pred.detach())

            return torch.cat(preds, dim=0)


class RedModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean red value, assuming x is of shape (N, C, H, W)
        return x[:, 0].mean(dim=(1, 2))[None]
