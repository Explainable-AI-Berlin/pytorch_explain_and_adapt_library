from tqdm import tqdm
import yaml
import torch
from torch import nn
from experiment import LitModel
from templates import (
    square64_autoenc,
    ffhq256_autoenc,
)


# Use their implementation
class DAE(nn.Module):
    def __init__(
        self,
        dataset: str,
        checkpoint_path: str,
        forward_t: int = 250,
        backward_t: int = 20,
    ):
        super().__init__()

        self.forward_t = forward_t
        self.backward_t = backward_t

        if dataset == "square":
            self.conf = square64_autoenc()
        else:  # Assume ffhq
            self.conf = ffhq256_autoenc()

        assert (
            self.conf.model_type.has_autoenc()
        ), "Model must have a latent autoencoder."

        model = LitModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, conf=self.conf, map_location="cpu"
        )

        self.model = model.eval()
        del self.model.model  # TODO: Make sure to use EMA model?

    def forward(self, x):
        return self.encode(x)

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        z_sem: torch.Tensor = self.model.encode(x)
        xT: torch.Tensor = self.model.encode_stochastic(x, z_sem)
        return z_sem, xT

    def to(self, device):
        self.model.ema_model.to(device)
        return self

    def decode(self, z_sem, xT) -> torch.Tensor:
        return self.model.render(xT, z_sem, T=self.backward_t, grads=True)

    @torch.no_grad()
    def generate_more_cf(self, z_sem: torch.Tensor, num_cfs: int) -> torch.Tensor:
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
