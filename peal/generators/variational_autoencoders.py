import torch

from torch import nn
from pydantic import PositiveInt

from peal.generators.interfaces import InvertibleGenerator
from peal.architectures.downstream_models import SequentialModel
from peal.configs.generators.template import VAEConfig


class VAE(InvertibleGenerator):
    """
    Implements a Variational Autoencoder (VAE) as a generator in a standartized way
    so that it can be used in the PEAL framework.

    Args:
        InvertibleGenerator (nn.Module): The base class for all invertible generators
    """

    def __init__(
        self,
        config: VAEConfig,
        input_channels: PositiveInt,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
    ):
        """
        The constructor of the VAE class.

        Args:
            config (VAEConfig): The configuration dictionary of the experiment.
            input_channels (PositiveInt): The number of input channels of the VAE.
            encoder (nn.Module, optional): The encoder of the VAE. Defaults to None.
            decoder (nn.Module, optional): The decoder of the VAE. Defaults to None.
        """
        super().__init__()
        self.config = config
        self.input_channels = input_channels
        if encoder is None:
            self.encoder = SequentialModel(self.config.encoder, self.input_channels)

        else:
            self.encoder = encoder

        if decoder is None:
            self.decoder = SequentialModel(
                self.config.decoder, self.encoder.output_channels, self.input_channels
            )

        else:
            self.decoder = decoder

        # The prior is a standard normal distribution
        # TODO this will become a problem with external encoders
        self.register_buffer(
            "mean",
            torch.zeros(
                self.encoder.output_channels,
            ),
        )
        self.register_buffer(
            "var",
            torch.ones(
                self.encoder.output_channels,
            ),
        )

    def prior(self):
        return torch.distributions.Normal(self.mean, self.var)

    def encode(self, x):
        """
        The encoder is a function that maps the input to the latent space.

        Args:
            x (torch.tensor): The input

        Returns:
            torch.tensor: The latent space representation of the input
        """
        return [self.encoder(x)]

    def decode(self, z):
        """
        The decoder is a function that maps the latent space to the input space.

        Args:
            z (torch.tensor): The latent space representation of the input

        Returns:
            torch.tensor: The input
        """
        return self.decoder(z[0] if len(z) == 1 else z)

    def sample_z(self):
        """
        The prior samples from the latent space.

        Returns:
            torch.tensor: The sample from the latent space
        """
        return [self.prior().sample()]

    def log_prob_z(self, z):
        """
        Estimates the log probability of a sample in latent space.

        Args:
            z (torch.tensor): The sample from the latent space

        Returns:
            torch.tensor: The log probability of the sample
        """
        return self.prior().log_prob(z[0])

    def forward(self, x):
        """
        The forward pass of the VAE.

        Args:
            x (torch.tensor): The input

        Returns:
            tuple(torch.tensor, torch.tensor): The reconstructed input and the latent space representation of the input
        """

        z = self.encode(x)
        if isinstance(self.decoder, Latent2SequenceDecoder):
            x_hat = self.decode([z[0] + torch.randn_like(z[0]), x])

        else:
            x_hat = self.decode([z[0] + torch.randn_like(z[0])])

        return x_hat, z
