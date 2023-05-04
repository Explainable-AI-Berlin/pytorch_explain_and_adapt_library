import torch

from torch import nn

from peal.generators.interfaces import InvertibleGenerator
from peal.architectures.model_parts import Img2LatentEncoder
from peal.architectures.model_parts import Sequence2LatentEncoder
from peal.architectures.model_parts import Vector2LatentEncoder
from peal.architectures.model_parts import Latent2ImgDecoder
from peal.architectures.model_parts import Latent2SequenceDecoder
from peal.architectures.model_parts import Latent2VectorDecoder


class VAE(InvertibleGenerator):
    """
    Implements a Variational Autoencoder (VAE) as a generator in a standartized way so that it can be used in the PEAL framework.

    Args:
        InvertibleGenerator (nn.Module): The base class for all invertible generators
    """

    def __init__(self, config, encoder=None, decoder=None):
        """
        The constructor of the VAE class.

        Args:
            config (dict): The configuration dictionary of the experiment.
            encoder (nn.Module, optional): The encoder of the VAE. Defaults to None.
            decoder (nn.Module, optional): The decoder of the VAE. Defaults to None.
        """
        super().__init__()
        self.config = config
        if encoder is None:
            if self.config["data"]["input_type"] == "image":
                self.encoder = Img2LatentEncoder(self.config)

            elif self.config["data"]["input_type"] == "sequence":
                self.encoder = Sequence2LatentEncoder(
                    num_blocks=config["architecture"]["num_blocks"],
                    embedding_dim=config["architecture"]["neuron_numbers_encoder"][-1],
                    num_heads=config["architecture"]["num_heads"],
                    input_channels=config["data"]["input_size"][-1] + 2,
                    activation=nn.ReLU,
                )

            elif self.config["data"]["input_type"] == "symbolic":
                self.encoder = Vector2LatentEncoder(
                    input_channels=self.config["data"]["input_size"][0],
                    activation=nn.ReLU,
                    neuron_numbers=self.config["architecture"][
                        "neuron_numbers_encoder"
                    ],
                )

        else:
            self.encoder = encoder

        if decoder is None:
            if self.config["data"]["input_type"] == "image":
                self.decoder = Latent2ImgDecoder(self.config)

            elif self.config["data"]["input_type"] == "sequence":
                self.decoder = Latent2SequenceDecoder(
                    num_blocks=config["architecture"]["num_blocks"],
                    embedding_dim=config["architecture"]["neuron_numbers_encoder"][-1],
                    num_heads=config["architecture"]["num_heads"],
                    input_channels=config["data"]["input_size"][-1] + 2,
                    activation=nn.ReLU,
                    max_length=self.config["data"]["input_size"][0],
                    embedding=list(self.encoder.children())[0],
                )

            elif self.config["data"]["input_type"] == "symbolic":
                self.decoder = Latent2VectorDecoder(
                    output_size=self.config["data"]["input_size"][0],
                    num_hidden_in=self.config["architecture"]["neuron_numbers_encoder"][
                        -1
                    ],
                    activation=nn.ReLU,
                    neuron_numbers=self.config["architecture"][
                        "neuron_numbers_decoder"
                    ],
                )

        else:
            self.decoder = decoder

        # The prior is a standard normal distribution
        self.register_buffer(
            "mean",
            torch.zeros(
                self.config["architecture"]["neuron_numbers_encoder"][-1],
            ),
        )
        self.register_buffer(
            "var",
            torch.ones(
                self.config["architecture"]["neuron_numbers_encoder"][-1],
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
