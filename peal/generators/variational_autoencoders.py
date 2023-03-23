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
    def __init__(self, config, device, encoder = None, decoder = None):
        super().__init__()
        self.config = config
        self.device = device
        if encoder is None:
            if self.config['data']['type'] == 'image':
                self.encoder = Img2LatentEncoder(self.config).to(device)

            elif self.config['data']['type'] == 'sequence':
                self.encoder = Sequence2LatentEncoder(self.config).to(device)

            elif self.config['data']['type'] == 'symbolic':
                self.encoder = Vector2LatentEncoder(
                    input_channels=self.config['data']['input_size'][0],
                    activation=nn.ReLU(),
                    neuron_numbers=self.config['architecture']['neuron_numbers_encoder'],
                ).to(device)
        
        else:
            self.encoder = encoder

        if decoder is None:
            if self.config['data']['type'] == 'image':
                self.decoder = Latent2ImgDecoder(self.config).to(device)

            elif self.config['data']['type'] == 'sequence':
                self.decoder = Latent2SequenceDecoder(self.config).to(device)

            elif self.config['data']['type'] == 'symbolic':
                self.decoder = Latent2VectorDecoder(
                    output_size=self.config['data']['input_size'][0],
                    num_hidden_in=self.config['architecture']['neuron_numbers_encoder'][-1],
                    activation=nn.ReLU(),
                    neuron_numbers=self.config['architecture']['neuron_numbers_decoder'],
                ).to(device)

        else:
            self.decoder = decoder
        
        self.prior = torch.distributions.Normal(
            self.config['architecture']['neuron_numbers_encoder'][-1],
            self.config['architecture']['neuron_numbers_encoder'][-1]
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample_z(self):
        return self.prior.sample()

    def log_prob_z(self, z):
        return self.prior.log_prob(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat
