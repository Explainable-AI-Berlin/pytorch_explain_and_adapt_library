import copy
import math
import torch
import numpy as np

from torch import nn

from peal.architectures.coupling_layers import (
    Block,
    gaussian_log_p
)
from peal.architectures.interfaces import InvertibleGenerator

class Glow(InvertibleGenerator):
    def __init__(
        self, config,
    ):
        super().__init__()
        self.config = config

        if isinstance(self.config['architecture']['n_block'], str) and self.config['architecture']['n_block'] == 'auto':
            self.n_block = int(math.log(self.config['data']['input_size'][1], 2)) - 2

        else:
            self.n_block = self.config['architecture']['n_block']

        self.blocks = nn.ModuleList()
        n_channel = self.config['data']['input_size'][0]
        for i in range(self.n_block - 1):
            self.blocks.append(Block(
                n_channel,
                self.config['architecture']['n_flow'],
                affine = self.config['architecture']['affine'],
                conv_lu = self.config['architecture']['conv_lu']
            ))
            n_channel *= 2

        self.blocks.append(Block(n_channel, self.config['architecture']['n_flow'], split=False, affine=self.config['architecture']['affine']))

    def forward(self, x):
        log_p_sum = 0
        logdet = 0
        out = x
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return z_outs, log_p_sum, logdet

    def encode(self, x):
        out = x
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)

        return z_outs

    def decode(self, z, reconstruct=True):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.reverse(z[-1], z[-1], reconstruct=reconstruct)

            else:
                x = block.reverse(x, z[-(i + 1)], reconstruct=reconstruct)

        return x

    def sample_z(self, n = 'auto'):
        if isinstance(n, str) and n == 'auto':
            n_sample = self.config['training']['val_batch_size']

        else:
            n_sample = n

        z_sample = []
        z_shapes = self.calc_z_shapes()
        for z in z_shapes:
            z_new = torch.randn(n_sample, *z) * self.config['architecture']['temp']
            z_sample.append(z_new.to(next(self.parameters()).device))

        return z_sample

    def log_prob_z(self, z):
        log_probs = []

        for it, block in enumerate(self.blocks):
            zero = torch.zeros_like(z[it])
            mean, log_sd = block.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(z[it], mean, log_sd)
            log_probs.append(torch.flatten(log_p, start_dim = 1).sum(1))

        log_p_sum = torch.sum(torch.stack(log_probs, dim = 0), dim = 0)

        n_pixel = np.prod(self.config['data']['input_size'])
        log_p = log_p_sum - self.config['architecture']['n_bits'] * n_pixel
        return - torch.tensor(log_p / (math.log(2) * n_pixel))

    def calc_z_shapes(self):
        z_shapes = []

        n_channel = copy.deepcopy(self.config['data']['input_size'][0])
        input_size = copy.deepcopy(self.config['data']['input_size'][1])

        for i in range(self.n_block - 1):
            input_size //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))

        return z_shapes