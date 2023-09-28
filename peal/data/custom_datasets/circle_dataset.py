import pandas as pd
import numpy as np
from peal.data.datasets import SymbolicDataset
from peal.global_utils import load_yaml_config
from peal.generators.interfaces import Generator
import torch


class CircleDataset(SymbolicDataset):

    def __init__(self):
        super().__init__()

    @staticmethod
    def circle_fid(samples):
        radius = 1
        return (((samples.pow(2)).sum(dim=-1) - radius).pow(2)).mean()

    @staticmethod
    def angle_cdf(samples):
        scores = abs(samples[:, 1] / samples[:, 0])

        first_quad_mask = (samples[:, 0] > 0) & (samples[:, 1] > 0)
        second_quad_mask = (samples[:, 0] < 0) & (samples[:, 1] > 0)
        third_quad_mask = (samples[:, 0] < 0) & (samples[:, 1] < 0)
        fourth_quad_mask = (samples[:, 0] > 0) & (samples[:, 1] < 0)
        theta_1 = torch.atan(scores) * first_quad_mask
        theta_1 = theta_1[theta_1 != 0]
        theta_2 = (torch.pi - torch.atan(scores)) * second_quad_mask
        theta_2 = theta_2[theta_2 != 0]
        theta_3 = (torch.pi + torch.atan(scores)) * third_quad_mask
        theta_3 = theta_3[theta_3 != 0]
        theta_4 = (2 * torch.pi - torch.atan(scores)) * fourth_quad_mask
        theta_4 = theta_4[theta_4 != 0]
        thetas, indices = torch.cat([theta_1, theta_2, theta_3, theta_4]).sort(dim=-1)

        return thetas

    def circle_ks(self, samples, true_data):
        true_thetas = CircleDataset.angle_cdf(true_data)
        sample_thetas = CircleDataset.angle_cdf(samples)

        ecdf = torch.arange(self.config['num_samples']) / self.config['num_samples']
        true_cdf = (sample_thetas[:, None] >= true_thetas[None, :]).sum(-1) / len(true_data)
        return torch.max(torch.abs((true_cdf - ecdf)))

    def track_generator_performance(self, generator: Generator, batch_size=1):
        samples = generator.sample_x(batch_size).detach()

        ks = self.circle_ks(samples, self.true_data)
        fid = CircleDataset.circle_fid(samples)

        harmonic_mean = 1 / (1 / fid + 1 / ks)

        return {
            'KS': ks,
            'FID': fid,
            'harmonic_mean_fid_ks': harmonic_mean

        }

    __name__ = "circle"
