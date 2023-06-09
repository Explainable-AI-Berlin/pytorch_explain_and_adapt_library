import torch

from torch import nn


class InvertibleGenerator(nn.Module):
    def encode(self, x):
        pass

    def decode(self, z):
        pass

    def sample_z(self):
        pass

    def log_prob_z(self, z):
        pass

    def sample_x(self):
        z = self.sample_z()
        return self.decode(z)

    def log_prob_x(self, x):
        z = self.encode(x)
        return self.log_prob_z(z)


class EditCapablGenerator(nn.Module):
    def edit(
        x_in: torch.Tensor,
        target_confidence_goal: float,
        target_classes: torch.Tensor,
        classifier: nn.Module,
    ):
        pass
