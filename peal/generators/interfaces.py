import torch

from torch import nn


class Generator(nn.Module):
    def sample_x(self, batch_size=1):
        '''
        This function samples a batch of data samples from the generator
        '''
        pass

class InvertibleGenerator(Generator):
    def encode(self, x):
        pass

    def decode(self, z):
        pass

    def sample_z(self, batch_size=1):
        '''
        This function samples a batch of latent vectors from the prior
        '''
        pass

    def log_prob_z(self, z):
        pass

    def sample_x(self, batch_size=1):
        '''
        This function samples a batch of data samples from the generator
        '''
        z = self.sample_z(batch_size)
        return self.decode(z)

    def log_prob_x(self, x):
        z = self.encode(x)
        return self.log_prob_z(z)


class EditCapableGenerator(Generator):
    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        target_classes: torch.Tensor,
        classifier: nn.Module,
    ):
        pass
