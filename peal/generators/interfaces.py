import torch

from torch import nn
from typing import Tuple


class Generator(nn.Module):
    def sample_x(self, batch_size=1):
        '''
        This function samples a batch of data samples from the generator
        '''
        pass

class InvertibleGenerator(Generator):
    def encode(self, x):
        '''
        This function encodes a batch of data samples to latent vectors
        Args:
            x: A batch of data samples

        Returns:
            A batch of latent vectors
        '''
        pass

    def decode(self, z):
        '''
        This function decodes a batch of latent vectors to data samples
        Args:
            z: A batch of latent vectors

        Returns:
            A batch of data samples
        '''
        pass

    def sample_z(self, batch_size=1):
        '''
        This function samples a batch of latent vectors from the prior
        '''
        pass

    def log_prob_z(self, z):
        '''
        This function computes the log probability of a batch of latent vectors
        Args:
            z: A batch of latent vectors

        Returns:
            The log probability of the batch of latent vectors
        '''
        pass

    def sample_x(self, batch_size=1):
        '''
        This function samples a batch of data samples from the generator
        '''
        z = self.sample_z(batch_size)
        return self.decode(z)

    def log_prob_x(self, x):
        '''
        This function computes the log probability of a batch of data samples
        Args:
            x: A batch of data samples

        Returns:
            The log probability of the batch of data samples
        '''
        z = self.encode(x)
        return self.log_prob_z(z)


class EditCapableGenerator(Generator):
    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        target_classes: torch.Tensor,
        classifier: nn.Module,
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        '''
        This function edits the input to match the target confidence goal and target classes
        Args:
            x_in: The input
            target_confidence_goal: The target confidence goal
            target_classes: The target classes
            classifier: The classifier according to which the confidence is measured

        Returns:
            list[torch.Tensor]: List of the counterfactuals
            list[torch.Tensor]: List of the differences in latent codes. In the simplest case just x_in - x_counterfactual
            list[torch.Tensor]: List of the achieved target confidences of the counterfactuals
            list[torch.Tensor]: List of x_in. This is necessary since the counterfactuals might be in a different order
        '''
        pass
