import torch

from torch import nn
from typing import Tuple

from peal.explainers.interfaces import ExplainerConfig

from pydantic import BaseModel
class GeneratorConfig(BaseModel):
    """
    This class defines the config of a generator.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str
    """
    The category of the config
    """
    category: str = 'generator'
    """
    The name of the class.
    """
    current_fid: float = float('inf')

class Generator(nn.Module):
    def sample_x(self, batch_size=1):
        """
        This function samples a batch of data samples from the generator.
        If not implemented, it will throw a NotImplementedError.
        """
        raise NotImplementedError

    def train_model(self):
        """
        This function trains the generator.
        If not implemented, it will throw a NotImplementedError.
        """
        raise NotImplementedError


class InvertibleGenerator(Generator):
    def encode(self, x):
        """
        This function encodes a batch of data samples to latent vectors
        Args:
            x: A batch of data samples

        Returns:
            A batch of latent vectors
        If not implemented, it will throw a NotImplementedError.
        """
        raise NotImplementedError

    def decode(self, z):
        """
        This function decodes a batch of latent vectors to data samples
        Args:
            z: A batch of latent vectors

        Returns:
            A batch of data samples
        If not implemented, it will throw a NotImplementedError.
        """
        raise NotImplementedError

    def sample_z(self, batch_size=1):
        """
        This function samples a batch of latent vectors from the prior
        If not implemented, it will throw a NotImplementedError.
        """
        raise NotImplementedError

    def log_prob_z(self, z):
        """
        This function computes the log probability of a batch of latent vectors
        Args:
            z: A batch of latent vectors

        Returns:
            The log probability of the batch of latent vectors
        If not implemented, it will throw a NotImplementedError.
        """
        raise NotImplementedError

    def sample_x(self, batch_size=1):
        """
        This function samples a batch of data samples from the generator
        """
        z = self.sample_z(batch_size)
        return self.decode(z)

    def log_prob_x(self, x):
        """
        This function computes the log probability of a batch of data samples
        Args:
            x: A batch of data samples

        Returns:
            The log probability of the batch of data samples
        """
        z = self.encode(x)
        return self.log_prob_z(z)


class EditCapableGenerator(Generator):
    def edit(
        self,
            x_in: torch.Tensor,
            target_confidence_goal: float,
            source_classes: torch.Tensor,
            target_classes: torch.Tensor,
            predictor: nn.Module,
            explainer_config: ExplainerConfig,
            predictor_datasets: list,
            pbar: object = None,
            mode: object = "",
            base_path: object = "",
    ) -> Tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        """
        This function edits the input to match the target confidence goal and target classes
        Args:
            predictor_datasets:
            explainer_config:
            base_path:
            x_in: The input
            target_confidence_goal: The target confidence goal
            source_classes: The source classes
            target_classes: The target classes
            predictor: The predictor according to which the confidence is measured
            pbar: A progress bar
            mode: The mode of the edit. This is used to determine the edit method

        Returns:
            list[torch.Tensor]: List of the counterfactuals
            list[torch.Tensor]: List of the differences in latent codes. In the simplest case just x_in - x_counterfactual
            list[torch.Tensor]: List of the achieved target confidences of the counterfactuals
            list[torch.Tensor]: List of x_in. This is necessary since the counterfactuals might be in a different order
        If not implemented, it will throw a NotImplementedError.
        """
        raise NotImplementedError
