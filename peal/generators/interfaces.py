"""Abstract interfaces every PEAL generator implements.

A generator is the model that produces counterfactual images for an
explainer or an adaptor. This module fixes the three capability levels the
rest of PEAL programs against: :class:`Generator` (can be sampled from and
trained), :class:`InvertibleGenerator` (also has a latent space one can
``encode`` into and ``decode`` from, which is what the diffusion
autoencoders provide) and :class:`EditCapableGenerator` (can run the whole
counterfactual search itself). :class:`GeneratorConfig` is the pydantic base
that every concrete generator config extends.
"""

import torch

from torch import nn
from typing import Tuple

from peal.explainers.interfaces import ExplainerConfig
from typing import Union
from pydantic import BaseModel
from abc import ABC, abstractmethod


class GeneratorConfig(BaseModel):
    """
    This class defines the config of a generator.
    """

    generator_type: str
    """
    The type of generator that shall be used.
    """
    category: str = "generator"
    """
    The category of the config
    """
    batch_size: int = 1
    """
    The batch size of the generator.
    """
    current_fid: float = float("inf")
    """
    The name of the class.
    """
    seed: int = 0


class Generator(nn.Module, ABC):
    """Base class of every PEAL generator.

    A ``torch.nn.Module`` that a concrete generator subclasses to provide the
    two operations the library needs from any generative model: drawing
    samples and fitting itself to a dataset. Both raise ``NotImplementedError``
    here, so a subclass only overrides what it actually supports.
    """

    @abstractmethod
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
    """A generator with an explicit, invertible latent space.

    Adds ``encode``/``decode`` and a prior (``sample_z``, ``log_prob_z``) to
    :class:`Generator`; the diffusion autoencoders, the VAEs and the RAE
    pipeline are all of this kind. Counterfactual methods such as DiDAE need
    this level because they perturb a latent code and decode the result.
    ``sample_x`` and ``log_prob_x`` are already implemented in terms of the
    latent operations, so a subclass only supplies the four abstract methods.

    Notes
    -----
    The optional ``t``, ``stochastic`` and ``num_steps`` arguments of
    ``encode``/``decode`` are the diffusion controls (how far to noise, whether
    to use the stochastic sampler, how many sampler steps); non-diffusion
    subclasses ignore them.
    """

    @abstractmethod
    def encode(self, x, t=1.0, stochastic=None, num_steps=None):
        """
        This function encodes a batch of data samples to latent vectors
        Args:
            x: A batch of data samples

        Returns:
            A batch of latent vectors
        If not implemented, it will throw a NotImplementedError.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, z, t=1.0, stochastic=None, num_steps=None):
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
    """A generator that can search for counterfactuals on its own.

    Where an explainer normally drives the search, these generators implement
    :meth:`edit` end to end: given a batch of inputs, the predictor and the
    source/target classes, they return the counterfactuals they found together
    with the latent differences, the confidences the predictor assigns them and
    the matching originals. The diffusion autoencoders and the tabular VAE use
    this to keep the search inside their own latent parameterisation.
    """

    @abstractmethod
    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        predictor: nn.Module,
        explainer_config: ExplainerConfig,
        predictor_datasets: list,
        boolmask_in: Union[torch.Tensor, type(None)] = None,
        attempt_number: Union[int, type(None)] = None,
        pbar: object = None,
        mode: object = "",
        base_path: object = "",
    ) -> Tuple[
        Tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ],
        torch.Tensor,
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
