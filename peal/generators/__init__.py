"""Generative models that render the counterfactuals the explainers need.

Every generator implements ``interfaces.Generator``; invertible ones
(``InvertibleGenerator``) add ``encode`` / ``decode`` and edit-capable ones
(``EditCapableGenerator``) an ``edit`` that moves an image across a
predictor's decision boundary. Implementations: the DiffAE-based
``diffusion_autoencoder`` and its RAE twin, pixel / latent DDPMs
(``ddpm_generator``, ``ddpm_pathldm``), Stable Diffusion 3, Flux and the
Stable Diffusion autoencoder, PathLDM for histopathology and a tabular DDPM.
``generator_factory.get_generator`` builds one
from a ``GeneratorConfig`` yaml; ``deeplift_resnet`` provides DeepLift-safe
ResNets used for attribution-guided generation.
"""
