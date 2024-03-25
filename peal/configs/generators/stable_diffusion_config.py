from typing import Union

from peal.configs.generators.generator_config import GeneratorConfig
from peal.configs.data.data_config import DataConfig


class StableDiffusionConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str = "StableDiffusion"