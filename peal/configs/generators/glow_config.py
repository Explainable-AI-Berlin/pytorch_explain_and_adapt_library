from typing import Union

from peal.configs.generators.generator_config import GeneratorConfig
from peal.configs.data.data_config import DataConfig


class GlowGeneratorConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str = "GlowGenerator"
    """
    The config of the data.
    """
    data: DataConfig = DataConfig()
    batch: int = 16
    iter: int = 400000
    n_flow: int = 32
    n_block: int = 4
    no_lu: bool = False
    affine: bool = False
    n_bits: int = 5
    lr: float = 1e-4
    img_size: int = 64
    temp: float = 0.7
    n_sample: int = 20
    base_path: str = "glow_run"
    x_selection: Union[str, type(None)] = None