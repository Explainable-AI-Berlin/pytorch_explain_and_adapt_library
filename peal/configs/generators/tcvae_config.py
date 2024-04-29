from typing import Union

from peal.configs.generators.generator_config import GeneratorConfig
from peal.configs.data.data_config import DataConfig


class DiveTCVAEConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str = "DiveTCVAE"
    """
    The config of the data.
    """
    data: DataConfig = DataConfig()
    full_args: dict = {}
    wrapper: str = "tcvae"
    # Hardware
    ngpu: int = 1
    amp: int = 0

    # Optimization
    batch_size: int = 64  # (ngpu=1 for now),
    target_loss: str = "val_loss"
    lr_tcvae: float = 0.0004
    max_epoch: int = 400  # constraint: >=4
    clip: bool = True

    # Model
    model: str = "biggan"
    backbone: str = "resnet"
    channels_width: int = 4
    z_dim: int = 128
    mlp_width: int = 4
    mlp_depth: int = 2
    # savedir : 'peal_runs/dive_tcvae'
    base_path: str = "/home/space/datasets/peal/peal_runs/dive_celeba"
    savedir: str = "/home/space/datasets/peal/peal_runs/dive_celeba"

    # TCVAE
    beta: float = (
        0.001  # the idea is to be able to interpolate while getting good reconstructions
    )
    tc_weight: int = (
        1  # we keep the total_correlation penalty high to encourage disentanglement
    )
    vgg_weight: float = 1.0
    pix_mse_weight: float = 0.0001
    beta_annealing: bool = True
    dp_prob: float = 0.3

    # Data
    height: int = 128
    width: int = 128
    crop_size: Union[int, None] = None

    # Attacks
    lr_dive: float = 0.01
    max_iters: int = 20
    cache_batch_size: int = 64
    force_cache: bool = False
    stop_batch_threshold: float = 0.9
    attribute: str = "Smiling"
    num_explanations: int = 8
    method: str = "fisher spectral inv"
    reconstruction_weight: float = 10.0
    lasso_weight: float = 1.0
    diversity_weight: float = 1
    n_samples: int = 10
    fisher_samples: int = 0
