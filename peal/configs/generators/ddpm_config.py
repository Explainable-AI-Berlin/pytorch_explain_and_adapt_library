from typing import Union

from peal.configs.generators.generator_config import GeneratorConfig
from peal.configs.data.data_config import DataConfig


class DDPMConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str = "DDPM"
    """
    The path where the generator is stored.
    """
    base_path: str = "peal_runs/ddpm"
    """
    The config of the data.
    """
    data: DataConfig = DataConfig()
    """
    The number of channels
    """
    num_channels: int = 128
    image_size: Union[int, type(None)] = None
    num_res_blocks: int = 2
    num_heads: int = 4
    num_heads_upsample: int = -1
    num_head_channels: int = -1
    attention_resolutions: str = "32,16,8"
    channel_mult: str = ""
    dropout: float = 0.0
    class_cond: bool = False
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    use_fp16: bool = False
    use_new_attention_order: bool = False
    schedule_sampler: str = "uniform"
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_anneal_steps: int = 0
    batch_size: int = 1
    microbatch: int = -1  # -1 disables microbatches
    ema_rate: str = "0.9999"  # comma-separated list of EMA values
    log_interval: int = 10
    save_interval: int = 10000
    max_steps: int = 1000000
    resume_checkpoint: str = ""
    fp16_scale_growth: float = 1e-3
    output_path: str = "peal_runs/ddpm/outputs"
    gpus: str = ""
    use_hdf5: bool = False
    learn_sigma : bool = True
    diffusion_steps : int = 1000
    noise_schedule : str = "linear"
    timestep_respacing : str = ""
    use_kl : bool = False
    predict_xstart : bool = False
    rescale_timesteps : bool = False
    rescale_learned_sigmas : bool = False
    full_args: dict = {}
