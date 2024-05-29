from typing import Union

from peal.configs.generators.generator_config import GeneratorConfig
from peal.configs.data.data_config import DataConfig
from peal.configs.models.model_config import TaskConfig


class StableDiffusionConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str = "StableDiffusion"
    base_path: str = "/home/space/datasets/peal/peal_runs/stable_diffusion"
    full_args: Union[None, dict] = None
    """
    The config of the data.
    """
    data: DataConfig = DataConfig()
    sd_model: str = "CompVis/stable-diffusion-v1-4"
    #
    revision: Union[str, type(None)] = None
    variant: Union[str, type(None)] = None
    dataset_name: Union[str, type(None)] = None
    dataset_config_name: Union[str, type(None)] = None
    train_data_dir: Union[str, type(None)] = None
    image_column: Union[str, type(None)] = "image"
    caption_column: Union[str, type(None)] = "text"
    validation_prompt: Union[str, type(None)] = None
    num_validation_images: int = 4
    validation_epochs: int = 1
    max_train_samples: Union[int, type(None)] = None
    cache_dir: Union[str, type(None)] = None
    seed: Union[int, type(None)] = None
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    train_batch_size: int = 16
    num_train_epochs: int = 100
    max_train_steps: Union[int, type(None)] = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    snr_gamma: Union[float, type(None)] = None
    use_8bit_adam: bool = False
    allow_tf32: bool = False
    dataloader_num_workers: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    push_to_hub: bool = False
    hub_token: Union[str, type(None)] = None
    prediction_type: Union[str, type(None)] = None
    hub_model_id: Union[str, type(None)] = None
    logging_dir: Union[str, type(None)] = "logs"
    mixed_precision: Union[str, type(None)] = None
    report_to: Union[str, type(None)] = "tensorboard"
    local_rank: int = 1
    checkpointing_steps: int = 500
    checkpoints_total_limit: Union[int, type(None)] = None
    resume_from_checkpoint: Union[str, type(None)] = None
    enable_xformers_memory_efficient_attention: bool = False
    noise_offset: float = 0.0
    rank: int = 4
    task_config: Union[TaskConfig, type(None)] = None

