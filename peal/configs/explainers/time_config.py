from typing import Union
from peal.configs.explainers.explainer_config import ExplainerConfig


class TIMEConfig(ExplainerConfig):
    """
    This class defines the config of a ACEConfig.
    """

    """
    The type of explanation that shall be used.
    Options: ['counterfactual', 'lrp']
    """
    explainer_type: str = "TIME"
    editing_type: str = "ddpm_inversion"
    sd_model: str = "CompVis/stable-diffusion-v1-4"
    use_negative_guidance_denoise: bool = True
    use_negative_guidance_inverse: bool = True
    guidance_scale_denoising: list = [4]
    guidance_scale_invertion: list = [4]
    num_inference_steps: list = [50]
    exp_name: str = "time"
    label_target: int = -1
    label_query: int = 31
    class_custom_token: list = [
        "|<A*01>| |<A*02>| |<A*03>|",
        "|<A*11>| |<A*12>| |<A*13>|",
    ]
    base_prompt: str = "A photo of a |<C*1>| |<C*2>| |<C*3>|"
    prompt_connector: str = " that is "
    chunks: int = 1
    chunk: int = 0
    enable_xformers_memory_efficient_attention: bool = False
    use_fp16: bool = False
    sd_image_size: int = 128
    custom_obj_token: str = "|<C*>|"
    p: float = 0.93
    l2: float = 0.0
    inference_batch_size: int = 1
    classifier_image_size: int = 128
    recover: bool = False
    num_samples: int = 9999999999999999
    merge_chunks: bool = False
    generic_custom_tokens: list = ["|<C*1>|", "|<C*2>|", "|<C*3>|"]
    total_num_inference_steps: int = 50
    custom_tokens_context: list = ["|<C*1>|", "|<C*2>|", "|<C*3>|"]
    custom_tokens_init: list = ["centered", "realistic", "celebrity"]
    mini_batch_size: int = 1
    gpu: str = "0"
    lr: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-9
    weight_decay: float = 1e-4
    iterations: int = 500
    max_epoch: int = 10
    train_batch_size: int = 64
    image_size: int = 128
    seed: int = 99999999
