from typing import Union

from peal.configs.explainers.explainer_config import ExplainerConfig


class ACEConfig(ExplainerConfig):
    """
    This class defines the config of a ACEConfig.
    """

    """
    The type of explanation that shall be used.
    Options: ['counterfactual', 'lrp']
    """
    explainer_type: str = "ACE"
    attack_iterations: Union[list, int] = [10,50]
    sampling_time_fraction: Union[list, float] = [0.1,0.3]
    dist_l1: Union[list, float] = [1.0, 0.0001]
    dist_l2: Union[list, float] = 0.0
    sampling_inpaint: Union[list, float] = [0.3, 0.1]
    sampling_dilation: Union[list, int] = 17
    timestep_respacing: Union[list, int] = 50
    attempts: int = 5
    clip_denoised: bool = True  # Clipping noise
    batch_size: int = 32  # Batch size
    gpu: str = "0"  # GPU index, should only be 1 gpu
    save_images: bool = False  # Saving all images
    num_samples: int = 500000000000  # useful to sample few examples
    # setting this to true will slow the computation time but will have identic results
    # hwhen using the checkpoint backwards
    cudnn_deterministic: bool = False
    # path args
    base_path: str = ""  # DDPM weights path
    # Experiment name (will store the results at Output/Results/exp_name)
    exp_name: str = "example_name"
    # attack args
    seed: int = 4  # Random seed
    # Attack method (currently 'PGD', 'C&W', 'GD' and 'None' supported)
    attack_method: str = "PGD"
    attack_epsilon: float = 255  # L inf epsilon bound (will be devided by 255)
    attack_step: float = 1.0  # Attack update step (will be devided by 255)
    attack_joint: bool = True  # Set to false to generate adversarial attacks
    # use checkpoint method for backward. Beware, this will substancially slow down the CE
    # generation!
    attack_joint_checkpoint: bool = False
    # number of DDPM iterations per backward process. We highly recommend have a larger
    # backward steps than batch size (e.g have 2 backward steps and batch size of 1 than 1
    # backward step and batch size 2)
    attack_checkpoint_backward_steps: int = 1
    # Use dime2 shortcut to transfer gradients. We do not recommend it.
    attack_joint_shortcut: bool = False
    # schedule for the distance loss. We did not used any for our results
    dist_schedule: str = "none"
    # filtering args
    sampling_stochastic: bool = True  # Set to False to remove the noise when sampling
    # dataset
    chunks: int = 1  # Chunking for spliting the CE generation into multiple gpus
    chunk: int = 0  # current chunk (between 0 and chunks - 1)
    merge_chunks: bool = False  # to merge all chunked results
    y_target_goal_confidence: float = 0.5  # to merge all chunked results
