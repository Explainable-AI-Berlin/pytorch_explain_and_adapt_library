from typing import Union

from peal.configs.data.data_config import DataConfig
from peal.configs.editors.editor_config import EditorConfig


class DDPMInversionConfig(EditorConfig):
    editor_type: str = "DDPMInversion"
    model_id: str = "CompVis/stable-diffusion-v1-4"
    generator_type: str = "DDPMInversionAdaptor"
    base_path: str = "peal_runs/ddpm_inversion"
    num_diffusion_steps: int = 100
    cfg_scale_src: float = 3.5
    cfg_scale_tar: float = 15
    eta: float = 1.0
    mode: str = "our_inv"  # modes: our_inv,p2pinv,p2pddim,ddim
    skip: int = 36
    xa: float = 0.6
    sa: float = 0.2
    data: Union[type(None), DataConfig] = None
