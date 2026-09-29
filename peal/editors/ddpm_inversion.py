"""Config for the edit-friendly DDPM inversion editor.

PEAL's Stable Diffusion generator can hand the actual image editing to the
"edit friendly DDPM inversion" code vendored under
``peal.dependencies.ddpm_inversion``. That code inverts an image into a
sequence of per-step noise maps with the source prompt and re-synthesises it
with the target prompt. This module only holds the pydantic config the editor
is built from; the editor class itself lives in the dependency package.
"""

from typing import Union

from peal.data.interfaces import DataConfig
from peal.editors.interfaces import EditorConfig


class DDPMInversionConfig(EditorConfig):
    """Settings of the edit-friendly DDPM inversion editor.

    Consumed by ``peal.dependencies.ddpm_inversion.ddpm_inversion.DDPMInversion``
    and by ``StableDiffusion.initialize`` in
    ``peal.generators.stable_diffusion_generator``, which overrides
    ``cfg_scale_src`` and ``cfg_scale_tar`` from the explainer config.

    Parameters
    ----------
    editor_type : str
        Registry name of the editor, ``"DDPMInversion"``.
    model_id : str
        Hugging Face id of the Stable Diffusion pipeline that is inverted.
    generator_type : str
        Name under which the editor is registered as a generator.
    base_path : str
        Directory for the editor's outputs.
    num_diffusion_steps : int
        Number of scheduler timesteps used for inversion and re-synthesis.
    cfg_scale_src : float
        Classifier-free guidance scale during the forward (inversion) pass.
    cfg_scale_tar : float
        Classifier-free guidance scale during the reverse (editing) pass.
    eta : float
        DDIM ``eta``; ``1.0`` gives the fully stochastic DDPM sampler the
        edit-friendly inversion is built on.
    mode : str
        Editing mode of the original inversion CLI (``our_inv``, ``p2pinv``,
        ``p2pddim`` or ``ddim``). The vendored editor does not read it.
    skip : int
        Number of the noisiest steps that are skipped: the reverse pass starts
        from ``wts[num_diffusion_steps - skip]`` and reuses the first
        ``num_diffusion_steps - skip`` stored noise maps.
    xa, sa : float
        Cross- and self-attention replacement fractions of the prompt-to-prompt
        modes. Not read by the vendored editor.
    data : DataConfig or None
        Data config whose ``normalization`` (mean, std) the editor uses to map
        predictor-space images to ``[0, 1]`` and back.
    """

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
