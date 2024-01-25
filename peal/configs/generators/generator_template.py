from typing import Union

from peal.configs.architectures.architecture_template import ArchitectureConfig


class GeneratorConfig:
    generator_type: str

class DDPMConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
    """

    """
    The config of the model.
    """
    model: ArchitectureConfig = None
    """
    The config of the diffusion.
    """
    diffusion: dict = None
    """
    The name of the class.
    """
    __name__: str = "peal.DDPMConfig"

    def __init__(
        self,
        model: Union[dict, ArchitectureConfig] = None,
        diffusion: dict = None,
    ):
        """
        The constructor of the DDPMConfig class.
        """
        if isinstance(model, ArchitectureConfig):
            self.model = model

        elif isinstance(model, dict):
            self.model = ArchitectureConfig(**model)

        if diffusion is not None:
            self.diffusion = diffusion

class VAEConfig(GeneratorConfig):
    """
    This class defines the config of a VAE.
    """

    """
    The config of the encoder.
    """
    encoder: ArchitectureConfig = None
    """
    The config of the decoder.
    """
    decoder: ArchitectureConfig = None
    """
    The name of the class.
    """
    __name__: str = "peal.VAEConfig"

    def __init__(
        self,
        encoder: Union[dict, ArchitectureConfig] = None,
        decoder: Union[dict, ArchitectureConfig] = None,
    ):
        """
        The constructor of the VAEConfig class.
        """
        if isinstance(encoder, ArchitectureConfig):
            self.encoder = encoder

        elif isinstance(encoder, dict):
            self.encoder = ArchitectureConfig(**encoder)

        if isinstance(decoder, ArchitectureConfig):
            self.decoder = decoder

        elif isinstance(decoder, dict):
            self.decoder = ArchitectureConfig(**decoder)
