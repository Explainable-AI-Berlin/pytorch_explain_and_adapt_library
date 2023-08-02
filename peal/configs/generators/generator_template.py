from typing import Union

from peal.configs.architectures.architecture_template import ArchitectureConfig

class VAEConfig:
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
    __name__ : str = 'peal.VAEConfig'
    def __init__(self, encoder : Union[dict, ArchitectureConfig] = None, decoder : Union[dict, ArchitectureConfig] = None):
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