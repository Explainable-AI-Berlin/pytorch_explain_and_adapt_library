from pydantic import BaseModel
class GeneratorConfig(BaseModel):
    """
    This class defines the config of a generator.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str
    """
    The category of the config
    """
    category: str = 'generator'

'''class VAEConfig(GeneratorConfig):
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
            self.decoder = ArchitectureConfig(**decoder)'''
