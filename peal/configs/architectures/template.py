from datetime import datetime
from pydantic import BaseModel, PositiveInt

class ArchitectureConfig(BaseModel):
    type : str
    '''
    A dict containing all variables that could not be given with the current config structure
    '''
    kwargs : dict = {}

class Img2VectorConfig(ArchitectureConfig):
    '''
    The type of block used:
    Options: ['vgg','resnet','transformer']
    '''
    block_type : str
    '''
    List of the number of neurons per superblock.
    Implicitly also defines the number of superblocks
    '''
    neuron_numbers : list[PositiveInt]
    '''
    Number of superblocks per block.
    '''
    blocks_per_layer: PositiveInt
    '''
    The type of dimension reduction used for compressing to a vector.
    Options: ['mean', 'flatten']
    '''
    dimension_reduction: str = 'mean'
    '''
    The activation function used in the architecture.
    Options: ['ReLU', 'LeakyReLU']
    '''
    activation: str = "ReLU"
    '''
    Whether to use batchnorm or not.
    '''
    use_batchnorm: bool = True
    '''
    The strength of the dropout applied.
    '''
    dropout: float = 0.0

class Symbolic2VectorConfig(ArchitectureConfig):
    # TODO
    pass

class GlowConfig(ArchitectureConfig):
    # TODO
    pass

class VAEConfig(ArchitectureConfig):
    # TODO
    pass

class DDMConfig(ArchitectureConfig):
    # TODO
    pass