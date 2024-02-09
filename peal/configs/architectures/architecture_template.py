from datetime import datetime
from pydantic import BaseModel, PositiveInt


class FCConfig:
    """
    The config template for a Fully Connected Layer.
    """

    """
    The number of neurons in the layer.
    """
    num_neurons: PositiveInt
    """
    Whether to use batchnorm or not.
    """
    dropout: float = 0.0
    """
    The dimension of the tensor.
    Options: [0, 1, 2, 3]
    """
    tensor_dim: int = 0

    def __init__(
        self,
        num_neurons: PositiveInt,
        dropout: float = 0.0,
        tensor_dim: int = 0,
    ):
        self.num_neurons = num_neurons
        self.dropout = dropout
        self.tensor_dim = tensor_dim

    def dict(self):
        return ["fc", self.num_neurons, self.dropout, self.tensor_dim]


class VGGConfig:
    """
    The config template for a VGG Layer.
    """

    """
    The number of neurons in the layer.
    """
    num_neurons: PositiveInt
    """
    Number of blocks per layer.
    """
    num_blocks: PositiveInt
    """
    Whether to use batchnorm or not.
    """
    use_batchnorm: bool = True
    """
    The size of the receptive field.
    """
    receptive_field: PositiveInt = 3
    """
    The dimension of the tensor.
    Options: [1, 2, 3]
    """
    tensor_dim: PositiveInt = 2

    def __init__(
        self,
        num_neurons: PositiveInt,
        num_blocks: PositiveInt,
        use_batchnorm: bool = True,
        receptive_field: PositiveInt = 3,
        tensor_dim: PositiveInt = 2,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_blocks = num_blocks
        self.use_batchnorm = use_batchnorm
        self.receptive_field = receptive_field
        self.tensor_dim = tensor_dim

    def dict(self):
        return [
            "vgg",
            self.num_neurons,
            self.num_blocks,
            self.use_batchnorm,
            self.receptive_field,
            self.tensor_dim,
        ]


class ResnetConfig:
    """
    The config template for a ResNet layer.
    """

    """
    The number of neurons in the layer.
    """
    num_neurons: PositiveInt
    """
    Number of blocks per layer.
    """
    num_blocks: PositiveInt
    """
    Whether to use batchnorm or not.
    """
    use_batchnorm: bool = True
    """
    The dimension of the tensor.
    Options: [1, 2, 3]
    """
    tensor_dim: PositiveInt = 2

    def __init__(
        self,
        num_neurons: PositiveInt,
        num_blocks: PositiveInt,
        use_batchnorm: bool = True,
        tensor_dim: PositiveInt = 2,
        **kwargs,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_blocks = num_blocks
        self.use_batchnorm = use_batchnorm
        self.tensor_dim = tensor_dim

    def dict(self):
        # TODO this does not seem to work...
        return [
            "resnet",
            self.num_neurons,
            self.num_blocks,
            self.use_batchnorm,
            self.tensor_dim,
        ]


# TODO complete this
class TransformerConfig:
    """
    The config template for a transformer layer.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


class ArchitectureConfig:
    """
    The config template for a neural architecture.
    """

    """
    The layers of the architecture.
    Elements of the list are tuples of the form (layer_type, *layer_config).
    Options for list elements: ['fc', 'vgg','resnet','transformer']
    """
    layers: list
    """
    The activation function used in the architecture.
    Options: ['ReLU', 'LeakyReLU', 'Softplus']
    """
    activation: str = "ReLU"
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
    """
    The name of the class.
    """
    __name__: str = "peal.ArchitectureConfig"

    def __init__(self, layers: list, activation: str = "ReLU", **kwargs):
        self.activation = activation
        self.kwargs = kwargs
        self.layers = []
        for layer in layers:
            """
            layer[0] contains the type of layer used:
            Options: ['fc', 'vgg','resnet','transformer']
            """
            if layer[0] == "resnet":
                self.layers.append(ResnetConfig(*layer[1:]))

            elif layer[0] == "fc":
                self.layers.append(FCConfig(*layer[1:]))

            elif layer[0] == "vgg":
                self.layers.append(VGGConfig(*layer[1:]))

            elif layer[0] == "transformer":
                self.layers.append(TransformerConfig(*layer[1:]))

            else:
                self.layers.append(layer[0])
