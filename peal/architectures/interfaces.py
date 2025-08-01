from typing import Union

from pydantic import BaseModel, PositiveInt


class TaskConfig(BaseModel):
    """
    A dict of critirion names (that have to be implemented in peal.training.criterions)
    mapped to the weight.
    Like this the loss function can be post_hoc attached without changing the code.
    """
    """
    The type of config. This is necessary to find config class from yaml config
    """
    config_name: str = "TaskConfig"
    """
    The criterions used for training the model.
    """
    criterions: dict = {'ce' : 1.0, 'l2' : 1.0}
    """
    The output_type that either can just be the output_type of the dataset or could be some
    possible subtype.
    E.g. when having a binary multiclass dataset one could use as task binary single class
    classification for one of the output variables.
    """
    output_type: str = "singleclass"
    """
    The output_channels that can be at most the output_channels of the dataset, but if a subtask is chosen
    the output_channels has also be adapted accordingly
    """
    output_channels: PositiveInt = 2
    """
    Gives the option to select a subset of the input variables. Only works for symbolic data.
    """
    x_selection: Union[list[str], type(None)] = None
    """
    Gives the option to select a subset of the output_variables.
    Can be used e.g. to transform binary multiclass into the subtask of predicting one of the
    binary variables with single class classification.
    """
    y_selection: Union[list[str], type(None)] = None
    """
    Gives the option to only use samples from one or more specified classes
    """
    class_restriction: Union[int ,list[int], type(None)] = None


class ArchitectureConfig(BaseModel):
    """
    The config template for a neural architecture.
    """
    """
    The type of config. This is necessary to find config class from yaml config
    """
    config_name: str = "ArchitectureConfig"
    """
    The layers of the architecture.
    Elements of the list are tuples of the form (layer_type, *layer_config).
    Options for list elements: ['fc', 'vgg','resnet','transformer']
    """
    layers: list
    """
    The activation function used in the architecture.
    Options: ['ReLU', 'LeakyReLU', 'LeakySoftplus']
    """
    activation: str = "ReLU"

    '''def __init__(self, layers: list, activation: str = "ReLU", **kwargs):
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
                self.layers.append(layer[0])'''


class FCConfig(BaseModel):
    """
    The config template for a Fully Connected Layer.
    """
    """
    The type of config. This is necessary to find config class from yaml config
    """
    config_name: str = "FCConfig"
    """
    The number of neurons in the layer.
    """
    num_neurons: PositiveInt = 512
    """
    Whether to use batchnorm or not.
    """
    dropout: float = 0.0
    """
    The dimension of the tensor.
    Options: [0, 1, 2, 3]
    """
    tensor_dim: int = 0


class VGGConfig(BaseModel):
    """
    The config template for a VGG Layer.
    """
    """
    The type of config. This is necessary to find config class from yaml config
    """
    config_name: str = "VGGConfig"
    """
    The number of neurons in the layer.
    """
    num_neurons: PositiveInt = 512
    """
    Number of blocks per layer.
    """
    num_blocks: PositiveInt = 2
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


class ResnetConfig(BaseModel):
    """
    The config template for a ResNet layer.
    """
    """
    The type of config. This is necessary to find config class from yaml config
    """
    config_name: str = "ResnetConfig"
    """
    The number of neurons in the layer.
    """
    num_neurons: PositiveInt = 512
    """
    Number of blocks per layer.
    """
    num_blocks: PositiveInt = 2
    """
    Whether to use batchnorm or not.
    """
    use_batchnorm: bool = True
    """
    The dimension of the tensor.
    Options: [1, 2, 3]
    """
    tensor_dim: PositiveInt = 2


class TransformerConfig(BaseModel):
    """
    The config template for a transformer layer.
    """
    """
    The type of config. This is necessary to find config class from yaml config
    """
    config_name: str = "TransformerConfig"
    """
    The number of neurons in the layer.
    """
    num_neurons: PositiveInt = 512
    """
    Number of blocks per layer.
    """
    num_blocks: PositiveInt = 2
