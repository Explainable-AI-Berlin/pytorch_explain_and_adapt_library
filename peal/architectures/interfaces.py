"""Pydantic config templates for PEAL architectures and training tasks.

:class:`ArchitectureConfig` describes a network as a list of layer entries
(``fc``, ``vgg``, ``resnet``, ``transformer``), each backed by one of the
per-layer config classes here; :mod:`peal.architectures.predictors` turns such a
config into a module.  :class:`TaskConfig` describes what the network is trained
and evaluated on: the weighted mix of criterions from
:mod:`peal.training.criterions`, the output type and channel count, and optional
restrictions of the input/output variables or classes.  Every class carries a
``config_name`` field so that a yaml config can be mapped back to its class.
"""

from typing import Union

from pydantic import BaseModel, PositiveInt


class TaskConfig(BaseModel):
    """
    A dict of critirion names (that have to be implemented in peal.training.criterions)
    mapped to the weight.
    Like this the loss function can be post_hoc attached without changing the code.
    """

    config_name: str = "TaskConfig"
    """
    The type of config. This is necessary to find config class from yaml config
    """
    criterions: dict = {"ce": 1.0, "l2": 100.0}
    """
    The criterions used for training the model.
    """
    focal_alpha: float = 0.25
    """
    The alpha parameter of the focal loss.
    """
    focal_gamma: float = 1.0
    """
    The gamma parameter of the focal loss.
    """
    focal_mse_mix: float = 1.0
    """
    The mix between focal loss and mse loss.
    """
    mixed_bce_pos_weight: float = 1.0
    """
    The pos_weight parameter of the BCEWithLogitsLoss used for mixed BCE and MSE loss.
    """
    bce_mse_mix: float = 1.0
    """
    The mix between binary cross entropy loss and mse loss.
    """
    output_type: str = "singleclass"
    """
    The output_type that either can just be the output_type of the dataset or could be some
    possible subtype.
    E.g. when having a binary multiclass dataset one could use as task binary single class
    classification for one of the output variables.
    """
    output_channels: PositiveInt = 2
    """
    The output_channels that can be at most the output_channels of the dataset, but if a subtask is chosen
    the output_channels has also be adapted accordingly
    """
    x_selection: Union[list[str], type(None)] = None
    """
    Gives the option to select a subset of the input variables. Only works for symbolic data.
    """
    y_selection: Union[list[str], type(None)] = None
    """
    Gives the option to select a subset of the output_variables.
    Can be used e.g. to transform binary multiclass into the subtask of predicting one of the
    binary variables with single class classification.
    """
    class_restriction: Union[int, list[int], type(None)] = None
    """
    Gives the option to only use samples from one or more specified classes
    """


class ArchitectureConfig(BaseModel):
    """
    The config template for a neural architecture.
    """

    config_name: str = "ArchitectureConfig"
    """
    The type of config. This is necessary to find config class from yaml config
    """
    layers: list
    """
    The layers of the architecture.
    Elements of the list are tuples of the form ``(layer_type, *layer_config)``.
    Options for list elements: ['fc', 'vgg','resnet','transformer']
    """
    activation: str = "ReLU"
    """
    The activation function used in the architecture.
    Options: ['ReLU', 'LeakyReLU', 'LeakySoftplus']
    """


class FCConfig(BaseModel):
    """
    The config template for a Fully Connected Layer.
    """

    config_name: str = "FCConfig"
    """
    The type of config. This is necessary to find config class from yaml config
    """
    num_neurons: PositiveInt = 512
    """
    The number of neurons in the layer.
    """
    dropout: float = 0.0
    """
    Whether to use batchnorm or not.
    """
    tensor_dim: int = 0
    """
    The dimension of the tensor.
    Options: [0, 1, 2, 3]
    """


class VGGConfig(BaseModel):
    """
    The config template for a VGG Layer.
    """

    config_name: str = "VGGConfig"
    """
    The type of config. This is necessary to find config class from yaml config
    """
    num_neurons: PositiveInt = 512
    """
    The number of neurons in the layer.
    """
    num_blocks: PositiveInt = 2
    """
    Number of blocks per layer.
    """
    use_batchnorm: bool = True
    """
    Whether to use batchnorm or not.
    """
    receptive_field: PositiveInt = 3
    """
    The size of the receptive field.
    """
    tensor_dim: PositiveInt = 2
    """
    The dimension of the tensor.
    Options: [1, 2, 3]
    """


class ResnetConfig(BaseModel):
    """
    The config template for a ResNet layer.
    """

    config_name: str = "ResnetConfig"
    """
    The type of config. This is necessary to find config class from yaml config
    """
    num_neurons: PositiveInt = 512
    """
    The number of neurons in the layer.
    """
    num_blocks: PositiveInt = 2
    """
    Number of blocks per layer.
    """
    use_batchnorm: bool = True
    """
    Whether to use batchnorm or not.
    """
    tensor_dim: PositiveInt = 2
    """
    The dimension of the tensor.
    Options: [1, 2, 3]
    """


class TransformerConfig(BaseModel):
    """
    The config template for a transformer layer.
    """

    config_name: str = "TransformerConfig"
    """
    The type of config. This is necessary to find config class from yaml config
    """
    num_neurons: PositiveInt = 512
    """
    The number of neurons in the layer.
    """
    num_blocks: PositiveInt = 2
    """
    Number of blocks per layer.
    """
