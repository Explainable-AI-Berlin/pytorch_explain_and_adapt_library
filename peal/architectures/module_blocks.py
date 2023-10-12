from torch import nn
from typing import Union, Type
from pydantic.types import PositiveInt

from peal.architectures.basic_modules import (
    SkipConnection,
    SelfAttentionLayer,
    Transpose,
)
from peal.configs.architectures.architecture_template import (
    VGGConfig,
    ResnetConfig,
    FCConfig,
    TransformerConfig,
)


class FCBlock(nn.Sequential):
    """
    The FCBlock class implements a fully connected block as a sequential module.
    """

    def __init__(
        self,
        layer_config: FCConfig,
        num_neurons_previous: PositiveInt,
        activation: Type[nn.Module] = None,
    ):
        """
        The __init__ method initializes the FCBlock class.
        Args:
            layer_config: The config of the last layer.
            num_neurons_previous: The number of neurons in the previous layer.
            activation: The activation function class.
        """
        submodules = []
        if layer_config.tensor_dim == 0:
            submodules.append(nn.Linear(num_neurons_previous, layer_config.num_neurons))

        elif layer_config.tensor_dim == 1:
            submodules.append(
                nn.Conv1d(num_neurons_previous, layer_config.num_neurons, 1)
            )

        elif layer_config.tensor_dim == 2:
            submodules.append(
                nn.Conv2d(num_neurons_previous, layer_config.num_neurons, 1)
            )

        elif layer_config.tensor_dim == 3:
            submodules.append(
                nn.Conv3d(num_neurons_previous, layer_config.num_neurons, 1)
            )

        if not activation is None:
            submodules.append(activation())

        if layer_config.dropout > 0.0:
            submodules.append(nn.Dropout(layer_config.dropout))

        super().__init__(*submodules)


class VGGBlock(nn.Sequential):
    """
    The VGGBlock class implements a VGG block as a sequential module.

    Args:
        nn.Sequential (nn.Module): The base class for all neural network modules.
    """

    def __init__(
        self,
        input_channels: PositiveInt,
        activation: Type[nn.Module],
        stride: PositiveInt,
        conv: Type[nn.Module],
        batchnorm: Type[nn.Module],
        config: VGGConfig,
    ):
        """
        The __init__ method initializes the VGGBlock class.
        Args:
            input_channels: The number of input channels.
            activation: The activation function class.
            stride: The stride of the convolution.
            conv: The convolution class.
            batchnorm: The batchnorm class.
            config: The config of the block.
        """
        submodules = []
        padding = int(config.receptive_field / 2)
        submodules.append(
            conv(
                input_channels,
                config.num_neurons,
                config.receptive_field,
                stride,
                padding,
            )
        )

        #
        if config.use_batchnorm:
            submodules.append(batchnorm(config.num_neurons))

        submodules.append(activation())

        super(VGGBlock, self).__init__(*submodules)


class ResnetBlock(nn.Sequential):
    """
    The ResnetBlock class implements a ResNet block as a sequential module.

    Args:
        nn.Sequential (nn.Module): The base class for all neural network modules.
    """

    def __init__(
        self,
        input_channels: PositiveInt,
        activation: Type[nn.Module],
        stride: PositiveInt,
        conv: Type[nn.Module],
        batchnorm: Type[nn.Module],
        config: ResnetConfig,
    ):
        """
        The __init__ method initializes the ResnetBlock class.
        Args:
            input_channels: The number of input channels.
            activation: The activation function class.
            stride: The stride of the convolution.
            conv: The convolution class.
            batchnorm: The batchnorm class.
            config: The config of the block.
        """
        submodules = []
        submodule_1 = []
        submodule_1.append(conv(input_channels, config.num_neurons, 3, stride, 1))
        if config.use_batchnorm:
            submodule_1.append(batchnorm(config.num_neurons))

        submodule_1.append(activation())
        submodule_1.append(conv(config.num_neurons, config.num_neurons, 3, 1, 1))
        if config.use_batchnorm:
            submodule_1.append(batchnorm(config.num_neurons))

        submodule_1 = nn.Sequential(*submodule_1)
        if stride > 1:
            pooling = nn.AvgPool2d(2)
            downsample_conv = conv(input_channels, config.num_neurons, 1)
            downsample = nn.Sequential(*[pooling, downsample_conv])
            submodule_1 = SkipConnection(submodule_1, downsample)

        else:
            submodule_1 = SkipConnection(submodule_1)

        submodules.append(submodule_1)
        submodules.append(activation())

        super(ResnetBlock, self).__init__(*submodules)


def create_cnn_layer(
    block_type: Union[VGGBlock, ResnetBlock],
    config: Union[ResnetConfig, VGGConfig],
    input_channels: PositiveInt,
    activation: nn.Module,
):
    """
    The create_cnn_layer function creates a CNN layer.
    Args:
        block_type: The type of the block.
        config: The config of the block.
        input_channels: The number of input channels.
        activation: The activation function.

    Returns:
        The created CNN layer.
    """
    if config.tensor_dim == 1:
        conv = nn.Conv1d
        batchnorm = nn.BatchNorm1d

    if config.tensor_dim == 2:
        conv = nn.Conv2d
        batchnorm = nn.BatchNorm2d

    elif config.tensor_dim == 3:
        conv = nn.Conv3d
        batchnorm = nn.BatchNorm3d

    blocks = []
    blocks.append(
        block_type(
            input_channels=input_channels,
            activation=activation,
            stride=2,
            conv=conv,
            batchnorm=batchnorm,
            config=config,
        )
    )
    for i in range(config.num_blocks - 1):
        blocks.append(
            block_type(
                input_channels=config.num_neurons,
                activation=activation,
                stride=1,
                conv=conv,
                batchnorm=batchnorm,
                config=config,
            )
        )

    return nn.Sequential(*blocks)


class TransformerBlock(nn.Sequential):
    """
    The TransformerBlock class implements a transformer block as a sequential module.

    Args:
        nn.Sequential (nn.Module): The base class for all neural network modules.
    """

    def __init__(self, embedding_dim, num_heads, activation, use_masking=False):
        """
        The __init__ method initializes the TransformerBlock class.

        Args:
            embedding_dim (int): The number of input channels.
            num_heads (int): The number of self-attention heads.
            activation (int): The activation function.
            use_masking (bool, optional): Whether to use masking. Defaults to False.
        """
        submodule_1 = []
        submodule_1.append(
            SelfAttentionLayer(
                embedding_dim,
                num_heads,
                use_masking,
            )
        )
        submodule_1.append(nn.LayerNorm(embedding_dim))
        submodule_1 = nn.Sequential(*submodule_1)
        submodule_1 = SkipConnection(submodule_1)

        submodule_2 = []
        submodule_2.append(Transpose(1, 2))
        submodule_2.append(nn.Conv1d(embedding_dim, embedding_dim, 1))
        submodule_2.append(Transpose(1, 2))
        submodule_2.append(activation())
        submodule_2.append(nn.LayerNorm(embedding_dim))
        submodule_2 = nn.Sequential(*submodule_2)
        submodule_2 = SkipConnection(submodule_2)

        super(TransformerBlock, self).__init__(*[submodule_1, submodule_2])
