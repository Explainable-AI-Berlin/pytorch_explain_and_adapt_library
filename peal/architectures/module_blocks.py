"""Building blocks from which PEAL's configurable predictors are assembled.

Each block is an ``nn.Sequential`` parameterised by a pydantic layer config
(``FCConfig``, ``VGGConfig``, ``ResnetConfig``) so that classifier
architectures can be described entirely in yaml. ``create_cnn_layer`` stacks
``num_blocks`` VGG or ResNet blocks into one stage whose first block
downsamples by stride 2.
"""

from torch import nn
from typing import Union, Type
from pydantic.types import PositiveInt

from peal.architectures.basic_modules import (
    SkipConnection,
    SelfAttentionLayer,
    Transpose,
)
from peal.architectures.interfaces import FCConfig, VGGConfig, ResnetConfig


class FCBlock(nn.Sequential):
    """Fully connected block: a 1x1 projection, optional activation and dropout.

    Depending on ``layer_config.tensor_dim`` the projection is an ``nn.Linear``
    (0) or a kernel-size-1 ``Conv1d``/``Conv2d``/``Conv3d`` (1/2/3), so the
    same config can act on flat vectors or on feature maps.

    Parameters
    ----------
    layer_config : FCConfig
        Provides ``tensor_dim``, ``num_neurons`` and ``dropout``.
    num_neurons_previous : int
        Input width (features or channels).
    activation : type of nn.Module, optional
        Activation class appended after the projection; none if ``None``.
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
    """VGG-style block: convolution, optional batchnorm, activation.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    activation : type of nn.Module
        Activation class, instantiated without arguments.
    stride : int
        Stride of the convolution.
    conv : type of nn.Module
        Convolution class (``nn.Conv1d``/``Conv2d``/``Conv3d``).
    batchnorm : type of nn.Module
        Batchnorm class matching ``conv``; used if ``config.use_batchnorm``.
    config : VGGConfig
        Provides ``num_neurons`` (output channels), ``receptive_field``
        (kernel size; padding is ``receptive_field // 2``) and
        ``use_batchnorm``.
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
    """Basic residual block with two 3x3 convolutions and a skip connection.

    The residual branch is conv-(bn)-act-conv-(bn). When ``stride > 1`` the
    identity path is replaced by ``AvgPool2d(2)`` followed by a 1x1 convolution
    so that shapes match; the sum is followed by a final activation.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    activation : type of nn.Module
        Activation class, instantiated without arguments.
    stride : int
        Stride of the first convolution (2 downsamples).
    conv : type of nn.Module
        Convolution class (``nn.Conv1d``/``Conv2d``/``Conv3d``).
    batchnorm : type of nn.Module
        Batchnorm class matching ``conv``; used if ``config.use_batchnorm``.
    config : ResnetConfig
        Provides ``num_neurons`` (output channels) and ``use_batchnorm``.

    Notes
    -----
    The downsampling path always uses ``nn.AvgPool2d`` regardless of
    ``tensor_dim``.
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
    """Pre-activation transformer block: attention and position-wise MLP, each residual.

    Sub-block 1 is ``SkipConnection(SelfAttentionLayer -> LayerNorm)``; sub-block
    2 is ``SkipConnection(1x1 Conv1d over the embedding axis -> activation ->
    LayerNorm)``. Inputs and outputs have shape ``(B, T, embedding_dim)``.

    Parameters
    ----------
    embedding_dim : int
        Token embedding width.
    num_heads : int
        Number of self-attention heads.
    activation : type of nn.Module
        Activation class for the feed-forward sub-block.
    use_masking : bool, optional
        Whether the attention layer applies a causal mask. Default ``False``.
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
