from torch import nn

from peal.architectures.basic_modules import (
    SkipConnection,
    SelfAttentionLayer,
    Transpose,
)


class VGGBlock(nn.Sequential):
    '''
    The VGGBlock class implements a VGG block as a sequential module.

    Args:
        nn.Sequential (nn.Module): The base class for all neural network modules.
    '''

    def __init__(
        self,
        inplanes,
        planes,
        stride,
        activation,
        use_batchnorm=False,
        receptive_field=3
    ):
        '''
        The __init__ method initializes the VGGBlock class.

        Args:
            inplanes (int): The number of input channels.
            planes (int): The number of output channels.
            stride (int): The stride of the convolution.
            activation (nn.Module): The activation function.
            use_batchnorm (bool, optional): The. Defaults to False.
            receptive_field (int, optional): The. Defaults to 3.
        '''
        submodules = []
        padding = int(receptive_field / 2)
        submodules.append(nn.Conv2d(
            inplanes,
            planes,
            receptive_field,
            1,
            padding
        ))
        #
        if stride > 1:
            submodules.append(nn.MaxPool2d(stride))

        #
        if use_batchnorm:
            submodules.append(nn.BatchNorm2d(planes))

        submodules.append(activation())

        super(VGGBlock, self).__init__(*submodules)


class TransformerBlock(nn.Sequential):
    '''
    The TransformerBlock class implements a transformer block as a sequential module.

    Args:
        nn.Sequential (nn.Module): The base class for all neural network modules.
    '''

    def __init__(
        self,
        embedding_dim,
        num_heads,
        activation,
        use_masking=False
    ):
        '''
        The __init__ method initializes the TransformerBlock class.

        Args:
            embedding_dim (int): The number of input channels.
            num_heads (int): The number of self-attention heads.
            activation (int): The activation function.
            use_masking (bool, optional): Whether to use masking. Defaults to False.
        '''
        submodule_1 = []
        submodule_1.append(SelfAttentionLayer(
            embedding_dim,
            num_heads,
            use_masking,
        ))
        submodule_1.append(nn.LayerNorm(embedding_dim))
        submodule_1 = nn.Sequential(*submodule_1)
        submodule_1 = SkipConnection(submodule_1)

        submodule_2 = []
        submodule_2.append(Transpose(1, 2))
        submodule_2.append(nn.Conv1d(
            embedding_dim,
            embedding_dim,
            1
        ))
        submodule_2.append(Transpose(1, 2))
        submodule_2.append(activation())
        submodule_2.append(nn.LayerNorm(embedding_dim))
        submodule_2 = nn.Sequential(*submodule_2)
        submodule_2 = SkipConnection(submodule_2)

        super(TransformerBlock, self).__init__(*[submodule_1, submodule_2])


class ResnetBlock(nn.Sequential):
    '''
    The ResnetBlock class implements a ResNet block as a sequential module.

    Args:
        nn.Sequential (nn.Module): The base class for all neural network modules.
    '''

    def __init__(
        self,
        inplanes,
        planes,
        stride,
        activation,
        use_batchnorm=True
    ):
        '''
        The __init__ method initializes the ResnetBlock class.

        Args:
            inplanes (int): The number of input channels.
            planes (int): The number of output channels.
            stride (int): The stride of the convolution.
            activation (int): The activation function.
            use_batchnorm (bool, optional): Whether to use batch normalization. Defaults to True.
        '''
        submodules = []
        submodule_1 = []
        submodule_1.append(nn.Conv2d(
            inplanes,
            planes,
            3,
            stride,
            1
        ))
        if use_batchnorm:
            submodule_1.append(nn.BatchNorm2d(planes))

        submodule_1.append(activation())
        submodule_1.append(nn.Conv2d(
            planes,
            planes,
            3,
            1,
            1
        ))
        if use_batchnorm:
            submodule_1.append(nn.BatchNorm2d(planes))

        submodule_1 = nn.Sequential(*submodule_1)
        if stride > 1:
            pooling = nn.AvgPool2d(2)
            downsample_conv = nn.Conv2d(
                inplanes,
                planes,
                1
            )
            downsample = nn.Sequential(*[pooling, downsample_conv])
            submodule_1 = SkipConnection(submodule_1, downsample)

        else:
            submodule_1 = SkipConnection(submodule_1)

        submodules.append(submodule_1)
        submodules.append(activation())

        super(ResnetBlock, self).__init__(*submodules)
