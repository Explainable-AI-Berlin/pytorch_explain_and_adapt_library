from torch import nn

from peal.architectures.basic_modules import (
    SkipConnection
)

class VGGBlock(nn.Sequential):
    def __init__(
        self,
        inplanes,
        planes,
        stride,
        activation,
        use_batchnorm = False,
        receptive_field = 3
    ):  
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


class ResnetBlock(nn.Sequential):
    def __init__(
        self,
        inplanes,
        planes,
        stride,
        activation,
        use_batchnorm = True
    ):
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