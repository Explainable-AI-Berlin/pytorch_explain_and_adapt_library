import code # code.interact(local=dict(globals(), **locals()))

from torch import nn

from peal.architectures.module_blocks import (
    VGGBlock,
    ResnetBlock
)
from peal.architectures.basic_modules import (
    PgelSequential,
    PgelConv2d,
    PgelConv1d,
    Squeeze,
    Mean,
    Flatten,
    PgelDropout,
    PgelBatchNorm,
    PgelIdentity,
    Sum,
    TwoPathNetwork,
    NoiseLayer,
)
from peal.architectures.advanced_modules import DimensionSwitchAttentionLayer


class Img2LatentEncoder(PgelSequential):
    '''

    '''
    def __init__(self, neuron_numbers, blocks_per_layer, block_type, input_channels, use_batchnorm, activation):
        #
        layers = []
        #
        sublayers = []

        #
        '''sublayers.append(PgelConv2d(
            input_channels,
            neuron_numbers[0],
            7,
            2,
            3
        ))
        sublayers.append(activation())
        layers.append(PgelSequential(*sublayers))'''

        #
        if block_type == 'resnet':
            block_factory = ResnetBlock
            layers.append(VGGBlock(input_channels, neuron_numbers[0], stride = 2, activation = activation, use_batchnorm = False, receptive_field = 7))

        elif block_type == 'vgg':
            block_factory = VGGBlock
            layers.append(VGGBlock(input_channels, neuron_numbers[0], stride = 2, activation = activation, use_batchnorm = False, receptive_field = 7))

        elif block_type == 'nf':
            block_factory = NFBlock
            layers.append(NFBlock(input_channels, neuron_numbers[0], stride = 2, activation = activation, use_batchnorm = False))
        
        #
        for i in range(len(neuron_numbers) - 1):
            #
            sublayers = []
            for j in range(blocks_per_layer - 1):
                sublayers.append(block_factory(neuron_numbers[i], neuron_numbers[i], stride = 1, activation = activation))

            sublayers.append(block_factory(neuron_numbers[i], neuron_numbers[i + 1], stride = 2, activation = activation, use_batchnorm = use_batchnorm))

            layers.append(PgelSequential(*sublayers))
        #
        super(Img2LatentEncoder, self).__init__(*layers)


class Latent2VectorDecoder(PgelSequential):
    '''

    '''
    def __init__(self, output_size, num_hidden_in, dimension_reduction, dropout, activation, latent_height):
        layers = {}
        if dimension_reduction == 'mean':
            layers['dimensionality_reductor'] = Mean([-2,-1], keepdim = True)
            kernel_size = 1
            num_hidden = num_hidden_in

        elif dimension_reduction == 'flatten':
            kernel_size = latent_height
            num_hidden = latent_height * num_hidden_in

        elif dimension_reduction == 'sum':
            layers['dimensionality_reductor'] = Sum([-2,-1], keepdim = True)
            kernel_size = 1

        elif dimension_reduction == 'flatten':
            num_hidden = num_hidden_in

        '''elif dimension_reduction == 'center':
            dimensionality_reductor = FunctionWrapper(lambda x: x[:, :, int(x.shape[2] / 2):int(x.shape[2] / 2) + 1, int(x.shape[3] / 2):int(x.shape[3] / 2) + 1])'''


        if dimension_reduction in ['mean', 'sum']:
            layers['layer2'] = PgelConv2d(num_hidden, output_size, 1)
            layers['squeezer'] = Squeeze([-1, -1])            

        elif dimension_reduction == ['flatten']:
            if dropout > 0.0: layers['dropout1'] = PgelDropout(dropout / 2)
            layers['layer1'] = PgelConv2d(num_hidden_in, num_hidden, kernel_size)
            layers['activation1'] = activation()
            if dropout > 0.0: layers['dropout2'] = PgelDropout(dropout)
            layers['layer2'] = PgelConv2d(num_hidden, output_size, 1)
            layers['squeezer'] = Squeeze([-1, -1])

        elif dimension_reduction == 'attention':
            layers['layer1'] = DimensionSwitchAttentionLayer(output_size, num_hidden, 2)
            layers['activation1'] = activation()
            if dropout > 0.0: layers['dropout2'] = PgelDropout(dropout)
            layers['layer2'] = PgelConv1d(num_hidden, 1, 1)
            layers['squeezer'] = Squeeze([-2])

        elif dimension_reduction == 'nf':
            # careful, only works for special hyperparameters
            layers['layer1'] = CouplingLayer()
            layers['layer2'] = CouplingLayer()
            layers['squeezer'] = SplitFlow((output_size - 1) / num_hidden_in)

        super(Latent2VectorDecoder, self).__init__(*layers.values())


class Latent2ImgDecoder(TwoPathNetwork):
    '''

    '''
    def __init__(self, neuron_numbers, blocks_per_layer, block_type, output_size, use_batchnorm, activation):
        '''

        '''
        #
        layers = []
        layers.append(NoiseLayer())
        #
        if block_type == 'resnet':
            print('not implemented')

        elif block_type == 'vgg':
            block_factory = DeconvVGGBlock
        #
        for i in range(len(neuron_numbers) - 1):
            sublayers = []
            for j in range(blocks_per_layer - 1):
                sublayers.append(block_factory(neuron_numbers[i], neuron_numbers[i], 1, activation))

            sublayers.append(block_factory(neuron_numbers[i], neuron_numbers[i + 1], 2, activation, use_batchnorm))
            layers.append(PgelSequential(*sublayers))
        #
        layers.append(nn.ConvTranspose2d(
            neuron_numbers[-1],
            output_size,
            3,
            2,
            1,
            1
        ))
        #
        network = PgelSequential(*layers)
        super().__init__(network)