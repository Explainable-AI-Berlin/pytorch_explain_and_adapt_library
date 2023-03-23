from torch import nn

from peal.architectures.module_blocks import (
    VGGBlock,
    ResnetBlock
)
from peal.architectures.basic_modules import (
    Squeeze,
    Mean,
    Sum,
    NoiseLayer,
)
from peal.architectures.advanced_modules import DimensionSwitchAttentionLayer

class Img2LatentEncoder(nn.Sequential):
    '''

    '''
    def __init__(self, neuron_numbers, blocks_per_layer, block_type, input_channels, use_batchnorm, activation):
        #
        layers = []
        #
        sublayers = []

        #
        if block_type == 'resnet':
            block_factory = ResnetBlock
            layers.append(VGGBlock(input_channels, neuron_numbers[0], stride = 2, activation = activation, use_batchnorm = False, receptive_field = 7))

        elif block_type == 'vgg':
            block_factory = VGGBlock
            layers.append(VGGBlock(input_channels, neuron_numbers[0], stride = 2, activation = activation, use_batchnorm = False, receptive_field = 7))
        
        #
        for i in range(len(neuron_numbers) - 1):
            #
            sublayers = []
            for j in range(blocks_per_layer - 1):
                sublayers.append(block_factory(neuron_numbers[i], neuron_numbers[i], stride = 1, activation = activation))

            sublayers.append(block_factory(neuron_numbers[i], neuron_numbers[i + 1], stride = 2, activation = activation, use_batchnorm = use_batchnorm))

            layers.append(nn.Sequential(*sublayers))
        #
        super(Img2LatentEncoder, self).__init__(*layers)


class Sequence2LatentEncoder(nn.Sequential):
    '''

    '''
    def __init__(self, neuron_numbers, blocks_per_layer, block_type, input_channels, use_batchnorm, activation):
        #
        layers = []
        #
        super(Img2LatentEncoder, self).__init__(*layers)


class Vector2LatentEncoder(nn.Sequential):
    '''

    '''

    def __init__(self, input_channels, activation, neuron_numbers = []):
        #
        layers = []
        neuron_numbers = [input_channels] + neuron_numbers
        for i in range(len(neuron_numbers) - 1):
            layers.append(nn.Linear(neuron_numbers[i], neuron_numbers[i + 1]))
            layers.append(activation())
        #
        super(Img2LatentEncoder, self).__init__(*layers)


class Latent2ImgDecoder(nn.Sequential):
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
            block_factory = None

        #
        for i in range(len(neuron_numbers) - 1):
            sublayers = []
            for j in range(blocks_per_layer - 1):
                sublayers.append(block_factory(neuron_numbers[i], neuron_numbers[i], 1, activation))

            sublayers.append(block_factory(neuron_numbers[i], neuron_numbers[i + 1], 2, activation, use_batchnorm))
            layers.append(nn.Sequential(*sublayers))
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
        super().__init__(*layers)


class Latent2SequenceDecoder:
    '''

    '''
    def __init__(self, neuron_numbers, blocks_per_layer, block_type, output_size, use_batchnorm, activation):
        '''

        '''
        pass


class Latent2VectorDecoder(nn.Sequential):
    '''

    '''

    def __init__(self, output_size, num_hidden_in, activation, dropout = False, latent_height=None, dimension_reduction=None, neuron_numbers=[]):
        layers = {}
        if dimension_reduction == 'mean':
            layers['dimensionality_reductor'] = Mean([-2, -1], keepdim=True)
            kernel_size = 1
            num_hidden = num_hidden_in

        elif dimension_reduction == 'flatten':
            kernel_size = latent_height
            num_hidden = latent_height * num_hidden_in

        elif dimension_reduction == 'sum':
            layers['dimensionality_reductor'] = Sum([-2, -1], keepdim=True)
            kernel_size = 1

        if dimension_reduction in ['mean', 'sum']:
            layers['layer2'] = nn.Conv2d(num_hidden, output_size, 1)
            layers['squeezer'] = Squeeze([-1, -1])

        elif dimension_reduction == ['flatten']:
            if dropout > 0.0:
                layers['dropout1'] = nn.Dropout(dropout / 2)
            layers['layer1'] = nn.Conv2d(
                num_hidden_in, num_hidden, kernel_size)
            layers['activation1'] = activation()
            if dropout > 0.0:
                layers['dropout2'] = nn.Dropout(dropout)
            layers['layer2'] = nn.Conv2d(num_hidden, output_size, 1)
            layers['squeezer'] = Squeeze([-1, -1])

        elif dimension_reduction == 'attention':
            layers['layer1'] = DimensionSwitchAttentionLayer(
                output_size, num_hidden, 2)
            layers['activation1'] = activation()
            if dropout > 0.0:
                layers['dropout2'] = nn.Dropout(dropout)
            layers['layer2'] = nn.Conv1d(num_hidden, 1, 1)
            layers['squeezer'] = Squeeze([-2])
        
        elif dimension_reduction == 'none':
            layers = {}
            neuron_numbers = [num_hidden_in] + neuron_numbers + [output_size]
            for i in range(len(neuron_numbers) - 1):
                layers['layer_'+str(i)] = nn.Linear(neuron_numbers[i], neuron_numbers[i + 1])
                if i < len(neuron_numbers) - 2:
                    layers['activation_' + str(i)] = activation()

        super(Latent2VectorDecoder, self).__init__(*layers.values())
