import os
import code # code.interact(local=dict(globals(), **locals()))
import math
import torch
import numpy as np

from peal.architectures.model_parts import (
	Img2LatentEncoder,
	Latent2VectorDecoder,
	Latent2ImgDecoder
)
from peal.architectures.basic_modules import (
	PgelSequential,
	PgelReLU,
	PgelLeakyReLU,
	PgelLinear,
	Flatten,
)

def load_model(model_config, model_path, device, discriminator = None):
	'''

	'''
	if model_config['architecture']['block_type'] in ['vgg', 'resnet']:
		model = ImgEncoderDecoderModel(model_config)

	elif model_config['architecture']['block_type'] == 'fc':
		model = FCModel(model_config)

	else:
		print(model_config['data']['output_type'] + ' is not available!')

	checkpoint = torch.load(os.path.join(model_path, 'checkpoints', 'final.cpl'), map_location=torch.device(device))
	model.load_state_dict(checkpoint)
		
	return model.to(device)


class FCModel(PgelSequential):
	'''

	'''
	def __init__(self, config):
		'''

		'''
		if config['architecture']['activation'] == 'LeakyReLU':
			activation = torch.nn.LeakyReLU

		elif config['architecture']['activation'] == 'ReLU':
			activation = torch.nn.ReLU

		elif config['architecture']['activation'] == 'Softplus':
			activation = torch.nn.Softplus

		layers = []
		neuron_numbers_with_inputs = [int(np.prod(config['data']['input_size']))] + config['architecture']['neuron_numbers'] + [config['data']['output_size']]
		layers.append(Flatten(keepdim = False))
		for idx in range(len(neuron_numbers_with_inputs) - 1):
			layers.append(PgelLinear(neuron_numbers_with_inputs[idx], neuron_numbers_with_inputs[idx + 1]))
			if idx != len(neuron_numbers_with_inputs) - 2:
				layers.append(activation)
				
		super(FCModel, self).__init__(*layers)


class ImgEncoderDecoderModel(PgelSequential):
	'''

	'''
	def __init__(self, config):
		'''

		'''
		#
		if config['architecture']['activation'] == 'LeakyReLU':
			activation = torch.nn.LeakyReLU

		elif config['architecture']['activation'] == 'ReLU':
			activation = torch.nn.ReLU

		elif config['architecture']['activation'] == 'Softplus':
			activation = torch.nn.Softplus

		img2latent_encoder = Img2LatentEncoder(
			neuron_numbers = config['architecture']['neuron_numbers'],
			blocks_per_layer = config['architecture']['blocks_per_layer'],
			block_type = config['architecture']['block_type'],
			input_channels = config['data']['input_size'][0],
			use_batchnorm = config['architecture']['use_batchnorm'],
			activation = activation
		)

		#
		latent_height = int(config['data']['input_size'][2] / (math.pow(2, len(config['architecture']['neuron_numbers']))))

		#
		if len(set(config['task']['criterions'].keys()).intersection(['ce', 'bce', 'mse', 'mae', 'mixed', 'supervised'])) >= 1:
			#
			latent2target_decoder = Latent2VectorDecoder(
				output_size = config['task']['output_size'],
				num_hidden_in = config['architecture']['neuron_numbers'][-1],
				dimension_reduction = config['architecture']['dimension_reduction'],
				dropout = config['architecture']['dropout'],
				activation = activation,
				latent_height = latent_height,
			)

		elif 'ldj' in config['task']['criterions'].keys():
			# solely generative normalizing flow
			downsampling_factor = math.pow(2, len(config['architecture']['neuron_numbers']))
			latent2target_decoder = NFFlatten(input_shape = [
				config['architecture']['neuron_numbers'][-1],
				int(config['data']['input_size'][1] / downsampling_factor),
				int(config['data']['input_size'][2] / downsampling_factor)
			])

		elif 'reconstruction' in config['task']['criterions'].keys():
			#
			latent2target_decoder = Latent2ImgDecoder(
				neuron_numbers = config['architecture']['neuron_numbers'][::-1],
				blocks_per_layer = config['architecture']['blocks_per_layer'],
				block_type = config['architecture']['block_type'],
				output_size = config['data']['input_size'][0],
				use_batchnorm = config['architecture']['use_batchnorm'],
				activation = activation
			)

		super(ImgEncoderDecoderModel, self).__init__(*[img2latent_encoder, latent2target_decoder])
