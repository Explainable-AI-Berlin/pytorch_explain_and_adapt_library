import os
import code  # code.interact(local=dict(globals(), **locals()))
import math
import torch
import numpy as np

from peal.architectures.model_parts import (
    Vector2LatentEncoder,
    Sequence2LatentEncoder,
    Img2LatentEncoder,
    Latent2VectorDecoder,
)


def load_model(model_config, model_path, device):
    """ """
    if model_config["architecture"]["block_type"] in ["vgg", "resnet"]:
        model = Img2VectorModel(model_config)

    elif model_config["architecture"]["block_type"] == "fc":
        model = Symbolic2VectorModel(model_config)

    else:
        print(model_config["data"]["output_type"] + " is not available!")

    checkpoint = torch.load(
        os.path.join(model_path, "checkpoints", "final.cpl"),
        map_location=torch.device(device),
    )
    model.load_state_dict(checkpoint)

    return model.to(device)


class Symbolic2VectorModel(torch.nn.Sequential):
    """ """

    def __init__(self, config):
        """ """
        if config["architecture"]["activation"] == "LeakyReLU":
            activation = torch.nn.LeakyReLU

        elif config["architecture"]["activation"] == "ReLU":
            activation = torch.nn.ReLU

        elif config["architecture"]["activation"] == "Softplus":
            activation = torch.nn.Softplus

        encoder = Vector2LatentEncoder(
            input_channels=config["data"]["input_size"][0],
            activation=activation,
            neuron_numbers=config["architecture"]["neuron_numbers_encoder"],
        )
        decoder = Latent2VectorDecoder(
            output_size=config["task"]["output_size"],
            activation=activation,
            num_hidden_in=config["architecture"]["neuron_numbers_encoder"][-1],
            neuron_numbers=config["architecture"]["neuron_numbers_decoder"],
        )

        super(Symbolic2VectorModel, self).__init__(*[encoder, decoder])


class Sequence2VectorModel(torch.nn.Sequential):
    """ """

    def __init__(self, config):
        """ """
        if config["architecture"]["activation"] == "LeakyReLU":
            activation = torch.nn.LeakyReLU

        elif config["architecture"]["activation"] == "ReLU":
            activation = torch.nn.ReLU

        elif config["architecture"]["activation"] == "Softplus":
            activation = torch.nn.Softplus

        encoder = Sequence2LatentEncoder(
            num_blocks=config["architecture"]['num_blocks'],
            embedding_dim=config["architecture"]["neuron_numbers_encoder"][-1],
            num_heads=config["architecture"]['num_heads'],
            input_channels=config["data"]['input_size'][-1] + 2,
            activation=activation,
        )
        decoder = Latent2VectorDecoder(
            output_size=config["task"]["output_size"],
            activation=activation,
            num_hidden_in=config["architecture"]["neuron_numbers_encoder"][-1],
            neuron_numbers=config["architecture"]["neuron_numbers_decoder"],
        )

        super(Sequence2VectorModel, self).__init__(*[encoder, decoder])


class Img2VectorModel(torch.nn.Sequential):
    '''
    _summary_

    Args:
        torch (_type_): _description_
    '''

    def __init__(self, config):
        '''
        _summary_

        Args:
            config (_type_): _description_
        '''
        #
        if config["architecture"]["activation"] == "LeakyReLU":
            activation = torch.nn.LeakyReLU

        elif config["architecture"]["activation"] == "ReLU":
            activation = torch.nn.ReLU

        elif config["architecture"]["activation"] == "Softplus":
            activation = torch.nn.Softplus

        img2latent_encoder = Img2LatentEncoder(
            neuron_numbers=config["architecture"]["neuron_numbers"],
            blocks_per_layer=config["architecture"]["blocks_per_layer"],
            block_type=config["architecture"]["block_type"],
            input_channels=config["data"]["input_size"][0],
            use_batchnorm=config["architecture"]["use_batchnorm"],
            activation=activation,
        )

        #
        latent_height = int(
            config["data"]["input_size"][2]
            / (math.pow(2, len(config["architecture"]["neuron_numbers"])))
        )

        #
        latent2target_decoder = Latent2VectorDecoder(
            output_size=config["task"]["output_size"],
            num_hidden_in=config["architecture"]["neuron_numbers"][-1],
            dimension_reduction=config["architecture"]["dimension_reduction"],
            dropout=config["architecture"]["dropout"],
            activation=activation,
            latent_height=latent_height,
        )

        super(Img2VectorModel, self).__init__(
            *[img2latent_encoder, latent2target_decoder]
        )
