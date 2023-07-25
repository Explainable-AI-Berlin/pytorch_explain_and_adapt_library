import os
import code  # code.interact(local=dict(globals(), **locals()))
import math
import torch
import numpy as np

from pydantic import PositiveInt

from peal.architectures.basic_modules import Mean
from peal.architectures.module_blocks import (
    FCBlock,
    ResnetBlock,
    TransformerBlock,
    VGGBlock,
)
from peal.configs.architectures.template import (
    ArchitectureConfig,
    FCConfig,
    VGGConfig,
    TransformerConfig,
    ResnetConfig,
)


def load_model(
    model_config: ArchitectureConfig,
    input_channels: PositiveInt,
    output_channels: PositiveInt,
    model_path,
    device,
):
    """
    This function loads a model from a given path.
    Args:
        model_config: The config of the model.
        input_channels: The number of input channels of the model.
        output_channels: The number of output channels of the model.
        model_path: The path to the model weights.
        device: The device the model is loaded on.

    Returns:
        The loaded model.
    """
    model = SequentialModel(model_config, input_channels, output_channels)
    checkpoint = torch.load(
        os.path.join(model_path, "checkpoints", "final.cpl"),
        map_location=torch.device(device),
    )
    model.load_state_dict(checkpoint)

    return model.to(device)


class SequentialModel(torch.nn.Sequential):
    """ A sequential model that is defined by a list of layers. """

    def __init__(
        self,
        architecture_config: ArchitectureConfig,
        input_channels: PositiveInt,
        output_channels: PositiveInt,
    ):
        """
        This function initializes the sequential model.
        Args:
            architecture_config: The config of the architecture.
            input_channels: The number of input channels of the model.
            output_channels: The number of output channels of the model.
        """
        if architecture_config.activation == "LeakyReLU":
            activation = torch.nn.LeakyReLU

        elif architecture_config.activation == "ReLU":
            activation = torch.nn.ReLU

        elif architecture_config.activation == "Softplus":
            activation = torch.nn.Softplus

        layers = []
        num_neurons_previous = input_channels
        for layer_config in architecture_config.layers:
            if isinstance(layer_config, ResnetConfig):
                layers.append(
                    ResnetBlock(layer_config, num_neurons_previous, activation)
                )
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, VGGConfig):
                layers.append(VGGBlock(layer_config, num_neurons_previous, activation))
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, FCConfig):
                layers.append(FCBlock(layer_config, num_neurons_previous, activation))
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, TransformerConfig):
                layers.append(
                    TransformerBlock(layer_config, num_neurons_previous, activation)
                )
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, str) and layer_config == "mean":
                layers.append(Mean())
                tensor_dim = 0

        last_layer_config = FCConfig(output_channels, tensor_dim=tensor_dim)
        layers.append(FCBlock(last_layer_config, num_neurons_previous, activation))
        super(SequentialModel, self).__init__(*layers)
