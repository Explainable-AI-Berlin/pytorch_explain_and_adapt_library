import torch
import os

import torchvision
from pydantic import BaseModel, PositiveInt
from typing import Union

from peal.architectures.basic_modules import Mean
from peal.architectures.module_blocks import (
    FCBlock,
    ResnetBlock,
    TransformerBlock,
    VGGBlock,
    create_cnn_layer, ResnetConfig, FCConfig, VGGConfig, TransformerConfig,
)
from peal.global_utils import load_yaml_config


class TaskConfig(BaseModel):
    """
    A dict of critirion names (that have to be implemented in peal.training.criterions)
    mapped to the weight.
    Like this the loss function can be post_hoc attached without changing the code.
    """

    criterions: dict = {'ce' : 1.0, 'l2' : 1.0}
    """
    The output_type that either can just be the output_type of the dataset or could be some
    possible subtype.
    E.g. when having a binary multiclass dataset one could use as task binary single class
    classification for one of the output variables.
    """
    output_type: str = "singleclass"
    """
    The output_channels that can be at most the output_channels of the dataset, but if a subtask is chosen
    the output_channels has also be adapted accordingly
    """
    output_channels: PositiveInt = 2
    """
    Gives the option to select a subset of the input variables. Only works for symbolic data.
    """
    x_selection: Union[list[str], type(None)] = None
    """
    Gives the option to select a subset of the output_variables.
    Can be used e.g. to transform binary multiclass into the subtask of predicting one of the
    binary variables with single class classification.
    """
    y_selection: Union[list[str], type(None)] = None
    """
    kwargs: dict = {}
    __name__: str = "peal.TaskConfig"

    def __init__(
        self,
        criterions: dict = None,
        output_type: str = None,
        output_channels: PositiveInt = None,
        x_selection: list[str] = None,
        y_selection: list[str] = None,
        **kwargs
    ):
        self.criterions = criterions if criterions is not None else self.criterions
        self.output_type = output_type if output_type is not None else self.output_type
        self.output_channels = output_channels if output_channels is not None else self.output_channels
        self.x_selection = x_selection if x_selection is not None else self.x_selection
        self.y_selection = y_selection if y_selection is not None else self.y_selection
        self.kwargs = kwargs"""


class ArchitectureConfig:
    """
    The config template for a neural architecture.
    """

    """
    The layers of the architecture.
    Elements of the list are tuples of the form (layer_type, *layer_config).
    Options for list elements: ['fc', 'vgg','resnet','transformer']
    """
    layers: list
    """
    The activation function used in the architecture.
    Options: ['ReLU', 'LeakyReLU', 'Softplus']
    """
    activation: str = "ReLU"
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
    """
    The name of the class.
    """
    __name__: str = "peal.ArchitectureConfig"

    def __init__(self, layers: list, activation: str = "ReLU", **kwargs):
        self.activation = activation
        self.kwargs = kwargs
        self.layers = []
        for layer in layers:
            """
            layer[0] contains the type of layer used:
            Options: ['fc', 'vgg','resnet','transformer']
            """
            if layer[0] == "resnet":
                self.layers.append(ResnetConfig(*layer[1:]))

            elif layer[0] == "fc":
                self.layers.append(FCConfig(*layer[1:]))

            elif layer[0] == "vgg":
                self.layers.append(VGGConfig(*layer[1:]))

            elif layer[0] == "transformer":
                self.layers.append(TransformerConfig(*layer[1:]))

            else:
                self.layers.append(layer[0])

def get_predictor(predictor, device="cpu"):
    if isinstance(predictor, torch.nn.Module):
        return predictor, None

    elif isinstance(predictor, str):
        if predictor[-4:] == ".cpl":
            return torch.load(predictor, map_location=device), None

        elif predictor[-5:] == ".onnx":
            return torch.onnx.load(predictor, map_location=device), None

    else:
        predictor_config = load_yaml_config(predictor)
        model_path = os.path.join(predictor_config.model_path, "model.cpl")
        predictor_out = torch.load(model_path, map_location=device)
        return predictor_out, predictor_config


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
    """A sequential model that is defined by a list of layers."""

    def __init__(
        self,
        architecture_config: ArchitectureConfig,
        input_channels: PositiveInt,
        output_channels: PositiveInt = None,
        dropout: float = 0.0,
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
                    create_cnn_layer(
                        ResnetBlock, layer_config, num_neurons_previous, activation
                    )
                )
                num_neurons_previous = layer_config.num_neurons
                tensor_dim = layer_config.tensor_dim

            elif isinstance(layer_config, VGGConfig):
                layers.append(
                    create_cnn_layer(
                        VGGBlock, layer_config, num_neurons_previous, activation
                    )
                )
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

            else:
                import pdb

                pdb.set_trace()
                raise ValueError("Unknown layer config: {}".format(layer_config))

        if not dropout == 0.0:
            layers.append(torch.nn.Dropout(dropout))

        if not output_channels is None:
            last_layer_config = FCConfig(output_channels, tensor_dim=tensor_dim)
            layers.append(FCBlock(last_layer_config, num_neurons_previous))#, activation))
            num_neurons_previous = output_channels

        self.output_channels = num_neurons_previous

        super(SequentialModel, self).__init__(*layers)


class TorchvisionModel(torch.nn.Module):
    def __init__(self, model, num_classes):
        super(TorchvisionModel, self).__init__()
        if model == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=True)

        elif model == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)

        else:
            raise ValueError("Unknown model: {}".format(model))

        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)