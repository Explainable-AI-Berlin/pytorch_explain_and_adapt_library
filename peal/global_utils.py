# This whole file contains all the stuff, that is written too bad and had no clear position where it should be located in the project!
import numpy as np
import sys
import types
import torch
import os
import yaml
import socket
import typing

from pkg_resources import resource_filename


def add_class_arguments(parser, config_class, base_str=""):
    for attr_name in config_class.__annotations__.keys():
        if attr_name[:2] == "__":
            continue

        attr_type = config_class.__annotations__[attr_name]
        parser.add_argument(
            f"--{base_str}{attr_name}",
            type=attr_type,
            default=None,
            help=str(attr_type), # TODO + getattr(config_class, attr_name).__doc__,
        )
        # TODO this is not thought to the end...
        if (
            isinstance(attr_type, typing._GenericAlias)
            and attr_type.__origin__ is typing.Union
        ):
            attr_types = [list(attr_type.__args__)[0]]

        else:
            attr_types = [attr_type]

        for attr_type in attr_types:
            if hasattr(attr_type, "__name__") and (
                getattr(attr_type, "__name__")[-6:] == "Config"
            ):
                add_class_arguments(parser, attr_type, f"{base_str}{attr_name}.")


def integrate_argument(arg_name, arg_value, config):
    if arg_value is None:
        pass

    elif "." in arg_name:
        arg_name1, arg_name2 = arg_name.split(".", 1)
        integrate_argument(arg_name2, arg_value, getattr(config, arg_name1))

    elif hasattr(getattr(config, arg_name).__class__, "__name__") and (
        getattr(getattr(config, arg_name).__class__, "__name__")[-6:] == "Config"
    ):
        setattr(
            config,
            arg_name,
            load_yaml_config(arg_value, getattr(config, arg_name).__class__),
        )

    else:
        setattr(config, arg_name, arg_value)


def integrate_arguments(args, config, exclude=[]):
    for arg_name, arg_value in args.__dict__.items():
        if not arg_name in exclude:
            integrate_argument(arg_name, arg_value, config)


def set_adaptive_batch_size(config, gigabyte_vram, samples_per_iteration):
    if not config.base_batch_size is None:
        multiplier = float(
            np.prod(config.assumed_input_size) / np.prod(config.data.input_size)
        )
        if not gigabyte_vram is None and not config.gigabyte_vram is None:
            multiplier = multiplier * (gigabyte_vram / config.gigabyte_vram)

        batch_size_adapted = max(1, int(config.base_batch_size * multiplier))
        if config.batch_size == -1:
            config.batch_size = batch_size_adapted
            config.num_batches = int(samples_per_iteration / batch_size_adapted) + 1


def embed_numberstring(number_str, num_digits=7):
    return "0" * (num_digits - len(number_str)) + number_str


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def request(name, default, is_asking=True):
    if not is_asking:
        return default

    answer = input(
        "Do you want to change value of " + str(name) + "==" + str(default) + "? [y/n]"
    )
    if answer == "n":
        return default

    else:
        if isinstance(default, bool):
            return not default

        elif isinstance(default, list):
            return input(
                "To what list of values do you want to change " + str(name) + "?"
            ).split(",")

        else:
            return input("To what value do you want to change " + str(name) + "?")


def get_project_resource_dir():
    return os.path.abspath(os.sep) + os.path.join(
        *resource_filename(__name__, "peal").replace("\\", "/").split("/")[:-1]
    )


def _load_yaml_config(config_path):
    def open_config(config_path):
        with open(config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                return config
            except yaml.YAMLError as exc:
                raise

    if not isinstance(config_path, str):
        # config_path is already a config object
        return config_path

    elif config_path[-5:] == ".yaml":
        split_path = config_path.split("/")
        if split_path[0] == "<PEAL_BASE>":
            config_path = os.path.join(get_project_resource_dir(), *split_path[1:])

        try:
            config = open_config(config_path)

        except Exception as e:
            error, _, _ = sys.exc_info()
            if error.__name__ == "OSError" or error.__name__ == "FileNotFoundError":
                config_path = os.path.abspath(os.path.join("..", *split_path))
                config_path = config_path.replace("<PEAL_BASE>", "peal")
                config = open_config(config_path)
            else:
                raise e

        for key in config.keys():
            if isinstance(config[key], str) and config[key][-5:] == ".yaml":
                config[key] = _load_yaml_config(os.path.join(*config[key].split("/")))

        return config

    else:
        raise Exception(config_path + " has no valid ending!")


def load_yaml_config(config_path, config_model=None):
    config_data = _load_yaml_config(config_path)
    if config_model is None and isinstance(config_data, dict):
        return types.SimpleNamespace(**config_data)

    elif isinstance(config_data, dict):
        return config_model(**config_data)

    else:
        return config_path


def move_to_device(X, device):
    if isinstance(X, list):
        return [torch.clone(x).to(device) for x in X]
    else:
        return torch.clone(X).to(device)


def requires_grad_(model, requires_grad):
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def orthogonal_initialization(model):
    """ """
    for parameter_idx, parameter in enumerate(model.parameters()):
        if len(parameter.shape) == 1:
            parameter.data = torch.randn(parameter.shape).to(parameter.device)
        else:
            torch.nn.init.orthogonal_(parameter)
