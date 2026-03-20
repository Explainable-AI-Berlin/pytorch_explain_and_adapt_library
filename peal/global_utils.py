# This whole file contains all the stuff, that is written too bad and had no clear position where it should be located in the project!
import random

import numpy as np
import sys
import types
import pandas as pd
import torch
import os

import torchvision
import yaml
import socket
import typing
import inspect
import pkgutil
import importlib
import importlib.util
import matplotlib.pyplot as plt

from pkg_resources import resource_filename
from pydantic import BaseModel
from tqdm import tqdm
from pathlib import Path


def onehot(label, n_classes):
    one_hots = torch.zeros(label.size(0), n_classes).to(label.device)
    return one_hots.scatter_(1, label.to(torch.int64).view(-1, 1), 1)


def cprint(s, a, b):
    if a >= b:
        print(s)


def dict_to_bar_chart(input_dict, name):
    """
    Creates a bar chart from a dictionary and saves it as a PNG image.

    Args:
      interpretation: A dictionary where keys are strings and values are integers.
      name: The desired filename (without the .png extension) to save the image.
    """

    # Extract labels and values from the dictionary
    labels = list(input_dict.keys())
    values = list(input_dict.values())

    # Create the bar chart
    plt.bar(labels, values)
    plt.xlabel("Interpretation")
    plt.ylabel("Count")
    plt.title("Interpretation Distribution")

    # Save the chart as a PNG image
    plt.savefig(f"{name}.png")

    # Clear the plot to avoid affecting subsequent plots
    plt.clf()


def find_subclasses(base_class, directory):
    directory_name = os.path.split(directory)[-1]
    if directory_name == "dependencies":
        return []

    subclasses = []

    def check_module(module_name):
        module = importlib.import_module(module_name)

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                if issubclass(obj, base_class):
                    subclasses.append(obj)

    project_base_dir = get_project_resource_dir()
    for dirpath, dirnames, filenames in os.walk(directory):
        if "dependencies" in dirpath:
            continue

        for filename in filenames:
            current_path = os.path.join(dirpath, filename)
            if filename.endswith(".py"):
                module_path = os.path.relpath(
                    os.path.join(dirpath, filename), project_base_dir
                )
                module_name = module_path.replace("/", ".")[:-3]
                check_module(module_name)

            elif os.path.isdir(current_path):
                subclasses.extend(find_subclasses(base_class, current_path))

    for importer, package_name, _ in pkgutil.iter_modules():
        if package_name.startswith(directory):
            check_module(package_name)

    return subclasses


def add_class_arguments(parser, config_class, base_str=""):
    for attr_name in config_class.__annotations__.keys():
        if attr_name[:2] == "__":
            continue

        attr_type = config_class.__annotations__[attr_name]
        parser.add_argument(
            f"--{base_str}{attr_name}",
            type=attr_type,
            default=None,
            help=str(attr_type),  # TODO + getattr(config_class, attr_name).__doc__,
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
    number_str = str(number_str)
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


"""def get_project_resource_dir():
    return os.path.abspath(os.sep) + os.path.join(
        *resource_filename(__name__, "peal").replace("\\", "/").split("/")[:-1]
    )"""


def get_project_resource_dir():
    return os.path.abspath(os.sep) + os.path.join(
        *resource_filename(__name__, ".").replace("\\", "/").split("/")[:-2]
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

    if config_path[: len("$PEAL_RUNS")] == "$PEAL_RUNS":
        peal_runs = os.environ.get("PEAL_RUNS", "peal_runs")
        config_path = config_path.replace("$PEAL_RUNS", peal_runs)

    if config_path[: len("$PEAL_DATA")] == "$PEAL_DATA":
        peal_data = os.environ.get("PEAL_DATA", "datasets")
        config_path = config_path.replace("$PEAL_DATA", peal_data)

    if config_path[-5:] == ".yaml":
        split_path = config_path.split("/")
        if split_path[0] == "<PEAL_BASE>":
            config_path = os.path.join(get_project_resource_dir(), *split_path[1:])

        try:
            config = open_config(config_path)

        except Exception as e:
            # TODO this seems like a rather odd error handling
            error, _, _ = sys.exc_info()
            if error.__name__ == "OSError" or error.__name__ == "FileNotFoundError":
                config_path = os.path.abspath(os.path.join("..", *split_path))
                config_path = config_path.replace("<PEAL_BASE>", "peal")
                config = open_config(config_path)

            else:
                raise e

        for key in config.keys():
            if isinstance(config[key], str):
                if config[key][: len("$PEAL_RUNS")] == "$PEAL_RUNS":
                    peal_runs = os.environ.get("PEAL_RUNS", "peal_runs")
                    config[key] = config[key].replace("$PEAL_RUNS", peal_runs)

                if config[key][: len("$PEAL_DATA")] == "$PEAL_DATA":
                    peal_data = os.environ.get("PEAL_DATA", "datasets")
                    config[key] = config[key].replace("$PEAL_DATA", peal_data)

                if config[key][-5:] == ".yaml":
                    config[key] = _load_yaml_config(config[key])

        return config

    else:
        raise Exception(config_path + " has no valid ending!")


def get_config_model(config_data):
    if "config_name" in config_data.keys():
        subclass_dir = os.path.join(
            get_project_resource_dir(),
            "peal",
        )
        class_list = find_subclasses(BaseModel, subclass_dir)
        class_dict = {c.__name__: c for c in class_list}
        config_model = class_dict[config_data["config_name"]]
        return config_model

    else:
        config_class_str = config_data[config_data["category"] + "_type"] + "Config"

        module_path = os.path.join(
            "peal",
            config_data["category"],
        )
        if not module_path[-1] == "s":
             module_path = module_path + "s"

        superclass_dir = os.path.join(module_path, "interfaces")
        superclass_module_name = superclass_dir.replace("/", ".")
        module = importlib.import_module(superclass_module_name)
        superclass = None
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                if (
                    obj.__name__[-6:] == "Config"
                    and obj.__module__ == superclass_module_name
                ):
                    superclass = obj

        subclass_dir = os.path.join(
            get_project_resource_dir(),
            module_path,
        )
        class_list = find_subclasses(superclass, subclass_dir)
        class_dict = {c.__name__: c for c in class_list}
        config_model = class_dict[config_class_str]

        return config_model


def load_yaml_config(config_path, config_model=None, return_namespace=True):
    config_data = _load_yaml_config(config_path)

    if (
        config_model is None
        and isinstance(config_data, dict)
        and (
            "category" in config_data.keys()
            and config_data["category"] + "_type" in config_data.keys()
            or "config_name" in config_data.keys()
        )
    ):
        config_model = get_config_model(config_data)

    if config_model is None and isinstance(config_data, dict) and return_namespace:
        config = types.SimpleNamespace(**config_data)

    elif not config_model is None and isinstance(config_data, dict):
        for key in config_data.keys():
            if isinstance(config_data[key], dict):
                config_data[key] = load_yaml_config(config_data[key], return_namespace=False)

            elif isinstance(config_data[key], list):
                for idx in range(len(config_data[key])):
                    if isinstance(config_data[key][idx], dict):
                        config_data[key][idx] = load_yaml_config(config_data[key][idx])

        config = config_model(**config_data)

    else:
        config = config_data

    return config


def save_yaml_config(config, config_path):
    """
    This function saves a config to a yaml file.
    Args:
        config: The config to save.
        config_path: The path to save the config to.
    """

    def process_object(obj):
        if hasattr(obj, "__dict__"):
            return process_object(obj.__dict__)

        elif hasattr(obj, "state"):
            return process_object(obj.state)

        elif isinstance(obj, (list, tuple)):
            return [process_object(item) for item in obj]

        elif isinstance(obj, dict):
            return {key: process_object(value) for key, value in obj.items()}

        return obj

    processed_data = process_object(config)
    directory_path = os.path.dirname(config_path)
    if not os.path.exists(directory_path):
        Path(directory_path).mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as outfile:
        yaml.dump(processed_data, outfile, default_flow_style=False)


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
            parameter.data = torch.zeros(parameter.shape).to(parameter.device)

        else:
            torch.nn.init.orthogonal_(parameter)


'''def reset_weights(model):
    """
    Resets all weights in the model recursively using reset_parameters
    """
    for parameter_idx, parameter in enumerate(model.parameters()):
        if len(parameter.shape) >= 2:
            #parameter.data = torch.nn.init.xavier_uniform_(parameter)
            parameter.data = torch.randn_like(parameter.data)

        else:
            parameter.data = torch.zeros_like(parameter.data)'''


def reset_weights(model):
    for idx, layer in enumerate(model.children()):
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

        else:
            reset_weights(layer)


class LeakySoftplus(torch.nn.Module):
    def __init__(self):
        super(LeakySoftplus, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)
        self.softplus = torch.nn.Softplus(beta=10.0)

    def forward(self, x):
        return 0.5 * (self.leaky_relu(x) + self.softplus(x))


def replace_relu_with_leakysoftplus(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, LeakySoftplus())

        else:
            replace_relu_with_leakysoftplus(child)

    return model


def replace_relu_with_leakyrelu(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.LeakyReLU(negative_slope=0.1))

        else:
            replace_relu_with_leakyrelu(child)

    return model


def get_predictions(args):
    torch.set_grad_enabled(False)

    device = torch.device("cuda:0")
    os.makedirs("utils", exist_ok=True)

    dataset = args.dataset

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=5, shuffle=False
    )

    classifier = args.classifier

    d = {"idx": [], "prediction": []}
    n = 0
    acc = 0

    for idx, sample in enumerate(tqdm(loader)):
        if (
            hasattr(args, "max_samples")
            and not args.max_samples is None
            and idx > args.max_samples
        ):
            break

        if len(sample) == 3:
            img, lab, img_file = sample

        else:
            img, y = sample
            if len(y) == 2:
                (lab, img_file) = y

            elif len(y) == 3:
                (lab, hint, img_file) = y

            elif len(y) == 4:
                (lab, hint, idx, img_file) = y

        if hasattr(args, "is_image_to_class") and args.is_image_to_class:
            img_file = list(img_file)
            for i in range(len(img_file)):
                img_file[i] = os.path.join(str(int(lab[i])), img_file[i])

            img_file = tuple(img_file)

        img = img.to(device)
        lab = lab.to(device)

        logits = classifier(img)
        if len(logits.shape) > 1:
            pred = logits.argmax(dim=1)

        else:
            pred = (logits > 0).int()

        try:
            acc += (pred == lab).float().sum().item()

        except:
            import pdb; pdb.set_trace()
        n += lab.size(0)

        d["prediction"] += [p.item() for p in pred]
        d["idx"] += list(img_file)

    print("distilled accuracy: " + str(acc / n))
    print("distilled accuracy: " + str(acc / n))
    print("distilled accuracy: " + str(acc / n))
    df = pd.DataFrame(data=d)

    df.to_csv(
        args.label_path,
        index=False,
    )

    torch.set_grad_enabled(True)


def high_contrast_heatmap(x, counterfactual):
    if x.shape[0] == 3:
        # Per-channel differences
        diff_channels = counterfactual - x
        
        # Intensity change (mean of channel changes)
        # Red = Intensity Up, Blue = Intensity Down
        delta_intensity = diff_channels.mean(dim=0)
        heatmap_red = torch.clamp(delta_intensity, min=0)
        heatmap_blue = torch.clamp(-delta_intensity, min=0)
        
        # Color change (deviance from the mean change)
        # We boost the green channel by a factor of 1.5 relative to intensity changes
        # to reflect the stronger subjective impression of color shifts.
        heatmap_green = (diff_channels - delta_intensity).abs().sum(dim=0) * 1.3
        
        x_in = torch.clone(x)
        counterfactual_rgb = torch.clone(counterfactual)
    else:
        # For single channel, all changes are intensity changes
        diff = (counterfactual - x)
        heatmap_red = torch.clamp(diff, min=0)[0]
        heatmap_blue = torch.clamp(-diff, min=0)[0]
        heatmap_green = torch.zeros_like(heatmap_red)
        x_in = torch.tile(x, [3, 1, 1])
        counterfactual_rgb = torch.tile(torch.clone(counterfactual), [3, 1, 1])

    # Stack channels: Red (Up), Green (Color Neutral), Blue (Down)
    heatmap = torch.stack([heatmap_red, heatmap_green, heatmap_blue], dim=0)

    # Dilate the heatmap to make connected regions more visible
    heatmap = torch.nn.functional.max_pool2d(
        heatmap.unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze(0)
    
    if heatmap.max() > 0:
        heatmap_high_contrast = torch.clamp(heatmap / heatmap.max(), 0.0, 1.0)
    else:
        heatmap_high_contrast = heatmap

    return heatmap_high_contrast, x_in, counterfactual_rgb


def generate_overlay(x, counterfactual, alpha_factor=2.0):
    heatmap, x_in, _ = high_contrast_heatmap(x, counterfactual)
    if x_in.shape[0] == 3:
        grey = x_in.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    else:
        grey = x_in  # already tiled in high_contrast_heatmap if 1 channel

    # Bring everything closer to the mean (0.5) so that black and white look grey
    grey = (grey - 0.5) * 0.5 + 0.5

    # alpha mask based on max difference across colors
    # Since heatmap reached at least one 1.0 value due to normalization in high_contrast_heatmap,
    # the relative strongest change will be clearly highlighted.
    alpha = torch.clamp(heatmap.max(dim=0, keepdim=True)[0] * alpha_factor, 0, 1)

    # Blend heatmap onto the low-contrast greyscale background
    overlay = grey * (1 - alpha) + heatmap * alpha
    return overlay


@torch.no_grad()
def generate_smooth_mask(x1, x2, dilation, max_avg_combination=0.5):
    assert (dilation % 2) == 1, "dilation must be an odd number"
    mask = (x1 - x2).abs().sum(dim=1, keepdim=True)
    # mask = mask / mask.view(mask.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    # dil_mask = mask
    blurring = torchvision.transforms.GaussianBlur(dilation, sigma=2.0)
    dil_mask = blurring(mask)

    dil_mask = torch.clamp(dil_mask, 0, 1)
    dil_mask = dil_mask / mask.view(dil_mask.size(0), -1).max(dim=1)[0].view(
        -1, 1, 1, 1
    )

    """dil_mask = max_avg_combination * F.avg_pool2d(mask, dilation, stride=1, padding=(dilation - 1) // 2)
    dil_mask += (1 - max_avg_combination) * F.max_pool2d(
        mask, dilation, stride=1, padding=(dilation - 1) // 2
    )"""
    # dil_mask = F.max_pool2d(mask, dilation, stride=1, padding=(dilation - 1) // 2)
    return mask, dil_mask


def get_intermediate_output(
    model: torch.nn.Module, x: torch.Tensor, distance_from_last_layer: int
):
    """
    Runs the model on input x and returns the activation tensor
    that is `distance_from_last_layer` layers away from the final layer.

    Args:
        model (nn.Module): The predictor model.
        x (torch.Tensor): Input tensor (image).
        distance_from_last_layer (int): Number of layers away from the final output layer.
                                        (1 = last layer before logits, 2 = two layers back, etc.)
    """
    activations = {}
    handles = []

    # Get a flat list of all modules in order
    layers = [
        m
        for m in model.modules()
        if not isinstance(m, torch.nn.Sequential)
        and not isinstance(m, torch.nn.ModuleList)
    ]
    layers = [
        m for m in layers if len(list(m.children())) == 0
    ]  # keep only leaf modules

    # Select the layer we want
    target_layer_index = len(layers) - distance_from_last_layer
    if target_layer_index < 0 or target_layer_index >= len(layers):
        raise ValueError("distance_from_last_layer is out of range.")

    target_layer = layers[target_layer_index]

    # Hook to capture the output
    def hook_fn(module, input, output):
        activations["out"] = output.detach()

    handle = target_layer.register_forward_hook(hook_fn)
    handles.append(handle)

    # Run forward pass
    _ = model(x)

    # Cleanup
    for h in handles:
        h.remove()

    return activations["out"]


def extract_penultima_activation(x, predictor, distance_from_last_layer=1):
    # Function to traverse the computation graph
    """predictor.train()
    def find_activation(tensor, depth=2):
        current_fn = tensor.grad_fn  # Start from the output tensor
        for _ in range(depth):
            if current_fn is None or not current_fn.next_functions:
                raise ValueError(
                    "Unable to traverse the graph; check the computation structure."
                )
            # Navigate to the next function
            current_fn = current_fn.next_functions[0][0]  # Access the next function
        # Return the associated variable (activation)
        if hasattr(current_fn, "variable"):
            return current_fn.variable
        raise ValueError("Activation not found at the specified depth.")

    # Example: Find the second-to-last activation
    output = predictor(x.requires_grad_())
    import pdb; pdb.set_trace()
    second_last_activation = find_activation(output, depth=2)
    predictor.eval()"""
    if distance_from_last_layer == 1:
        if hasattr(predictor, "feature_extractor"):
            return predictor.feature_extractor(x)

        else:
            submodules = list(predictor.children())
            while len(submodules) == 1:
                submodules = list(submodules[0].children())

            feature_extractor = torch.nn.Sequential(*submodules[:-1])
            return feature_extractor(x)

    else:
        return get_intermediate_output(predictor, x, distance_from_last_layer)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Ensure deterministic results.
    torch.backends.cudnn.benchmark = (
        False  # Disable the optimization for specific architectures.
    )
    os.environ["PYTHONHASHSEED"] = str(seed)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

class DINOEvaluator:
    def __init__(self, model_name='facebook/dinov2-small', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Loading {model_name} on {self.device}...")

        # We use the base model (no classification head) as we only need embeddings
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Storage for distribution statistics (PyTorch Tensors)
        self.mu_real = None
        self.sigma_real = None
        self.sigma_inv_real = None # For Mahalanobis
        self.fitted = False

    @torch.no_grad()
    def _get_features(self, inputs):
        """
        Internal helper to extract [CLS] token embeddings.
        inputs: can be a DataLoader or a Tensor (B, C, H, W) or List of Tensors
        """
        # If inputs is a dataloader, we iterate and concatenate
        if isinstance(inputs, DataLoader):
            features_list = []
            for batch in tqdm(inputs, desc="Extracting features"):
                # Handle standard Tuple (img, label) or just img
                if isinstance(batch, (list, tuple)):
                    imgs = batch[0]
                else:
                    imgs = batch

                imgs = imgs.to(self.device).float()
                outputs = self.model(pixel_values=imgs)
                # DINOv2: The [CLS] token is usually the first token or pooled output
                # transformers implementation usually puts pooled_output in .pooler_output
                # or we take last_hidden_state[:, 0]
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    feats = outputs.pooler_output
                else:
                    feats = outputs.last_hidden_state[:, 0, :]

                features_list.append(feats)
            return torch.cat(features_list, dim=0)

        # If inputs is a direct tensor batch
        else:
            inputs = inputs.to(self.device).float()
            outputs = self.model(pixel_values=inputs)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                return outputs.last_hidden_state[:, 0, :]

    def fit(self, real_dataloader):
        """
        Step 1: Precompute statistics for the Real (Training) distribution.
        Run this once at the beginning.
        """
        print("Fitting evaluator to real data...")
        real_feats = self._get_features(real_dataloader) # (N, D)

        # 1. Compute Mean
        self.mu_real = torch.mean(real_feats, dim=0)

        # 2. Compute Covariance
        # torch.cov requires shape (Variables, Observations), so we transpose
        self.sigma_real = torch.cov(real_feats.T)

        # 3. Compute Inverse Covariance for Mahalanobis
        # We add a small epsilon to diagonal for numerical stability (regularization)
        D = self.sigma_real.shape[0]
        epsilon = 1e-6
        reg_sigma = self.sigma_real + torch.eye(D, device=self.device) * epsilon

        # Use pseudo-inverse or inverse (cholesky is faster if positive definite)
        try:
            self.sigma_inv_real = torch.linalg.inv(reg_sigma)
        except torch.linalg.LinAlgError:
            print("Warning: Covariance matrix singular, using pseudo-inverse.")
            self.sigma_inv_real = torch.linalg.pinv(reg_sigma)

        self.fitted = True
        print(f"Fitted on {len(real_feats)} samples. Feature Dim: {D}")

    def compute_fid(self, fake_inputs):
        """
        Calculates FID score between the fitted real distribution and the provided fake inputs.
        """
        if not self.fitted:
            raise ValueError("Please run .fit(real_dataloader) before computing FID.")

        fake_feats = self._get_features(fake_inputs)

        mu_fake = torch.mean(fake_feats, dim=0)
        sigma_fake = torch.cov(fake_feats.T)

        # --- FID Formula in PyTorch ---
        # FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r * Sigma_g)^(1/2))

        # 1. Diff squared term
        diff = self.mu_real - mu_fake
        diff_sq = diff.dot(diff)

        # 2. Trace term
        # Matrix square root of product
        covmean = self.sigma_real @ sigma_fake

        # Calculating matrix square root in PyTorch
        # We use SVD for stability: M = U S V*, sqrt(M) approx via eigen decomp
        # Or usually eigenvalues: M = V L V^{-1}, M^{1/2} = V L^{1/2} V^{-1}
        vals, vecs = torch.linalg.eig(covmean)

        # Suppress complex parts (numerical noise)
        vals = vals.real
        vecs = vecs.real

        # sqrt of eigenvalues
        sqrt_vals = torch.sqrt(torch.clamp(vals, min=0)) # clamp to avoid nan

        # Reconstruct sqrt matrix
        # This part can be tricky in pure PyTorch without scipy.linalg.sqrtm
        # Approximation: Tr((Sigma_r * Sigma_g)^(1/2)) = sum(sqrt(eigenvalues))
        # strictly true if matrices effectively commute or via trace properties

        trace_sqrt_product = torch.sum(sqrt_vals)

        trace_term = torch.trace(self.sigma_real) + torch.trace(sigma_fake) - 2 * trace_sqrt_product

        fid = diff_sq + trace_term
        return fid.item()

    def compute_mahalanobis(self, single_image_or_batch):
        """
        Calculates Mahalanobis distance for specific samples against the real distribution.
        Returns a tensor of distances (one per image).
        """
        if not self.fitted:
            raise ValueError("Please run .fit(real_dataloader) first.")

        # Ensure inputs are tensor (preprocess if needed before passing here)
        feats = self._get_features(single_image_or_batch) # (B, D)

        # Distance = sqrt( (x - mu) * Sigma^-1 * (x - mu)^T )
        # Centering
        diff = feats - self.mu_real.unsqueeze(0) # Broadcasting (B, D) - (1, D)

        # Matrix multiplication: (B, D) @ (D, D) -> (B, D)
        left = diff @ self.sigma_inv_real

        # Dot product: sum(left * diff, dim=1)
        # This is effectively doing the row-wise dot product
        mahalanobis_sq = torch.sum(left * diff, dim=1)

        return torch.sqrt(torch.clamp(mahalanobis_sq, min=0))