# This whole file contains all the stuff, that is written too bad and had no clear position where it should be located in the project!
"""Cross-cutting helpers shared by the whole PEAL library.

Collected here are the utilities that do not belong to one subpackage: yaml
config loading (including the ``$PEAL_RUNS``/``$PEAL_DATA``/``<PEAL_BASE>``
placeholders, recursive inclusion of nested yamls and the mapping of a config
onto its pydantic model), argparse integration for those configs, seed handling
and device moves, small module surgery such as replacing ReLUs or resetting
weights, the difference heatmaps and change masks used to visualise
counterfactuals, and a DINOv2-based evaluator for FID, Mahalanobis distance and
a LPIPS-style perceptual distance.
"""

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

from pydantic import BaseModel
from tqdm import tqdm
import pathlib
from pathlib import Path
from peal.log import get_logger

_log = get_logger(__name__)


def onehot(label, n_classes):
    """One-hot encode a batch of class indices.

    Parameters
    ----------
    label : torch.Tensor
        Class indices of shape ``(B,)`` (or anything reshapeable to
        ``(B, 1)``).
    n_classes : int
        Width of the encoding.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(B, n_classes)`` on the device of ``label``.
    """
    one_hots = torch.zeros(label.size(0), n_classes).to(label.device)
    return one_hots.scatter_(1, label.to(torch.int64).view(-1, 1), 1)


def cprint(s, a, b):
    """Print ``s`` only if the tracking level ``a`` reaches the threshold ``b``.

    Parameters
    ----------
    s : object
        What to print.
    a : int
        The configured verbosity (``tracking_level``).
    b : int
        Verbosity from which on this message is of interest.
    """
    if a >= b:
        _log.info("%s", s)


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
    """Import every module under ``directory`` and collect subclasses.

    Used by the config machinery to find, for example, all
    ``ExplainerConfig`` subclasses so that a config file can be mapped onto
    its pydantic model by name. Every ``.py`` file below ``directory`` is
    imported as a module relative to the project root; anything under a
    ``dependencies`` directory is skipped, since the vendored third-party code
    is expensive and often not importable. Import errors are printed and the
    module is skipped. Afterwards, top-level packages whose name starts with
    ``directory`` are imported as well.

    Parameters
    ----------
    base_class : type
        The class whose subclasses are wanted (``base_class`` itself matches
        too, as ``issubclass`` is reflexive).
    directory : str
        Directory to walk.

    Returns
    -------
    list of type
        The classes found, possibly with duplicates.
    """
    directory_name = os.path.split(directory)[-1]
    if directory_name == "dependencies":
        return []

    subclasses = []

    def check_module(module_name):
        """Import a module and append its ``base_class`` subclasses."""
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
                try:
                    # print(f"Importing module {module_name}...")
                    if "dependencies" in module_name:
                        continue
                    check_module(module_name)
                except Exception as e:
                    _log.info("%s", f"Error importing module {module_name}: {e}")

            elif os.path.isdir(current_path):
                subclasses.extend(find_subclasses(base_class, current_path))

    for importer, package_name, _ in pkgutil.iter_modules():
        if package_name.startswith(directory):
            try:
                check_module(package_name)
            except Exception as e:
                _log.info("%s", f"Error importing package {package_name}: {e}")
                raise

    return subclasses


def add_class_arguments(parser, config_class, base_str=""):
    """Add one command line argument per annotated config field.

    Every annotation of ``config_class`` becomes ``--<base_str><name>`` with
    default ``None``, so that only the flags actually passed override the
    config file. Fields whose type is itself a ``*Config`` class are recursed
    into with a dotted prefix (``--data.input_size``); for ``Union`` types the
    first member is inspected.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser the arguments are added to, in place.
    config_class : type
        A pydantic config class.
    base_str : str, optional
        Dotted prefix for nested configs.
    """
    for attr_name in config_class.__annotations__.keys():
        if attr_name[:2] == "__":
            continue

        attr_type = config_class.__annotations__[attr_name]
        # Resolve Union and Optional BEFORE registering the argument. argparse
        # calls `type` on the raw string, and `Optional[int]("1")` is not
        # callable, so `--seed 1` used to die with
        # "invalid typing.Optional[int] value: '1'" instead of setting a seed.
        if (
            isinstance(attr_type, typing._GenericAlias)
            and attr_type.__origin__ is typing.Union
        ):
            attr_types = [list(attr_type.__args__)[0]]

        else:
            attr_types = [attr_type]

        # Only a plain callable can coerce a command line string; anything else
        # (a nested *Config, a bare list or dict annotation) is taken verbatim.
        parse_type = attr_types[0]
        if not callable(parse_type) or isinstance(parse_type, typing._GenericAlias):
            parse_type = str
        parser.add_argument(
            f"--{base_str}{attr_name}",
            type=parse_type,
            default=None,
            help=str(attr_type),  # TODO + getattr(config_class, attr_name).__doc__,
        )

        for attr_type in attr_types:
            if hasattr(attr_type, "__name__") and (
                getattr(attr_type, "__name__")[-6:] == "Config"
            ):
                add_class_arguments(parser, attr_type, f"{base_str}{attr_name}.")


def integrate_argument(arg_name, arg_value, config):
    """Write one parsed command line value into a (possibly nested) config.

    A dotted ``arg_name`` descends into the corresponding sub-config. If the
    field currently holds a ``*Config`` object, ``arg_value`` is treated as
    the path of a yaml file and loaded into that config class; otherwise it is
    set directly. ``None`` means "not given on the command line" and is
    ignored.

    Parameters
    ----------
    arg_name : str
        Field name, possibly dotted.
    arg_value : object
        Parsed value, or ``None``.
    config : object
        Config to modify in place.
    """
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
    """Apply all parsed command line arguments to a config.

    Calls :func:`integrate_argument` for every attribute of ``args`` that is
    not listed in ``exclude`` and finally re-runs :func:`propagate_seed`, so
    that a seed given on the command line also reaches the sub-configs.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    config : object
        Config to modify in place.
    exclude : list of str, optional
        Argument names that are not config fields (for example the config
        path itself).
    """
    for arg_name, arg_value in args.__dict__.items():
        if not arg_name in exclude:
            integrate_argument(arg_name, arg_value, config)

    # Always ensure that any updated seed is propagated
    propagate_seed(config)


def set_adaptive_batch_size(config, gigabyte_vram, samples_per_iteration):
    """Scale a reference batch size to the current resolution and GPU.

    Only acts when ``config.base_batch_size`` is set and
    ``config.batch_size == -1``. The reference batch size, which was measured
    for ``config.assumed_input_size`` on a GPU with ``config.gigabyte_vram``,
    is scaled by the ratio of the input volumes and by the ratio of the
    available memory, and ``config.num_batches`` is set so that
    ``samples_per_iteration`` samples are covered.

    Parameters
    ----------
    config : object
        Config modified in place; reads ``base_batch_size``,
        ``assumed_input_size``, ``data.input_size``, ``gigabyte_vram`` and
        ``batch_size``.
    gigabyte_vram : float or None
        Memory of the GPU actually used; ``None`` disables the memory term.
    samples_per_iteration : int
        Number of samples one epoch should see.
    """
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
    """Left-pad a number with zeros so that filenames sort lexicographically.

    Parameters
    ----------
    number_str : int or str
        The number to pad.
    num_digits : int, optional
        Target width; longer inputs are returned unchanged.

    Returns
    -------
    str
        The zero-padded number, e.g. ``"0000042"``.
    """
    number_str = str(number_str)
    return "0" * (num_digits - len(number_str)) + number_str


def is_port_in_use(port: int) -> bool:
    """Return ``True`` if something is already listening on ``localhost:port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def request(name, default, is_asking=True):
    """Ask on the console whether a value should be changed.

    Used by the interactive config scripts. Booleans are simply negated when
    the answer is yes, lists are read back as a comma separated string, and
    everything else is read as free text (so the return type is ``str``).

    Parameters
    ----------
    name : str
        Name shown in the prompt.
    default : object
        Current value, returned when the answer is ``"n"``.
    is_asking : bool, optional
        ``False`` returns the default without prompting, which makes the
        scripts usable non-interactively.

    Returns
    -------
    object
        The default or the value entered by the user.
    """
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
    """Return the repository root, i.e. the directory containing ``peal``.

    Serves as the anchor for the ``<PEAL_BASE>`` placeholder in config paths
    and as the base for the module names built in :func:`find_subclasses`.

    ``$PEAL_BASE`` wins when it is set. Without it the answer is the parent of
    the ``peal`` package, which is the repository root in a checkout but
    ``site-packages`` once PEAL is pip-installed -- and ``configs/`` lives
    outside the package, so every ``<PEAL_BASE>/configs/...`` path in the 1294
    shipped configs would resolve into ``site-packages``. Point ``$PEAL_BASE``
    at an unpacked source tree to use them from an installed PEAL.

    ``peal.generators.rae_pipeline._peal_base`` reads the same variable, so the
    two placeholder expanders agree.

    Returns
    -------
    str
        Directory that ``<PEAL_BASE>`` expands to.
    """
    override = os.environ.get("PEAL_BASE")
    if override:
        return str(Path(override).expanduser().resolve())
    return str(Path(__file__).resolve().parents[1])


def _missing_configs_hint(config_path):
    """Explain a missing ``<PEAL_BASE>/configs/...`` file on an installed PEAL.

    The 1294 shipped configs live outside the ``peal`` package, so they travel
    in the source distribution but not in the wheel. A wheel install therefore
    resolves ``<PEAL_BASE>/configs/...`` into ``site-packages``, where nothing
    is found. Without this hint the failure reads as a missing file and gives
    no clue that ``$PEAL_BASE`` is the knob.

    Parameters
    ----------
    config_path : str
        The path that could not be opened, after placeholder expansion.

    Returns
    -------
    str or None
        A sentence to append to the error, or ``None`` when the situation is
        an ordinary missing file and the standard message is already right.
    """
    base = pathlib.Path(get_project_resource_dir())
    if os.environ.get("PEAL_BASE") or (base / "configs").is_dir():
        return None
    if "configs" not in str(config_path):
        return None
    return (
        f"No configs/ directory under {base}. PEAL was probably installed as a "
        "wheel, which does not carry the experiment configs: they ship in the "
        "source distribution only. Clone the repository (or unpack the sdist) "
        "and point $PEAL_BASE at it."
    )


def _load_yaml_config(config_path):

    def open_config(config_path):
        """Read one yaml file into a dict, re-raising parse errors."""
        try:
            with open(config_path, "r") as stream:
                try:
                    config = yaml.safe_load(stream)
                    return config
                except yaml.YAMLError as exc:
                    _log.info("%s", exc)
                    raise
        except FileNotFoundError as exc:
            hint = _missing_configs_hint(config_path)
            if hint:
                raise FileNotFoundError(f"{exc}. {hint}") from exc
            raise
        except:
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
            error, _, _ = sys.exc_info()
            if error.__name__ == "OSError" or error.__name__ == "FileNotFoundError":
                if os.path.isabs(config_path):
                    raise e

                config_path = os.path.abspath(os.path.join("..", *split_path))
                config_path = config_path.replace("<PEAL_BASE>", "peal")
                config = open_config(config_path)

            else:
                raise e

        # Keys whose value is a raw file path consumed directly by third-party
        # loaders (e.g. OmegaConf.load) and must NOT be pre-parsed into a dict
        # here, even though it ends in ".yaml". stage1_config/stage2_config are
        # the RAEv2 yamls that RAEDiffusionAutoencoder hands to RAEv2's own
        # train scripts and OmegaConf by path.
        RAW_PATH_KEYS = {"config_path", "stage1_config", "stage2_config"}

        def expand_recursive(cfg, key=None):
            """Expand path placeholders and inline nested yaml files.

            Walks the loaded structure and, for every string, substitutes
            ``$PEAL_RUNS``, ``$PEAL_DATA`` and ``<PEAL_BASE>``. A string that
            then still ends in ``.yaml`` is loaded as a sub-config, unless its
            key is in ``RAW_PATH_KEYS``, where the path itself is the value.
            """
            if isinstance(cfg, dict):
                for k in list(cfg.keys()):
                    cfg[k] = expand_recursive(cfg[k], key=k)
            elif isinstance(cfg, list):
                for i in range(len(cfg)):
                    cfg[i] = expand_recursive(cfg[i], key=key)
            elif isinstance(cfg, str):
                if cfg.startswith("$PEAL_RUNS"):
                    peal_runs = os.environ.get("PEAL_RUNS", "peal_runs")
                    cfg = cfg.replace("$PEAL_RUNS", peal_runs)

                if cfg.startswith("$PEAL_DATA"):
                    peal_data = os.environ.get("PEAL_DATA", "datasets")
                    cfg = cfg.replace("$PEAL_DATA", peal_data)

                if "<PEAL_BASE>" in cfg:
                    cfg = cfg.replace("<PEAL_BASE>", get_project_resource_dir())

                if cfg.endswith(".yaml") and key not in RAW_PATH_KEYS:
                    cfg = _load_yaml_config(cfg)
            return cfg

        config = expand_recursive(config)
        return config

    else:
        raise Exception(config_path + " has no valid ending!")


def get_config_model(config_data):
    """Find the pydantic config class that a raw config dict describes.

    Two conventions are supported. If the dict has a ``config_name`` key, that
    name is looked up among all ``BaseModel`` subclasses found under ``peal``.
    Otherwise the class name is built as ``<category>_type`` + ``"Config"``
    (for example ``category: explainer`` and ``explainer_type: SCE`` give
    ``SCEConfig``), and the search is narrowed to the subclasses of the
    ``*Config`` base class defined in ``peal/<category>s/interfaces.py``.

    Parameters
    ----------
    config_data : dict
        The raw yaml content.

    Returns
    -------
    type
        The config class.

    Raises
    ------
    KeyError
        If no class of the derived name exists.
    """
    if "config_name" in config_data.keys():
        _log.info(
            "%s",
            "Config data contains config_name, trying to get config model from it.",
        )
        # Registry first: the scan below imports every module under peal/ and
        # ran on every config load with a config_name (2 - 6 s warm, minutes cold).
        from peal.registry import CONFIGS, resolve

        if config_data["config_name"] in CONFIGS:
            return resolve(CONFIGS[config_data["config_name"]])

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
        from peal.registry import CONFIGS, resolve

        if config_class_str in CONFIGS:
            return resolve(CONFIGS[config_class_str])

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


def propagate_seed(config):
    """Copy the top-level seed into every nested config that has one.

    Walks all attributes, list entries and dict values of ``config`` and sets
    their ``seed`` attribute to ``config.seed``, so that a single seed in the
    top-level config governs data, generator, predictor and explainer alike.
    Does nothing when the top-level config has no seed or it is ``None``, and
    silently skips attributes that refuse assignment.

    Parameters
    ----------
    config : object
        Config tree, modified in place.
    """
    if hasattr(config, "seed") and getattr(config, "seed") is not None:
        seed_val = getattr(config, "seed")

        def _propagate(obj, s):
            if isinstance(obj, BaseModel) or hasattr(obj, "__dict__"):
                if hasattr(obj, "seed"):
                    try:
                        obj.seed = s
                    except Exception:
                        pass
                for k, v in obj.__dict__.items():
                    if k != "seed":
                        _propagate(v, s)
            elif isinstance(obj, list) or isinstance(obj, tuple):
                for item in obj:
                    _propagate(item, s)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _propagate(v, s)

        for key, value in config.__dict__.items():
            if key != "seed":
                _propagate(value, seed_val)


def load_yaml_config(config_path, config_model=None, return_namespace=True):
    """Load a config file (or dict) into its pydantic model.

    The central entry point used all over PEAL. The yaml is read with all
    path placeholders expanded and nested yamls inlined. If no
    ``config_model`` is given but the data identifies itself through
    ``config_name`` or ``category``/``<category>_type``, the model is looked
    up with :func:`get_config_model`; a failure there is only warned about.
    With a model, nested dicts and lists of dicts are converted first, the
    model is instantiated and :func:`propagate_seed` is run. Without one, the
    dict is either wrapped in a ``SimpleNamespace`` or returned as is.

    Parameters
    ----------
    config_path : str or dict or object
        Path of a ``.yaml`` file, an already loaded dict, or a config object,
        which is passed through unchanged.
    config_model : type, optional
        The pydantic class to instantiate; inferred when omitted.
    return_namespace : bool, optional
        Wrap a model-less dict in a ``types.SimpleNamespace`` instead of
        returning the plain dict. Set to ``False`` for nested dicts, which
        pydantic wants to receive as dicts.

    Returns
    -------
    object
        The config: an instance of ``config_model``, a ``SimpleNamespace`` or
        the raw data.

    Raises
    ------
    Exception
        If ``config_path`` is a string that does not end in ``.yaml``.
    """
    _log.info("%s", "Loading config from " + str(config_path))
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
        _log.info(
            "%s",
            "No config model provided, but config_data contains category and type. "
            "Trying to get config model from config_data.",
        )
        try:
            config_model = get_config_model(config_data)
        except Exception as e:
            _log.info("%s", f"Warning: Failed to infer config_model: {e}")

    if config_model is None and isinstance(config_data, dict) and return_namespace:
        _log.info(
            "%s",
            "No config model provided, but config_data is a dict. "
            "Returning a SimpleNamespace object.",
        )
        config = types.SimpleNamespace(**config_data)

    elif not config_model is None and isinstance(config_data, dict):
        _log.info(
            "%s",
            "Config model provided and config_data is a dict. "
            "Loading config data into the config model.",
        )
        for key in config_data.keys():
            if isinstance(config_data[key], dict):
                config_data[key] = load_yaml_config(
                    config_data[key], return_namespace=False
                )

            elif isinstance(config_data[key], list):
                for idx in range(len(config_data[key])):
                    if isinstance(config_data[key][idx], dict):
                        config_data[key][idx] = load_yaml_config(config_data[key][idx])

        config = config_model(**config_data)
        propagate_seed(config)

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
        """Convert a config tree into plain yaml-serialisable data.

        Pydantic models and objects with a ``__dict__`` become dicts (private
        attributes are dropped), lists and dicts are mapped element-wise, and
        anything that yaml has no representer for falls back to ``str(obj)``.
        """
        if isinstance(obj, BaseModel):
            data = {}
            for k, v in obj.__dict__.items():
                if not k.startswith("_"):
                    data[k] = process_object(v)
            return data

        if hasattr(obj, "__dict__"):
            return process_object(obj.__dict__)

        elif hasattr(obj, "state"):
            return process_object(obj.state)

        elif isinstance(obj, (list, tuple)):
            return [process_object(item) for item in obj]

        elif isinstance(obj, dict):
            return {key: process_object(value) for key, value in obj.items()}

        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # Anything else has no yaml representer and would abort the dump with
        # the config half-written. A SAE config's nested cfg dict, for
        # instance, carries cfg["dtype"] = torch.float32, which killed every
        # such run right after training had finished and the weights were
        # already on disk.
        # A config dump is for the record, so a readable repr is enough.
        return str(obj)

    processed_data = process_object(config)
    directory_path = os.path.dirname(config_path)
    if not os.path.exists(directory_path):
        Path(directory_path).mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as outfile:
        yaml.dump(processed_data, outfile, default_flow_style=False)


def move_to_device(X, device):
    """Clone a tensor, or a list of tensors, onto ``device``.

    Parameters
    ----------
    X : torch.Tensor or list of torch.Tensor
        The data to move.
    device : torch.device or str
        Target device.

    Returns
    -------
    torch.Tensor or list of torch.Tensor
        Copies, so the originals keep their device and graph.
    """
    if isinstance(X, list):
        return [torch.clone(x).to(device) for x in X]
    else:
        return torch.clone(X).to(device)


def requires_grad_(model, requires_grad):
    """Set ``requires_grad`` on every parameter of ``model``, in place."""
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def orthogonal_initialization(model):
    """Orthogonally initialise all weights and zero all biases.

    Every parameter with at least two dimensions gets
    ``torch.nn.init.orthogonal_``; one-dimensional parameters (biases, norm
    scales) are set to zero.

    Parameters
    ----------
    model : torch.nn.Module
        Model modified in place.
    """
    for parameter_idx, parameter in enumerate(model.parameters()):
        if len(parameter.shape) == 1:
            parameter.data = torch.zeros(parameter.shape).to(parameter.device)

        else:
            torch.nn.init.orthogonal_(parameter)


def reset_weights(model):
    """Re-initialise a model by calling ``reset_parameters`` recursively.

    Descends through the module tree and calls ``reset_parameters`` on every
    layer that has one, so that a pretrained model can be turned back into a
    randomly initialised one of the same architecture.

    Parameters
    ----------
    model : torch.nn.Module
        Model modified in place.
    """
    for idx, layer in enumerate(model.children()):
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

        else:
            reset_weights(layer)


class LeakySoftplus(torch.nn.Module):
    """Smooth, everywhere non-zero gradient replacement for ReLU.

    Computes ``0.5 * (LeakyReLU(x, 0.1) + Softplus(x, beta=10))``. Swapping
    the ReLUs of a predictor for this activation (see
    :func:`replace_relu_with_leakysoftplus`) keeps the function close to the
    original one while removing the flat region that stalls the gradient-based
    counterfactual search.
    """

    def __init__(self):
        """Build the leaky ReLU and softplus halves of the activation."""
        super(LeakySoftplus, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)
        self.softplus = torch.nn.Softplus(beta=10.0)

    def forward(self, x):
        """Return the mean of the leaky ReLU and the softplus of ``x``."""
        return 0.5 * (self.leaky_relu(x) + self.softplus(x))


def replace_relu_with_leakysoftplus(model):
    """Replace every ``ReLU`` in the module tree by a :class:`LeakySoftplus`.

    Parameters
    ----------
    model : torch.nn.Module
        Model modified in place.

    Returns
    -------
    torch.nn.Module
        The same model, for convenience.

    Notes
    -----
    Functional ``torch.relu`` calls inside a ``forward`` are not touched, only
    ``ReLU`` submodules.
    """
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, LeakySoftplus())

        else:
            replace_relu_with_leakysoftplus(child)

    return model


def replace_relu_with_leakyrelu(model):
    """Replace every ``ReLU`` submodule by a ``LeakyReLU`` with slope 0.1.

    Parameters
    ----------
    model : torch.nn.Module
        Model modified in place.

    Returns
    -------
    torch.nn.Module
        The same model, for convenience.
    """
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.LeakyReLU(negative_slope=0.1))

        else:
            replace_relu_with_leakyrelu(child)

    return model


def get_predictions(args):
    """Predict a whole dataset and write the labels to a csv file.

    Runs ``args.classifier`` over ``args.dataset`` with gradients disabled,
    accepting the several sample layouts used in PEAL and in the vendored
    code (``(img, lab, file)`` triples, dicts with ``x``/``y``/``url`` and
    ``(img, y)`` pairs whose ``y`` bundles label, hint, index and filename).
    Multi-logit outputs are argmaxed, single-logit outputs thresholded at 0.
    The accuracy against the dataset labels is printed.

    Parameters
    ----------
    args : object
        Needs ``dataset``, ``classifier``, ``batch_size`` and ``label_path``;
        optionally ``max_samples`` to stop early and
        ``is_image_to_class``, which prefixes each filename with its class
        directory so the csv matches an ImageFolder layout.

    Returns
    -------
    None
        A csv with the columns ``idx`` and ``prediction`` is written to
        ``args.label_path``; ``idx`` falls back to the running sample number
        when the dataset yields no filenames. Note that a directory named
        ``utils`` is created in the working directory as a side effect.
    """
    torch.set_grad_enabled(False)

    device = torch.device("cuda:0")
    os.makedirs("utils", exist_ok=True)

    dataset = args.dataset

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=0, shuffle=False
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
        elif isinstance(sample, dict):
            img = sample["x"]
            lab = sample["y"]
            img_file = sample.get("url", None)
        else:
            img, y = sample
            if isinstance(y, (tuple, list)):
                if len(y) == 2:
                    (lab, img_file) = y
                elif len(y) == 3:
                    (lab, hint, img_file) = y
                elif len(y) == 4:
                    (lab, hint, idx, img_file) = y
                else:
                    lab = y[0]
                    img_file = y[-1]
            else:
                lab = y
                img_file = None

        if hasattr(args, "is_image_to_class") and args.is_image_to_class:
            img_file = list(img_file)
            for i in range(len(img_file)):
                img_file[i] = os.path.join(str(int(lab[i])), img_file[i])

            img_file = tuple(img_file)

        img = move_to_device(img, device)
        lab = move_to_device(lab, device)

        logits = classifier(img)
        if len(logits.shape) > 1:
            pred = logits.argmax(dim=1)

        else:
            pred = (logits > 0).int()

        try:
            if lab.dim() > 1 and lab.shape[-1] == 1:
                lab_match = lab.squeeze(-1)
            else:
                lab_match = lab
            if pred.shape == lab_match.shape:
                acc += (pred == lab_match).float().sum().item()
        except Exception:
            pass
        n += lab.size(0)

        d["prediction"] += [p.item() for p in pred]
        if img_file is not None:
            if isinstance(img_file, (list, tuple)):
                d["idx"] += list(img_file)
            else:
                d["idx"].append(img_file)
        else:
            d["idx"] += list(range(n - len(pred), n))

    _log.info("%s", acc / n)

    df = pd.DataFrame(data=d)

    df.to_csv(
        args.label_path,
        index=False,
    )

    torch.set_grad_enabled(True)


def high_contrast_heatmap(x, counterfactual):
    """Colour-code the change from an image to its counterfactual.

    The difference is split into an intensity part, shown in red where the
    image got brighter and in blue where it got darker, and a colour part,
    the deviation of the per-channel change from the mean change, shown in
    green and boosted by 1.3 because colour shifts read weaker. The result is
    max-pooled with a 3x3 window so that connected regions stay visible and
    normalised to its maximum. Greyscale inputs only produce the red/blue
    intensity part and are tiled to three channels.

    Parameters
    ----------
    x, counterfactual : torch.Tensor
        A single image each, shape ``(C, H, W)`` with ``C`` 1 or 3.

    Returns
    -------
    heatmap_high_contrast : torch.Tensor
        RGB heatmap of shape ``(3, H, W)`` in ``[0, 1]``.
    x_in : torch.Tensor
        The original, tiled to three channels.
    counterfactual_rgb : torch.Tensor
        The counterfactual, tiled to three channels.
    """
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
        diff = counterfactual - x
        heatmap_red = torch.clamp(diff, min=0)[0]
        heatmap_blue = torch.clamp(-diff, min=0)[0]
        heatmap_green = torch.zeros_like(heatmap_red)
        x_in = torch.tile(x, [3, 1, 1])
        counterfactual_rgb = torch.tile(torch.clone(counterfactual), [3, 1, 1])
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
    """Blend the change heatmap onto a washed-out copy of the image.

    The original is turned greyscale and pulled towards 0.5 so that black and
    white areas read as grey, and the heatmap of
    :func:`high_contrast_heatmap` is blended in with an alpha taken from its
    strongest channel.

    Parameters
    ----------
    x, counterfactual : torch.Tensor
        A single image each, shape ``(C, H, W)``.
    alpha_factor : float, optional
        Multiplier on the alpha mask; larger values make weaker changes
        opaque as well.

    Returns
    -------
    torch.Tensor
        RGB overlay of shape ``(3, H, W)`` in ``[0, 1]``.
    """
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


def ssim_map(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Computes a simplified Structural Similarity Index Measure (SSIM) map between two images locally.
    img1, img2: (C, H, W) tensors.
    """
    import torch.nn.functional as F

    # Add batch dimension and convert to greyscale for structural check
    if img1.shape[0] == 3:
        img1_grey = img1.mean(dim=0, keepdim=True).unsqueeze(0)
        img2_grey = img2.mean(dim=0, keepdim=True).unsqueeze(0)
    else:
        img1_grey = img1.unsqueeze(0)
        img2_grey = img2.unsqueeze(0)

    window = torch.ones((1, 1, window_size, window_size)).to(img1.device) / (
        window_size**2
    )

    mu1 = F.conv2d(img1_grey, window, padding=window_size // 2)
    mu2 = F.conv2d(img2_grey, window, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1_grey * img1_grey, window, padding=window_size // 2) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2_grey * img2_grey, window, padding=window_size // 2) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1_grey * img2_grey, window, padding=window_size // 2) - mu1_mu2
    )

    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    # SSIM map: 1.0 is identical, <1.0 is different.
    # Return 1 - SSIM to highlight differences.
    return torch.clamp(1.0 - ssim, 0.0, 1.0).squeeze(0).squeeze(0)


def generate_ssim_overlay(x, counterfactual, alpha_factor=1.5):
    """Highlight structural, rather than pixelwise, change in red.

    Uses the ``1 - SSIM`` map of :func:`ssim_map`, normalised to its maximum,
    as both the red channel and the alpha of an overlay on a washed-out
    greyscale copy of the image. Unlike :func:`generate_overlay` this ignores
    uniform brightness or colour shifts and reacts to changed structure.

    Parameters
    ----------
    x, counterfactual : torch.Tensor
        A single image each, shape ``(C, H, W)``.
    alpha_factor : float, optional
        Multiplier on the alpha mask.

    Returns
    -------
    torch.Tensor
        RGB overlay of shape ``(3, H, W)``.
    """
    diff = ssim_map(x, counterfactual)
    if diff.max() > 0:
        diff = diff / diff.max()

    if x.shape[0] == 3:
        grey = x.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    else:
        grey = torch.tile(x, [3, 1, 1])

    # Low-contrast greyscale background (slightly lighter for nuance)
    grey = (grey - 0.5) * 0.4 + 0.6

    # nuancing: scale red intensity directly by SSIM change map
    heatmap_red = torch.zeros_like(grey)
    heatmap_red[0] = diff  # Red channel scaled by SSIM diff map

    # alpha mask for blending (less aggressive factor)
    alpha = torch.clamp(diff * alpha_factor, 0, 1)

    overlay = grey * (1 - alpha) + heatmap_red * alpha
    return overlay


@torch.no_grad()
def generate_smooth_mask(x1, x2, dilation, max_avg_combination=0.5):
    """Build the change mask that drives the repaint step.

    The raw mask is the channel-summed absolute difference of the two
    batches. The dilated mask is that mask blurred with a Gaussian kernel of
    width ``dilation`` and sigma 2, clamped to ``[0, 1]`` and divided by the
    per-sample maximum of the raw mask, so that the strongest changed pixel
    reaches 1 and its neighbourhood decays smoothly. Callers threshold the
    dilated mask to decide which region stays untouched.

    Parameters
    ----------
    x1, x2 : torch.Tensor
        Batches of shape ``(B, C, H, W)``, typically the original and the
        edited image in the same normalisation.
    dilation : int
        Odd Gaussian kernel size.
    max_avg_combination : float, optional
        Kept for the commented-out max/average pooling variant; it has no
        effect on the returned masks.

    Returns
    -------
    mask : torch.Tensor
        Raw absolute difference, shape ``(B, 1, H, W)``.
    dil_mask : torch.Tensor
        The blurred and normalised mask, same shape.

    Raises
    ------
    AssertionError
        If ``dilation`` is even.
    """
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

    # dil_mask = F.max_pool2d(mask, dilation, stride=1, padding=(dilation - 1) // 2)
    return mask, dil_mask


def get_intermediate_output(
    model: torch.nn.Module,
    x: torch.Tensor,
    distance_from_last_layer: int = None,
    layer_name: str = None,
):
    """
    Runs the model on input x and returns the activation tensor.
    Can specify distance from end or a specific layer name.
    """
    activations = {}
    handles = []

    if layer_name is not None:
        target_layer = dict(model.named_modules())[layer_name]
    else:
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
        """Forward hook storing the target layer's detached output."""
        activations["out"] = output.detach()

    handle = target_layer.register_forward_hook(hook_fn)
    handles.append(handle)

    # Run forward pass
    try:
        _ = model(x)
    except Exception:
        pass  # Some models might fail on full forward pass if we only need intermediate

    # Cleanup
    for h in handles:
        h.remove()

    return activations["out"]


def generate_feature_similarity_overlay(
    x, counterfactual, model, layer_name="model.layer2", alpha_factor=2.0
):
    """
    Visualizes similarity of intermediate features.
    """
    device = next(model.parameters()).device
    x_batch = x.unsqueeze(0).to(device)
    cf_batch = counterfactual.unsqueeze(0).to(device)

    # Get features
    try:
        feat_x = get_intermediate_output(model, x_batch, layer_name=layer_name)
        feat_cf = get_intermediate_output(model, cf_batch, layer_name=layer_name)
    except Exception:
        # Fallback for different model structures
        if "model." in layer_name:
            layer_name = layer_name.replace("model.", "")
        feat_x = get_intermediate_output(model, x_batch, layer_name=layer_name)
        feat_cf = get_intermediate_output(model, cf_batch, layer_name=layer_name)

    # Cosine similarity across channels
    # feat shape: [1, C, H, W]
    sim = torch.nn.functional.cosine_similarity(feat_x, feat_cf, dim=1)

    # Change map: 1 - similarity
    diff = torch.clamp(1.0 - sim, 0, 2).squeeze(0)
    if diff.max() > 0:
        diff = diff / diff.max()

    # Interpolate to image size
    diff = (
        torch.nn.functional.interpolate(
            diff.unsqueeze(0).unsqueeze(0), size=x.shape[1:], mode="bilinear"
        )
        .squeeze(0)
        .squeeze(0)
        .cpu()
    )

    # Standard overlay style
    if x.shape[0] == 3:
        grey = x.mean(dim=0, keepdim=True).repeat(3, 1, 1).cpu()
    else:
        grey = torch.tile(x, [3, 1, 1]).cpu()

    grey = (grey - 0.5) * 0.4 + 0.6
    heatmap_red = torch.zeros_like(grey)
    heatmap_red[0] = diff

    alpha = torch.clamp(diff * alpha_factor, 0, 1)
    overlay = grey * (1 - alpha) + heatmap_red * alpha
    return overlay


def extract_penultima_activation(x, predictor, distance_from_last_layer=1):
    """Return the activation feeding the predictor's last layer.

    For ``distance_from_last_layer == 1`` the model's own
    ``feature_extractor`` is used when it has one (see
    ``peal.architectures.predictors.TorchvisionModel``); otherwise the module
    tree is unwrapped until it branches and everything but the last child is
    re-assembled into a ``Sequential``. Larger distances are served by
    :func:`get_intermediate_output`, which hooks the corresponding leaf
    module.

    Parameters
    ----------
    x : torch.Tensor
        Input batch in the predictor's normalisation, ``(B, C, H, W)``.
    predictor : torch.nn.Module
        The classifier.
    distance_from_last_layer : int, optional
        How many layers before the output to tap.

    Returns
    -------
    torch.Tensor
        The activations; the trailing spatial dimensions are kept, so a
        ResNet yields ``(B, D, 1, 1)``. The result stays attached to the
        graph, which the orthogonality penalty of the explainer relies on.
    """
    # Function to traverse the computation graph
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
    """Seed every random source PEAL uses and force deterministic cuDNN.

    Seeds ``random``, ``numpy``, torch on CPU and on all GPUs, sets
    ``PYTHONHASHSEED`` and switches cuDNN to deterministic mode with
    benchmarking off, which costs some speed but makes runs comparable.

    Parameters
    ----------
    seed : int
        The seed to use.
    """
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
from torch.utils.data import DataLoader


class DINOEvaluator:
    """Distribution metrics computed in a DINOv2 feature space.

    Instead of the usual Inception features, the CLS embedding of a DINOv2
    model is used, which is cheap to run locally and works on non-natural
    images as well. After :meth:`fit` has stored mean and covariance of a
    reference (real) distribution, the evaluator provides a FID-style score
    for a set of generated images, a per-sample Mahalanobis distance that says
    how far one counterfactual left the data manifold, and a LPIPS-style
    cosine distance between two batches.

    Parameters
    ----------
    model_name : str, optional
        Hugging Face id of the backbone; the bare model without a
        classification head is used.
    device : str, optional
        Falls back to ``"cpu"`` when no GPU is available.

    Attributes
    ----------
    mu_real, sigma_real, sigma_inv_real : torch.Tensor or None
        Statistics of the reference distribution, filled by :meth:`fit`.
    fitted : bool
        Whether those statistics are available.
    """

    def __init__(self, model_name="facebook/dinov2-small", device="cuda"):
        """Load the DINOv2 backbone and its processor in eval mode."""
        self.device = device if torch.cuda.is_available() else "cpu"
        _log.info("%s", f"Loading {model_name} on {self.device}...")

        # We use the base model (no classification head) as we only need embeddings
        # Imported here, not at module level: transformers costs ~0.5 s per
        # process and only this evaluator needs it.
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Storage for distribution statistics (PyTorch Tensors)
        self.mu_real = None
        self.sigma_real = None
        self.sigma_inv_real = None  # For Mahalanobis
        self.fitted = False

    @torch.no_grad()
    def _get_features(self, inputs):
        """
        Internal helper to extract [CLS] token embeddings.
        inputs: can be a DataLoader or a Tensor (B, C, H, W) or List of Tensors/dicts
        """

        def _extract_tensor(item):
            if isinstance(item, dict):
                if "img" in item:
                    return item["img"]
                elif "x" in item:
                    return item["x"]
                else:
                    return next(iter(item.values()))
            elif isinstance(item, (list, tuple)):
                return item[0]
            return item

        # If inputs is a dataloader, we iterate and concatenate
        if isinstance(inputs, DataLoader):
            features_list = []
            for batch in tqdm(inputs, desc="Extracting features"):
                imgs = _extract_tensor(batch)
                imgs = imgs.to(self.device).float()
                if imgs.dim() == 4 and imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                if imgs.min() < 0:
                    imgs = (imgs + 1) / 2

                outputs = self.model(pixel_values=imgs)
                if (
                    hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    feats = outputs.pooler_output
                else:
                    feats = outputs.last_hidden_state[:, 0, :]

                features_list.append(feats)
            return torch.cat(features_list, dim=0)

        # If inputs is a direct tensor batch or list of items
        else:
            if isinstance(inputs, (list, tuple)):
                inputs = [_extract_tensor(item) for item in inputs]
                inputs = torch.stack(inputs, dim=0)
            else:
                inputs = _extract_tensor(inputs)

            inputs = inputs.to(self.device).float()
            if inputs.dim() == 4 and inputs.shape[1] == 1:
                inputs = inputs.repeat(1, 3, 1, 1)
            if inputs.min() < 0:
                inputs = (inputs + 1) / 2

            outputs = self.model(pixel_values=inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                return outputs.last_hidden_state[:, 0, :]

    def _save_state(self, save_path="dino_evaluation_state.pt"):

        state = {
            "mu_real": self.mu_real,
            "sigma_real": self.sigma_real,
            "sigma_inv_real": self.sigma_inv_real,
            "fitted": self.fitted,
            "device": str(self.device),
        }
        torch.save(state, save_path)

        _log.info("%s", f"Evaluator state saved to: {save_path}")

    def _load_state(self, path="dino_evaluation_state.pt"):
        state = torch.load(path, map_location=self.device)
        self.mu_real = state["mu_real"].to(self.device)
        self.sigma_real = state["sigma_real"].to(self.device)
        self.sigma_inv_real = state["sigma_inv_real"].to(self.device)
        self.fitted = state["fitted"]
        _log.info("%s", f"loaded real eval state from {path}")

    def fit(self, real_dataloader, save_path=None):
        """
        Step 1: Precompute statistics for the Real (Training) distribution.
        Run this once at the beginning. The fitted state is written to
        ``save_path`` only when one is given; the old default, a relative
        ``dino_evaluation_state.pt``, landed in the working directory (the
        read-only checkout in the web demo container).
        """
        _log.info("%s", "Fitting evaluator to real data...")
        real_feats = self._get_features(real_dataloader)  # (N, D)

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
            _log.info(
                "%s", "Warning: Covariance matrix singular, using pseudo-inverse."
            )
            self.sigma_inv_real = torch.linalg.pinv(reg_sigma)

        self.fitted = True
        _log.info("%s", f"Fitted on {len(real_feats)} samples. Feature Dim: {D}")
        if save_path is not None:
            self._save_state(save_path=save_path)

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
        sqrt_vals = torch.sqrt(torch.clamp(vals, min=0))  # clamp to avoid nan

        # Reconstruct sqrt matrix
        # This part can be tricky in pure PyTorch without scipy.linalg.sqrtm
        # Approximation: Tr((Sigma_r * Sigma_g)^(1/2)) = sum(sqrt(eigenvalues))
        # strictly true if matrices effectively commute or via trace properties

        trace_sqrt_product = torch.sum(sqrt_vals)

        trace_term = (
            torch.trace(self.sigma_real)
            + torch.trace(sigma_fake)
            - 2 * trace_sqrt_product
        )

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
        feats = self._get_features(single_image_or_batch)  # (B, D)

        # Distance = sqrt( (x - mu) * Sigma^-1 * (x - mu)^T )
        # Centering
        diff = feats - self.mu_real.unsqueeze(0)  # Broadcasting (B, D) - (1, D)

        # Matrix multiplication: (B, D) @ (D, D) -> (B, D)
        left = diff @ self.sigma_inv_real

        # Dot product: sum(left * diff, dim=1)
        # This is effectively doing the row-wise dot product
        mahalanobis_sq = torch.sum(left * diff, dim=1)

        return torch.sqrt(torch.clamp(mahalanobis_sq, min=0))

    @torch.no_grad()
    def compute_lpips(self, x1, x2):
        """
        Calculates LPIPS perceptual distance using DINOv2 feature representations
        between two image batches x1 and x2.
        """
        feats1 = self._get_features(x1)
        feats2 = self._get_features(x2)

        # Normalize along feature dimension
        feats1_norm = torch.nn.functional.normalize(feats1, dim=-1)
        feats2_norm = torch.nn.functional.normalize(feats2, dim=-1)

        # Cosine distance: 1 - cosine_similarity per sample
        cosine_dist = 1.0 - torch.sum(feats1_norm * feats2_norm, dim=-1)
        return torch.mean(cosine_dist).item()
