import torch
import os

from typing import Union

from peal.generators.interfaces import (
    InvertibleGenerator,
    EditCapableGenerator,
    Generator,
)
from peal.global_utils import (
    load_yaml_config,
    find_subclasses,
    get_project_resource_dir,
)
from peal.training.trainers import ModelTrainer


def get_generator(
    generator: Union[InvertibleGenerator, str, dict],
    device: Union[str, torch.device] = "cuda",
    predictor_dataset=None,
) -> InvertibleGenerator:
    """
    This function returns a generator.

    Args:
        generator (Union[InvertibleGenerator, str, dict]): The generator to use.
        data_config (Union[str, dict]): The data config.
        predictor_train_dataloader (torch.utils.data.DataLoader): The train dataloader of the predictor.
        dataloaders_val (torch.utils.data.DataLoader): The validation dataloader.
        base_dir (str): The base directory.
        gigabyte_vram (float): The amount of VRAM to use.
        device (Union[str, torch.device]): The device to use.

    Returns:
        InvertibleGenerator: The generator.
    """
    if isinstance(generator, str) and generator[-4:] == ".cpl":
        generator_out = torch.load(generator, map_location=device)

    elif not (
        isinstance(generator, InvertibleGenerator)
        or isinstance(generator, EditCapableGenerator)
        or generator is None
    ):
        generator_config = load_yaml_config(generator)
        generator_class_list = find_subclasses(
            Generator,
            os.path.join(get_project_resource_dir(), "peal", "generators"),
        )
        generator_class_dict = {
            generator_class.__name__: generator_class
            for generator_class in generator_class_list
        }
        if (
            hasattr(generator_config, "generator_type")
            and generator_config.generator_type in generator_class_dict.keys()
        ):
            generator_out = generator_class_dict[generator_config.generator_type](
                config=generator_config,
                device=device,
                predictor_dataset=predictor_dataset,
            )

    else:
        generator_out = generator

    if not generator_out is None:
        generator_out.eval().to(device)

    return generator_out
