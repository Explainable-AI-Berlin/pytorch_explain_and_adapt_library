import torch
import os

from typing import Union

from peal.generators.interfaces import (
    InvertibleGenerator,
    EditCapableGenerator,
    Generator,
)
from peal.generators.normalizing_flows import Glow
from peal.global_utils import (
    load_yaml_config,
    find_subclasses,
    get_project_resource_dir,
)
from peal.training.trainers import ModelTrainer


def get_generator(
    generator: Union[InvertibleGenerator, str, dict],
    device: Union[str, torch.device] = "cuda",
    classifier_dataset=None,
) -> InvertibleGenerator:
    """
    This function returns a generator.

    Args:
        generator (Union[InvertibleGenerator, str, dict]): The generator to use.
        data_config (Union[str, dict]): The data config.
        classifier_train_dataloader (torch.utils.data.DataLoader): The train dataloader of the classifier.
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
    ):
        generator_config = load_yaml_config(generator)
        generator_class_list = find_subclasses(
            Generator,
            os.path.join(get_project_resource_dir(), "generators", "custom_generators"),
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
                classifier_dataset=classifier_dataset,
            )

        """elif hasattr(generator_config.architecture, "n_flow"):
            # TODO this should be moved into the glow class
            #generator_config.data = data_config
            if os.path.exists(os.path.join(generator_config.base_path, "model.cpl")):
                generator_out = torch.load(os.path.join(generator_config.base_path, "model.cpl"))
                generator_out.config = generator_config

            else:
                generator_out = Glow(generator_config).to(device)
                generator_trainer = ModelTrainer(
                    config=generator_config,
                    model=generator_out,
                    datasource=(
                        train_dataloader.dataset,
                        dataloaders_val[0].dataset,
                    ),
                    base_dir=base_dir,
                    model_name="generator",
                    gigabyte_vram=gigabyte_vram,
                )
                print("Train generator model!")
                generator_trainer.fit()"""

    else:
        generator_out = generator

    generator_out.eval()

    return generator_out
