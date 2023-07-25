import torch

from typing import Union

from peal.generators.interfaces import InvertibleGenerator, EditCapableGenerator
from peal.generators.normalizing_flows import Glow
from peal.utils import load_yaml_config
from peal.training.trainers import ModelTrainer


def get_generator(
    generator: Union[InvertibleGenerator, str, dict],
    data_config: Union[str, dict],
    train_dataloader: torch.utils.data.DataLoader,
    dataloaders_val: torch.utils.data.DataLoader,
    base_dir: str,
    gigabyte_vram: float,
    device: Union[str, torch.device],
) -> InvertibleGenerator:
    """
    This function returns a generator.

    Args:
        generator (Union[InvertibleGenerator, str, dict]): The generator to use.
        data_config (Union[str, dict]): The data config.
        train_dataloader (torch.utils.data.DataLoader): The train dataloader.
        dataloaders_val (torch.utils.data.DataLoader): The validation dataloader.
        base_dir (str): The base directory.
        gigabyte_vram (float): The amount of VRAM to use.
        device (Union[str, torch.device]): The device to use.

    Returns:
        InvertibleGenerator: The generator.
    """
    if not (isinstance(generator, InvertibleGenerator) or isinstance(generator, EditCapableGenerator)):
        generator_config = load_yaml_config(generator)
        generator_config.data = data_config
        generator = Glow(generator_config).to(device)
        generator_trainer = ModelTrainer(
            config=generator_config,
            model=generator,
            datasource=(
                train_dataloader.dataset,
                dataloaders_val[0].dataset,
            ),
            base_dir=base_dir,
            model_name="generator",
            gigabyte_vram=gigabyte_vram,
        )
        print("Train generator model!")
        generator_trainer.fit()

    generator.eval()

    return generator
