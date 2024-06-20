import torch
import os

from typing import Union

from peal.explainers.interfaces import (
    ExplainerInterface,
)
from peal.global_utils import (
    load_yaml_config,
    find_subclasses,
    get_project_resource_dir,
)
from peal.training.trainers import ModelTrainer


def get_explainer(
    explainer: Union[ExplainerInterface, str, dict],
    device: Union[str, torch.device] = "cuda",
    classifier_dataset=None,
) -> ExplainerInterface:
    """
    This function returns a explainer.

    Args:
        explainer (Union[Invertibleexplainer, str, dict]): The explainer to use.
        data_config (Union[str, dict]): The data config.
        classifier_train_dataloader (torch.utils.data.DataLoader): The train dataloader of the classifier.
        dataloaders_val (torch.utils.data.DataLoader): The validation dataloader.
        base_dir (str): The base directory.
        gigabyte_vram (float): The amount of VRAM to use.
        device (Union[str, torch.device]): The device to use.

    Returns:
        Invertibleexplainer: The explainer.
    """
    if not isinstance(explainer, ExplainerInterface):
        explainer_config = load_yaml_config(explainer)
        explainer_class_list = find_subclasses(
            explainer,
            os.path.join(get_project_resource_dir(), "peal", "explainers"),
        )
        explainer_class_dict = {
            explainer_class.__name__: explainer_class
            for explainer_class in explainer_class_list
        }
        if (
            hasattr(explainer_config, "explainer_type")
            and explainer_config.explainer_type in explainer_class_dict.keys()
        ):
            explainer_out = explainer_class_dict[explainer_config.explainer_type](
                config=explainer_config,
                device=device,
                classifier_dataset=classifier_dataset,
            )

    else:
        explainer_out = explainer

    return explainer_out
