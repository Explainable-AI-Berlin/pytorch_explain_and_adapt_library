import torch
import os

from typing import Union

from peal.explainers.counterfactual_explainer import CounterfactualExplainer
from peal.explainers.interfaces import (
    ExplainerInterface,
)
from peal.global_utils import (
    load_yaml_config,
    find_subclasses,
    get_project_resource_dir,
)


def get_explainer(
    explainer: Union[ExplainerInterface, str, dict],
    device: Union[str, torch.device] = "cuda",
    predictor_datasets=None,
    **kwargs

) -> ExplainerInterface:
    """
    This function returns a explainer.

    Args:
        explainer (Union[Invertibleexplainer, str, dict]): The explainer to use.
        device (Union[str, torch.device]): The device to use.
        predictor_datasets: The datasets to use for the predictor.
    Returns:
        ExplainerInterface: The explainer.
    """
    if not isinstance(explainer, ExplainerInterface):
        explainer_config = load_yaml_config(explainer)
        if explainer_config.explainer_type in ["SCE", "ACE", "TIME", "DAEdistill"]:
            explainer_out = CounterfactualExplainer(
                explainer_config=explainer_config,
                datasets=predictor_datasets,
                **kwargs
            )

        else:
            explainer_class_list = find_subclasses(
                ExplainerInterface,
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
                    predictor_dataset=predictor_datasets,
                    **kwargs
                )

    else:
        explainer_out = explainer

    return explainer_out
