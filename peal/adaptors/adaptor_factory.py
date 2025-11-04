import torch
import os

from typing import Union

from peal.adaptors.interfaces import Adaptor
from peal.global_utils import (
    load_yaml_config,
    find_subclasses,
    get_project_resource_dir,
)


def get_adaptor(
    adaptor: Union[Adaptor, str, dict],
) -> Adaptor:
    """
    This function returns a adaptor.

    Args:
        adaptor (Union[Invertibleadaptor, str, dict]): The adaptor to use.

    Returns:
        Invertibleadaptor: The adaptor.
    """
    if isinstance(adaptor, Adaptor):
        adaptor_out = adaptor

    else:
        adaptor_config = load_yaml_config(adaptor)
        adaptor_class_list = find_subclasses(
            Adaptor,
            os.path.join(get_project_resource_dir(), "peal", "adaptors"),
        )
        adaptor_class_dict = {
            adaptor_class.__name__: adaptor_class
            for adaptor_class in adaptor_class_list
        }
        if (
            hasattr(adaptor_config, "adaptor_type")
            and adaptor_config.adaptor_type in adaptor_class_dict.keys()
        ):
            adaptor_out = adaptor_class_dict[adaptor_config.adaptor_type](
                adaptor_config=adaptor_config,
            )

    return adaptor_out
