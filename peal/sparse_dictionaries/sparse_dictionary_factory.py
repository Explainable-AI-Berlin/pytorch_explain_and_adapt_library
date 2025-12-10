import torch
import os

from typing import Union

from peal.sparse_dictionaries.interfaces import (
    SparseDictionary,
    SparseDictionaryConfig,
)
from peal.global_utils import (
    load_yaml_config,
    find_subclasses,
    get_project_resource_dir,
)


def get_sparse_dictionary(
    sparse_dictionary: Union[SparseDictionary, str, dict],
    device: Union[str, torch.device] = "cuda",
    predictor_datasets=None,
) -> SparseDictionary:
    """
    This function returns a SparseDictionary.

    Args:
        SparseDictionary (Union[InvertibleSparseDictionary, str, dict]): The SparseDictionary to use.
        device (Union[str, torch.device]): The device to use.
        predictor_datasets: The datasets to use for the predictor.
    Returns:
        SparseDictionaryInterface: The SparseDictionary.
    """
    if not isinstance(sparse_dictionary, SparseDictionary):
        sparse_dictionary_config = load_yaml_config(sparse_dictionary)
        sparse_dictionary_class_list = find_subclasses(
            SparseDictionary,
            os.path.join(get_project_resource_dir(), "peal", "sparse_dictionaries"),
        )
        sparse_dictionary_class_dict = {
            sparse_dictionary_class.__name__: sparse_dictionary_class
            for sparse_dictionary_class in sparse_dictionary_class_list
        }
        if (
            hasattr(sparse_dictionary_config, "sparse_dictionaries_type")
            and sparse_dictionary_config.sparse_dictionaries_type in sparse_dictionary_class_dict.keys()
        ):
            sparse_dictionary_out = sparse_dictionary_class_dict[sparse_dictionary_config.sparse_dictionaries_type](
                config=sparse_dictionary_config,
            )

    else:
        sparse_dictionary_out = sparse_dictionary

    return sparse_dictionary_out
