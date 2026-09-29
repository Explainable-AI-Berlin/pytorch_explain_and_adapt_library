"""Factory that builds a ``SparseDictionary`` from a config, path or instance.

The yaml key ``sparse_dictionaries_type`` names the concrete class; the
factory discovers every ``SparseDictionary`` subclass under
``peal/sparse_dictionaries`` by name and instantiates the matching one.
Weights are loaded from ``config.weights_path`` when that file exists, so a
dictionary fitted by a previous run is reused instead of refitted.
"""

import torch
import os

from typing import Union

from peal.sparse_dictionaries.interfaces import (
    SparseDictionary,
)
from peal.registry import lookup
from peal.global_utils import (
    load_yaml_config,
    get_project_resource_dir,
)


def get_sparse_dictionary(
    sparse_dictionary: Union[SparseDictionary, str, dict],
    device: Union[str, torch.device] = "cuda",
    predictor_datasets=None,
) -> SparseDictionary:
    """Build (or pass through) a sparse dictionary and load its weights.

    Parameters
    ----------
    sparse_dictionary : SparseDictionary or str or dict
        Either an already constructed dictionary, which is returned as is, or a
        yaml path / dict that ``load_yaml_config`` turns into a
        ``SparseDictionaryConfig``. Its ``sparse_dictionaries_type`` must match
        the class name of a ``SparseDictionary`` subclass found under
        ``peal/sparse_dictionaries``.
    device : str or torch.device
        Accepted for interface symmetry with the other factories; not used.
    predictor_datasets : optional
        Accepted for interface symmetry with the other factories; not used.

    Returns
    -------
    SparseDictionary
        The dictionary, with ``load_from_disk(config.weights_path)`` applied when
        ``weights_path`` is set and exists on disk.

    Notes
    -----
    An unknown ``sparse_dictionaries_type`` raises
    ``peal.registry.UnknownComponentError`` naming the known types.
    """
    if not isinstance(sparse_dictionary, SparseDictionary):
        sparse_dictionary_config = load_yaml_config(sparse_dictionary)
        sparse_dictionary_class = lookup(
            "sparse_dictionaries",
            getattr(sparse_dictionary_config, "sparse_dictionaries_type", None),
            base_class=SparseDictionary,
            scan_dir=os.path.join(
                get_project_resource_dir(), "peal", "sparse_dictionaries"
            ),
        )
        sparse_dictionary_out = sparse_dictionary_class(config=sparse_dictionary_config)

    else:
        sparse_dictionary_out = sparse_dictionary

    if not sparse_dictionary_out.config.weights_path is None:
        if os.path.exists(sparse_dictionary_out.config.weights_path):
            sparse_dictionary_out.load_from_disk(
                sparse_dictionary_out.config.weights_path
            )

    return sparse_dictionary_out
