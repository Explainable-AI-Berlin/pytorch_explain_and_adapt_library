"""Adaptor that repairs a classifier by projecting concept directions out of its head.

The ``ProjectionAdaptor`` is a training-free repair: it loads a trained
predictor, takes the component directions of a sparse dictionary (SAE, MSAE,
SVD, ...) fitted on the penultimate features, and registers a forward pre-hook
on the final ``nn.Linear`` layer that removes the subspace spanned by the
selected components from its input. Accuracy and group statistics on the chosen
data partition are printed before and after the projection so that the effect
of removing a (spurious) concept can be read off directly.
"""

import pathlib
import types
import torch
import os
import numpy as np
import torch.nn as nn

from typing import Union

from peal.adaptors.interfaces import AdaptorConfig, Adaptor
from peal.sparse_dictionaries.interfaces import SparseDictionaryConfig
from peal.sparse_dictionaries.sparse_dictionary_factory import get_sparse_dictionary
from peal.architectures.interfaces import TaskConfig
from peal.data.interfaces import DataConfig
from peal.training.trainers import ModelTrainer
from peal.training.interfaces import TrainingConfig, PredictorConfig
from peal.data.dataloaders import create_dataloaders_from_datasource
from peal.global_utils import load_yaml_config
from peal.training.trainers import calculate_test_accuracy
from peal.log import get_logger

_log = get_logger(__name__)


# dict_keys(['adaptor_type', 'category', 'data', 'test_data', 'model_path', 'base_dir'])
class ProjectionAdaptorConfig(AdaptorConfig):
    """Config of ``ProjectionAdaptor``.

    Parameters
    ----------
    adaptor_type : str
        Fixed to ``"ProjectionAdaptor"``; used by ``get_adaptor`` for dispatch.
    category : str
        Fixed to ``"adaptor"``.
    base_model_config : PredictorConfig or dict or str
        Predictor config (or yaml path) of the classifier to repair; its
        ``model_path`` must contain ``model.cpl``.
    base_dir : str
        Output directory; created on construction.
    data, test_data : DataConfig or dict or None
        ``test_data`` (if given) replaces the predictor's data config so the
        evaluation runs on a different dataset; ``data`` is currently unused.
    sparse_dictionary : SparseDictionaryConfig or dict or None
        Config passed to ``get_sparse_dictionary``; its ``get_components()``
        provides the candidate directions.
    projected_component_index_list : list of int
        Indices of the dictionary components whose span is projected out.
    partition : int
        Which dataloader to evaluate on: 0 train, 1 validation, 2 test.
    """

    adaptor_type: str = "ProjectionAdaptor"
    category: str = "adaptor"
    base_model_config: Union[PredictorConfig, dict, str]
    base_dir: str
    data: Union[DataConfig, dict, type(None)]
    test_data: Union[DataConfig, dict, type(None)]
    sparse_dictionary: Union[SparseDictionaryConfig, dict, type(None)] = None
    projected_component_index_list: list = []
    partition: int = 2


class ProjectionAdaptor(Adaptor):
    """Training-free repair: project dictionary components out of the classifier head.

    Parameters
    ----------
    adaptor_config : ProjectionAdaptorConfig
        See ``ProjectionAdaptorConfig``. ``base_dir`` is created and the sparse
        dictionary is built immediately.

    Attributes
    ----------
    config : ProjectionAdaptorConfig
    sparse_dictionary
        Dictionary object returned by ``get_sparse_dictionary``.
    """

    def __init__(self, adaptor_config: ProjectionAdaptorConfig):
        """Create ``base_dir`` and instantiate the sparse dictionary."""
        pathlib.Path(adaptor_config.base_dir).mkdir(parents=True, exist_ok=True)
        self.config = adaptor_config
        self.sparse_dictionary = get_sparse_dictionary(self.config.sparse_dictionary)

    def run(self):
        """Evaluate the predictor, install the projection hook and evaluate again.

        Loads ``model.cpl`` from ``base_model_config.model_path`` (either a
        pickled ``nn.Module`` or a state dict that is loaded into a freshly built
        ``ModelTrainer`` model), builds the dataloader of ``config.partition``
        and prints overall, per-group, worst-group and average group accuracy
        (the latter always divides by 4). It then wraps the final linear layer
        (``model.fc`` or ``model.model.model.fc``) with
        ``projection_wrap_model`` and prints the same statistics again.

        Returns
        -------
        None
            Results are only printed; the hook stays registered on the
            in-memory model and nothing is written to disk.
        """
        # TODO this can't be done properly before bug is fixed...
        model_config = load_yaml_config(self.config.base_model_config)

        if not isinstance(model_config.training, TrainingConfig):
            model_config.training = TrainingConfig(**model_config.training)

        if not isinstance(model_config.task, TaskConfig):
            model_config.task = TaskConfig(**model_config.task)

        if not self.config.test_data is None:
            model_config.data = load_yaml_config(self.config.test_data)

        if not isinstance(model_config.data, DataConfig):
            if isinstance(model_config.data, types.SimpleNamespace):
                model_config.data = vars(model_config.data)
            model_config.data = DataConfig(**model_config.data)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.join(model_config.model_path, "model.cpl")

        model = torch.load(model_path, map_location=device, weights_only=False)
        if not isinstance(model, torch.nn.Module):
            predictor_config = load_yaml_config(model_config, PredictorConfig)
            model_weights = model
            model = ModelTrainer(predictor_config).model
            model.load_state_dict(model_weights)

        model.eval()
        test_dataloader = create_dataloaders_from_datasource(model_config)[
            self.config.partition
        ]

        _log.info("%s", "before projection:")
        _log.info("%s", "before projection:")
        _log.info("%s", "before projection:")
        correct, group_accuracies, group_distribution, groups, worst_group_accuracy = (
            calculate_test_accuracy(model, test_dataloader, device, True)
        )
        partitions = ["Training", "Validation", "Test"]
        _log.info(
            "%s", partitions[self.config.partition] + " accuracy: " + str(correct)
        )
        _log.info("%s", "Group accuracies: " + str(group_accuracies))
        _log.info("%s", "Group distribution: " + str(group_distribution))
        _log.info("%s", "Samples per Group: " + str(groups))
        _log.info("%s", "Worst group accuracy: " + str(worst_group_accuracy))
        _log.info(
            "%s",
            "Average group accuracy: "
            + str(float(np.sum(np.array(group_accuracies))) / 4),
        )

        components = self.sparse_dictionary.get_components()
        if hasattr(model, "fc"):
            fc = model.fc

        else:
            fc = model.model.model.fc

        model_handle = projection_wrap_model(
            fc, components.t(), self.config.projected_component_index_list
        )

        _log.info("%s", "after projection:")
        _log.info("%s", "after projection:")
        _log.info("%s", "after projection:")
        correct, group_accuracies, group_distribution, groups, worst_group_accuracy = (
            calculate_test_accuracy(model, test_dataloader, device, True)
        )
        partitions = ["Training", "Validation", "Test"]
        _log.info(
            "%s", partitions[self.config.partition] + " accuracy: " + str(correct)
        )
        _log.info("%s", "Group accuracies: " + str(group_accuracies))
        _log.info("%s", "Group distribution: " + str(group_distribution))
        _log.info("%s", "Samples per Group: " + str(groups))
        _log.info("%s", "Worst group accuracy: " + str(worst_group_accuracy))
        _log.info(
            "%s",
            "Average group accuracy: "
            + str(float(np.sum(np.array(group_accuracies))) / 4),
        )


def projection_wrap_model(fc_layer, components, projected_component_index_list):
    """
    Modifies a torch.nn.Linear layer to project out specific components from the input
    before the standard forward pass.

    Args:
        fc_layer (torch.nn.Linear): The fully connected layer to wrap.
        components (torch.Tensor): A tensor of shape (N, D) containing N potential
                                   direction vectors, where D is the input dimension
                                   of fc_layer.
        projected_component_index_list (list): A list of indices indicating which
                                               rows in 'components' to project out.

    Returns:
        torch.utils.hooks.RemovableHandle: The handle for the registered hook.
    """

    # 1. Validation
    if not isinstance(fc_layer, nn.Linear):
        raise ValueError(f"fc_layer must be a torch.nn.Linear, got {type(fc_layer)}")

    if not projected_component_index_list:
        _log.info(
            "%s",
            "Warning: projected_component_index_list is empty. No projection will be applied.",
        )
        return None

    device = fc_layer.weight.device
    dtype = fc_layer.weight.dtype

    # 2. Select the specific components
    # Shape: (k, D) where k is the number of selected indices
    selected_components = components[projected_component_index_list].to(
        device=device, dtype=dtype
    )

    # 3. Compute the Projection Matrix
    # We transpose to (D, k) because we want to find an orthonormal basis for the column space
    # QR decomposition ensures we have an orthogonal basis even if input vectors are not orthogonal.
    # Q will have shape (D, k) with orthonormal columns.
    Q, _ = torch.linalg.qr(selected_components.T)

    # The projection matrix onto the subspace spanned by Q is P = Q @ Q.T
    # Shape: (D, D)
    projection_matrix = Q @ Q.T

    # 4. Define the Pre-Forward Hook
    def projection_hook(module, input_tuple):
        """
        PyTorch forward_pre_hook receives: (module, input_tuple)
        It must return: a tuple of modified inputs or a single modified input.
        Linear layers take a single tensor, but it comes wrapped in a tuple.
        """
        x = input_tuple[0]

        # Apply projection: x_new = x - proj_subspace(x)
        # x shape: (Batch, ..., D)
        # projection_matrix shape: (D, D)
        # We calculate the component of x in the subspace: (x @ P)
        x_projected = x @ projection_matrix

        # Subtract it to "project out"
        x_out = x - x_projected

        return (x_out,)

    # 5. Register the hook
    handle = fc_layer.register_forward_pre_hook(projection_hook)

    return handle
