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

# dict_keys(['adaptor_type', 'category', 'data', 'test_data', 'model_path', 'base_dir'])
class ProjectionAdaptorConfig(AdaptorConfig):
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
    def __init__(self, adaptor_config: ProjectionAdaptorConfig):
        pathlib.Path(adaptor_config.base_dir).mkdir(exist_ok=True)
        self.config = adaptor_config
        self.sparse_dictionary = get_sparse_dictionary(self.config.sparse_dictionary)

    def run(self):
        # TODO this can't be done properly before bug is fixed...
        model_config = load_yaml_config(self.config.base_model_config)

        if not isinstance(model_config.training, TrainingConfig):
            model_config.training = TrainingConfig(**model_config.training)
    
        if not isinstance(model_config.task, TaskConfig):
            model_config.task = TaskConfig(**model_config.task)
    
        if not self.config.test_data is None:
            model_config.data = load_yaml_config(self.config.test_data)
    
        if not isinstance(model_config.data, DataConfig):
            if type(model_config.data) == types.SimpleNamespace:
                model_config.data = vars(model_config.data)
            model_config.data = DataConfig(**model_config.data)
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.join(model_config.model_path, "model.cpl")
    
        model = torch.load(model_path, map_location=device)
        if not isinstance(model, torch.nn.Module):
            predictor_config = load_yaml_config(model_config, PredictorConfig)
            model_weights = model
            model = ModelTrainer(predictor_config).model
            model.load_state_dict(model_weights)
    
        model.eval()
        test_dataloader = create_dataloaders_from_datasource(model_config)[self.config.partition]


        print('before projection:')
        print('before projection:')
        print('before projection:')
        correct, group_accuracies, group_distribution, groups, worst_group_accuracy = (
            calculate_test_accuracy(model, test_dataloader, device, True)
        )
        partitions = ["Training", "Validation", "Test"]
        print(partitions[self.config.partition] + " accuracy: " + str(correct))
        print("Group accuracies: " + str(group_accuracies))
        print("Group distribution: " + str(group_distribution))
        print("Samples per Group: " + str(groups))
        print("Worst group accuracy: " + str(worst_group_accuracy))
        print(
            "Average group accuracy: " + str(float(np.sum(np.array(group_accuracies))) / 4)
        )

        components = self.sparse_dictionary.get_components()
        if hasattr(model, "fc"):
            fc = model.fc

        else:
            fc = model.model.model.fc

        model_handle = projection_wrap_model(fc, components.t(), self.config.projected_component_index_list)

        print('after projection:')
        print('after projection:')
        print('after projection:')
        correct, group_accuracies, group_distribution, groups, worst_group_accuracy = (
            calculate_test_accuracy(model, test_dataloader, device, True)
        )
        partitions = ["Training", "Validation", "Test"]
        print(partitions[self.config.partition] + " accuracy: " + str(correct))
        print("Group accuracies: " + str(group_accuracies))
        print("Group distribution: " + str(group_distribution))
        print("Samples per Group: " + str(groups))
        print("Worst group accuracy: " + str(worst_group_accuracy))
        print(
            "Average group accuracy: " + str(float(np.sum(np.array(group_accuracies))) / 4)
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
        print("Warning: projected_component_index_list is empty. No projection will be applied.")
        return None

    device = fc_layer.weight.device
    dtype = fc_layer.weight.dtype

    # 2. Select the specific components
    # Shape: (k, D) where k is the number of selected indices
    selected_components = components[projected_component_index_list].to(device=device, dtype=dtype)

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