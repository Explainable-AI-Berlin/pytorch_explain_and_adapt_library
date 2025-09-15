"""
Adapted from the code for "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization
for Worst-Case Generalization"

doi: https://doi.org/10.48550/arXiv.1911.08731
Repository: https://github.com/kohpangwei/group_DRO/tree/master
"""

import os
import torch
import numpy as np
from torch.utils.data import Subset

# from data.label_shift_utils import prepare_label_shift_data
from peal.dependencies.group_dro.data.confounder_utils import prepare_confounder_data

# TODO: How do I set the root directory for the datasets?
root_dir = "./datasets"

dataset_attributes = {
    "CelebA": {"root_dir": "celebA"},
    "CUB": {"root_dir": "cub"},
    #    'CIFAR10': {
    #        'root_dir': 'CIFAR10/data'
    #    },
    #    'MultiNLI': {
    #        'root_dir': 'multinli'
    #    }
}

for dataset in dataset_attributes:
    dataset_attributes[dataset]["root_dir"] = os.path.join(
        root_dir, dataset_attributes[dataset]["root_dir"]
    )

shift_types = ["confounder"]  # , 'label_shift_step']


def prepare_data(
    root_dir,
    dataset,
    target_name,
    confounder_names,
    model,
    augment_data,
    fraction,
    shift_type="confounder",
    train=False,
    return_full_dataset=False,
):
    # Set root_dir to defaults if necessary
    if root_dir is None:
        root_dir = dataset_attributes[dataset]["root_dir"]
    if shift_type == "confounder":
        return prepare_confounder_data(
            root_dir,
            dataset,
            target_name,
            confounder_names,
            model,
            augment_data,
            fraction,
            train,
            return_full_dataset,
        )


#    elif args.shift_type.startswith('label_shift'):
#        assert not return_full_dataset
#        return prepare_label_shift_data(args, train)

# def log_data(data, logger):
#     logger.write('Training Data...\n')
#     for group_idx in range(data['train_data'].n_groups):
#         logger.write(f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
#     logger.write('Validation Data...\n')
#     for group_idx in range(data['val_data'].n_groups):
#         logger.write(f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
#     if data['test_data'] is not None:
#         logger.write('Test Data...\n')
#         for group_idx in range(data['test_data'].n_groups):
#             logger.write(f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')
