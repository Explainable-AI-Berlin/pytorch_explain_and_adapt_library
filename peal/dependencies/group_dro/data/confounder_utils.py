"""
Adapted from the code for "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization
for Worst-Case Generalization"

doi: https://doi.org/10.48550/arXiv.1911.08731
Repository: https://github.com/kohpangwei/group_DRO/tree/master
"""

import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from peal.dependencies.group_dro.models import model_attributes
from torch.utils.data import Dataset, Subset
from peal.dependencies.group_dro.data.celebA_dataset import CelebADataset
from peal.dependencies.group_dro.data.cub_dataset import CUBDataset
from peal.dependencies.group_dro.data.dro_dataset import DRODataset
# from data.multinli_dataset import MultiNLIDataset

################
### SETTINGS ###
################

confounder_settings = {
    'CelebA':{
        'constructor': CelebADataset
    },
    'CUB':{
        'constructor': CUBDataset
    },
#    'MultiNLI':{
#        'constructor': MultiNLIDataset
#    }
}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(
        root_dir,
        dataset,
        target_name,
        confounder_names,
        model,
        augment_data,
        fraction,
        train=False,
        return_full_dataset=False
):
    full_dataset = confounder_settings[dataset]['constructor'](
        root_dir=root_dir,
        target_name=target_name,
        confounder_names=confounder_names,
        model_type=model,
        augment_data=augment_data)
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(splits, train_frac=fraction)
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets