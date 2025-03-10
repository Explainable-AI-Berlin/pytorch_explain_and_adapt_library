"""
"""

import copy
from datetime import datetime

import torch
import os
import psutil
import types
import shutil
import inspect
import platform
import numpy as np
import gc

from pathlib import Path

import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pydantic import BaseModel, PositiveInt
from typing import Union

from peal.data.dataset_factory import get_datasets
from peal.data.datasets import DataConfig, Image2MixedDataset, Image2ClassDataset
from peal.dependencies.attacks.attacks import PGD_L2
from peal.global_utils import (
    orthogonal_initialization,
    move_to_device,
    load_yaml_config,
    save_yaml_config,
    reset_weights,
    requires_grad_,
    get_predictions,
    replace_relu_with_leakysoftplus,
    replace_relu_with_leakyrelu,
)
from peal.training.loggers import log_images_to_writer
from peal.training.loggers import Logger
from peal.training.criterions import get_criterions, available_criterions
from peal.training.trainers import PredictorConfig
from peal.data.dataloaders import create_dataloaders_from_datasource, DataloaderMixer
from peal.generators.interfaces import Generator
from peal.architectures.interfaces import ArchitectureConfig, TaskConfig
from peal.architectures.predictors import (
    SequentialModel,
    TorchvisionModel,
)
from peal.adaptors.interfaces import Adaptor, AdaptorConfig

from torch import nn

from peal.dependencies.group_dro.loss import LossComputer

from memory_profiler import profile


dro_criterions = {
    "ce": nn.CrossEntropyLoss(reduction='none'),
    "bce": nn.BCEWithLogitsLoss(reduction='none'),
    "mse": nn.MSELoss(reduction='none'),
    "mae": nn.MSELoss(reduction='none'),
}


class GroupDROConfig(AdaptorConfig):
    """
    Config template for running the DRO adaptor.
    """

    """
    The config template for an adaptor
    """

    adaptor_type: str = "GroupDRO"
    """
    The config of the predictor to be adapted.
    """
    predictor: PredictorConfig = None
    """
    Parameters for the Gorup_DRO loss computer
    """
    is_robust: bool = False
    alpha: float = None
    gamma: float = 0.1
    adj: float = None
    min_var_weight: float = 0 # I'm guessing this is a float
    step_size: float = 0.01
    normalize_loss: bool = False
    btl: bool = False
    """
    Resets weights if true.
    """
    reset_weights: bool = True
    """
    The path where the model is to be stored. Explicitly overwrites model_path in predictor.
    """
    model_path: str = None
    """
    Sets the seed of the run. Expliticly overwrites the seed parameter in predictor.
    """
    seed: int = 0
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
    """
    The name of the class.
    """
    __name__: str = "peal.GroupDROConfig"


    def __init__(
        self,
        predictor: Union[dict, PredictorConfig] = None,
        is_robust: bool = False,
        alpha: float = None,
        gamma: float = 0.1,
        adj: float = None,
        min_var_weight: float = 0,
        step_size: float = 0.01,
        normalize_loss: bool = False,
        btl: bool = False,
        reset_weights: bool = True,
        model_path: str = None,
        seed: int = 0,
        **kwargs,
    ):
        """
        The config template for the DRO adaptor.
        Sets the values of the config that are listed above.

        TODO: Run checks to assure all values are filled, including with defaults, if necessary
        Args:
            so weiter und so fort
        """

        # TODO: We are using pydantic to create the config file. Be sure to check that it's written in this style

        if isinstance(predictor, PredictorConfig):
            self.predictor = predictor
        elif isinstance(predictor, dict):
            self.predictor = PredictorConfig(**predictor)
        else:
            raise TypeErorr(f"predictor is of type {type(predictor)}; expecting type dict or PredictorConfig")

        self.is_robust = is_robust
        self.alpha = alpha
        self.gamma = gamma
        self.adj = adj
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.reset_weights = reset_weights
        self.model_path = model_path
        self.seed = seed
        self.kwargs = kwargs


class GroupDRO(Adaptor):
    """ GroupDRO Adaptor """

    def __init__(
        self,
        adaptor_config: Union[
            dict, str, Path, AdaptorConfig
        ] = "<PEAL_BASE>/configs/adaptors/test_dro.yaml",
        model_path=None,
        model=None,
        datasource=None,
        optimizer=None,
        criterions=None,
        logger=None,
        only_last_layer=False,
        unit_test_train_loop=False,
        unit_test_single_sample=False,
        log_frequency=1000,
        gigabyte_vram=None,
        val_dataloader_weights=[1.0],
    ):



        pass
