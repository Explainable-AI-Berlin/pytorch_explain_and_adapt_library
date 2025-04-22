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

#from group_DRO.train import train
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig
from peal.data.datasets import Image2MixedDataset, Image2ClassDataset
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
from peal.dependencies.group_dro.data.data import prepare_data


# from memory_profiler import profile

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


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
    generalization_adjustment: float = 0.0
    minimum_variational_weight: float = 0.0
    robust_step_size: float = 0.01
    use_normalized_loss: bool = False
    btl: bool = False
    """
    Reweights groups during training so the training set is balanced.
    """
    reweight_groups: bool = False
    """
    Use paper's datasets. Only accepts "celebA", "waterbirds" and None, with None indicating to not use the paper's 
    datasets, but rather the one specified in predictor's data config.
    """
    replication_dataset: str = None
    """
    Resets weights of the model if true.
    """
    reset_weights: bool = True
    """
    Use ReduceLROnPlateau scheduler if true.
    """
    scheduler: bool = False
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

        # Unpack Config
        self.adaptor_config = load_yaml_config(adaptor_config, AdaptorConfig)
        self.predictor_config = self.adaptor_config.predictor
        self.architecture_config = self.predictor_config.architecture
        self.training_config = self.predictor_config.training
        self.data_config = self.predictor_config.data
        self.task_config = self.predictor_config.task

        # Set basic parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Instantiate directory to store the model
        if model_path is None:
            self.model_path = self.predictor_config.model_path
        else:
            self.model_path = model_path


        # TODO: Check for unsupported parameters in configs
        # config.training.class_balanced
        # config.training.adv_training

        # Get model path
        if model_path is not None:
            self.model_path = model_path
        elif self.adaptor_config.model_path is not None:
            self.model_path = self.adaptor_config.model_path
        else:
            self.model_path = Path(self.predictor_config.model_path)
            name = "DRO_" + str(self.model_path.name)
            self.model_path = str(self.model_path.with_name(name))

        # TODO: Get logs dir

        # TODO: Instantiate logs dir if it doesn't exist

        # TODO: Set the global seed
        # Seed is set in run_adaptor.py

        ############################################################
        # Load dataset
        ############################################################
        if self.adaptor_config.replication_dataset is None:
            self._load_PEAL_dataset(datasource)
        elif self.adaptor_config.replication_dataset in ["celeba", "waterbirds"]:
            print("Loading replication dataset", self.adaptor_config.replication_dataset)
            self._load_replication_dataset(self.adaptor_config.replication_dataset)
        else:
            raise NotImplementedError(
                f"Dataset {self.adaptor_config.replication_dataset} not implemented. Only celebA and waterbirds are implemented."
            )


        #########################################################
        # Initialize model
        #########################################################

        if model is None:
            if (
                not self.task_config.x_selection is None
                and not self.data_config.input_type == "image"
            ):
                input_channels = len(self.task_config.x_selection)
            elif not(self.adaptor_config.replication_dataset is None):
                input_channels = 3
            else:
                input_channels = self.data_config.input_size[0]

            if not self.task_config.output_channels is None:
                output_channels = self.task_config.output_channels
            elif not(self.adaptor_config.replication_dataset is None):
                output_channels = 2
            else:
                output_channels = self.data_config.output_size[0]

            if (isinstance(self.architecture_config, ArchitectureConfig)
                    and not(self.adaptor_config.replication_dataset is None)):
                self.model = SequentialModel(
                    self.architecture_config,
                    input_channels,
                    output_channels,
                    self.training_config.dropout,
                )
            elif (
                isinstance(self.architecture_config, str)
                and self.architecture_config[:12] == "torchvision_"
            ):
                self.model = TorchvisionModel(
                    self.architecture_config[12:], output_channels
                )

            else:
                raise Exception("Architecture not available!")

        else:
            self.model = model

        if self.adaptor_config.reset_weights:
            reset_weights(self.model)

        self.model.to(self.device)

        # Initialize objective function
        objective_found = False
        for k in self.task_config.criterions.keys():
            if k in dro_criterions.keys() and not objective_found:
                self.objective = dro_criterions[k]
                objective_found = True
            elif k not in ['l2']:
                if objective_found:
                    raise Exception(
                        f"Multiple objectives found. Only one (non l2) objective is allowed. Found: {self.task_config.criterions.keys()}"
                    )
                else:
                    raise Exception(
                        f"Criterion {k} not supported. Supported criteria are: {dro_criterions.keys()}"
                    )


    def run(self, continue_training=False, is_initialized=False):

        # TODO: Create setup for the run. Roughly should be the contents of run_expt.py

        if continue_training:
            raise NotImplementedError(
                "Continue training not implemented. Please implement this."
            )

        # Move previous run directory if it exists
        if not is_initialized and os.path.exists(self.model_path):
                shutil.move(
                    self.model_path,
                    self.model_path + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                )

        # Create directories where model history will be located
        os.makedirs(os.path.join(self.model_path, "checkpoints"))

        # Log config file
        save_yaml_config(self.adaptor_config, os.path.join(self.model_path, "config.yaml"))

        # Set generalization adjustment
        if self.adaptor_config.replication_dataset is None:
            n_groups = 2 ** self.train_dataloader.dataset.output_size
        else:
            n_groups = self.train_dataloader.dataset.n_groups
        adjustments = self.adaptor_config.generalization_adjustment
        if (type(adjustments) in [float, int]):
            adjustments = [adjustments]
        if len(adjustments) == 1:
            adjustments = np.array(adjustments * n_groups)
        elif len(adjustments) == n_groups:
            adjustments = np.array(adjustments)
        else:
            raise ValueError(
                f"generalization_adjustment must be a float, int or list of length 1 or {n_groups}. This is found in DROConfig."
            )

        # Instantiate training loss computer
        train_loss_computer = LossComputer(
            self.objective,
            is_robust=self.adaptor_config.is_robust,
            dataset=self.train_dataloader.dataset,
            alpha=self.adaptor_config.alpha,
            gamma=self.adaptor_config.gamma,
            adj=adjustments,
            normalize_loss=self.adaptor_config.use_normalized_loss,
            btl=self.adaptor_config.btl,
            min_var_weight=self.adaptor_config.minimum_variational_weight,
            replication=(self.adaptor_config.replication_dataset is not None),
        )

        # Instantiate optimizer
        if self.training_config.optimizer != "sgd":
            raise NotImplementedError(
                f"Optimizer {self.training_config.optimizer} not implemented. Only SGD is implemented."
            )

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.training_config.learning_rate,
            momentum=0.9,
            weight_decay=self.task_config.criterions['l2']
        )

        # Instantiate scheduler
        # TODO: Integrate scheduler. Currently there are no references to it in the code.
        if self.adaptor_config.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08)
        else:
            scheduler = None

        # Instantiate progress bar
        pbar = tqdm(
            total = self.training_config.max_epochs
            * (len(self.train_dataloader) + len(self.val_dataloaders)),
            ncols=80
        )
        pbar.stored_values = {}


        val_accuracy_max = 0.0
        # TODO: Set range argument to self.training_config.epochs (or similar)
        self.log_memory_usage(pbar)
        for epoch in range(self.training_config.max_epochs):

            pbar.stored_values["Epoch"] = epoch

            # Train epoch
            self.run_epoch(
                epoch,
                self.model,
                optimizer,
                self.train_dataloader,
                train_loss_computer,
                is_training=True,
                pbar=pbar)

            gc.collect()

            val_loss_computer = LossComputer(
                self.objective,
                is_robust=self.adaptor_config.is_robust,
                dataset=self.val_dataloaders.dataset,
                step_size=self.adaptor_config.robust_step_size,
                alpha=self.adaptor_config.alpha,
                replication=(self.adaptor_config.replication_dataset is not None),
            )

            # Validation epoch
            self.run_epoch(
                epoch,
                self.model,
                optimizer,
                self.val_dataloaders,
                val_loss_computer,
                is_training=False,
                pbar=pbar)

            self.log_memory_usage(pbar)
            gc.collect()

            # TODO: Inspect learning rate (low priority) (I think the GroupDRO code is just logging it)

            # TODO: Save model in checkpoints
            torch.save(
                self.model.state_dict(),
                os.path.join(self.model_path, "checkpoints", f"{epoch}.cpl")
            )

            val_accuracy = val_loss_computer.avg_acc

            if val_accuracy > val_accuracy_max:
                val_accuracy_max = val_accuracy
                # Save the model
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_path, "checkpoints", "final.cpl")
                )

            # TODO: Implement automatic adjustment



    def run_epoch(self, epoch, model, optimizer, loader, loss_computer, is_training, pbar=None):

        if is_training:
            model.train()
        else:
            model.eval()

        self.log_memory_usage(pbar)

        with torch.set_grad_enabled(is_training):
            for batch_idx, batch in enumerate(loader):

                conf = None
                if self.adaptor_config.replication_dataset is None:
                    x, y = batch
                    y, conf = y # Group is included in the label, need to separate
                    group = y * loader.dataset.output_size + conf
                else:
                    x, y, group = batch

                x = x.to(self.device)
                y = y.to(self.device)
                group = group.to(self.device)

                outputs = model(x)

                loss_main = loss_computer.loss(outputs, y , group, is_training)

                if is_training:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

                if batch_idx % 10 == 0:
                    self.log_memory_usage(pbar)

                # Update progress bar
                pbar.update(1)

                del x, y, group, conf, batch

        # Output state of the epoch to the terminal
        current_state = "Model Training: " + ("train" if is_training else "val") + "_it:" + str(batch_idx)
        current_state += ", loss:" + f"{loss_computer.avg_actual_loss.item():.8f}"
        current_state += ", acc:" + f"{loss_computer.avg_acc.item():.8f}"
        current_state += ", "
        current_state += ", ".join(
            [key + ": " + str(pbar.stored_values[key]) for key in pbar.stored_values]
        )
        pbar.write(current_state)
        loss_computer.reset_stats()


    def log_memory_usage(self, pbar=None):
        """
        Logs the memory usage of the model.
        """

        to_out = f"Memory usage: CPU: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB"

        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated() / 1024 ** 2
            reserved_memory = torch.cuda.memory_reserved() / 1024 ** 2
            to_out += f", GPU: {allocated_memory:.2f} MB allocated, {reserved_memory:.2f} MB reserved"

        if pbar is None:
            print(to_out)
        else:
            pbar.write(to_out)


    def _load_PEAL_dataset(self, datasource=None):
        if datasource is None:
            datasource = self.data_config.dataset_path

        if isinstance(datasource, str):
            dataset_train, dataset_val, dataset_test = get_datasets(
                config=self.data_config,
                base_dir=datasource
            )
        elif isinstance(datasource[0], torch.utils.data.Dataset):
            if len(datasource) == 2:
                dataset_train, dataset_val = datasource
                dataset_test = dataset_val
            else:
                dataset_train, dataset_val, dataset_test = datasource

        dataset_train.enable_groups()
        dataset_val.enable_groups()

        dataset_train.task_config = self.task_config
        dataset_val.task_config = self.task_config
        dataset_test.task_config = self.task_config

        shuffle = True
        sampler = None

        if self.adaptor_config.reweight_groups:
            group_array = []
            # TODO: This n_groups calculation seems dubious
            n_groups = 2 ** dataset_train.output_size

            for x, y in dataset_train:
                y, confounder = y
                group_idx = y * dataset_train.output_size + confounder
                group_array.append(group_idx)

            _group_array = torch.LongTensor([g.item() for g in group_array])
            _group_counts = (torch.arange(n_groups).unsqueeze(1) == _group_array).sum(1).float()

            group_weights = len(dataset_train) / _group_counts
            weights = group_weights[_group_array]

            shuffle = False
            sampler = WeightedRandomSampler(weights, len(dataset_train), replacement=True)

        self.train_dataloader = DataLoader(
            dataset_train,
            shuffle=shuffle,
            sampler=sampler,
            batch_size=self.training_config.train_batch_size,
            num_workers=0
        )

        self.val_dataloaders = DataLoader(
            dataset_val,
            shuffle=False,
            batch_size=self.training_config.val_batch_size,
            num_workers=0
        )

        self.test_dataloader = DataLoader(
            dataset_test,
            shuffle=False,
            batch_size=self.training_config.test_batch_size,
            num_workers=0
        )

    def _load_replication_dataset(self, replication_dataset):

        peal_data = os.environ.get("PEAL_DATA", "datasets")

        # TODO: Perhpaps these are better set as function arguments?
        if replication_dataset == "celeba":
            dataset = "CelebA"
            target_name = "Blond_Hair"
            confounder_names = ["Male"]
        elif replication_dataset == "waterbirds":
            dataset = "CUB"
            target_name = "y"
            confounder_names = ["place"]

        if self.predictor_config.architecture.startswith("torchvision_"):
            model = self.predictor_config.architecture[12:]
        else:
            raise ValueError(f"Architecture {self.predictor_config.architecture} not supported. "
                             + "Supported architectures are: torchvision_wideresnet50, torchvision_resnet50 "
                             + "and torchvision_vision34.")

        if model not in ['wideresnet50', 'resnet50', 'resnet34']:
            raise ValueError(f"Architecture {self.predictor_config.architecture} not supported. "
                             + "Supported architectures are: torchvision_wideresnet50, torchvision_resnet50 "
                             + "and torchvision_vision34.")

        train_data, val_data, test_data = prepare_data(
            root_dir=peal_data,
            dataset=dataset,
            target_name=target_name,
            confounder_names=confounder_names,
            model=model,
            augment_data=False,
            fraction=1.0,
            shift_type='confounder',
            train=True
        )

        self.train_dataloader = train_data.get_loader(
            train=True,
            reweight_groups=self.adaptor_config.reweight_groups,
            batch_size=self.training_config.train_batch_size,
            num_workers=0
        )
        self.val_dataloaders = val_data.get_loader(
            train=False,
            reweight_groups=None,
            batch_size=self.training_config.val_batch_size,
            num_workers=0
        )
        self.test_dataloader = test_data.get_loader(
            train=False,
            reweight_groups=None,
            batch_size=self.training_config.test_batch_size,
            num_workers=0
        )