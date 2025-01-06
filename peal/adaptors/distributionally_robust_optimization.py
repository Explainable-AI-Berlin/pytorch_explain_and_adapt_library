"""
Notes:

All GroupDRO does is modify the loss function as such: Instead of taking the expected (average) loss
for all samples, partition your samples and take the maximum expected loss of an individual group.

In PEAL loss calculation during training can be found in peal/training/trainers.py in the function
run_epoch in the class ModelTrainer around line 594.

The loss is calculated by using criterions which are found in config.task.criterions.

It is not possible to simply write a criterion that implements DRO for two reasons:
    1. DRO should take an arbitrary loss function as the base loss. DRO simply partitions the batch into groups and
       runs the loss on these groups individually, taking the largest loss. Writing a criterion for this would
       require writing individual DRO loss functions for every potential base loss function. These criterions are found
       in peal.training.criterions.

    2. It is not possible to pass group information into a criterion function. Initially I thought it would be possible
       to include group information in the 'y' variable, as one is able to include this information by calling
       enable_groups on the dataset. However, in the run_epoch function of the ModelTrainer, only the zeroth element
       of 'y' is selected (line 540 in trainers.py) if 'y' is a list or tuple, removing any group information that could
       be included. It is possible to return a dictionary, however this does not allow for the tuple expansion on
       line 536 in trainers.py.

Idea (DOESN'T WORK): Given a training config, and the datasets which are defined within it, for the dataset, select
    return_dict=True and groups_enabled=True. This will hopefully allow for the use of group information in a criterion.

New idea: Use code from the ModelTrainer class (essentially all of it), and modify it to apply DRO


Groups can be enabled in the existing Dataset class (groups_enabled). Adds "has_confounder" entry to the
returned item dictionary. To enable groups in a PEAL dataset, call dataset.enable_groups().


Training should continue as normal, we just need to adjust the loss criterion.


What is needed in the GroupDRO adaptor?

Predictor Config -- Contains the dataset (DataConfig), training procedure (TrainingConfig), and architecture
    (ArchitectureConfig),

"""

import copy
from datetime import datetime

import torch
import os
import types
import shutil
import inspect
import platform
import numpy as np

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
from peal.training.criterions import get_criterions
from peal.training.trainers import PredictorConfig
from peal.data.dataloaders import create_dataloaders_from_datasource, DataloaderMixer
from peal.generators.interfaces import Generator
from peal.architectures.predictors import (
    SequentialModel,
    ArchitectureConfig,
    TaskConfig,
    TorchvisionModel,
)
from peal.adaptors.interfaces import Adaptor, AdaptorConfig


class DROConfig(AdaptorConfig):
    """
    Config template for running the DRO adaptor.
    """

    """
    The config template for an adaptor
    """
    adaptor_type: str = "DRO"
    """
    The config of the predictor to be adapted.
    """
    predictor: PredictorConfig = None
    """
    The base directory where the run of GroupDRO is stored.
    """
    base_dir: str = "peal_runs/dro"
    """
    Number of epochs to train model for.
    """
    max_epochs: PositiveInt = None
    """
    
    """
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
    """
    The name of the class.
    """
    __name__: str = "peal.DROConfig"

    def __init__(
        self,
        predictor: Union[dict, PredictorConfig] = None,
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

        self.kwargs = kwargs



class DRO(Adaptor):
    """
    DRO Adaptor docstring.
    """

    # Instantiate the dataset with return_dict=True

    # Pass an instantiated datasets to the model trainer as a list of datasets
    # [train, val, test] under the datasource keyword

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

        self.adaptor_config = load_yaml_config(adaptor_config, AdaptorConfig)
        config = self.adaptor_config.predictor
        self.config = load_yaml_config(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.val_dataloader_weights = val_dataloader_weights

        #
        if model_path is not None:
            self.model_path = model_path

        else:
            self.model_path = self.config.model_path

        if model is None:
            if (
                not self.config.task.x_selection is None
                and not self.config.data.input_type == "image"
            ):
                input_channels = len(self.config.task.x_selection)

            else:
                input_channels = self.config.data.input_size[0]

            if not self.config.task.output_channels is None:
                output_channels = self.config.task.output_channels

            else:
                output_channels = self.config.data.output_size[0]

            if isinstance(self.config.architecture, ArchitectureConfig):
                self.model = SequentialModel(
                    self.config.architecture,
                    input_channels,
                    output_channels,
                    self.config.training.dropout,
                )

            elif (
                isinstance(self.config.architecture, str)
                and self.config.architecture[:12] == "torchvision_"
            ):
                self.model = TorchvisionModel(
                    self.config.architecture[12:], output_channels
                )

            else:
                raise Exception("Architecture not available!")

        else:
            self.model = model

        self.model.to(self.device)

        # either the dataloaders have to be given or the path to the dataset
        (
            self.train_dataloader,
            self.val_dataloaders,
            test_dataloader,
        ) = create_dataloaders_from_datasource(
            config=self.config, datasource=datasource
        )

        # BEGIN: DRO Alterations
        self.train_dataloader.dataset.enable_groups()
        self.val_dataloaders.dataset.enable_groups()
        # END: DRO Alterations

        if self.config.training.train_on_test:
            self.train_dataloader = test_dataloader

        if isinstance(self.val_dataloaders, tuple):
            self.val_dataloaders = list(self.val_dataloaders)

        if not isinstance(self.val_dataloaders, list):
            self.val_dataloaders = [self.val_dataloaders]

        if self.config.training.class_balanced:
            new_train_dataloaders = []
            new_val_dataloaders = []
            for i in range(self.config.task.output_channels):
                train_dataloader_copy = copy.deepcopy(self.train_dataloader)
                val_dataloader_copy = copy.deepcopy(self.val_dataloaders[0])
                train_dataloader_copy.dataset.enable_class_restriction(i)
                val_dataloader_copy.dataset.enable_class_restriction(i)
                new_train_dataloaders.append(train_dataloader_copy)
                new_val_dataloaders.append(val_dataloader_copy)

            new_config = copy.deepcopy(self.config.training)
            new_config.steps_per_epoch = 200
            new_config.concatenate_batches = True
            self.train_dataloader = DataloaderMixer(
                new_config, new_train_dataloaders[0]
            )
            for i in range(1, len(new_train_dataloaders)):
                # TODO this only works for two classes!
                self.train_dataloader.append(
                    new_train_dataloaders[i], weight_added_dataloader=0.5
                )

            self.val_dataloaders = new_val_dataloaders
            self.val_dataloader_weights = [1.0 / len(self.val_dataloaders)] * len(
                self.val_dataloaders
            )

        #
        if optimizer is None:
            param_list = [param for param in self.model.parameters()]
            if only_last_layer:
                param_list_trained = []
                if len(param_list[-1].shape) == 1:
                    num_unfrozen = 2

                else:
                    num_unfrozen = 1

                assert (
                    len(param_list[-num_unfrozen].shape) == 2
                ), "Wrong layer was chosen!"
                for param in param_list[-num_unfrozen:]:
                    param_list_trained.append(param)

                param_list = param_list_trained

            print("trainable parameters: ", len(param_list))
            if self.config.training.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(
                    param_list,
                    lr=self.config.training.learning_rate,
                    momentum=0.9,
                    weight_decay=0.0001,
                )
                lambda1 = lambda epoch: 0.95**epoch
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=lambda1
                )

            elif self.config.training.optimizer == "adam":
                self.optimizer = torch.optim.Adam(
                    param_list, lr=self.config.training.learning_rate
                )

            elif self.config.training.optimizer[:5] == "adamw":
                if len(self.config.training.optimizer) > 5:
                    weight_decay = float(self.config.training.optimizer[5:])

                else:
                    weight_decay = 0.01

                self.optimizer = torch.optim.AdamW(
                    param_list,
                    lr=self.config.training.learning_rate,
                    weight_decay=weight_decay,
                )

            else:
                raise Exception("optimizer not available!")

        else:
            self.optimizer = optimizer

        if not model_path is None:
            self.config.model_path = model_path

        if criterions is None:
            criterions = get_criterions(config)
            self.criterions = {}
            for criterion_key in self.config.task.criterions:
                if inspect.isclass(criterions[criterion_key]):
                    # and issubclass(criterions[criterion_key], nn.Module):
                    self.criterions[criterion_key] = criterions[criterion_key](
                        self.config, None, self.device
                    )

                else:
                    self.criterions[criterion_key] = criterions[criterion_key]

        else:
            self.criterions = criterions

        if logger is None:
            self.logger = Logger(
                config=self.config,
                model=self.model,
                optimizer=self.optimizer,
                base_dir=self.model_path,
                criterions=self.criterions,
                val_dataloader=self.val_dataloaders[0],
                writer=None,
            )

        else:
            self.logger = logger

        self.unit_test_train_loop = unit_test_train_loop
        self.unit_test_single_sample = unit_test_single_sample
        self.log_frequency = log_frequency
        self.regularization_level = 0

        if self.config.training.adv_training:
            self.attacker = PGD_L2(
                steps=self.config.training.attack_num_steps,
                device=torch.device(self.device),
                max_norm=self.config.training.attack_epsilon,
            )

        # import pdb; pdb.set_trace();


    def run(self, continue_training=False, is_initialized=False):
        """
        Runs GroupDRO method.
        """
        # import pdb; pdb.set_trace()
        self.fit(continue_training=continue_training, is_initialized=is_initialized)


    def run_epoch(self, dataloader, mode="train", pbar=None):
        """ """
        sources = {}
        for batch_idx, sample in enumerate(dataloader):
            if hasattr(dataloader, "return_src") and dataloader.return_src:
                sample, source = sample
                source = str(source)
                while isinstance(sample[0], tuple) or isinstance(sample[0], list):
                    sample, inner_source = sample
                    source = str(source) + str(inner_source)

                if not source in sources.keys():
                    sources[source] = 1

                else:
                    sources[source] += 1

                source_distibution = ""
                for key in sources.keys():
                    source_distibution += (
                        key + ": " + str(sources[key] / (batch_idx + 1)) + ", "
                    )

            else:
                source_distibution = None

            X, y = sample

            ### BEGIN: DRO Alterations
            # TODO this is a dirty fix!!!
            # if isinstance(y, list) or isinstance(y, tuple):
            #     y = y[0]
            y, has_confounder = y
            ### END: DRO Alterations

            #
            if self.unit_test_train_loop and batch_idx >= 2:
                break

            if self.unit_test_single_sample and not self.logger is None:
                X = self.logger.test_X
                y = self.logger.test_y

            X = move_to_device(X, self.device)
            y_original = y
            if self.config.training.label_smoothing > 0.0 and mode == "train":
                y_dist = torch.ones(y.size(0), self.config.task.output_channels)
                y_dist *= self.config.training.label_smoothing / (
                    self.config.task.output_channels - 1
                )
                for i in range(y.size(0)):
                    y_dist[i, y[i]] = 1 - self.config.training.label_smoothing

                y = torch.distributions.categorical.Categorical(y_dist).sample()

            if self.config.training.use_mixup and mode == "train":
                X, y = mixup(
                    X,
                    y,
                    self.config.training.mixup_alpha,
                    self.config.task.output_channels,
                )

            if self.config.training.adv_training and mode == "train":
                noise = (
                    torch.randn_like(X, device=self.device)
                    * self.config.training.input_noise_std
                )

                requires_grad_(self.model, False)
                self.model.eval()
                X = self.attacker.attack(
                    self.model,
                    X,
                    y.to(self.device),
                    noise=noise,
                    num_noise_vectors=self.config.training.num_noise_vec,
                    no_grad=self.config.training.no_grad_attack,
                )
                self.model.train()
                requires_grad_(self.model, True)

            self.optimizer.zero_grad()
            # Compute prediction and loss
            pred = self.model(X)
            loss = torch.tensor(0.0).to(self.device)
            loss_logs = {}

            ### BEGIN: DRO Alterations
            groups = y + dataloader.dataset.output_size * has_confounder
            unique_groups = torch.unique(groups)

            for criterion in self.config.task.criterions.keys():

                if criterion in ["l1", "l2", "orthogonality"]:
                    criterion_loss = self.config.task.criterions[
                        criterion
                    ] * self.criterions[criterion](self.model, pred, y.to(self.device)) * self.regularization_level
                else:
                    group_criterion_loss = torch.zeros(unique_groups.shape)

                    for i, g in enumerate(unique_groups):
                        group_y = y[groups==g]
                        group_pred = pred[groups==g]

                        group_criterion_loss[i] = self.config.task.criterions[
                            criterion
                        ] * self.criterions[criterion](self.model, group_pred, group_y.to(self.device))

                    criterion_loss = torch.max(group_criterion_loss)

                # if criterion in ["l1", "l2", "orthogonality"]:
                #    criterion_loss *= self.regularization_level

                ### END: DRO Alterations

                loss_logs[criterion] = criterion_loss.detach().item()
                loss += criterion_loss

            loss_logs["loss"] = loss.detach().item()

            self.logger.log_step(mode, pred, y_original, loss_logs)

            # Backpropagation
            loss.backward()
            current_state = "Model Training: " + mode + "_it: " + str(batch_idx)
            current_state += ", loss: " + str(loss.detach().item())
            current_state += ", lr: " + str(
                self.scheduler.get_last_lr()
                if hasattr(self, "scheduler")
                else self.optimizer.param_groups[0]["lr"]
            )
            current_state += (
                ", source_distibution: " + source_distibution
                if not source_distibution is None
                else ""
            )
            current_state += ", ".join(
                [
                    key + ": " + str(pbar.stored_values[key])
                    for key in pbar.stored_values
                ]
            )

            pbar.write(current_state)
            pbar.update(1)

            #
            if mode == "train":
                self.optimizer.step()

        #
        accuracy = self.logger.log_epoch(mode, pbar=pbar)

        return loss.detach().item(), accuracy

    def fit(self, continue_training=False, is_initialized=False):
        """ """
        print("Training Config: " + str(self.config))
        if not continue_training:
            if "orthogonality" in self.config.task.criterions.keys():
                print("Orthogonal intialization!!!")
                print("Orthogonal intialization!!!")
                print("Orthogonal intialization!!!")
                orthogonal_initialization(self.model)

            else:
                print("reset weights!!!")
                print("reset weights!!!")
                print("reset weights!!!")
                reset_weights(self.model)

        if not is_initialized:
            if os.path.exists(self.model_path):
                shutil.move(
                    self.model_path,
                    self.model_path
                    + "_old_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S"),
                )

            Path(os.path.join(self.model_path, "logs")).mkdir(
                parents=True, exist_ok=True
            )
            print(os.path.join(self.model_path, "logs"))
            print(os.path.join(self.model_path, "logs"))
            print(os.path.join(self.model_path, "logs"))
            writer = SummaryWriter(os.path.join(self.model_path, "logs"))
            self.logger.writer = writer
            os.makedirs(os.path.join(self.model_path, "outputs"))
            os.makedirs(os.path.join(self.model_path, "checkpoints"))
            open(os.path.join(self.model_path, "platform.txt"), "w").write(
                platform.node()
            )

            log_images_to_writer(self.train_dataloader, self.logger.writer, "train")
            log_images_to_writer(
                self.val_dataloaders[0], self.logger.writer, "validation0_"
            )
            if len(self.val_dataloaders) > 1:
                log_images_to_writer(
                    self.val_dataloaders[1], self.logger.writer, "validation1_"
                )

            self.config.is_loaded = True
            save_yaml_config(self.config, os.path.join(self.model_path, "config.yaml"))

        else:
            writer = SummaryWriter(os.path.join(self.model_path, "logs"))
            self.logger.writer = writer

        pbar = tqdm(
            total=self.config.training.max_epochs
            * (
                len(self.train_dataloader)
                + int(np.sum(list(map(lambda dl: len(dl), self.val_dataloaders))))
            )
        )
        pbar.stored_values = {}
        val_accuracy_max = 0.0
        val_accuracy_previous = 0.0
        train_accuracy_previous = 0.0
        self.model.eval()
        val_accuracy = 0.0
        self.config.training.epoch = -1
        for idx, val_dataloader in enumerate(self.val_dataloaders):
            if len(val_dataloader) >= 1:
                val_loss, val_accuracy_current = self.run_epoch(
                    val_dataloader, mode="validation_" + str(idx), pbar=pbar
                )
                val_accuracy += self.val_dataloader_weights[idx] * val_accuracy_current

        self.logger.writer.add_scalar("epoch_validation_accuracy", val_accuracy, -1)

        self.config.training.epoch = 0
        while self.config.training.epoch < self.config.training.max_epochs:
            pbar.stored_values["Epoch"] = self.config.training.epoch
            self.logger.writer.add_scalar(
                "regularization_level",
                self.regularization_level,
                self.config.training.epoch,
            )
            #
            self.model.train()
            train_loss, train_accuracy = self.run_epoch(
                self.train_dataloader, pbar=pbar
            )
            if isinstance(self.model, Generator):
                train_generator_performance = (
                    self.train_dataloader.dataset.track_generator_performance(
                        self.model, self.train_dataloader.batch_size
                    )
                )
                print(train_generator_performance)
                for key in train_generator_performance.keys():
                    self.logger.writer.add_scalar(
                        "epoch_train_" + key,
                        train_generator_performance[key],
                        self.config.training.epoch,
                    )
            #
            self.model.eval()
            val_accuracy = 0.0
            for idx, val_dataloader in enumerate(self.val_dataloaders):
                if len(val_dataloader) >= 1:
                    val_loss, val_accuracy_current = self.run_epoch(
                        val_dataloader, mode="validation_" + str(idx), pbar=pbar
                    )
                    val_accuracy += (
                        self.val_dataloader_weights[idx] * val_accuracy_current
                    )

            self.logger.writer.add_scalar(
                "epoch_validation_accuracy", val_accuracy, self.config.training.epoch
            )
            if isinstance(self.model, Generator):
                val_generator_performance = self.val_dataloaders[
                    0
                ].dataset.track_generator_performance(
                    self.model, self.val_dataloaders[0].batch_size
                )
                print(val_generator_performance)
                for key in val_generator_performance.keys():
                    self.logger.writer.add_scalar(
                        "epoch_val_" + key,
                        val_generator_performance[key],
                        self.config.training.epoch,
                    )
            #
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.model_path,
                    "checkpoints",
                    str(self.config.training.epoch) + ".cpl",
                ),
            )

            if val_accuracy > val_accuracy_max:
                torch.save(
                    self.model.to("cpu").state_dict(),
                    os.path.join(self.model_path, "checkpoints", "final.cpl"),
                )
                torch.save(
                    self.model.to("cpu"), os.path.join(self.model_path, "model.cpl")
                )
                val_accuracy_max = val_accuracy

                """
                dummy_input = next(iter(self.val_dataloaders[0]))[0]  # Batch size = 1, input size = 10
                # Export to ONNX
                onnx_file_path = os.path.join(self.model_path, "model.onnx")
                torch.onnx.export(
                    self.model,                     # Model to export
                    dummy_input,               # Dummy input
                    onnx_file_path,            # Output file
                    export_params=True,        # Store the trained parameters
                    opset_version=11,          # ONNX version to export (e.g., 11)
                    do_constant_folding=True,  # Optimize constants
                    input_names=['input'],     # Input tensor name(s)
                    output_names=['output'],   # Output tensor name(s)
                    dynamic_axes={             # Support variable-length axes (if applicable)
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                """
                self.model.to(self.device)

            # increase regularization and reset checkpoint if overfitting occurs
            if (
                train_accuracy >= train_accuracy_previous
                and val_accuracy < val_accuracy_previous
            ):
                if self.regularization_level == 0:
                    self.regularization_level = 1

                else:
                    self.regularization_level *= 1.3

                checkpoint = torch.load(
                    os.path.join(
                        self.model_path,
                        "checkpoints",
                        str(self.config.training.epoch - 1) + ".cpl",
                    ),
                    map_location=torch.device(self.device),
                )
                self.model.load_state_dict(checkpoint)

            else:
                train_accuracy_previous = train_accuracy
                val_accuracy_previous = val_accuracy
                if hasattr(self, "scheduler"):
                    self.scheduler.step()

            save_yaml_config(self.config, os.path.join(self.model_path, "config.yaml"))

            self.config.training.epoch += 1