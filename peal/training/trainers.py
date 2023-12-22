import torch
import os
import yaml
import torchvision
import shutil
import inspect
import platform
import numpy as np
import sys

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from peal.global_utils import (
    orthogonal_initialization,
    move_to_device,
    load_yaml_config,
    save_yaml_config,
    log_images_to_writer,
)
from peal.training.loggers import Logger
from peal.training.criterions import get_criterions
from peal.data.dataloaders import create_dataloaders_from_datasource
from peal.generators.interfaces import Generator
from peal.architectures.downstream_models import SequentialModel
from peal.generators.variational_autoencoders import VAE
from peal.generators.normalizing_flows import Glow
from peal.configs.architectures.architecture_template import ArchitectureConfig
from peal.configs.generators.generator_template import VAEConfig


def calculate_test_accuracy(model, test_dataloader, device):
    # determine the test accuracy of the student
    correct = 0
    num_samples = 0
    pbar = tqdm(total=test_dataloader.dataset.__len__())
    for it, (X, y) in enumerate(test_dataloader):
        y_pred = model(X.to(device)).argmax(-1).detach().to("cpu")
        correct += float(torch.sum(y_pred == y))
        num_samples += X.shape[0]
        pbar.set_description(
            "test_correct: "
            + str(correct / num_samples)
            + ", it: "
            + str(it * X.shape[0])
        )
        pbar.update(1)

    return correct / test_dataloader.dataset.__len__()


class ModelTrainer:
    """ """

    def __init__(
        self,
        config,
        model_name=None,
        model=None,
        datasource=None,
        optimizer=None,
        base_dir="peal_runs",
        criterions=None,
        logger=None,
        unit_test_train_loop=False,
        unit_test_single_sample=False,
        log_frequency=1000,
        gigabyte_vram=None,
        val_dataloader_weights=[1.0],
    ):
        """ """
        #
        self.config = load_yaml_config(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.val_dataloader_weights = val_dataloader_weights

        #
        if model_name is not None:
            self.model_name = model_name

        else:
            self.model_name = self.config.model_name

        if model is None:
            if not self.config.task.x_selection is None:
                input_channels = len(self.config.task.x_selection)

            else:
                input_channels = self.config.data.input_size[0]

            if not self.config.task.output_channels is None:
                output_channels = self.config.task.output_channels

            else:
                output_channels = self.config.data.output_size[0]

            if isinstance(self.config.architecture, ArchitectureConfig):
                self.model = SequentialModel(
                    self.config.architecture, input_channels, output_channels
                )

            elif isinstance(self.config.architecture, VAEConfig):
                self.model = VAE(self.config.architecture, input_channels)

            elif hasattr(self.config.architecture, "n_flow"):
                self.model = Glow(self.config)


        else:
            self.model = model

        self.model.to(self.device)

        # either the dataloaders have to be given or the path to the dataset
        (
            self.train_dataloader,
            self.val_dataloaders,
            _,
        ) = create_dataloaders_from_datasource(
            config=self.config, datasource=datasource, gigabyte_vram=gigabyte_vram
        )

        if isinstance(self.val_dataloaders, tuple):
            self.val_dataloaders = list(self.val_dataloaders)

        if not isinstance(self.val_dataloaders, list):
            self.val_dataloaders = [self.val_dataloaders]

        #
        if optimizer is None:
            if self.config.training.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.config.training.learning_rate
                )

            elif self.config.training.optimizer == "adam":
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=self.config.training.learning_rate
                )

            else:
                raise Exception("optimizer not available!")

        else:
            self.optimizer = optimizer

        if not model_name is None:
            self.config.model_name = model_name

        self.base_dir = os.path.join(base_dir, self.model_name)
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
                base_dir=self.base_dir,
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

    def run_epoch(self, dataloader, mode="train", pbar=None):
        """ """
        for batch_idx, (X, y) in enumerate(dataloader):
            #
            if self.unit_test_train_loop and batch_idx >= 2:
                break

            if self.unit_test_single_sample and not self.logger is None:
                X = self.logger.test_X
                y = self.logger.test_y

            self.optimizer.zero_grad()
            # Compute prediction and loss
            pred = self.model(move_to_device(X, self.device))
            loss = torch.tensor(0.0).to(self.device)
            loss_logs = {}
            for criterion in self.config.task.criterions.keys():
                criterion_loss = self.config.task.criterions[
                    criterion
                ] * self.criterions[criterion](self.model, pred, y.to(self.device))
                if criterion in ["l1", "l2", "orthogonality"]:
                    criterion_loss *= self.regularization_level

                loss_logs[criterion] = criterion_loss.detach().item()
                loss += criterion_loss

            loss_logs["loss"] = loss.detach().item()

            self.logger.log_step(mode, pred, y, loss_logs)

            # Backpropagation
            loss.backward()

            pbar.set_description(
                "Model Training: "
                + mode
                + "_it: "
                + str(batch_idx)
                + ", loss: "
                + str(loss.detach().item())
            )
            pbar.write(
                ", ".join(
                    [
                        key + ": " + str(pbar.stored_values[key])
                        for key in pbar.stored_values
                    ]
                )
            )
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
        if (
            not continue_training
            and "orthogonality" in self.config.task.criterions.keys()
        ):
            orthogonal_initialization(self.model)

        if not is_initialized:
            shutil.rmtree(self.base_dir, ignore_errors=True)
            Path(os.path.join(self.base_dir, "logs")).mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(os.path.join(self.base_dir, "logs"))
            self.logger.writer = writer
            os.makedirs(os.path.join(self.base_dir, "outputs"))
            os.makedirs(os.path.join(self.base_dir, "checkpoints"))
            open(os.path.join(self.base_dir, "platform.txt"), "w").write(
                platform.node()
            )

            log_images_to_writer(self.train_dataloader, self.logger.writer, "train")
            log_images_to_writer(
                self.val_dataloaders[0], self.logger.writer, "validation"
            )

            self.config.is_loaded = True
            save_yaml_config(self.config, os.path.join(self.base_dir, "config.yaml"))

        else:
            writer = SummaryWriter(os.path.join(self.base_dir, "logs"))
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
                    val_accuracy += self.val_dataloader_weights[idx] * val_accuracy_current

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
                    self.base_dir,
                    "checkpoints",
                    str(self.config.training.epoch) + ".cpl",
                ),
            )

            if val_accuracy > val_accuracy_max:
                torch.save(
                    self.model.to("cpu").state_dict(),
                    os.path.join(self.base_dir, "checkpoints", "final.cpl"),
                )
                torch.save(
                    self.model.to("cpu"), os.path.join(self.base_dir, "model.cpl")
                )
                val_accuracy_max = val_accuracy
                self.model.to(self.device)

            # increase regularization and reset checkpoint if overfitting occurs
            if train_accuracy >= train_accuracy_previous and val_accuracy < val_accuracy_previous:
                if self.regularization_level == 0:
                    self.regularization_level = 1

                else:
                    self.regularization_level *= 1.3

                checkpoint = torch.load(
                    os.path.join(
                        self.base_dir,
                        "checkpoints",
                        str(self.config.training.epoch - 1) + ".cpl",
                    ),
                    map_location=torch.device(self.device),
                )
                self.model.load_state_dict(checkpoint)

            else:
                train_accuracy_previous = train_accuracy
                val_accuracy_previous = val_accuracy

            save_yaml_config(self.config, os.path.join(self.base_dir, "config.yaml"))

            self.config.training.epoch += 1
