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
)
from peal.training.loggers import log_images_to_writer
from peal.training.loggers import Logger
from peal.training.criterions import get_criterions
from peal.data.dataloaders import create_dataloaders_from_datasource
from peal.generators.interfaces import Generator
from peal.architectures.downstream_models import SequentialModel

# from peal.generators.variational_autoencoders import VAE
from peal.generators.normalizing_flows import Glow
from peal.configs.architectures.architecture_template import ArchitectureConfig

# from peal.configs.generators.generator_config import VAEConfig


def calculate_test_accuracy(
    model, test_dataloader, device, calculate_group_accuracies=False, max_test_batches=None
):
    # determine the test accuracy of the student
    correct = 0
    num_samples = 0
    pbar = tqdm(total=int(test_dataloader.dataset.__len__() / test_dataloader.batch_size))
    if calculate_group_accuracies:
        test_dataloader.dataset.enable_groups()
        return_dict_buffer = bool(test_dataloader.dataset.return_dict)
        test_dataloader.dataset.return_dict = True
        groups = np.zeros([2 * test_dataloader.dataset.output_size, 2])

    for it, sample in enumerate(test_dataloader):
        if not max_test_batches is None and it >= max_test_batches:
            break

        if calculate_group_accuracies:
            x = sample["x"]
            y = sample["y"]
            has_confounder = sample["has_confounder"]
            group = y + test_dataloader.dataset.output_size * has_confounder

        else:
            x, y = sample
            if test_dataloader.dataset.idx_enabled:
                y = y[0]

        y_pred = model(x.to(device)).argmax(-1).detach().to("cpu")
        correct += float(torch.sum(y_pred == y))
        num_samples += x.shape[0]
        pbar.set_description(
            "test_correct: "
            + str(correct / num_samples)
            + ", it: "
            + str(it * x.shape[0])
        )
        pbar.update(1)
        if calculate_group_accuracies:
            for idx in range(x.shape[0]):
                groups[int(group[idx])][0] += int(y_pred[idx] == y[idx])
                groups[int(group[idx])][1] += 1

    if calculate_group_accuracies:
        test_dataloader.dataset.return_dict = return_dict_buffer
        test_dataloader.dataset.disable_groups()
        group_accuracies = []
        group_distribution = []
        for idx in range(len(groups)):
            group_accuracies.append(float(groups[idx][0] / groups[idx][1]))
            group_distribution.append(float(groups[idx][1] / num_samples))

        worst_group_accuracy = min(group_accuracies)
        return (
            correct / test_dataloader.dataset.__len__(),
            group_accuracies,
            group_distribution,
            worst_group_accuracy,
        )

    else:
        return correct / test_dataloader.dataset.__len__()


class ModelTrainer:
    """ """

    def __init__(
        self,
        config,
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
        """ """
        #
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
                    self.config.architecture, input_channels, output_channels
                )

            """elif isinstance(self.config.architecture, VAEConfig):
                self.model = VAE(self.config.architecture, input_channels)

            elif hasattr(self.config.architecture, "n_flow"):
                self.model = Glow(self.config)"""

        else:
            self.model = model

        self.model.to(self.device)

        # either the dataloaders have to be given or the path to the dataset
        (
            self.train_dataloader,
            self.val_dataloaders,
            _,
        ) = create_dataloaders_from_datasource(
            config=self.config, datasource=datasource
        )

        if isinstance(self.val_dataloaders, tuple):
            self.val_dataloaders = list(self.val_dataloaders)

        if not isinstance(self.val_dataloaders, list):
            self.val_dataloaders = [self.val_dataloaders]

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

            print('trainable parameters: ', len(param_list))
            if self.config.training.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(
                    param_list, lr=self.config.training.learning_rate
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
                    param_list, lr=self.config.training.learning_rate, weight_decay=weight_decay
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
            # TODO this is a dirty fix!!!
            if isinstance(y, list) or isinstance(y, tuple):
                y = y[0]

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
        if (
            not continue_training
            and "orthogonality" in self.config.task.criterions.keys()
        ):
            orthogonal_initialization(self.model)

        if not is_initialized:
            shutil.rmtree(self.model_path, ignore_errors=True)
            Path(os.path.join(self.model_path, "logs")).mkdir(parents=True, exist_ok=True)
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
                self.val_dataloaders[0], self.logger.writer, "validation"
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
