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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pydantic import BaseModel, PositiveInt
from typing import Union

from peal.data.dataset_factory import get_datasets
from peal.data.datasets import DataConfig
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
from peal.data.dataloaders import create_dataloaders_from_datasource, DataloaderMixer
from peal.generators.interfaces import Generator
from peal.architectures.predictors import (
    SequentialModel,
    ArchitectureConfig,
    TaskConfig,
    TorchvisionModel,
)


class TrainingConfig(BaseModel):
    """
    The maxmimum number of epochs that the model is trained.
    """

    """
    The learning rate the model is trained with.
    """
    max_epochs: PositiveInt = 15
    """
    The learning rate the model is trained with.
    """
    learning_rate: float = 0.0001
    """
    The dropout rate the model is trained with.
    """
    dropout: float = 0.5
    """
    Logs how many steps the model was trained with already.
    """
    global_train_step: int = 0
    """
    Logs how many steps the model was validated for.
    """
    global_validation_step: int = 0
    """
    The current epoch of the model training.
    """
    epoch: int = -1
    """
    The optimizer used for training the model.
    """
    optimizer: str = "Adam"
    """
    The train batch size. Can either be set manually or be left empty and calulated by adaptive batch_size.
    """
    train_batch_size: PositiveInt = 1
    """
    The val batch size. Can either be set manually or be left empty and calulated by adaptive batch_size.
    """
    val_batch_size: PositiveInt = 1
    """
    The test batch size. Can either be set manually or be left empty and calulated by adaptive batch_size.
    """
    test_batch_size: PositiveInt = 1
    """
    The number of iterations per episode when using DataloaderMixer.
    If it is not set, the DataloaderMixer is not used and the value implicitly becomes
    dataset_size / batch_size
    """
    steps_per_epoch: Union[type(None), PositiveInt] = None
    concatenate_batches: bool = False
    adv_training: bool = False
    input_noise_std: float = 0.1
    num_noise_vec: int = 1
    no_grad_attack: bool = False
    attack_epsilon: float = 1.0
    attack_num_steps: int = 5
    train_on_test: bool = False
    class_balanced: bool = False
    use_mixup: bool = False
    mixup_alpha: float = 1.0
    label_smoothing: float = 0.0


class PredictorConfig:
    """
    The config template for a model.
    """

    """
    The config of the training of the model.
    """
    training: TrainingConfig
    """
    The config of the task the model shall solve.
    """
    task: TaskConfig
    """
    The config of the architecture of the model.
    """
    architecture: Union[ArchitectureConfig, types.SimpleNamespace, str, type(None)] = (
        None
    )
    """
    The config of the data used for training the model.
    """
    data: DataConfig = None
    """
    The name of the model.
    """
    model_path: str = "peal_runs/predictor1"
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
    """
    A flag that indicates if the model is loaded from a checkpoint.
    """
    is_loaded: bool = False
    """
    The name of the class.
    """
    model_type: str = "discriminator"
    """
    The name of the class.
    """
    base_path: str = None

    def __init__(
        self,
        training: Union[dict, TrainingConfig],
        task: Union[dict, TaskConfig],
        architecture: Union[dict, ArchitectureConfig] = None,
        data: Union[dict, DataConfig] = None,
        model_path: str = None,
        model_type: str = None,
        **kwargs
    ):
        if isinstance(architecture, dict):
            self.architecture = ArchitectureConfig(**architecture)

        else:
            self.architecture = architecture

        self.training = (
            training
            if isinstance(training, TrainingConfig)
            else TrainingConfig(**training)
        )
        self.task = task if isinstance(task, TaskConfig) else TaskConfig(**task)
        if isinstance(data, DataConfig):
            self.data = data

        elif data is None:
            self.data = None

        else:
            self.data = DataConfig(**data)

        if not model_path is None:
            self.model_path = model_path

        if not model_type is None:
            self.model_type = model_type

        self.kwargs = kwargs


def onehot(label, n_classes):
    one_hots = torch.zeros(label.size(0), n_classes).to(label.device)
    return one_hots.scatter_(1, label.to(torch.int64).view(-1, 1), 1)


def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0))
    data2 = data[indices].to(data)
    targets2 = targets[indices].to(data)

    targets_onehot = onehot(targets.to(data), n_classes)
    targets2_onehot = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)]).to(data)
    data = data * lam + data2 * (1 - lam)
    targets_new = targets_onehot * lam + targets2_onehot * (1 - lam)

    return data, targets_new


def calculate_test_accuracy(
    model,
    test_dataloader,
    device,
    calculate_group_accuracies=False,
    max_test_batches=None,
):
    # determine the test accuracy of the student
    correct = 0
    num_samples = 0
    pbar = tqdm(
        total=int(test_dataloader.dataset.__len__() / test_dataloader.batch_size)
    )
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
            groups[:, 1],
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

            X = move_to_device(X, self.device)
            y_original = y
            if self.config.training.use_mixup and mode=="train":
                X, y = mixup(
                    X,
                    y,
                    self.config.training.mixup_alpha,
                    self.config.task.output_channels,
                )

            if self.config.training.adv_training and mode=="train":
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
            for criterion in self.config.task.criterions.keys():
                criterion_loss = self.config.task.criterions[
                    criterion
                ] * self.criterions[criterion](self.model, pred, y.to(self.device))

                if criterion in ["l1", "l2", "orthogonality"]:
                    criterion_loss *= self.regularization_level

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


def distill_predictor(explainer_config, base_path, predictor, predictor_datasets):
    distillation_datasets = []
    for i in range(2):
        class_predictions_path = os.path.join(
            base_path, "explainer", str(i) + "predictions.csv"
        )
        Path(os.path.join(base_path, "explainer")).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(class_predictions_path):
            predictor_datasets[i].enable_url()
            prediction_args = types.SimpleNamespace(
                batch_size=32,
                dataset=predictor_datasets[i],
                classifier=predictor,
                label_path=class_predictions_path,
                partition="train",
                label_query=0,
                max_samples=explainer_config.max_samples,
            )
            get_predictions(prediction_args)
            predictor_datasets[i].disable_url()

        distilled_dataset_config = copy.deepcopy(predictor_datasets[i].config)
        distilled_dataset_config.split = [1.0, 1.0] if i == 0 else [0.0, 1.0]
        distilled_dataset_config.img_name_idx = 0
        distilled_dataset_config.confounding_factors = None
        distilled_dataset_config.confounder_probability = None
        distilled_dataset_config.dataset_class = None
        distillation_datasets.append(
            get_datasets(
                config=distilled_dataset_config, data_dir=class_predictions_path
            )[i]
        )
        distilled_predictor_config = load_yaml_config(
            explainer_config.distilled_predictor, PredictorConfig
        )
        distilled_predictor_config.data = distilled_dataset_config
        explainer_config.distilled_predictor = distilled_predictor_config
        distillation_datasets[i].task_config = explainer_config.distilled_predictor.task
        distillation_datasets[i].task_config.x_selection = predictor_datasets[
            i
        ].task_config.x_selection

    predictor_distilled = copy.deepcopy(predictor)
    if explainer_config.replace_with_activation == "leakysoftplus":
        predictor_distilled = replace_relu_with_leakysoftplus(predictor_distilled)

    elif explainer_config.replace_with_activation == "leakyrelu":
        predictor_distilled = replace_relu_with_leakyrelu(predictor_distilled)

    distillation_trainer = ModelTrainer(
        config=explainer_config.distilled_predictor,
        model=predictor_distilled,
        datasource=distillation_datasets,
        model_path=os.path.join(base_path, "explainer", "distilled_predictor"),
    )
    distillation_trainer.fit()
    return predictor_distilled
