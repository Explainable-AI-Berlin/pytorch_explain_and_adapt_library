"""Supervised training loop for PEAL predictors and predictor distillation.

:class:`ModelTrainer` trains a classifier (or a generator exposing
``track_generator_performance``) from a ``PredictorConfig``: it builds the
model, dataloaders, optimizer, criterions and TensorBoard logger, runs epochs
with optional mixup, label smoothing and PGD adversarial training, applies an
early-stopping/regularisation scheme and writes ``model.cpl``,
``checkpoints/*.cpl`` and ``config.yaml`` into ``model_path``. The
``distill_*`` helpers relabel datasets with a teacher predictor's outputs and
train a student on them, which is how CFKD/DiDAE turn a repaired predictor
into a stand-alone model. :func:`calculate_test_accuracy` is the shared
(worst-)group accuracy evaluator.
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
from tqdm import tqdm

from peal.data.dataset_factory import get_datasets
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
    cprint,
    onehot,
)
from peal.training.interfaces import PredictorConfig
from peal.training.loggers import log_images_to_writer
from peal.training.loggers import Logger
from peal.training.criterions import get_criterions
from peal.data.dataloaders import (
    create_dataloaders_from_datasource,
    DataloaderMixer,
    WeightedDataloaderList,
)
from peal.generators.interfaces import Generator
from peal.architectures.predictors import (
    SequentialModel,
    TorchvisionModel,
)
from peal.architectures.interfaces import ArchitectureConfig
from peal.log import get_logger

_log = get_logger(__name__)


def mixup(data, targets, alpha, n_classes):
    """Mix each sample with a random partner (Zhang et al. mixup).

    Parameters
    ----------
    data : torch.Tensor
        Input batch of shape ``(B, ...)``.
    targets : torch.Tensor
        Integer class labels of shape ``(B,)``.
    alpha : float
        Beta distribution parameter; one ``lam ~ Beta(alpha, alpha)`` is drawn
        for the whole batch.
    n_classes : int
        Width of the one-hot targets.

    Returns
    -------
    tuple of torch.Tensor
        ``(mixed_data, mixed_onehot_targets)`` with targets of shape
        ``(B, n_classes)``.
    """
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
    tracking_level=2,
):
    """Compute the accuracy of ``model`` on a dataloader, optionally per group.

    Groups are ``y + output_size * has_confounder``, i.e. label x single
    binary confounder; datasets without ``has_confounder`` fall back to
    per-class groups and empty groups are skipped. The dataset is switched to
    dict output with groups enabled for the duration of the call and restored
    afterwards.

    Parameters
    ----------
    model : torch.nn.Module
        Classifier returning logits.
    test_dataloader : torch.utils.data.DataLoader
        Loader over a PEAL dataset (needs ``enable_groups`` etc. for groups).
    device : str or torch.device
        Device the inputs are moved to.
    calculate_group_accuracies : bool, optional
        Also compute per-group accuracies. Default ``False``.
    max_test_batches : int, optional
        Stop after this many batches (accuracy is still divided by the full
        dataset length).
    tracking_level : int, optional
        Progress bar is updated when ``>= 1``.

    Returns
    -------
    float or tuple
        Accuracy alone, or ``(accuracy, group_accuracies, group_distribution,
        group_counts, worst_group_accuracy)`` when
        ``calculate_group_accuracies`` is set.

    Raises
    ------
    TypeError
        When the data config names more than one confounding factor, so
        ``has_confounder`` is a list rather than a tensor.
    """
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
    with torch.no_grad():
        for it, sample in enumerate(test_dataloader):
            if not max_test_batches is None and it >= max_test_batches:
                break

            if calculate_group_accuracies:

                x = sample["x"]
                y = sample["y"]
                # datasets without a labelled confounder (e.g. ImageNet class pairs)
                # carry no has_confounder: fall back to per-class groups
                has_confounder = (
                    sample["has_confounder"]
                    if "has_confounder" in sample
                    else torch.zeros_like(y)
                )
                # Group accuracies are defined for a label plus ONE confounder, giving
                # 2 * output_size groups. When a data config names more than two
                # confounding factors the dataset returns has_confounder as a list of
                # per-factor targets, and the arithmetic below silently became list
                # repetition followed by "Tensor + list". Say what is actually wrong.
                if not isinstance(has_confounder, torch.Tensor):
                    factors = getattr(
                        getattr(test_dataloader.dataset, "config", None),
                        "confounding_factors",
                        None,
                    )
                    raise TypeError(
                        "group accuracies need exactly one confounder, but this data "
                        "config names "
                        + (f"{len(factors)}: {factors}" if factors else "several")
                        + ". Evaluate against a data config whose confounding_factors "
                        "are the label and the single confounder the model was trained "
                        "against."
                    )
                group = y + test_dataloader.dataset.output_size * has_confounder

            else:
                x, y = sample
                if test_dataloader.dataset.idx_enabled:
                    y = y[0]

            y_pred = model(x.to(device)).argmax(-1).detach().to("cpu")
            correct += float(torch.sum(y == y_pred))
            num_samples += x.shape[0]

            if calculate_group_accuracies:
                for idx in range(x.shape[0]):
                    groups[int(group[idx])][0] += int(y_pred[idx] == y[idx])
                    groups[int(group[idx])][1] += 1

            if tracking_level >= 1:
                pbar.set_description(
                    "test_correct: "
                    + str(round(correct / num_samples, 4))
                    + ", it: "
                    + str((it + 1) * x.shape[0])
                )
                pbar.update(1)

    if calculate_group_accuracies:
        test_dataloader.dataset.return_dict = return_dict_buffer
        test_dataloader.dataset.disable_groups()
        group_accuracies = []
        group_distribution = []
        for idx in range(len(groups)):
            if groups[idx][1] == 0:  # empty group (no confounder labels)
                continue
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


def get_predictor(config, model=None):
    """Instantiate the predictor described by a ``PredictorConfig``.

    Input channels come from ``task.x_selection`` (non-image data) or
    ``data.input_size``; output channels from ``task.output_channels`` or
    ``data.output_size``. ``config.architecture`` may be an
    ``ArchitectureConfig`` (-> ``SequentialModel``), a ``torchvision_<name>``
    string (-> ``TorchvisionModel``) or a ``.cpl`` path to a pickled module.

    Parameters
    ----------
    config : PredictorConfig
        Config with ``task``, ``data``, ``architecture`` and ``training``.
    model : torch.nn.Module, optional
        When given it is returned unchanged.

    Returns
    -------
    torch.nn.Module
        The predictor.

    Raises
    ------
    Exception
        When ``config.architecture`` matches none of the supported forms.
    """
    if model is None:
        if (
            not config.task.x_selection is None
            and not config.data.input_type == "image"
        ):
            input_channels = len(config.task.x_selection)

        else:
            input_channels = config.data.input_size[0]

        if not config.task.output_channels is None:
            output_channels = config.task.output_channels

        else:
            output_channels = config.data.output_size[0]

        if isinstance(config.architecture, ArchitectureConfig):
            model = SequentialModel(
                config.architecture,
                input_channels,
                output_channels,
                config.training.dropout,
            )

        elif (
            isinstance(config.architecture, str)
            and config.architecture[:12] == "torchvision_"
        ):
            model = TorchvisionModel(
                config.architecture[12:],
                output_channels,
                config.data.input_size[-1],
                config=config,
            )

        elif (
            isinstance(config.architecture, str) and config.architecture[-4:] == ".cpl"
        ):
            # torch >= 2.6 defaults to weights_only=True, which cannot unpickle a whole nn.Module
            model = torch.load(config.architecture, weights_only=False)

        else:
            raise Exception("Architecture not available!")

    return model


class ModelTrainer:
    """Train a predictor from a ``PredictorConfig`` with early stopping.

    On construction the model, dataloaders (via
    ``create_dataloaders_from_datasource``), optimizer (``sgd`` with a 0.95
    exponential schedule, ``adam`` or ``adamw``), criterions and ``Logger``
    are built from the config. ``training.class_balanced`` wraps the train
    loader in a ``DataloaderMixer`` and splits every validation loader into
    one class-restricted copy per class. Training happens in :meth:`fit`.

    Parameters
    ----------
    config : str, dict or PredictorConfig
        Loaded with ``load_yaml_config``; consumes ``model_path``,
        ``only_last_layer``, ``tracking_level`` and the ``training``, ``task``,
        ``data`` and ``architecture`` sections.
    model_path : str, optional
        Output directory; overrides ``config.model_path``.
    model : torch.nn.Module, optional
        Pre-built model; otherwise :func:`get_predictor` is used.
    datasource : optional
        Datasets or dataloaders handed to ``create_dataloaders_from_datasource``.
    optimizer : torch.optim.Optimizer, optional
        Overrides the optimizer built from ``training.optimizer``.
    criterions : dict, optional
        Overrides the criterions built from ``task.criterions``.
    logger : Logger, optional
        Overrides the default ``Logger``.
    only_last_layer : bool, optional
        Train only the final linear layer (``model.fc`` or the last weight and
        bias). Defaults to ``config.only_last_layer``.
    unit_test_train_loop : bool, optional
        Stop every epoch after two batches.
    unit_test_single_sample : bool, optional
        Replace every batch by the logger's fixed test sample.
    log_frequency : int, optional
        Stored on the instance; not used by the loop itself.
    val_dataloader_weights : list of float, optional
        Weights for averaging validation accuracies across loaders.

    Attributes
    ----------
    regularization_level : float
        Multiplier for the ``l1``/``l2``/``orthogonality`` criterions; raised
        by :meth:`fit` when overfitting is detected.
    attacker : PGD_L2
        Only present when ``training.adv_training`` is set.
    """

    def __init__(
        self,
        config,
        model_path=None,
        model=None,
        datasource=None,
        optimizer=None,
        criterions=None,
        logger=None,
        only_last_layer=None,
        unit_test_train_loop=False,
        unit_test_single_sample=False,
        log_frequency=1000,
        val_dataloader_weights=[1.0],
    ):
        """Build model, dataloaders, optimizer, criterions and logger.

        See the class docstring for the parameters.
        """
        #
        self.config = load_yaml_config(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.val_dataloader_weights = val_dataloader_weights
        if only_last_layer is None:
            only_last_layer = self.config.only_last_layer

        #
        if model_path is not None:
            self.model_path = model_path

        else:
            self.model_path = self.config.model_path

        self.model = get_predictor(self.config, model)
        self.model.config = self.config

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

        if isinstance(self.val_dataloaders, WeightedDataloaderList):
            self.val_dataloader_weights = list(self.val_dataloaders.weights)
            self.val_dataloaders = self.val_dataloaders.dataloaders

        elif isinstance(self.val_dataloaders, tuple):
            self.val_dataloaders = list(self.val_dataloaders)

        if not isinstance(self.val_dataloaders, list):
            self.val_dataloaders = [self.val_dataloaders]
        if self.config.training.class_balanced:
            if not isinstance(self.train_dataloader, DataloaderMixer):
                new_config = copy.deepcopy(self.config.training)
                new_config.steps_per_epoch = 200
                new_config.concatenate_batches = True
                self.train_dataloader = DataloaderMixer(
                    new_config, self.train_dataloader
                )

            self.train_dataloader.enable_class_balancing()
            new_val_dataloaders = []
            new_val_dataloader_weights = []
            for j in range(len(self.val_dataloaders)):
                for i in range(self.config.task.output_channels):
                    val_dataloader_copy = copy.deepcopy(self.val_dataloaders[j])
                    val_dataloader_copy.dataset.enable_class_restriction(i)
                    new_val_dataloaders.append(val_dataloader_copy)
                    new_val_dataloader_weights.append(
                        self.val_dataloader_weights[j]
                        / self.config.task.output_channels
                    )

            self.val_dataloaders = new_val_dataloaders
            self.val_dataloader_weights = new_val_dataloader_weights

        #
        if optimizer is None:
            param_list = [param for param in self.model.parameters()]
            if only_last_layer:
                if hasattr(self.model, "fc"):
                    param_list = [self.model.fc.weight]

                else:
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

            cprint(
                "trainable parameters: " + str(len(param_list)),
                self.config.tracking_level,
                4,
            )
            if self.config.training.optimizer == "sgd":
                _log.info("%s", "Using SGD optimizer")
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
                _log.info("%s", "Using Adam optimizer")
                self.optimizer = torch.optim.Adam(
                    param_list, lr=self.config.training.learning_rate
                )

            elif self.config.training.optimizer[:5] == "adamw":
                _log.info("%s", "Using AdamW optimizer")
                if hasattr(self.config.training, "weight_decay"):
                    if self.config.training.weight_decay is not None:
                        weight_decay = self.config.training.weight_decay

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

    def run_epoch(self, dataloader, mode="train", pbar=None, debug=False):
        """Run one pass over ``dataloader`` in train or validation mode.

        Per batch: optional diffusion augmentation, label smoothing, mixup and
        PGD adversarial attack (train mode only), forward pass (with latents
        when the ``lc`` criterion is configured), the weighted sum of
        ``task.criterions`` (regularisers scaled by ``regularization_level``),
        ``loss.backward()`` and, in train mode, an optimizer step. Every step
        is reported to ``self.logger`` and the progress bar; the loop stops
        after ``training.steps_per_epoch`` batches when that is set.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader or DataloaderMixer
            Batches of ``(X, y)``; when the loader has ``return_src`` the
            source distribution is tracked for the progress bar.
        mode : str, optional
            ``"train"`` or ``"validation_<i>"``; passed to the logger.
        pbar : tqdm
            Progress bar with a ``stored_values`` dict (required).
        debug : bool, optional
            Unused.

        Returns
        -------
        tuple
            ``(last_batch_loss, accuracy)`` where the accuracy comes from
            ``logger.log_epoch``; the loss is 0.0 when no batch was yielded.
        """
        # # Reset dataloader to ensure all nested iterators are fresh
        # if isinstance(dataloader, DataloaderMixer):
        #     dataloader.reset()

        # if mode == "train":
        #     # breakpoint()
        sources = {}
        # A dataloader can yield no batch at all even when len() reports one
        # (a weighted list whose inner loaders are exhausted, or a split smaller
        # than its batch size with drop_last). Such a pass has no loss; report it
        # as 0.0 instead of raising UnboundLocalError below.
        loss = None
        for batch_idx, sample in enumerate(dataloader):
            if (
                not self.config.training.steps_per_epoch is None
                and batch_idx >= self.config.training.steps_per_epoch
            ):
                break

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

            # if debug:
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

            if mode == "train" and hasattr(
                self.train_dataloader.dataset, "diffusion_augmentation"
            ):
                X = self.train_dataloader.dataset.diffusion_augmentation(X)

            if (
                "bce" in self.config.task.criterions.keys()
                and len(self.config.task.y_selection) == 1
                and isinstance(dataloader, DataloaderMixer)
                and len(y.shape) == 1
            ):
                y = y.unsqueeze(-1)

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
                project_to_default = (
                    lambda x: dataloader.dataset.project_to_pytorch_default(x)
                )
                project_to_pytorch = (
                    lambda x: dataloader.dataset.project_from_pytorch_default(x)
                )

                X = self.attacker.attack(
                    self.model,
                    torch.clamp(X, 0, 1),
                    y.to(self.device),
                    noise=noise,
                    num_noise_vectors=self.config.training.num_noise_vec,
                    no_grad=self.config.training.no_grad_attack,
                )

                self.model.train()
                requires_grad_(self.model, True)

            self.optimizer.zero_grad()
            # Validation and test passes need no autograd graph. Before
            # 2026-09-25 the forward, the loss and loss.backward() ran in every
            # mode and only optimizer.step() was guarded, so each validation
            # epoch paid a full backward pass and held the training
            # activations in memory.
            with torch.set_grad_enabled(mode == "train"):
                # Compute prediction and loss
                if "lc" in self.config.task.criterions.keys():
                    latent_code, pred = self.model(X, return_latents=True)
                else:
                    pred = self.model(X)
                    latent_code = None

                loss = torch.tensor(0.0).to(self.device)
                loss_logs = {}
                for criterion in self.config.task.criterions.keys():
                    criterion_loss = self.config.task.criterions[
                        criterion
                    ] * self.criterions[criterion](
                        self.model, pred, y.to(self.device), latent_code
                    )

                    if criterion in ["l1", "l2", "orthogonality"]:
                        criterion_loss *= self.regularization_level

                    loss_logs[criterion] = criterion_loss.detach().item()
                    loss += criterion_loss

            loss_logs["loss"] = loss.detach().item()

            self.logger.log_step(mode, pred, y_original, loss_logs)

            # Backpropagation
            if mode == "train":
                loss.backward()
            current_state = "MT: " + mode + "_it: " + str(batch_idx)
            if "val_acc" in pbar.stored_values.keys():
                current_state += ", val_acc: " + str(
                    round(float(pbar.stored_values["val_acc"]), 3)
                )

            current_state += ", loss: " + str(loss.detach().item())
            current_state += ", ".join(
                [
                    key + ": " + str(pbar.stored_values[key])
                    for key in pbar.stored_values
                ]
            )
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

            if self.config.tracking_level < 4:
                current_state = current_state[:199]

            if self.config.tracking_level >= 2:
                pbar.set_postfix_str(current_state, refresh=False)
                pbar.update(1)

            #
            if mode == "train":
                self.optimizer.step()

        accuracy = self.logger.log_epoch(mode, pbar=pbar)

        return (loss.detach().item() if loss is not None else 0.0), accuracy

    def fit(self, continue_training=False, is_initialized=False):
        """Run :meth:`_fit` with every parameter the optimizer does not update frozen.

        ``only_last_layer`` hands the optimizer just the head, but the backbone
        kept ``requires_grad=True``, so every step stored its activations and
        computed gradients that were thrown away: a DINOv3 ViT-L linear probe
        needed a full backward pass (and ran out of memory next to other jobs on
        2026-09-28). The flags are restored afterwards because the same model
        object can later be fine-tuned in full (CFKD), where permanently frozen
        parameters would silently stop training.

        Parameters
        ----------
        continue_training : bool, optional
            Passed to :meth:`_fit`.
        is_initialized : bool, optional
            Passed to :meth:`_fit`.
        """
        trained = {
            id(p) for group in self.optimizer.param_groups for p in group["params"]
        }
        previous = [(p, p.requires_grad) for p in self.model.parameters()]
        for p, _ in previous:
            if id(p) not in trained:
                p.requires_grad_(False)
        try:
            return self._fit(
                continue_training=continue_training, is_initialized=is_initialized
            )
        finally:
            for p, flag in previous:
                p.requires_grad_(flag)

    def _fit(self, continue_training=False, is_initialized=False):
        """Train for ``training.max_epochs`` epochs with early stopping.

        Unless ``continue_training`` is set the weights are reset (orthogonal
        initialisation when the ``orthogonality`` criterion is configured).
        Unless ``is_initialized`` is set an existing ``model_path`` is moved
        aside with an ``_old_<timestamp>`` suffix and ``logs/``, ``outputs/``
        and ``checkpoints/`` are created; at ``tracking_level >= 3`` sample
        train/validation images are logged to TensorBoard.

        Each epoch runs the train loader, then every validation loader; the
        validation score is the weighted average or, for
        ``early_stopping_goal == "worst_group_accuracy"``, the minimum. The
        epoch's state_dict is saved to ``checkpoints/<epoch>.cpl``, and a new
        best score additionally writes ``checkpoints/final.cpl`` and the whole
        module to ``model.cpl``. When train accuracy rises while validation
        accuracy falls, ``regularization_level`` is increased and the previous
        epoch's checkpoint is restored; otherwise the LR scheduler steps.
        Hyper-parameters and the best scores are written with
        ``add_hparams`` at the end.

        Parameters
        ----------
        continue_training : bool, optional
            Keep the current weights instead of resetting them.
        is_initialized : bool, optional
            The output directory already exists and must not be moved aside.
        """
        cprint("Training Config: " + str(self.config), self.config.tracking_level, 4)
        if not continue_training:
            if "orthogonality" in self.config.task.criterions.keys():
                cprint("Orthogonal initialization!!!", self.config.tracking_level, 4)
                orthogonal_initialization(self.model)

            else:
                _log.info("%s", "Training Config: " + str(self.config))
                _log.info("%s", "reset weights!")
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
            cprint(os.path.join(self.model_path, "logs"), self.config.tracking_level, 4)
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(os.path.join(self.model_path, "logs"))
            self.logger.writer = writer
            os.makedirs(os.path.join(self.model_path, "outputs"))
            os.makedirs(os.path.join(self.model_path, "checkpoints"))
            if self.config.tracking_level >= 3:
                open(os.path.join(self.model_path, "platform.txt"), "w").write(
                    platform.node()
                )

                _log.info("%s", "log train images!")
                if hasattr(self.train_dataloader, "remove_empty_dataloaders"):
                    self.train_dataloader.remove_empty_dataloaders()
                log_images_to_writer(self.train_dataloader, self.logger.writer, "train")
                for i in range(len(self.val_dataloaders)):
                    _log.info("%s", "log validation" + str(i) + " images!")
                    log_images_to_writer(
                        self.val_dataloaders[i],
                        self.logger.writer,
                        "validation" + str(i) + "_",
                    )

            self.config.is_loaded = True
            save_yaml_config(self.config, os.path.join(self.model_path, "config.yaml"))

        else:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(os.path.join(self.model_path, "logs"))
            self.logger.writer = writer

        pbar = tqdm(
            total=self.config.training.max_epochs
            * (
                len(self.train_dataloader)
                + int(np.sum(list(map(lambda dl: len(dl), self.val_dataloaders))))
            ),
            # ncols=200,
            dynamic_ncols=True,
        )
        pbar.stored_values = {}
        val_accuracy_previous = 0.0
        train_accuracy_previous = 0.0
        best_train = 0.0
        best_epoch = 0
        self.model.eval()
        val_accuracy = None
        val_weight_sum = 0.0
        self.config.training.epoch = -1
        for idx, val_dataloader in enumerate(self.val_dataloaders):
            if len(val_dataloader) >= 1:
                val_loss, val_accuracy_current = self.run_epoch(
                    val_dataloader, mode="validation_" + str(idx), pbar=pbar
                )
                if self.config.training.early_stopping_goal == "average_accuracy":
                    if val_accuracy is None:
                        val_accuracy = 0.0

                    val_accuracy += (
                        self.val_dataloader_weights[idx] * val_accuracy_current
                    )
                    val_weight_sum += self.val_dataloader_weights[idx]

                elif self.config.training.early_stopping_goal == "worst_group_accuracy":
                    if val_accuracy is None:
                        val_accuracy = val_accuracy_current

                    val_accuracy = min(val_accuracy, val_accuracy_current)

        if (
            self.config.training.early_stopping_goal == "average_accuracy"
            and val_accuracy is not None
            and val_weight_sum > 0
        ):
            val_accuracy = val_accuracy / val_weight_sum

        torch.save(
            self.model.to("cpu").state_dict(),
            os.path.join(self.model_path, "checkpoints", "final.cpl"),
        )
        self.model.to(self.device)
        val_accuracy_max = val_accuracy
        self.logger.writer.add_scalar("0_epoch_validation_accuracy", val_accuracy, -1)
        pbar.stored_values["val_acc"] = val_accuracy

        self.config.training.epoch = 0
        while self.config.training.epoch < self.config.training.max_epochs:
            epoch = self.config.training.epoch
            pbar.stored_values["Epoch"] = self.config.training.epoch
            self.logger.writer.add_scalar(
                "3_regularization_level",
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
                cprint(train_generator_performance, self.config.tracking_level, 4)
                for key in train_generator_performance.keys():
                    self.logger.writer.add_scalar(
                        "3_epoch_train_" + key,
                        train_generator_performance[key],
                        self.config.training.epoch,
                    )
            #
            self.model.eval()
            val_accuracy = None
            val_weight_sum = 0.0
            for idx, val_dataloader in enumerate(self.val_dataloaders):
                if len(val_dataloader) >= 1:
                    val_loss, val_accuracy_current = self.run_epoch(
                        val_dataloader, mode="validation_" + str(idx), pbar=pbar
                    )
                    if self.config.training.early_stopping_goal == "average_accuracy":
                        if val_accuracy is None:
                            val_accuracy = 0.0

                        val_accuracy += (
                            self.val_dataloader_weights[idx] * val_accuracy_current
                        )
                        val_weight_sum += self.val_dataloader_weights[idx]

                    elif (
                        self.config.training.early_stopping_goal
                        == "worst_group_accuracy"
                    ):
                        if val_accuracy is None:
                            val_accuracy = val_accuracy_current

                        val_accuracy = min(val_accuracy, val_accuracy_current)

            if (
                self.config.training.early_stopping_goal == "average_accuracy"
                and val_accuracy is not None
                and val_weight_sum > 0
            ):
                val_accuracy = val_accuracy / val_weight_sum

            self.logger.writer.add_scalar(
                "0_epoch_validation_accuracy", val_accuracy, self.config.training.epoch
            )
            pbar.stored_values["val_acc"] = val_accuracy
            if isinstance(self.model, Generator):
                val_generator_performance = self.val_dataloaders[
                    0
                ].dataset.track_generator_performance(
                    self.model, self.val_dataloaders[0].batch_size
                )
                cprint(val_generator_performance, self.config.tracking_level, 4)
                for key in val_generator_performance.keys():
                    self.logger.writer.add_scalar(
                        "3_epoch_val_" + key,
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
                _log.info(
                    "%s",
                    "New best validation accuracy: "
                    + str(val_accuracy)
                    + ", previous: "
                    + str(val_accuracy_max),
                )
                torch.save(
                    self.model.to("cpu").state_dict(),
                    os.path.join(self.model_path, "checkpoints", "final.cpl"),
                )
                try:
                    torch.save(
                        self.model.to("cpu"), os.path.join(self.model_path, "model.cpl")
                    )

                except Exception:
                    _log.info("%s", "model could not be serialized!!!")
                    _log.info("%s", "model could not be serialized!!!")
                    _log.info("%s", "model could not be serialized!!!")
                    _discard_unloadable_model_file(
                        os.path.join(self.model_path, "model.cpl")
                    )

                val_accuracy_max = val_accuracy
                best_epoch = epoch
                best_train = train_accuracy

                self.model.to(self.device)

            # increase regularization and reset checkpoint if overfitting occurs
            if (
                train_accuracy >= train_accuracy_previous
                and val_accuracy < val_accuracy_previous
            ):
                if self.regularization_level == 0:
                    self.regularization_level = 1

                else:
                    regulization_level = self.config.training.regulization_level
                    self.regularization_level *= regulization_level

                try:
                    checkpoint = torch.load(
                        os.path.join(
                            self.model_path,
                            "checkpoints",
                            str(self.config.training.epoch - 1) + ".cpl",
                        ),
                        map_location=torch.device(self.device),
                    )
                except Exception:
                    checkpoint = torch.load(
                        os.path.join(
                            self.model_path,
                            "checkpoints",
                            str(self.config.training.epoch - 1) + ".cpl",
                        ),
                        map_location=torch.device(self.device),
                        weights_only=False,
                    )
                self.model.load_state_dict(checkpoint)

            else:
                train_accuracy_previous = train_accuracy
                val_accuracy_previous = val_accuracy
                if hasattr(self, "scheduler"):
                    self.scheduler.step()

            save_yaml_config(self.config, os.path.join(self.model_path, "config.yaml"))

            self.config.training.epoch += 1
        writer.add_hparams(
            dict(self.config.training),
            {
                "best_val_accuracy": val_accuracy_max,
                "train_accuracy": best_train,
                "epoch": best_epoch,
            },
        )
        epoch += 1

        if not os.path.exists(os.path.join(self.model_path, "model.cpl")):
            try:
                torch.save(
                    self.model.to("cpu"), os.path.join(self.model_path, "model.cpl")
                )
            except Exception:
                _log.info("%s", "model could not be serialized!!!")
                _discard_unloadable_model_file(
                    os.path.join(self.model_path, "model.cpl")
                )
            self.model.to(self.device)


def distill_binary_dataset(
    predictor_distillation, base_path, predictor, predictor_datasets
):
    """Relabel ``Image2MixedDataset``-style datasets with a predictor's outputs.

    For every dataset ``i`` the predictions are written once to
    ``<base_path>/<i>predictions.csv`` via ``get_predictions``; a copy of the
    dataset config with confounder information removed and
    ``output_type="multiclass"`` is then loaded from that csv. The first
    dataset becomes the train split, all others validation splits.

    Parameters
    ----------
    predictor_distillation : str, dict or PredictorConfig
        Config of the student; its ``task`` is attached to the new datasets.
    base_path : str
        Directory the prediction csv files are written to.
    predictor : torch.nn.Module
        Teacher that produces the labels.
    predictor_datasets : list
        Datasets or dataloaders to relabel.

    Returns
    -------
    list
        One relabelled dataset per input dataset.
    """
    _log.info("%s", "distill_binary_dataset")
    distillation_datasource = []
    for i in range(len(predictor_datasets)):
        if isinstance(predictor_datasets[i], torch.utils.data.DataLoader):
            predictor_dataset = predictor_datasets[i].dataset

        else:
            predictor_dataset = predictor_datasets[i]

        class_predictions_path = os.path.join(base_path, str(i) + "predictions.csv")
        Path(base_path).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(class_predictions_path):
            predictor_dataset.enable_url()
            prediction_args = types.SimpleNamespace(
                batch_size=32,
                dataset=predictor_dataset,
                classifier=predictor,
                label_path=class_predictions_path,
                partition="train",
                label_query=0,
                is_image_to_class=isinstance(predictor_dataset, Image2ClassDataset),
            )
            get_predictions(prediction_args)
            predictor_dataset.disable_url()

        distilled_dataset_config = copy.deepcopy(predictor_dataset.config)
        distilled_dataset_config.delimiter = ","  # update delimiter to ',' in case of different delimiter in original dataset.
        distilled_dataset_config.split = [1.0, 1.0] if i == 0 else [0.0, 1.0]
        distilled_dataset_config.confounding_factors = None
        distilled_dataset_config.confounder_probability = None
        distilled_dataset_config.dataset_class = None
        distilled_dataset_config.output_type = "multiclass"
        distillation_datasource.append(
            get_datasets(
                config=distilled_dataset_config, data_dir=class_predictions_path
            )[i]
        )
        distilled_predictor_config = load_yaml_config(
            predictor_distillation, PredictorConfig
        )
        distilled_predictor_config.data = distilled_dataset_config
        predictor_distillation = distilled_predictor_config
        distillation_datasource[i].task_config = predictor_distillation.task
        distillation_datasource[i].task_config.x_selection = (
            predictor_dataset.task_config.x_selection
        )
        try:
            sample = distillation_datasource[-1][0]
        except Exception as e:
            _log.info("%s", f"Warning indexing distillation datasource sample: {e}")

    return distillation_datasource


def distill_1ofn_dataset(
    predictor_distillation, base_path, predictor, predictor_datasets
):
    """Relabel ``Image2ClassDataset`` (folder-per-class) data with a predictor.

    Every sample of the first two datasets is classified and saved as
    ``<base_path>/dataset_<i>/<predicted class>/<idx>.png`` (skipped when the
    directory exists), then reloaded as a dataset without confounder
    information; index 0 is the train split, index 1 the validation split.
    Samples with more than three channels, or that fail to load or save, are
    skipped with a printed message.

    Parameters
    ----------
    predictor_distillation : str, dict or PredictorConfig
        Config of the student.
    base_path : str
        Directory the relabelled image folders are written to.
    predictor : torch.nn.Module
        Teacher that produces the labels.
    predictor_datasets : list
        Two datasets, dataloader mixers or weighted dataloader lists.

    Returns
    -------
    list
        The two relabelled datasets.
    """
    _log.info("%s", "distill_1ofn_dataset")
    distillation_datasource = []
    for i in range(2):
        class_predictions_path = os.path.join(base_path, "dataset_" + str(i))
        if isinstance(predictor_datasets[i], DataloaderMixer):
            dataset = predictor_datasets[i].dataset
        elif isinstance(predictor_datasets[i], WeightedDataloaderList):
            dataset = predictor_datasets[i].dataloaders[0].dataset
        else:
            dataset = predictor_datasets[i]
        if not os.path.exists(class_predictions_path):

            for sample_idx in range(dataset.__len__()):
                _log.info("%s", sample_idx)
                try:

                    X, y = dataset[sample_idx]
                    # get device of predictor torch.nn.Module
                    device = next(predictor.parameters()).device
                    if X.shape[0] > 3:
                        continue
                    y_pred = str(
                        int(predictor(X.unsqueeze(0).to(device))[0].argmax(-1))
                    )
                except Exception as e:
                    _log.info("%s", f"Error at sample {sample_idx}: {e}")
                    continue
                Path(os.path.join(class_predictions_path, y_pred)).mkdir(
                    exist_ok=True, parents=True
                )
                sample_url = os.path.join(
                    class_predictions_path, y_pred, str(sample_idx) + ".png"
                )
                try:

                    x_default = dataset.project_to_pytorch_default(X)
                    torchvision.utils.save_image(x_default, sample_url)

                except Exception as e:
                    _log.info("%s", f"Error at sample {sample_idx}: {e}")
                    continue

        distilled_dataset_config = copy.deepcopy(dataset.config)
        distilled_dataset_config.split = [1.0, 1.0] if i == 0 else [0.0, 1.0]
        # distilled_dataset_config.img_name_idx = 0
        distilled_dataset_config.confounding_factors = None
        distilled_dataset_config.confounder_probability = None
        distilled_dataset_config.dataset_class = None
        distillation_datasource.append(
            get_datasets(
                config=distilled_dataset_config, data_dir=class_predictions_path
            )[i]
        )
        distilled_predictor_config = load_yaml_config(
            predictor_distillation, PredictorConfig
        )
        distilled_predictor_config.data = distilled_dataset_config
        predictor_distillation = distilled_predictor_config
        distillation_datasource[i].task_config = dataset.task_config
    return distillation_datasource


def distill_dataloader_mixer(
    predictor_distillation, base_path, predictor, predictor_datasource
):
    """Recursively relabel every loader inside a ``DataloaderMixer``.

    Nested mixers are handled recursively under ``<base_path>/<i>``; plain
    loaders go through :func:`distill_binary_dataset` and are re-wrapped in a
    ``DataLoader`` with the original batch size. The copied mixer is reset
    before being returned.

    Parameters
    ----------
    predictor_distillation : str, dict or PredictorConfig
        Config of the student.
    base_path : str
        Root directory for the prediction files.
    predictor : torch.nn.Module
        Teacher that produces the labels.
    predictor_datasource : DataloaderMixer
        Mixer to relabel (deep-copied, not modified).

    Returns
    -------
    DataloaderMixer
        A mixer over the relabelled loaders.
    """
    distillation_datasource = copy.deepcopy(predictor_datasource)
    for i in range(len(distillation_datasource.dataloaders)):
        if isinstance(distillation_datasource.dataloaders[i], DataloaderMixer):
            distillation_datasource.dataloaders[i] = distill_dataloader_mixer(
                predictor_distillation=predictor_distillation,
                base_path=os.path.join(base_path, str(i)),
                predictor=predictor,
                predictor_datasource=distillation_datasource.dataloaders[i],
            )

        else:
            dataset = distill_binary_dataset(
                predictor_distillation=predictor_distillation,
                base_path=os.path.join(base_path, str(i)),
                predictor=predictor,
                predictor_datasets=[distillation_datasource.dataloaders[i]],
            )
            distillation_datasource.dataloaders[i] = torch.utils.data.DataLoader(
                dataset[0],
                batch_size=distillation_datasource.dataloaders[i].batch_size,
            )
    distillation_datasource.reset()
    return distillation_datasource


def _discard_unloadable_model_file(path):
    """Delete a model.cpl that torch.save left behind after failing to pickle.

    torch.save opens the zip archive before it pickles, and the archive is still
    finalised on the way out of the failed call, so a failure leaves a ~700 byte
    file containing `version` and `byteorder` but no `data.pkl`. Every guard
    downstream tests only that the path exists, so that stub is taken for a
    trained model from then on and every later run dies in torch.load with
    "PytorchStreamReader failed locating file data.pkl". The state_dict written
    to checkpoints/final.cpl is unaffected and is the real artefact.
    """
    try:
        if os.path.exists(path):
            os.remove(path)
            _log.info(
                "%s", f"[PEAL] Removed the unusable {path} left by the failed save."
            )
    except OSError as exc:
        _log.info("%s", f"[PEAL] Could not remove the unusable {path}: {exc}")


def load_first_loadable(paths, map_location=None):
    """First checkpoint in `paths` that actually loads, or None if none does.

    Returns whatever was saved -- a module or a state_dict -- so the caller must
    handle both. Callers that cannot rebuild an architecture from a state_dict
    should pass only paths holding a whole module.
    """
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        last = None
        for extra in ({}, {"weights_only": False}):
            try:
                return torch.load(path, map_location=map_location, **extra)
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                last = exc
        _log.info(
            "%s", f"[PEAL] {path} exists but does not load ({last}); trying the next."
        )
    return None


def distill_predictor(
    predictor_distillation,
    base_path,
    predictor,
    predictor_datasource,
    replace_with_activation=None,
    tracking_level=4,
    predictor_distilled=None,
    only_last_layer=False,
    continue_training=False,
    task_config=None,
):
    """Train a student predictor on data relabelled by ``predictor``.

    The relabelling strategy is chosen from the datasource: with
    ``distill_from == "dataset"`` the original labels are kept; a
    ``(DataloaderMixer, WeightedDataloaderList)`` pair is relabelled with
    :func:`distill_dataloader_mixer` / :func:`distill_binary_dataset`;
    ``Image2MixedDataset`` data with :func:`distill_binary_dataset`; and
    ``Image2ClassDataset`` data with :func:`distill_1ofn_dataset`. The
    student is built from ``predictor_distillation.architecture``, or is a
    deep copy of the teacher, and is trained by a :class:`ModelTrainer` into
    ``<base_path>/distilled_predictor``.

    Parameters
    ----------
    predictor_distillation : str, dict or PredictorConfig
        Student config; ``distill_from`` and ``architecture`` are read here.
    base_path : str
        Directory for relabelled data and the trained student.
    predictor : torch.nn.Module
        Teacher whose predictions become the labels.
    predictor_datasource : sequence
        ``(train, validation)`` datasets or loaders of the teacher.
    replace_with_activation : {"leakysoftplus", "leakyrelu", None}, optional
        Swap the student's ReLUs (leakysoftplus only for a freshly built
        student, leakyrelu only for a given ``predictor_distilled``).
    tracking_level : int, optional
        Verbosity written into the student config. Default 4.
    predictor_distilled : torch.nn.Module, optional
        Student to train instead of building one.
    only_last_layer : bool, optional
        Forwarded to :class:`ModelTrainer`.
    continue_training : bool, optional
        Forwarded to :meth:`ModelTrainer.fit`.
    task_config : TaskConfig, optional
        Overrides the task config of the relabelled datasets.

    Returns
    -------
    torch.nn.Module
        The trained student.

    Raises
    ------
    Exception
        When the datasource type matches no relabelling strategy.
    """
    predictor_distillation = load_yaml_config(
        predictor_distillation,
        PredictorConfig,
    )
    predictor_distillation.tracking_level = tracking_level
    if predictor_distillation.distill_from == "dataset":
        _log.info("%s", "distill_from_dataset")
        distillation_datasource = predictor_datasource

    elif isinstance(predictor_datasource[0], DataloaderMixer) and isinstance(
        predictor_datasource[1], WeightedDataloaderList
    ):
        _log.info("%s", "distill dataloader mixer")
        distillation_datasource = []
        distillation_datasource.append(
            distill_dataloader_mixer(
                predictor_distillation,
                os.path.join(base_path, "training"),
                predictor,
                copy.deepcopy(predictor_datasource[0]),
            )
        )
        cprint("distill validation dataset!", tracking_level, 2)
        distillation_datasource.append(copy.deepcopy(predictor_datasource[1]))
        validation_datasets = distill_binary_dataset(
            predictor_distillation,
            os.path.join(base_path, "validation"),
            predictor,
            distillation_datasource[1].dataloaders,
        )
        for i in range(len(validation_datasets)):
            distillation_datasource[1].dataloaders[i] = torch.utils.data.DataLoader(
                validation_datasets[i],
                batch_size=distillation_datasource[1].dataloaders[i].batch_size,
            )
            if not task_config is None:
                distillation_datasource[1].dataloaders[
                    i
                ].dataset.task_config = task_config

    elif isinstance(predictor_datasource[0], Image2MixedDataset) or isinstance(
        predictor_datasource[0].dataset, Image2MixedDataset
    ):
        distillation_datasource = distill_binary_dataset(
            predictor_distillation, base_path, predictor, predictor_datasource
        )
        if not task_config is None:
            for i in range(len(distillation_datasource)):
                distillation_datasource[i].task_config = task_config

    elif isinstance(predictor_datasource[0].dataset, Image2ClassDataset):
        _log.info("%s", "distill_1ofn_dataset")
        distillation_datasource = distill_1ofn_dataset(
            predictor_distillation, base_path, predictor, predictor_datasource
        )
        if isinstance(predictor_datasource[0], DataloaderMixer):
            task_config = predictor_datasource[0].dataset.task_config
        else:
            task_config = predictor_datasource[0].task_config
        predictor_distillation.task = task_config

    else:
        raise Exception(
            "Either distill from dataset or use available dataset type for relabeling"
        )

    if predictor_distilled is None:
        if not predictor_distillation.architecture is None:
            predictor_distilled = get_predictor(predictor_distillation)

        elif isinstance(predictor, torch.nn.Module):
            predictor_distilled = copy.deepcopy(predictor)

        else:
            # TODO how can I determine that there are no gradients anymore?
            predictor_distilled = get_predictor(predictor_distillation)

        if replace_with_activation == "leakysoftplus":
            predictor_distilled = replace_relu_with_leakysoftplus(predictor_distilled)

    elif replace_with_activation == "leakyrelu":
        predictor_distilled = replace_relu_with_leakyrelu(predictor_distilled)
    distillation_trainer = ModelTrainer(
        config=predictor_distillation,
        model=predictor_distilled,
        datasource=distillation_datasource,
        model_path=os.path.join(base_path, "distilled_predictor"),
        only_last_layer=only_last_layer,
    )
    distillation_trainer.fit(continue_training=continue_training)
    return predictor_distilled
