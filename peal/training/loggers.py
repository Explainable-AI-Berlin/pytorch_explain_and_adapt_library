"""TensorBoard logging for predictor (and generator) training.

``Logger`` is owned by ``ModelTrainer``: it accumulates the per-step losses
and predictions of one epoch, writes scalar / histogram summaries with the
``0_`` (validation), ``1_`` (train), ``2_`` (per-attribute) and ``3_``
(everything else) tag prefixes that order the TensorBoard panels, and
returns the epoch metric early stopping is judged by.
``log_images_to_writer`` dumps a few sample batches of a dataloader as image
grids so a run's inputs can be inspected in TensorBoard.
"""

import torch
import os
import math
import torchvision
import torch.nn as nn

from peal.data.dataloaders import DataloaderMixer
from peal.generators.interfaces import InvertibleGenerator
from peal.log import get_logger

_log = get_logger(__name__)


class Logger:
    """Accumulate step results of one epoch and write TensorBoard summaries.

    The logger reads ``config.task.criterions`` to decide which metrics apply:
    ``ce`` / ``bce`` runs collect predicted classes and per-sample correctness
    (accuracy), ``mixed`` / ``focal_mixed`` / ``focal`` runs treat the first
    ``config.data.output_split`` outputs as multi-label logits and the rest as
    regression targets (micro / macro F1, precision, recall, MAE, MSE, focal
    loss, balanced accuracy). ``InvertibleGenerator`` models only log the loss.
    Step counters ``config.training.global_<mode>_step`` are incremented as a
    side effect of ``log_step``.

    Parameters
    ----------
    config : PredictorConfig
        Run config; ``task``, ``data`` and ``training`` are consulted.
    model : torch.nn.Module
        The model being trained; used for its device and to detect generators.
    optimizer : torch.optim.Optimizer
        Kept for reference only.
    base_dir : str
        Run directory (images would be written under ``base_dir/outputs``).
    criterions : dict
        Loss functions of the run; kept for reference only.
    val_dataloader : torch.utils.data.DataLoader
        Validation loader; its first batch is stored as ``test_X`` / ``test_y``
        and its dataset's ``attributes`` name the per-attribute scalars.
    attacker : optional
        Adversarial attacker of the trainer; kept for reference only.
    writer : torch.utils.tensorboard.SummaryWriter
        Destination of all summaries.

    Attributes
    ----------
    output_channels : int
        ``config.task.output_channels`` or, if unset, ``config.data.output_size[0]``;
        the width used for one-hot encoding predictions.
    losses, predictions, targets, predicted_classes, correct : list
        Per-step buffers that ``log_epoch`` reduces and resets.
    """

    def __init__(
        self,
        config,
        model,
        optimizer,
        base_dir,
        criterions,
        val_dataloader,
        attacker=None,
        writer=None,
    ):
        """Store the trainer's objects and initialise the per-epoch buffers.

        Parameters
        ----------
        config, model, optimizer, base_dir, criterions, val_dataloader, attacker, writer
            See the class docstring.
        """
        #
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.base_dir = base_dir
        self.criterions = criterions
        self.attacker = attacker
        self.writer = writer
        self.val_dataloader = val_dataloader

        if not self.config.task.output_channels is None:
            self.output_channels = self.config.task.output_channels

        else:
            self.output_channels = self.config.data.output_size[0]

        self.device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        #
        if "ce" in config.task.criterions.keys():
            #
            # self.test_X, self.test_y = create_class_ordered_batch(val_dataloader.dataset, config)
            self.test_X, self.test_y = next(iter(val_dataloader))

        else:
            #
            self.test_X, self.test_y = next(iter(val_dataloader))

        if isinstance(self.model, InvertibleGenerator):
            self.latent_code = self.model.sample_z()

        # temporary variables
        self.losses = []
        #
        if self.config.data.output_type in ["singleclass", "multiclass", "mixed"]:
            self.predictions = []
            self.targets = []
            self.predicted_classes = []
            self.correct = []

    def log_step(self, mode, pred, y, loss_logs):
        """Record one training / validation step.

        Writes every entry of ``loss_logs`` as a ``3_step_<mode>_<name>`` scalar at
        ``config.training.global_train_step``, buffers the loss, targets and
        predictions needed for the epoch metrics and increments
        ``config.training.global_<mode>_step`` when that attribute exists.

        Parameters
        ----------
        mode : str
            ``"train"``, ``"validation"`` or a variant such as ``"validation_cf"``.
        pred : torch.Tensor
            Model outputs of the step, ``[B, output_channels]``.
        y : torch.Tensor
            Targets of the step (class indices for ``ce``, multi-hot / mixed vectors
            otherwise).
        loss_logs : dict
            Named loss values; must contain ``"loss"``.
        """
        for criterion in loss_logs:
            self.writer.add_scalar(
                "3_step_" + mode + "_" + criterion,
                loss_logs[criterion],
                self.config.training.global_train_step,
            )

        self.losses.append(loss_logs["loss"])

        if any(
            k in self.config.task.criterions.keys()
            for k in ["mixed", "focal_mixed", "focal"]
        ):
            self.targets.append(y.detach())
            self.predictions.append(pred.detach())

        if len(
            set(["ce", "bce"]).intersection(self.config.task.criterions.keys())
        ) >= 1 and not isinstance(self.model, InvertibleGenerator):
            #
            self.targets.append(y.detach())
            self.predictions.append(pred.detach().cpu())
            #
            if "ce" in self.config.task.criterions.keys():
                class_prediction = pred.detach().argmax(-1)

            elif "bce" in self.config.task.criterions.keys():
                class_prediction = nn.Sigmoid()(pred) >= 0.5

            self.correct.append(
                (class_prediction == y.to(self.device)).type(torch.float)
            )
            self.predicted_classes.append(class_prediction.detach().to(torch.float32))

        #
        if hasattr(
            self.config.training,
            "global_" + mode + "_step",
        ):
            setattr(
                self.config.training,
                "global_" + mode + "_step",
                getattr(self.config.training, "global_" + mode + "_step") + 1,
            )

    def log_epoch(self, mode, pbar=None):
        """Reduce the buffered steps, write the epoch summaries and reset buffers.

        The mean loss is written as ``3_epoch_<mode>_loss_accumulated``; classifier
        runs additionally get ``<0_|1_>epoch_<mode>_accuracy`` (or F1 / MAE / MSE /
        focal loss / balanced accuracy for mixed runs), per-channel prediction rates,
        a correct-per-class histogram and, for validation modes, per-attribute F1 and
        MAE scalars under ``2_epoch_<mode>_*``. Selected values are also stored in
        ``pbar.stored_values`` for the progress bar.

        Parameters
        ----------
        mode : str
            Same mode string as passed to ``log_step``; modes starting with
            ``"validation"`` use the ``0_`` prefix, others ``1_``.
        pbar : optional
            Progress bar with a ``stored_values`` dict to update.

        Returns
        -------
        float
            The epoch score early stopping compares: accuracy (``ce`` / ``bce``),
            micro F1 (mixed criterions) or ``exp(-mean loss)`` when no classification
            metric applies. An epoch with no collected steps yields ``exp(0) = 1``.
        """
        accuracy = None
        # An epoch can end with nothing collected (a validation dataloader that
        # yielded no batch at all -- e.g. a counterfactual split smaller than its
        # batch size). The one-hot tensors below are only built when targets were
        # seen, so keep them defined: reporting them is optional, crashing is not.
        predictions_one_hot = None
        targets_one_hot = None
        is_val = mode.startswith("validation")
        global_prefix = "0_" if is_val else "1_"

        if len(self.losses) == 0:
            loss_accumulated = torch.tensor(0.0)
        else:
            loss_accumulated = torch.mean(torch.tensor(self.losses))

        self.writer.add_scalar(
            "3_epoch_" + mode + "_loss_accumulated",
            loss_accumulated.item(),
            self.config.training.epoch,
        )
        if not pbar is None:
            pbar.stored_values[mode + "_loss_accumulated"] = loss_accumulated.item()

        if "ce" in self.config.task.criterions.keys() and not isinstance(
            self.model, InvertibleGenerator
        ):
            if len(self.targets) > 0:
                targets_one_hot = torch.nn.functional.one_hot(
                    torch.cat(self.targets).to(torch.int64), self.output_channels
                ).to(torch.float32)

                predictions_one_hot = torch.nn.functional.one_hot(
                    torch.cat(self.predicted_classes).to(torch.int64),
                    self.output_channels,
                ).to(torch.float32)

        if "bce" in self.config.task.criterions.keys() and not isinstance(
            self.model, InvertibleGenerator
        ):
            if len(self.targets) > 0:
                targets_one_hot = torch.cat(self.targets)
                predictions_one_hot = torch.cat(self.predicted_classes).cpu()
                correct_per_class = (
                    torch.cat(self.correct).mean(0)
                    if len(self.correct) > 0
                    else torch.tensor([0.0])
                )
                if not pbar is None:
                    pbar.stored_values["correct_per_class"] = correct_per_class

                self.writer.add_histogram(
                    "3_epoch_" + mode + "_correct_per_class",
                    correct_per_class,
                    self.config.training.epoch,
                )

        if any(
            k in self.config.task.criterions.keys()
            for k in ["mixed", "focal_mixed", "focal"]
        ) and not isinstance(self.model, InvertibleGenerator):
            if len(self.targets) > 0 and len(self.predictions) > 0:
                targets_one_hot = torch.cat(self.targets).cpu()
                predictions_cat = torch.cat(self.predictions)
                split = self.model.config.data.output_split

                class_preds = (nn.Sigmoid()(predictions_cat[:, :split]) >= 0.5).to(
                    torch.float32
                )
                regression_preds = predictions_cat[:, split:]

                predictions_one_hot = torch.cat(
                    [
                        class_preds,
                        regression_preds,
                    ],
                    dim=-1,
                ).cpu()
                self.correct = (
                    torch.abs(targets_one_hot - predictions_one_hot) < 0.5
                ).to(torch.float32)
                correct_per_class = self.correct.mean(0)

                targets1 = targets_one_hot[:, :split]
                targets2 = targets_one_hot[:, split:]

                class_preds = class_preds.cpu()
                regression_preds = regression_preds.cpu()

                # (micro) F1 score
                tp = (class_preds * targets1).sum()
                fp = (class_preds * (1 - targets1)).sum()
                fn = ((1 - class_preds) * targets1).sum()

                f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

                recall = (tp / (tp + fn + 1e-8)).item()
                precision = (tp / (tp + fp + 1e-8)).item()

                mse = nn.functional.mse_loss(regression_preds, targets2)
                mae = nn.functional.l1_loss(regression_preds, targets2)

                # macro F1 score
                tp_class = (class_preds * targets1).sum(dim=0)
                fp_class = (class_preds * (1 - targets1)).sum(dim=0)
                fn_class = ((1 - class_preds) * targets1).sum(dim=0)

                f1_per_class = (
                    2 * tp_class / (2 * tp_class + fp_class + fn_class + 1e-8)
                )

                accuracy = f1.item()

                # Calculate Balanced Accuracy for the discrete part
                disc_targets = targets_one_hot[:, :split]
                disc_preds = predictions_one_hot[:, :split]

                # Sensitivity (TPR) and Specificity (TNR)
                pos_mask = disc_targets == 1
                neg_mask = disc_targets == 0

                tp_bal = (disc_preds == 1) & pos_mask
                tn_bal = (disc_preds == 0) & neg_mask

                sensitivity = tp_bal.sum(0).float() / pos_mask.sum(0).float().clamp(
                    min=1e-6
                )
                specificity = tn_bal.sum(0).float() / neg_mask.sum(0).float().clamp(
                    min=1e-6
                )

                balanced_acc_per_class = (sensitivity + specificity) / 2
                avg_balanced_acc = balanced_acc_per_class.mean().item()

                # Calculate Focal Loss for sparse features
                focal_alpha = getattr(self.config.task, "focal_alpha", 0.25)
                focal_gamma = getattr(self.config.task, "focal_gamma", 1.0)
                bce_loss_tensor = nn.functional.binary_cross_entropy_with_logits(
                    predictions_cat[:, :split].cpu(), targets1, reduction="none"
                )
                pt = torch.exp(-bce_loss_tensor)
                focal_loss_val = (
                    (focal_alpha * ((1 - pt) ** focal_gamma) * bce_loss_tensor)
                    .mean()
                    .item()
                )

                if not pbar is None:
                    pbar.stored_values["f1"] = round(f1.item(), 3)
                    pbar.stored_values["recall"] = round(recall, 3)
                    pbar.stored_values["precision"] = round(precision, 3)
                    pbar.stored_values["mae"] = round(mae.item(), 3)
                    pbar.stored_values["mse"] = round(mse.item(), 3)
                    pbar.stored_values["focal_loss"] = round(focal_loss_val, 3)
                    pbar.stored_values[mode + "_balanced_accuracy"] = round(
                        avg_balanced_acc, 3
                    )

                # 1) global validation scores (0_*) & 2) global train scores (1_*)
                self.writer.add_scalar(
                    global_prefix + "epoch_" + mode + "_f1",
                    f1.item(),
                    self.config.training.epoch,
                )
                self.writer.add_scalar(
                    global_prefix + "epoch_" + mode + "_mae",
                    mae.item(),
                    self.config.training.epoch,
                )
                self.writer.add_scalar(
                    global_prefix + "epoch_" + mode + "_mse",
                    mse.item(),
                    self.config.training.epoch,
                )
                self.writer.add_scalar(
                    global_prefix + "epoch_" + mode + "_focal_loss",
                    focal_loss_val,
                    self.config.training.epoch,
                )
                self.writer.add_scalar(
                    global_prefix + "epoch_" + mode + "_balanced_accuracy",
                    avg_balanced_acc,
                    self.config.training.epoch,
                )

                # 3) local validation scores (2_*)
                if is_val:
                    attributes = getattr(
                        self.val_dataloader.dataset, "attributes", None
                    )
                    n_disc = split
                    if n_disc <= 10:
                        disc_indices = list(range(n_disc))
                    else:
                        disc_indices = list(range(5)) + list(range(n_disc - 5, n_disc))

                    for idx in disc_indices:
                        attr_name = (
                            attributes[idx]
                            if attributes and idx < len(attributes)
                            else f"var_{idx}"
                        )
                        self.writer.add_scalar(
                            f"2_epoch_{mode}_f1_{attr_name}",
                            f1_per_class[idx].item(),
                            self.config.training.epoch,
                        )

                    n_reg = targets2.shape[1]
                    mae_per_reg = torch.abs(regression_preds - targets2).mean(dim=0)
                    if n_reg <= 10:
                        reg_indices = list(range(n_reg))
                    else:
                        reg_indices = list(range(5)) + list(range(n_reg - 5, n_reg))

                    for reg_idx in reg_indices:
                        global_idx = n_disc + reg_idx
                        attr_name = (
                            attributes[global_idx]
                            if attributes and global_idx < len(attributes)
                            else f"var_{global_idx}"
                        )
                        self.writer.add_scalar(
                            f"2_epoch_{mode}_mae_{attr_name}",
                            mae_per_reg[reg_idx].item(),
                            self.config.training.epoch,
                        )

                # 4) everything else (3_*)
                self.writer.add_scalar(
                    "3_epoch_" + mode + "_precision",
                    precision,
                    self.config.training.epoch,
                )
                self.writer.add_scalar(
                    "3_epoch_" + mode + "_recall", recall, self.config.training.epoch
                )

                self.writer.add_histogram(
                    "3_epoch_" + mode + "_correct_per_class",
                    correct_per_class,
                    self.config.training.epoch,
                )

        if len(
            set(["ce", "bce"]).intersection(self.config.task.criterions.keys())
        ) >= 1 and not isinstance(self.model, InvertibleGenerator):
            if isinstance(self.correct, list):
                self.correct = (
                    torch.cat(self.correct)
                    if len(self.correct) > 0
                    else torch.tensor([0.0])
                )
            if accuracy is None:
                accuracy = self.correct.mean().item()

            self.writer.add_scalar(
                global_prefix + "epoch_" + mode + "_accuracy",
                accuracy,
                self.config.training.epoch,
            )

            channels_to_log = sorted(
                list(
                    set(
                        list(range(min(50, self.output_channels)))
                        + [
                            self.output_channels // 2 - 1,
                            self.output_channels // 2,
                            self.output_channels // 2 + 1,
                        ]
                        + list(
                            range(
                                max(0, self.output_channels - 8), self.output_channels
                            )
                        )
                    )
                )
            )
            channels_to_log = [
                c for c in channels_to_log if 0 <= c < self.output_channels
            ]

            for channel in channels_to_log:
                try:
                    self.writer.add_scalar(
                        "3_epoch_" + mode + "_predicted_classes" + str(channel),
                        predictions_one_hot.float().mean(0)[channel],
                        self.config.training.epoch,
                    )
                except Exception:
                    pass

                try:
                    self.writer.add_scalar(
                        "3_epoch_" + mode + "_classes_difference" + str(channel),
                        torch.abs(
                            predictions_one_hot.cpu().float() - targets_one_hot.float()
                        )
                        .mean(0)
                        .cpu()[channel],
                        self.config.training.epoch,
                    )
                except Exception:
                    pass
            if not pbar is None:
                pbar.stored_values[mode + "_accuracy"] = accuracy
            if (
                not pbar is None
                and predictions_one_hot is not None
                and targets_one_hot is not None
            ):
                pbar.stored_values[mode + "_predicted_classes"] = (
                    predictions_one_hot.float().mean(0).cpu()[:2]
                )
                pbar.stored_values[mode + "_targets"] = (
                    targets_one_hot.float().mean(0).cpu()[:2]
                )
                pbar.stored_values[mode + "_classes_difference"] = (
                    predictions_one_hot.float().mean(0).cpu()
                    - targets_one_hot.float().mean(0).cpu()
                )[:2]

        # Reset lists unconditionally at end of epoch
        self.predictions = []
        self.targets = []
        self.predicted_classes = []
        self.correct = []
        self.losses = []

        if accuracy is None:
            accuracy = torch.exp(-loss_accumulated).item()

        return accuracy


def log_images_to_writer(dataloader, writer, tag="train"):
    """Write up to three sample batches of a dataloader to TensorBoard.

    For a ``DataloaderMixer`` that does not concatenate batches, batches are
    drawn from its first (and, for the third grid, second) underlying loader.
    Images are mapped to ``[0, 1]`` with the dataset's
    ``project_to_pytorch_default`` when available; if the dataset offers
    ``diffusion_augmentation`` the augmented batch is written as a second grid.
    The tag encodes the batch index and, for 1-D labels, the label list. The
    mixer is reset before and after so training starts from fresh iterators.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader or DataloaderMixer
        Source of the batches; ``dataloader.dataset`` must be reachable.
    writer : torch.utils.tensorboard.SummaryWriter
        Destination of the image grids.
    tag : str
        Prefix of the image tags, e.g. ``"train"`` or ``"val"``.
    """
    if isinstance(dataloader, DataloaderMixer):
        dataloader.reset()
    dataloader_mixer_treatment = isinstance(dataloader, DataloaderMixer)
    if dataloader_mixer_treatment:
        dataloader_mixer_treatment &= (
            not hasattr(dataloader.train_config, "concatenate_batches")
            or not dataloader.train_config.concatenate_batches
        )

    if dataloader_mixer_treatment:
        iterator = iter(dataloader.dataloaders[0])

    else:
        iterator = iter(dataloader)

    for i in range(3):
        if i == 1:
            if dataloader_mixer_treatment:
                iterator = iter(dataloader.dataloaders[0])

            else:
                iterator = iter(dataloader)

        if i == 2 and dataloader_mixer_treatment and len(dataloader.dataloaders) > 1:
            iterator = iter(dataloader.dataloaders[1])
        try:
            sample_train_imgs, sample_train_y = next(iterator)
        except Exception:
            continue

        if isinstance(sample_train_imgs, list):
            sample_train_imgs, sample_train_y = sample_train_imgs

        if hasattr(dataloader.dataset, "diffusion_augmentation"):
            # Apply diffusion augmentation if available
            sample_train_imgs_augmented = dataloader.dataset.diffusion_augmentation(
                sample_train_imgs
            )
            if hasattr(dataloader.dataset, "project_to_pytorch_default"):
                sample_train_imgs_augmented = (
                    dataloader.dataset.project_to_pytorch_default(
                        sample_train_imgs_augmented
                    )
                )

        if hasattr(dataloader.dataset, "project_to_pytorch_default"):
            sample_train_imgs = dataloader.dataset.project_to_pytorch_default(
                sample_train_imgs
            )

        else:
            _log.info(
                "%s",
                "Warning! If your dataloader uses another normalization than the PyTorch default [0,1]"
                + "range data might be visualized incorrect!"
                + "In that case add function project_to_pytorch_default() to your underlying dataset to correct visualization!",
            )

        sample_batch_label_str = "sample_" + tag + "_batch" + str(i) + "_"
        if isinstance(sample_train_y, torch.Tensor) and len(sample_train_y.shape) == 1:
            sample_batch_label_str += "_" + str(
                list(map(lambda x: int(x), list(sample_train_y)))
            )

        elif isinstance(sample_train_y, list) and len(sample_train_y) == 1:
            sample_batch_label_str += "_" + str(
                list(map(lambda x: int(x), list(sample_train_y[0])[:128]))
            )

        writer.add_image(
            sample_batch_label_str,
            torchvision.utils.make_grid(sample_train_imgs[:128], 128),
        )

        if hasattr(dataloader.dataset, "diffusion_augmentation"):
            writer.add_image(
                sample_batch_label_str + "_augmented",
                torchvision.utils.make_grid(sample_train_imgs_augmented[:128], 128),
            )

    # Reset dataloader to ensure fresh iterators for training
    if isinstance(dataloader, DataloaderMixer):
        dataloader.reset()
