"""Group Distributionally Robust Optimization as a PEAL repair adaptor.

A second, self-contained implementation of GroupDRO (Sagawa et al., 2020)
next to :mod:`peal.adaptors.group_distributionally_robust_optimization`: it
brings its own training loop instead of reusing PEAL's ``ModelTrainer``, and
it assumes the four groups of a binary-label / binary-confounder setup
(``group = 2 * y + has_confounder``). Instead of minimising the average loss
it keeps an adversarial distribution over these groups, up-weights whichever
group currently performs worst, and so repairs a classifier that relies on a
confounder. The dataset must support ``enable_groups()`` so that every batch
carries ``has_confounder``.
"""

import copy
import os
from datetime import datetime
from typing import Union

from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import numpy as np

from peal.adaptors.interfaces import Adaptor, AdaptorConfig
from peal.architectures.interfaces import TaskConfig
from peal.data.dataloaders import create_dataloaders_from_datasource, get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig
from peal.global_utils import save_yaml_config
from peal.training.interfaces import TrainingConfig
from peal.log import get_logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotation only, keeps TensorFlow out of import
    from torch.utils.tensorboard import SummaryWriter

_log = get_logger(__name__)


class GroupDroLossComputer:
    """Group-robust loss plus the running per-group statistics it needs.

    Wraps a per-sample cross-entropy: the batch is split into the four groups,
    per-group mean losses and accuracies are computed, and the batch loss is
    the dot product of the group losses with an adversarial weight vector
    ``adv_probs`` that is updated multiplicatively (exponentiated gradient
    ascent with ``step_size``) so that hard groups get more weight. The
    running statistics double as the epoch's evaluation metrics, which is why
    the same class is also instantiated for the validation and test passes.

    Parameters
    ----------
    group_counts : torch.Tensor
        Number of training samples per group, shape ``(4,)``; used for the
        group fractions and the generalization adjustment.
    alpha : float
        Tail size of the "best-that-can-happen" (BTL) / greedy variant: the
        fraction of the data the worst-case average is taken over.
    gamma : float, optional
        Step size of the exponential moving average of the group losses,
        which the BTL variant ranks groups by. Default 0.1.
    adj : numpy.ndarray, optional
        Per-group generalization adjustment ``C_g``; added as
        ``adj / sqrt(group_counts)`` before the weight update. Default zeros.
    min_var_weight : float, optional
        Mixes the greedy weights back towards the empirical group fractions.
    step_size : float, optional
        Learning rate of the adversarial weight update. Default 0.01.
    normalize_loss : bool, optional
        Normalise the adjusted group losses to sum to one before the update.
    btl : bool, optional
        Use the BTL/greedy worst-case weighting instead of the exponentiated
        gradient weights.
    device : str, optional
        Device all running statistics live on.

    Attributes
    ----------
    n_groups : int
        Fixed to 4 (label x confounder).
    group_str : list of str
        Group names ``"<y>-<has_confounder>"`` used in the logged keys.
    adv_probs : torch.Tensor
        Current adversarial distribution over the groups.
    exp_avg_loss : torch.Tensor
        Exponential moving average of the per-group losses.
    avg_group_acc, avg_group_loss : torch.Tensor
        Running per-group averages since the last :meth:`reset_stats`.

    Notes
    -----
    ``adv_probs`` is never reset by :meth:`reset_stats`; it is meant to
    persist across epochs. It is also only applied when every entry of
    ``adj`` is positive, mirroring the reference implementation.
    """

    def __init__(
        self,
        group_counts,
        alpha,
        gamma=0.1,
        adj=None,
        min_var_weight=0,
        step_size=0.01,
        normalize_loss=False,
        btl=False,
        device="cpu",
    ):
        """Set up the per-sample criterion, the adjustment and the running stats."""
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.device = device

        self.n_groups = 4
        self.group_counts = group_counts
        self.group_frac = self.group_counts / self.group_counts.sum()
        self.group_str = ["0-0", "0-1", "1-0", "1-1"]

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().to(self.device)
        else:
            self.adj = torch.zeros(self.n_groups, device=device).float()

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups, device=device) / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups, device=device)
        self.exp_avg_initialized = torch.zeros(self.n_groups, device=device).byte()

        self.reset_stats()

    def loss(self, yhat, y, group_idx):
        """Group-robust loss of one batch, updating all running statistics.

        Parameters
        ----------
        yhat : torch.Tensor
            Logits of shape ``(N, C)``.
        y : torch.Tensor
            Integer labels of shape ``(N,)``.
        group_idx : torch.Tensor
            Group index in ``0..3`` per sample, shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Scalar loss to backpropagate: the group losses weighted by the
            (updated) adversarial distribution.
        """
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg(
            (torch.argmax(yhat, 1) == y).float(), group_idx
        )

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss)
        else:
            actual_loss, weights = self.compute_robust_loss(group_loss)

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss):
        """Exponentiated-gradient update of ``adv_probs`` and the weighted loss.

        Parameters
        ----------
        group_loss : torch.Tensor
            Mean loss per group, shape ``(4,)``.

        Returns
        -------
        tuple of torch.Tensor
            The scalar robust loss and the updated group weights.
        """
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss):
        """Robust loss ranking groups by their historical (moving-average) loss.

        Parameters
        ----------
        group_loss : torch.Tensor
            Mean loss per group of the current batch, shape ``(4,)``.

        Returns
        -------
        tuple of torch.Tensor
            The scalar robust loss and the weights used, in group order.
        """
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        """Average the ``alpha`` worst fraction of the data, ranked by ``ref_loss``.

        Groups are sorted by ``ref_loss`` descending and taken until their
        cumulative data fraction reaches ``alpha``; the group that straddles
        the boundary gets the remaining weight. The result is then mixed with
        the empirical group fractions according to ``min_var_weight``.

        Parameters
        ----------
        group_loss : torch.Tensor
            Mean loss per group of the current batch, shape ``(4,)``.
        ref_loss : torch.Tensor
            Scores the groups are ranked by, shape ``(4,)``.

        Returns
        -------
        tuple of torch.Tensor
            The scalar robust loss and the weights, unsorted back into group
            order.
        """
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (
            1 - self.min_var_weight
        )

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        """Mean of a per-sample quantity within each group, plus the group counts.

        Parameters
        ----------
        losses : torch.Tensor
            Per-sample values of shape ``(N,)`` (losses or 0/1 correctness).
        group_idx : torch.Tensor
            Group index per sample, shape ``(N,)``.

        Returns
        -------
        tuple of torch.Tensor
            Per-group mean and per-group sample count, both shape ``(4,)``.
            Empty groups get a mean of zero instead of a NaN.
        """
        # compute observed counts and mean loss for each group
        group_map = (
            group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(self.device)
        ).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        """Advance the moving average of the per-group losses by one batch.

        Only groups present in the batch are updated, and a group's first
        observation replaces the initial zero outright rather than being
        blended into it.

        Parameters
        ----------
        group_loss, group_count : torch.Tensor
            Output of :meth:`compute_group_avg` for this batch.
        """
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (
            self.exp_avg_initialized > 0
        ).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def reset_stats(self):
        """Zero the per-epoch running statistics.

        ``adv_probs`` and ``exp_avg_loss`` persist across epochs and are kept.
        """
        self.processed_data_counts = torch.zeros(self.n_groups, device=self.device)
        self.update_data_counts = torch.zeros(self.n_groups, device=self.device)
        self.update_batch_counts = torch.zeros(self.n_groups, device=self.device)
        self.avg_group_loss = torch.zeros(self.n_groups, device=self.device)
        self.avg_group_acc = torch.zeros(self.n_groups, device=self.device)
        self.avg_per_sample_loss = 0.0
        self.avg_actual_loss = 0.0
        self.avg_acc = 0.0
        self.batch_count = 0.0

    def update_stats(
        self, actual_loss, group_loss, group_acc, group_count, weights=None
    ):
        """Fold one batch into the running averages reported by :meth:`get_stats`.

        Per-group losses and accuracies are accumulated as sample-count
        weighted averages, the actual (robust) loss as a batch average, and
        the overall per-sample loss and accuracy as group-fraction weighted
        averages over the groups seen so far.

        Parameters
        ----------
        actual_loss : torch.Tensor
            The scalar robust loss of the batch.
        group_loss, group_acc, group_count : torch.Tensor
            Per-group mean loss, mean accuracy and sample count, shape
            ``(4,)``.
        weights : torch.Tensor, optional
            The group weights used; only their positivity is recorded, in
            ``update_data_counts`` and ``update_batch_counts``.
        """
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = (
            prev_weight * self.avg_group_loss + curr_weight * group_loss
        )

        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (
            1 / denom
        ) * actual_loss

        # counts
        self.processed_data_counts += group_count
        self.update_data_counts += group_count * ((weights > 0).float())
        self.update_batch_counts += ((group_count * weights) > 0).float()
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, weight_decay, stats_dict):
        """Add the squared parameter norm and the implied L2 penalty to ``stats_dict``.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose parameters are summed over.
        weight_decay : float
            Coefficient used for the reported ``reg_loss``.
        stats_dict : dict
            Dictionary that is updated in place.

        Returns
        -------
        dict
            The same dictionary, with ``model_norm_sq`` and ``reg_loss``.
        """
        model_norm_sq = 0.0
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict["model_norm_sq"] = model_norm_sq.item()
        stats_dict["reg_loss"] = weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, weight_decay=None):
        """Collect the running statistics as a flat dict for TensorBoard.

        Contains per-group average loss, moving-average loss, accuracy and
        update-batch count keyed by ``<name>_group:<y>-<confounder>``, plus
        the aggregates ``avg_actual_loss``, ``avg_per_sample_loss``,
        ``avg_acc``, ``avg_group_acc`` and ``worst_group_acc``.

        Parameters
        ----------
        model : torch.nn.Module, optional
            When given, :meth:`get_model_stats` is appended; then
            ``weight_decay`` is required.
        weight_decay : float, optional
            Coefficient for the reported regularisation loss.

        Returns
        -------
        dict of str to float
        """
        stats_dict = {}
        for idx in range(self.n_groups):
            group_str = self.group_str[idx]
            stats_dict[f"avg_loss_group:{group_str}"] = self.avg_group_loss[idx].item()
            stats_dict[f"exp_avg_loss_group:{group_str}"] = self.exp_avg_loss[
                idx
            ].item()
            stats_dict[f"avg_acc_group:{group_str}"] = self.avg_group_acc[idx].item()
            stats_dict[f"update_batch_count_group:{group_str}"] = (
                self.update_batch_counts[idx].item()
            )

        stats_dict["avg_actual_loss"] = self.avg_actual_loss.item()
        stats_dict["avg_per_sample_loss"] = self.avg_per_sample_loss.item()
        stats_dict["avg_acc"] = self.avg_acc.item()
        stats_dict["avg_group_acc"] = torch.mean(self.avg_group_acc).item()
        stats_dict["worst_group_acc"] = torch.min(self.avg_group_acc).item()

        # Model stats
        if model is not None:
            assert weight_decay is not None
            stats_dict = self.get_model_stats(model, weight_decay, stats_dict)

        return stats_dict


class GroupDROv2Config(AdaptorConfig):
    """Config for :class:`GroupDROv2`.

    Parameters
    ----------
    model_path : str
        ``torch.load``-able classifier to repair.
    base_dir : str
        Run directory; an existing one is renamed with a timestamp suffix.
    data : DataConfig
        Training/validation data, which must provide group labels.
    unpoisoned_data : DataConfig, optional
        Second dataset whose test split is used for the unbiased evaluation.
    training : TrainingConfig
        Supplies ``learning_rate``, ``max_epochs``, ``test_batch_size`` and
        ``early_stopping_goal`` (``"worst_group_accuracy"`` or
        ``"average_group_accuracy"``).
    task : TaskConfig
        Task of the classifier, passed on to the test dataloader.
    alpha : float, optional
        Tail fraction of the BTL/greedy weighting. Default 0.2.
    generalization_adjustment : float or list of float, optional
        Per-group adjustment ``C_g``; a scalar is broadcast to all 4 groups.
    automatic_adjustment : bool, optional
        Re-estimate the adjustment after every epoch from the train/validation
        generalization gap.
    robust_step_size : float, optional
        Step size of the adversarial weight update. Default 0.01.
    use_normalized_loss, btl, gamma, minimum_variational_weight
        Forwarded to :class:`GroupDroLossComputer`.
    weight_decay : float, optional
        L2 penalty of the SGD optimizer. Default 5e-5.
    track_test_acc : int, optional
        Evaluate on the unpoisoned test split every ``n`` epochs; ``None``
        disables the intermediate test passes.
    save_intermediate : bool, optional
        Write ``checkpoints/model_epoch<n>.cpl`` after every epoch.
    """

    __name__: str = "peal.AdaptorConfig"
    model_path: str
    base_dir: str
    data: DataConfig
    unpoisoned_data: DataConfig = None
    training: TrainingConfig
    task: TaskConfig
    alpha: float = 0.2
    generalization_adjustment: Union[float, list[float]] = 0.0
    automatic_adjustment: bool = False
    robust_step_size: float = 0.01
    use_normalized_loss: bool = False
    btl: bool = False
    weight_decay: float = 5e-5
    gamma: float = 0.1
    minimum_variational_weight: float = 0.0
    track_test_acc: int = None
    save_intermediate: bool = True


class GroupDROv2(Adaptor):
    """Adaptor that retrains a classifier with the group-robust objective.

    Parameters
    ----------
    adaptor_config : GroupDROv2Config
        See :class:`GroupDROv2Config`.

    Attributes
    ----------
    original_model : torch.nn.Module
        The loaded classifier, kept untouched.
    model : torch.nn.Module
        The deep copy that is actually retrained.
    train_group_counts, val_group_counts, test_group_counts : torch.Tensor
        Samples per group, counted once by :func:`compute_group_sizes`.
    test_data_unpoisoned : DataLoader or None
        Test loader built from ``config.unpoisoned_data``.
    """

    def __init__(self, adaptor_config: GroupDROv2Config):
        """Prepare the run directory, load the model and count the group sizes.

        Seeds torch, moves an existing ``base_dir`` aside with a timestamp
        suffix, writes ``config.yaml`` into the fresh one, loads the
        classifier, builds the train/validation dataloaders with groups
        enabled and dict-style batches, and optionally the unpoisoned test
        loader. Counting the group sizes walks each loader once.
        """
        self.config = adaptor_config
        torch.manual_seed(self.config.seed)

        if os.path.isdir(adaptor_config.base_dir):
            dest = f"{adaptor_config.base_dir}_old_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            os.rename(adaptor_config.base_dir, dest)
        assert not os.path.exists(adaptor_config.base_dir)
        os.makedirs(adaptor_config.base_dir)

        save_yaml_config(
            self.config, os.path.join(adaptor_config.base_dir, "config.yaml")
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _log.info("%s %s", "running on device: ", self.device)
        self.original_model = torch.load(
            self.config.model_path, map_location=self.device, weights_only=False
        )
        self.model = copy.deepcopy(self.original_model)

        self.train_dataloader, self.val_dataloader, _ = (
            create_dataloaders_from_datasource(self.config)
        )
        self.train_dataloader.dataset.return_dict = True
        self.train_dataloader.dataset.enable_groups()
        self.train_group_counts = torch.as_tensor(
            compute_group_sizes(self.train_dataloader), device=self.device
        )
        self.val_dataloader.dataset.return_dict = True
        self.val_dataloader.dataset.enable_groups()
        self.val_group_counts = torch.as_tensor(
            compute_group_sizes(self.val_dataloader), device=self.device
        )

        self.test_data_unpoisoned = None
        self.test_group_counts = None
        if self.config.unpoisoned_data is not None:
            self.test_data_unpoisoned = get_datasets(
                self.config.unpoisoned_data, return_dict=True
            )[-1]
            self.test_data_unpoisoned.enable_groups()
            self.test_data_unpoisoned = get_dataloader(
                self.test_data_unpoisoned,
                mode="test",
                batch_size=self.config.training.test_batch_size,
                task_config=self.config.task,
            )
            self.test_group_counts = torch.as_tensor(
                compute_group_sizes(self.test_data_unpoisoned), device=self.device
            )

    def run(self):
        """Entry point of the adaptor interface; runs :meth:`train`."""
        self.train()

    def train(self):
        """Retrain the model with the group-robust objective and evaluate it.

        Runs one validation (and, if configured, test) pass at epoch ``-1`` as
        a baseline, then ``training.max_epochs`` train/validation rounds with
        SGD (momentum 0.9, ``config.weight_decay``). The epoch's validation
        score is the worst or the average group accuracy depending on
        ``training.early_stopping_goal``; a new best writes ``model.cpl`` into
        ``base_dir``. With ``automatic_adjustment`` the generalization
        adjustment is re-estimated each epoch as the train/validation gap
        scaled by ``sqrt(group_counts)``. Finally the best checkpoint is
        scored on the unpoisoned test split and its average and worst group
        accuracy are logged as ``test_final_*``.

        Side effects: TensorBoard logs under ``<base_dir>/logs``, per-epoch
        checkpoints under ``<base_dir>/checkpoints`` when
        ``config.save_intermediate``.

        Raises
        ------
        NotImplementedError
            If ``training.early_stopping_goal`` is neither
            ``"worst_group_accuracy"`` nor ``"average_group_accuracy"``.
        """
        from torch.utils.tensorboard import SummaryWriter

        log_writer = SummaryWriter(log_dir=os.path.join(self.config.base_dir, "logs"))

        adjustments = (
            [self.config.generalization_adjustment]
            if isinstance(self.config.generalization_adjustment, float)
            else self.config.generalization_adjustment
        )
        assert len(adjustments) in (1, 4)
        if len(adjustments) == 1:
            adjustments = np.array(adjustments * 4)
        else:
            adjustments = np.array(adjustments)

        train_loss_computer = GroupDroLossComputer(
            self.train_group_counts,
            self.config.alpha,
            gamma=self.config.gamma,
            adj=adjustments,
            min_var_weight=self.config.minimum_variational_weight,
            step_size=self.config.robust_step_size,
            normalize_loss=self.config.use_normalized_loss,
            btl=self.config.btl,
            device=self.device,
        )

        self.model.train()
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.training.learning_rate,
            momentum=0.9,
            weight_decay=self.config.weight_decay,
        )

        best_val_acc = 0
        best_epoch = -1
        checkpoint_dir = os.path.join(self.config.base_dir, "checkpoints")
        os.makedirs(checkpoint_dir)

        val_loss_computer = GroupDroLossComputer(
            self.val_group_counts,
            self.config.alpha,
            step_size=self.config.robust_step_size,
            device=self.device,
        )
        self.run_epoch(
            -1,
            self.model,
            optimizer,
            self.val_dataloader,
            val_loss_computer,
            log_writer,
            mode="val",
        )

        if self.test_data_unpoisoned is not None:
            test_loss_computer = GroupDroLossComputer(
                self.test_group_counts,
                self.config.alpha,
                step_size=self.config.robust_step_size,
                device=self.device,
            )
            self.run_epoch(
                -1,
                self.model,
                optimizer,
                self.test_data_unpoisoned,
                test_loss_computer,
                log_writer,
                mode="test",
            )

        for epoch in range(self.config.training.max_epochs):
            _log.info("%s", f"epoch {epoch}/{self.config.training.max_epochs}")
            self.run_epoch(
                epoch,
                self.model,
                optimizer,
                self.train_dataloader,
                train_loss_computer,
                log_writer=log_writer,
                mode="train",
            )

            val_loss_computer = GroupDroLossComputer(
                self.val_group_counts,
                self.config.alpha,
                step_size=self.config.robust_step_size,
                device=self.device,
            )
            self.run_epoch(
                epoch,
                self.model,
                optimizer,
                self.val_dataloader,
                val_loss_computer,
                log_writer=log_writer,
                mode="val",
            )

            if (
                self.test_data_unpoisoned is not None
                and self.config.track_test_acc is not None
                and (epoch + 1) % self.config.track_test_acc == 0
            ):
                test_loss_computer = GroupDroLossComputer(
                    self.test_group_counts,
                    self.config.alpha,
                    step_size=self.config.robust_step_size,
                    device=self.device,
                )
                self.run_epoch(
                    epoch,
                    self.model,
                    optimizer,
                    self.test_data_unpoisoned,
                    test_loss_computer,
                    log_writer=log_writer,
                    mode="test",
                )

            if self.config.save_intermediate:
                torch.save(
                    self.model.to("cpu"),
                    os.path.join(checkpoint_dir, f"model_epoch{epoch}.cpl"),
                )

            if self.config.training.early_stopping_goal == "worst_group_accuracy":
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            elif self.config.training.early_stopping_goal == "average_group_accuracy":
                curr_val_acc = torch.mean(val_loss_computer.avg_group_acc).item()
            else:
                raise NotImplementedError

            # if args.reweight_groups:
            #     curr_val_acc = min(val_loss_computer.avg_group_acc)
            # else:
            #     curr_val_acc = val_loss_computer.avg_acc
            _log.info(
                "%s",
                f"Current validation accuracy: {curr_val_acc} (best={best_val_acc} after epoch {best_epoch})",
            )

            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                best_epoch = epoch
                torch.save(
                    self.model.to("cpu"),
                    os.path.join(self.config.base_dir, "model.cpl"),
                )

            if self.config.automatic_adjustment:
                gen_gap = (
                    val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
                )
                adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
                train_loss_computer.adj = adjustments.detach()
                for group_idx in range(train_loss_computer.n_groups):
                    log_writer.add_scalar(
                        f"group_{train_loss_computer.group_str[group_idx]}_adj",
                        train_loss_computer.adj[group_idx],
                        epoch,
                    )

            self.model.to(self.device)

        if self.test_data_unpoisoned is not None:
            test_loss_computer = GroupDroLossComputer(
                self.test_group_counts,
                self.config.alpha,
                step_size=self.config.robust_step_size,
                device=self.device,
            )
            best_model = torch.load(
                os.path.join(self.config.base_dir, "model.cpl"),
                map_location=self.device,
                weights_only=False,
            )
            if self.config.track_test_acc is not None:
                self.run_epoch(
                    best_epoch,
                    best_model,
                    optimizer,
                    self.test_data_unpoisoned,
                    test_loss_computer,
                    mode="test",
                )
            else:
                self.run_epoch(
                    best_epoch,
                    self.model,
                    optimizer,
                    self.test_data_unpoisoned,
                    test_loss_computer,
                    log_writer,
                    mode="test",
                )
            stats = test_loss_computer.get_stats()
            log_writer.add_scalar(
                "test_final_avg_group_acc", stats["avg_group_acc"], best_epoch
            )
            log_writer.add_scalar(
                "test_final_worst_group_acc", stats["worst_group_acc"], best_epoch
            )

        log_writer.close()

    def run_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        loss_computer: GroupDroLossComputer,
        log_writer: "SummaryWriter" = None,
        mode: str = "train",
    ):
        """Run one pass over ``loader``, training or evaluating.

        Each batch is expected as a dict with ``x``, ``y`` and
        ``has_confounder``; the group index is ``2 * y + has_confounder``.
        Gradients are only enabled, and the optimizer only steps, in
        ``"train"`` mode. After the pass the loss computer's statistics are
        written to TensorBoard under the ``<mode>_`` prefix, and reset for the
        next training epoch (the evaluation computers are short-lived and
        discarded by the caller instead).

        Parameters
        ----------
        epoch : int
            Step the statistics are logged at; ``-1`` for the baseline pass.
        model : torch.nn.Module
            Model to train or evaluate.
        optimizer : torch.optim.Optimizer
            Only used in ``"train"`` mode.
        loader : torch.utils.data.DataLoader
            Loader yielding dict batches with group information.
        loss_computer : GroupDroLossComputer
            Computes the loss and accumulates the statistics.
        log_writer : torch.utils.tensorboard.SummaryWriter, optional
            When ``None``, nothing is logged.
        mode : str, optional
            ``"train"``, ``"val"`` or ``"test"``.
        """
        is_training = mode == "train"
        if is_training:
            model.train()
        else:
            model.eval()

        with torch.set_grad_enabled(is_training):
            for batch_idx, batch in enumerate(tqdm(loader)):

                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device).squeeze().long()
                g = batch["has_confounder"].to(self.device).squeeze().long()
                g = 2 * y + g

                outputs = model(x)

                loss_main = loss_computer.loss(outputs, y, g)

                if is_training:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

        if log_writer is not None:
            for key, val in (
                loss_computer.get_stats(model, self.config.weight_decay)
                if is_training
                else loss_computer.get_stats()
            ).items():
                log_writer.add_scalar(f"{mode}_{key}", val, epoch)
        if is_training:
            loss_computer.reset_stats()


def compute_group_sizes(dataloader) -> list[int]:
    """Count the samples per (label, confounder) group in a dataloader.

    Walks the whole loader once, so this is as expensive as one epoch of data
    loading. The batches must be dicts carrying ``y`` and ``has_confounder``,
    both binary.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Loader over a dataset with ``enable_groups()`` active.

    Returns
    -------
    list of int
        Counts in the order ``[y=0/no, y=0/yes, y=1/no, y=1/yes]``, i.e.
        indexed by ``2 * y + has_confounder``.
    """
    group_sizes = [0, 0, 0, 0]
    with tqdm(dataloader) as pbar:
        pbar.set_description("determining group sizes...")
        for batch in pbar:
            y = batch["y"].int()
            for i, has_confounder in enumerate(batch["has_confounder"].int()):
                has_confounder = has_confounder.item()
                if y[i].item() == 0:
                    if has_confounder == 0:
                        group_sizes[0] += 1
                    elif has_confounder == 1:
                        group_sizes[1] += 1
                elif y[i].item() == 1:
                    if has_confounder == 0:
                        group_sizes[2] += 1
                    elif has_confounder == 1:
                        group_sizes[3] += 1
    _log.info("%s %s", "group sizes: ", group_sizes)
    return group_sizes
