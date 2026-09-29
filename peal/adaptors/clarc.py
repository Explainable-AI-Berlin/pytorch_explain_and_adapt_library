"""Class Artifact Compensation (ClArC) adaptors for removing known shortcuts.

ClArC (Anders et al.) corrects a classifier that relies on an artifact (a
spurious "confounder" feature) by estimating a Concept Activation Vector (CAV)
for the artifact from group-annotated activations and then either

* projecting the artifact direction out of the activations at a chosen layer
  (:class:`PClArC`, optionally followed by fine-tuning the downstream head), or
* fine-tuning the downstream head with a right-for-the-right-reasons penalty
  that pushes the input gradient orthogonal to the CAV (:class:`RRClArC`).

Both are PEAL adaptors driven by a :class:`ClArCConfig`: they load a trained
predictor, sweep ``layer_index`` x ``correction_strength``, evaluate group
accuracies on the validation (and optionally an unpoisoned test) split, write
one csv per evaluation dataset plus ``best_model_result.txt`` into
``base_dir`` and save every corrected model as ``*.cpl``. Module-level helpers
split a model into feature extractor / head and compute pattern or SVM CAVs.
"""

import copy
import math
import os
import pathlib
import sys
import traceback
from collections import defaultdict, namedtuple

from sklearn.svm import LinearSVC
from torch.nn import Module, CrossEntropyLoss, Sequential
from torch.utils.data import DataLoader
from torchvision.models import ResNet
from tqdm import tqdm

from peal.adaptors.interfaces import AdaptorConfig, Adaptor
from peal.architectures.interfaces import TaskConfig
from peal.data.dataloaders import create_dataloaders_from_datasource, get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig
from peal.training.interfaces import TrainingConfig

import torch
import numpy as np
import pandas as pd
from peal.log import get_logger

_log = get_logger(__name__)


class ClArCConfig(AdaptorConfig):
    """Configuration shared by the ClArC adaptors.

    Parameters
    ----------
    model_path : str
        ``torch.load``-able predictor; a wrapping ``.model`` attribute is
        unwrapped.
    base_dir : str
        Output directory for csv results, corrected models and logs.
    data : DataConfig
        Training / validation data with ``has_confounder`` group labels; if
        ``data.spray_label_file`` is set the labels are treated as SpRAy
        (estimated) rather than true annotations.
    unpoisoned_data : DataConfig, optional
        Extra clean test set evaluated as ``"unpoisoned-test"``.
    training : TrainingConfig
        Supplies ``learning_rate``, ``max_epochs`` and ``test_batch_size``.
    task : TaskConfig
        Task description handed to the dataloader factory.
    projection_type : str, default "pcav"
        CAV estimator: ``"pcav"`` (pattern CAV), ``"svm"`` (LinearSVC) or
        ``"simple"`` (mean difference projection, P-ClArC only).
    layer_index : list of int, default [-1]
        Layers (indices into the flattened child list) at which to correct;
        ``0`` means the input.
    correction_strength : list of float, default [1]
        Scaling of the projection (P-ClArC) or RR loss weight (RR-ClArC).
    attacked_class : int, default 0
        Class whose samples are used to estimate the CAV; None uses all.
    cav_mode : str, optional
        How conv activations are pooled before the CAV: ``"cavs_max"``,
        ``"cavs_mean"`` or None (flatten everything).
    save_model : bool, default True
        Save every corrected model into ``base_dir``.
    max_samples : int, default 999999
        Cap on confounder / non-confounder activations collected per group.
    reverse_cav_direction : bool, default False
        Swap the group labels before estimating the CAV.
    """

    __name__: str = "peal.AdaptorConfig"
    category: str = "adaptor"
    model_path: str
    base_dir: str
    data: DataConfig
    unpoisoned_data: DataConfig = None
    training: TrainingConfig
    task: TaskConfig
    projection_type: str = "pcav"
    layer_index: list[int] = [-1]
    correction_strength: list[float] = [1]
    attacked_class: int = 0
    cav_mode: str = None
    save_model: bool = True
    max_samples: int = 999999
    reverse_cav_direction: bool = False


class PClArCConfig(ClArCConfig):
    """Config of :class:`PClArC`.

    Parameters
    ----------
    finetune : bool, default False
        Fine-tune the downstream head after inserting the projection layer.
    """

    adaptor_type: str = "PClArC"
    finetune: bool = False


class RRClArCConfig(ClArCConfig):
    """Config of :class:`RRClArC`.

    Parameters
    ----------
    gradient_target : str, default "all"
        Which logits are differentiated for the RR penalty: ``"max"``,
        ``"attacked_class"``, ``"all"`` or ``"all_random"`` (random signs).
    mean_grad : bool, default False
        Spatially average the gradient before projecting it on the CAV.
    rrc_loss : str, default "l2"
        Penalty form: ``"l2"`` (squared projection) or ``"cosine"``.
    """

    adaptor_type: str = "RRClArC"
    gradient_target: str = "all"
    mean_grad: bool = False
    rrc_loss: str = "l2"


class ClArC(Adaptor):
    """Base class of the ClArC adaptors: data setup, sweep loop and evaluation.

    Subclasses implement :meth:`_run` (one correction for a given layer and
    strength) and :meth:`get_evaluation_filename`.

    Parameters
    ----------
    adaptor_config : ClArCConfig
        See :class:`ClArCConfig`.

    Attributes
    ----------
    original_model : torch.nn.Module
        The loaded, uncorrected predictor; ``model`` is a deep copy that is
        reset after every sweep step.
    train_dataloader, val_dataloader, test_dataloader
        Built with ``create_dataloaders_from_datasource``; train and val return
        dict batches with ``x``, ``y`` and ``has_confounder`` and the train set
        is restricted to ``attacked_class`` when that is set.
    test_data_unpoisoned : DataLoader or None
        Test loader of ``config.unpoisoned_data``.
    cav_cache : CavCache
        CAV, annotations and activations of the last processed layer so a
        layer is only encoded once across correction strengths.
    """

    def __init__(self, adaptor_config: ClArCConfig):
        """Load the model, build the dataloaders and create ``base_dir``."""
        pathlib.Path(adaptor_config.base_dir).mkdir(exist_ok=True)

        self.config = adaptor_config
        torch.manual_seed(self.config.seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _log.info("%s %s", "running on device: ", self.device)
        self.original_model = torch.load(
            self.config.model_path, map_location=self.device, weights_only=False
        )
        if hasattr(self.original_model, "model"):
            self.original_model = self.original_model.model
        self.model = copy.deepcopy(self.original_model)
        self.attacked_class = adaptor_config.attacked_class
        self.use_perfect_annotations = (
            True if self.config.data.spray_label_file is not None else False
        )

        self.cav_cache = CavCache("", "", "", "")

        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            create_dataloaders_from_datasource(self.config)
        )
        self.train_dataloader.dataset.return_dict = True
        self.train_dataloader.dataset.url_enabled = True
        self.train_dataloader.dataset.enable_groups()
        self.val_dataloader.dataset.return_dict = True
        self.val_dataloader.dataset.url_enabled = True
        self.val_dataloader.dataset.enable_groups()
        # self.test_dataloader.dataset.return_dict = True
        # self.test_dataloader.dataset.url_enabled = True
        # self.test_dataloader.dataset.enable_groups()
        self.train_dataloader.dataset.disable_class_restriction()
        self.val_dataloader.dataset.disable_class_restriction()
        # self.test_dataloader.dataset.disable_class_restriction()
        if self.config.attacked_class is not None:
            self.train_dataloader.dataset.enable_class_restriction(
                self.config.attacked_class
            )

        self.test_data_unpoisoned = None
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

    def run(self):
        """Sweep all layers and correction strengths and record the results.

        Evaluates the uncorrected model first, then every
        ``(layer_index, correction_strength)`` pair via :meth:`_run`, keeping
        the model with the best ``avg_group_acc`` on ``original-val``. Writes
        one csv per evaluation dataset (named by
        :meth:`get_evaluation_filename`) and ``best_model_result.txt`` into
        ``config.base_dir``.
        """

        eval_dataloaders = [("original-val", self.val_dataloader)]
        # eval_dataloaders.append(("original-test", self.test_dataloader))
        evaluation = {"original-val": defaultdict(list)}
        # evaluation["original-test"] = defaultdict(list)

        if self.test_data_unpoisoned is not None:
            eval_dataloaders.append(("unpoisoned-test", self.test_data_unpoisoned))
            evaluation["unpoisoned-test"] = defaultdict(list)

        self.model.eval()
        for description, dataloader in eval_dataloaders:
            evaluation[description]["projection_location"].append("uncorrected")
            evaluation[description]["correction_strength"].append("uncorrected")
            evaluation[description]["epochs_finetuned"].append("uncorrected")
            for k, v in self.get_stats(dataloader, self.model).items():
                evaluation[description][k].append(v)

        best_model = (self.model, 0, "uncorrected", "uncorrected")
        for layer in self.config.layer_index:
            for cs in self.config.correction_strength:
                model, number_epochs_finetuned = self._run(
                    layer_index=layer, correction_strength=cs
                )
                model.eval()
                for description, dataloader in eval_dataloaders:
                    evaluation[description]["projection_location"].append(layer)
                    evaluation[description]["correction_strength"].append(cs)
                    evaluation[description]["epochs_finetuned"].append(
                        number_epochs_finetuned
                    )

                    for k, v in self.get_stats(dataloader, model).items():
                        if k == "accuracy":
                            _log.info("%s %s", "accuracy: ", v)
                        evaluation[description][k].append(v)

                current_acc = evaluation["original-val"]["avg_group_acc"][-1]
                if current_acc > best_model[1]:
                    best_model = (model, current_acc, layer, cs)

                self.model = copy.deepcopy(self.original_model)

        for dataset_name, results in evaluation.items():
            filename = self.get_evaluation_filename(dataset_name)
            results = pd.DataFrame(results)
            results.fillna("empty", inplace=True)
            results.to_csv(os.path.join(self.config.base_dir, filename), index=False)

        result = f"best model stats (layer={best_model[2]}, correction_strength={best_model[3]}):"
        for description, dataloader in eval_dataloaders:
            result += f"\n\n{description}:\n"
            model_stats = self.get_stats(dataloader, best_model[0])
            result += f"c0-nonconfounder accuracy: {model_stats.get('c0_non-artifact_accuracy', '---')}\n"
            result += f"c0-confounder accuracy: {model_stats.get('c0_artifact_accuracy', '---')}\n"
            result += f"c1-nonconfounder accuracy: {model_stats.get('c1_non-artifact_accuracy', '---')}\n"
            result += f"c1-confounder accuracy: {model_stats.get('c1_artifact_accuracy', '---')}\n"
            result += (
                f"average group accuracy: {model_stats.get('avg_group_acc', '---')}\n"
            )
            result += (
                f"worst group accuracy: {model_stats.get('worst_group_acc', '---')}"
            )

        _log.info("%s", result)
        with open(
            os.path.join(self.config.base_dir, "best_model_result.txt"), "w"
        ) as result_file:
            result_file.write(result)

    def _run(self, *args, **kwargs) -> (Module, int):
        """Apply one correction; return ``(corrected_model, epochs_finetuned)``."""

    def get_evaluation_filename(self, dataset_name: str) -> str:
        """Return the csv file name for the results on ``dataset_name``."""

    def get_annotations_and_activations(
        self, feature_extractor: Module = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect pooled activations and artifact labels from the train loader.

        Parameters
        ----------
        feature_extractor : torch.nn.Module, optional
            Prefix of the model up to the correction layer; None means the raw
            input is used as activation.

        Returns
        -------
        activations : torch.Tensor
            Shape ``(N, D)``; non-confounder samples first, then confounder
            samples, each group capped at ``config.max_samples``. Pooling
            follows ``config.cav_mode``.
        annotations : torch.Tensor
            Shape ``(N,)`` with 0 for non-confounder and 1 for confounder
            (flipped if ``config.reverse_cav_direction``).
        """

        confounders = []
        non_confounders = []
        for it, batch in enumerate(self.train_dataloader):
            x = batch["x"].to(self.device)
            activation = (
                x if feature_extractor is None else feature_extractor(x).detach()
            )

            if self.config.cav_mode == "cavs_max":
                activation = activation.flatten(start_dim=2).max(2).values
            elif self.config.cav_mode == "cavs_mean":
                activation = activation.mean((2, 3))
            else:
                activation = activation.flatten(start_dim=1)

            group_labels = batch["has_confounder"].squeeze()

            if self.config.reverse_cav_direction:
                group_labels = group_labels * -1 + 1

            if len(non_confounders) < self.config.max_samples:
                non_confounders.extend(activation[group_labels == 0])
            if len(confounders) < self.config.max_samples:
                confounders.extend(activation[group_labels == 1])

        confounders = torch.stack(confounders)
        non_confounders = torch.stack(non_confounders)
        activations = torch.cat((non_confounders, confounders))
        annotations = torch.cat(
            (torch.zeros(len(non_confounders)), torch.ones(len(confounders)))
        ).to(self.device)

        num_artifact_samples = torch.sum(annotations == 1).item()
        _log.info(
            "%s",
            f"Number of artifact samples: {num_artifact_samples}; Number of non-artifact samples: {len(annotations) - num_artifact_samples}",
        )
        return activations, annotations

    @torch.no_grad()
    def get_stats(self, dataloader: DataLoader, model: Module) -> dict:
        """Compute accuracy per class and artifact group on a dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            Yields dict batches with ``x``, ``y`` and ``has_confounder``.
        model : torch.nn.Module
            Model to evaluate (switched to eval mode).

        Returns
        -------
        dict
            ``n``, ``artifact_freq``, ``accuracy``, ``artifact_accuracy``,
            ``non-artifact_accuracy`` plus per class ``c{y}_n``,
            ``c{y}_artifact_freq``, ``c{y}_accuracy``,
            ``c{y}_artifact_accuracy``, ``c{y}_non-artifact_accuracy``, and the
            aggregates ``avg_group_acc`` / ``worst_group_acc`` over all
            (class, artifact) groups that are non-empty.
        """
        model.eval()
        annotations = []
        targets = []
        acc = []
        with tqdm(dataloader) as pbar:
            pbar.set_description("evaluation")
            for batch in pbar:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device).squeeze()
                targets.append(y)
                annotations.append(batch["has_confounder"].squeeze())

                prediction = model(x)
                acc.append((prediction.argmax(dim=1) == y).to(torch.int))

        annotations = torch.cat(annotations, dim=0).to(self.device).int()
        targets = torch.cat(targets, dim=0).to(self.device).int()
        acc = torch.cat(acc, dim=0).to(self.device).float()

        results = {
            "n": len(targets),
            "artifact_freq": (
                annotations.sum().item() / len(annotations)
                if len(targets) == len(annotations)
                else "---"
            ),
            "accuracy": acc.mean().item(),
            "avg_group_acc": "empty",
            "worst_group_acc": "empty",
            "artifact_accuracy": acc[annotations == 1].mean().item(),
            "non-artifact_accuracy": acc[annotations == 0].mean().item(),
        }

        group_accuracies = []
        for y in torch.unique(targets):
            idx = targets == y
            results[f"c{y.item()}_n"] = len(targets[idx])
            results[f"c{y.item()}_artifact_freq"] = (
                annotations[idx].sum().item() / len(annotations[idx])
                if len(targets) == len(annotations)
                else "---"
            )
            results[f"c{y.item()}_accuracy"] = acc[idx].mean().item()
            results[f"c{y.item()}_artifact_accuracy"] = (
                acc[(annotations == 1) * idx].mean().item()
            )
            group_accuracies.append(results[f"c{y.item()}_artifact_accuracy"])
            results[f"c{y.item()}_non-artifact_accuracy"] = (
                acc[(annotations == 0) * idx].mean().item()
            )
            group_accuracies.append(results[f"c{y.item()}_non-artifact_accuracy"])
            # print(f"class {y}: {results[f'c{y.item()}_n']} items, artfiact freq: {results[f'c{y.item()}_artifact_freq']}")

        _log.info("%s %s", "group accuracies:", group_accuracies)
        group_accuracies = [num for num in group_accuracies if not math.isnan(num)]
        results["avg_group_acc"] = np.mean(group_accuracies).item()
        results["worst_group_acc"] = np.min(group_accuracies).item()
        return results


class PClArC(ClArC):
    """Projective ClArC: remove the CAV direction from activations at a layer.

    Inserts a :class:`CavProjection` (or :class:`SimpleProjection`) between the
    feature extractor and the downstream head, optionally fine-tunes the head
    with SGD on cross entropy, and saves the corrected ``nn.Sequential``.

    Parameters
    ----------
    adaptor_config : PClArCConfig
        See :class:`PClArCConfig`.
    """

    def __init__(self, adaptor_config: PClArCConfig):
        super().__init__(adaptor_config)
        self.adaptor_config = adaptor_config

    def get_evaluation_filename(self, dataset_name: str) -> str:
        """Build ``correction_<dataset>-dataset_<labels>_<proj>...csv``."""
        attacked_class = (
            f"_attacked-c{self.adaptor_config.attacked_class}"
            if self.adaptor_config.attacked_class is not None
            else ""
        )
        label_type = (
            "true" if self.config.data.spray_label_file is None else "spray"
        ) + "-group-labels"
        finetune = (
            f"_{self.config.training.max_epochs}epochs-finetune"
            if self.adaptor_config.finetune
            else ""
        )
        return f"correction_{dataset_name}-dataset_{label_type}_{self.adaptor_config.projection_type}-projection_mode-{self.adaptor_config.cav_mode}{attacked_class}{finetune}.csv"

    def run(self, *args, **kwargs):
        """Run the sweep of :meth:`ClArC.run`; extra arguments are ignored."""
        super().run()

    def _run(
        self, layer_index: int = -1, correction_strength: float = 1.0, **kwargs
    ) -> (Module, int):
        """Correct the model at one layer with one strength.

        Parameters
        ----------
        layer_index : int, default -1
            Split point for :func:`split_model`; ``0`` projects the input.
        correction_strength : float, default 1.0
            Scaling of the CAV inside :class:`CavProjection`.

        Returns
        -------
        model : torch.nn.Module
            ``Sequential(feature_extractor, projection, head)``; also saved to
            ``base_dir/corrected_model_*.cpl`` when ``save_model`` is set.
        number_epochs_finetuned : int
            Epoch of the best fine-tuned head, 0 without fine-tuning.
        """
        torch.manual_seed(self.config.seed)

        _log.info(
            "%s",
            f"\n\nperforming p-clarc in layer {layer_index} with correction strength {correction_strength} and projection type {self.adaptor_config.projection_type}",
        )
        self.model.eval()

        feature_extractor, downstream_head = None, None
        if layer_index != 0:
            feature_extractor, downstream_head = split_model(
                self.model, layer_index, self.device
            )

        if layer_index != self.cav_cache.layer:
            activations, annotations = self.get_annotations_and_activations(
                feature_extractor=feature_extractor
            )
            cav = calculate_cav(
                activations, annotations.clone(), self.adaptor_config.projection_type
            ).to(self.device, dtype=activations.dtype)
            self.cav_cache = CavCache(
                layer=layer_index,
                cav=cav,
                annotations=annotations,
                activations=activations,
            )

        projection = (
            SimpleProjection
            if self.adaptor_config.projection_type == "simple"
            else CavProjection
        )
        projection = projection(
            self.cav_cache.activations,
            self.cav_cache.annotations,
            cav=self.cav_cache.cav,
            cav_mode=self.adaptor_config.cav_mode,
            correction_strength=correction_strength,
        )

        number_epochs_finetuned = 0
        if layer_index != 0:
            if self.adaptor_config.finetune:
                downstream_head, number_epochs_finetuned = self.finetune(
                    projection, downstream_head, feature_extractor=feature_extractor
                )
            self.model = torch.nn.Sequential(
                feature_extractor, projection, downstream_head
            )
        else:
            if self.adaptor_config.finetune:
                self.model, number_epochs_finetuned = self.finetune(
                    projection, self.model
                )
            self.model = torch.nn.Sequential(projection, self.model)

        if self.adaptor_config.save_model:
            attacked_class = (
                f"_attacked-c{self.adaptor_config.attacked_class}"
                if self.adaptor_config.attacked_class is not None
                else ""
            )
            filename = f"corrected_model{attacked_class}_{self.adaptor_config.projection_type}{layer_index}_mode-{self.adaptor_config.cav_mode}_cs{correction_strength}_epoch{number_epochs_finetuned}.cpl"
            corrected_model_path = os.path.join(self.adaptor_config.base_dir, filename)
            _log.info("%s", "saving corrected model to: " + corrected_model_path)
            torch.save(self.model.to("cpu"), corrected_model_path)
            self.model.to(self.device)

        return self.model, number_epochs_finetuned

    def finetune(
        self,
        projection_layer: Module,
        downstream_head: Module,
        feature_extractor: Module = None,
    ) -> (Module, int):
        """Fine-tune the head behind a frozen projection with cross entropy.

        Parameters
        ----------
        projection_layer : torch.nn.Module
            Frozen CAV projection.
        downstream_head : torch.nn.Module
            Part of the model that is trained (SGD, momentum 0.9, weight decay
            1e-4, ``training.learning_rate``).
        feature_extractor : torch.nn.Module, optional
            Frozen prefix; None when projecting at the input.

        Returns
        -------
        best_model : torch.nn.Module
            Deep copy of the head with the best validation ``avg_group_acc``.
        number_epochs_trained : int
            Epoch at which that best head was found (0 = untouched head).

        Notes
        -----
        Temporarily lifts the class restriction of the train loader and
        re-enables it afterwards; a failing backward pass returns early.
        """

        if feature_extractor is not None:
            feature_extractor.eval()
        projection_layer.eval()
        downstream_head.train()
        self.train_dataloader.dataset.disable_class_restriction()

        optimizer = torch.optim.SGD(
            downstream_head.parameters(),
            lr=self.adaptor_config.training.learning_rate,
            momentum=0.9,
            weight_decay=0.0001,
        )
        loss = CrossEntropyLoss()

        composite = (
            Sequential(feature_extractor, projection_layer, downstream_head)
            if feature_extractor is not None
            else Sequential(projection_layer, downstream_head)
        )
        val_accuracies = self.get_stats(self.val_dataloader, composite)
        _log.info(
            "%s",
            f"Epoch 0: avg_group_acc={val_accuracies['avg_group_acc']}, worst_group_acc={val_accuracies['worst_group_acc']}",
        )
        best_model = copy.deepcopy(downstream_head)
        best_val_group_acc = val_accuracies["avg_group_acc"]
        number_epochs_trained = 0

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.adaptor_config.training.max_epochs):
            losses = []
            accuracies = []

            with tqdm(self.train_dataloader) as pbar:
                for batch in pbar:
                    optimizer.zero_grad()

                    x = batch["x"].to(self.device)
                    y = batch["y"].to(self.device).squeeze().long()
                    if feature_extractor is None:
                        prediction = downstream_head(projection_layer(x))
                    else:
                        prediction = downstream_head(
                            projection_layer(feature_extractor(x))
                        )

                    ce_loss = loss(prediction, y)
                    try:
                        ce_loss.backward()
                    except Exception:
                        _log.info("%s", traceback.format_exc())
                        if self.config.attacked_class is not None:
                            self.train_dataloader.dataset.enable_class_restriction(
                                self.attacked_class
                            )
                        return best_model, number_epochs_trained

                    optimizer.step()

                    accuracy = (prediction.argmax(1) == y).float().detach()
                    losses.append(ce_loss.item())
                    accuracies.append(accuracy)

                    pbar.set_description(
                        f"fine tuning epoch {epoch+1}/{self.adaptor_config.training.max_epochs}: accuracy={accuracy.mean()}, loss={ce_loss}"
                    )

            epoch_accuracy = torch.cat(accuracies).mean().item()
            epoch_ce_loss = torch.tensor(losses).mean().item()
            composite = (
                Sequential(feature_extractor, projection_layer, downstream_head)
                if feature_extractor is not None
                else Sequential(projection_layer, downstream_head)
            )
            val_accuracies = self.get_stats(self.val_dataloader, composite)

            if val_accuracies["avg_group_acc"] > best_val_group_acc:
                best_val_group_acc = val_accuracies["avg_group_acc"]
                best_model = copy.deepcopy(downstream_head)
                number_epochs_trained = epoch + 1

            _log.info(
                "%s",
                f"Epoch {epoch+1}: train_acc={epoch_accuracy}, avg_group_acc={val_accuracies['avg_group_acc']}, worst_group_acc={val_accuracies['worst_group_acc']}, ce_loss={epoch_ce_loss}",
            )

        if self.config.attacked_class is not None:
            self.train_dataloader.dataset.enable_class_restriction(self.attacked_class)

        return best_model, number_epochs_trained


class RRClArC(ClArC):
    """Right-for-the-right-reasons ClArC.

    Instead of projecting, fine-tunes the downstream head with
    ``ce_loss + correction_strength * rrc_loss`` where ``rrc_loss`` penalises
    the alignment between the gradient of the logits w.r.t. the layer
    activations and the CAV. Training curves go to TensorBoard under
    ``base_dir/finetuning-logs``.

    Parameters
    ----------
    adaptor_config : RRClArCConfig
        See :class:`RRClArCConfig`.
    """

    def __init__(self, adaptor_config: RRClArCConfig):
        super().__init__(adaptor_config)
        self.adaptor_config = adaptor_config
        from torch.utils.tensorboard import SummaryWriter

        self.log_writer = SummaryWriter(
            log_dir=self.adaptor_config.base_dir + "/finetuning-logs"
        )

    def get_evaluation_filename(self, dataset_name: str) -> str:
        """Build the csv name including loss type, gradient target and epochs."""
        attacked_class = (
            f"_attacked-c{self.adaptor_config.attacked_class}"
            if self.adaptor_config.attacked_class is not None
            else ""
        )
        mean_grad = "_mean-grad" if self.adaptor_config.mean_grad else ""
        label_type = (
            "true" if self.config.data.spray_label_file is None else "spray"
        ) + "-group-labels"
        return f"correction_{dataset_name}-dataset_{label_type}_{self.adaptor_config.projection_type}-projection_mode-{self.adaptor_config.cav_mode}{attacked_class}_{self.adaptor_config.rrc_loss}-loss_target-{self.adaptor_config.gradient_target}{mean_grad}_{self.adaptor_config.training.max_epochs}-epochs.csv"

    def run(self):
        """Run the sweep of :meth:`ClArC.run` and close the TensorBoard writer."""
        super().run()
        self.log_writer.close()

    def _run(
        self, layer_index: int = -2, correction_strength: float = 1.0
    ) -> (Module, int):
        """Estimate the CAV at ``layer_index`` and RR-fine-tune the head.

        Parameters
        ----------
        layer_index : int, default -2
            Split point for :func:`split_model`; ``0`` uses the whole model as
            head and the input as representation.
        correction_strength : float, default 1.0
            Weight ``lamb`` of the RR penalty.

        Returns
        -------
        model : torch.nn.Module
            Corrected model (saved as ``base_dir/<model_name>.cpl`` when
            ``save_model`` is set).
        number_epochs_finetuned : int
            Epoch of the selected head.
        """

        torch.manual_seed(self.config.seed)
        _log.info(
            "%s",
            f"\n\nperforming rr-clarc in layer {layer_index} with cav_mode={self.adaptor_config.cav_mode} and correction_strength={correction_strength}",
        )

        attacked_class = (
            f"_attacked-c{self.adaptor_config.attacked_class}"
            if self.adaptor_config.attacked_class is not None
            else ""
        )
        mean_grad = "_mean-grad" if self.adaptor_config.mean_grad else ""
        model_name = f"corrected_model_layer-{layer_index}_mode-{self.adaptor_config.cav_mode}_lamb{correction_strength}_{self.adaptor_config.rrc_loss}-loss{mean_grad}{attacked_class}_{self.adaptor_config.training.max_epochs}-epochs"

        self.model.eval()
        feature_extractor, downstream_head = None, None
        if layer_index == 0:
            downstream_head = self.model
        else:
            feature_extractor, downstream_head = split_model(
                self.model, layer_index, self.device
            )

        if layer_index != self.cav_cache.layer:
            activations, annotations = self.get_annotations_and_activations(
                feature_extractor=feature_extractor
            )
            cav = calculate_cav(
                activations, annotations, self.adaptor_config.projection_type
            )
            cav = cav.to(self.device, activations.dtype)
            self.cav_cache = CavCache(
                layer=layer_index,
                cav=cav,
                annotations=annotations,
                activations=activations,
            )

        downstream_head, number_epochs_finetuned = self.finetune(
            self.cav_cache.cav,
            downstream_head,
            correction_strength,
            feature_extractor=feature_extractor,
            model_name=model_name,
        )
        if layer_index == 0:
            self.model = downstream_head
        else:
            self.model = torch.nn.Sequential(*[feature_extractor, downstream_head])
        self.model.eval()

        if self.adaptor_config.save_model:
            corrected_model_path = os.path.join(
                self.adaptor_config.base_dir, model_name + ".cpl"
            )
            _log.info("%s", "saving corrected model to: " + corrected_model_path)
            torch.save(self.model.to("cpu"), corrected_model_path)
            self.model.to(self.device)

        return self.model, number_epochs_finetuned

    def finetune(
        self,
        cav: torch.Tensor,
        downstream_head: Module,
        lamb: float,
        feature_extractor: Module = None,
        model_name: str = "corrected_model",
    ):
        """Fine-tune the head with cross entropy plus the CAV-gradient penalty.

        Parameters
        ----------
        cav : torch.Tensor
            Unit CAV in the (pooled) activation space of the split layer.
        downstream_head : torch.nn.Module
            Module trained with SGD (momentum 0.95, ``training.learning_rate``).
        lamb : float
            Weight of the RR penalty.
        feature_extractor : torch.nn.Module, optional
            Frozen prefix producing the representation; None uses the input.
        model_name : str
            Prefix of the TensorBoard scalar tags.

        Returns
        -------
        best_model : torch.nn.Module
            Head with the best validation ``avg_group_acc`` (ties broken by
            lower epoch RR loss).
        number_epochs_finetuned : int
            Epoch at which it was selected.

        Notes
        -----
        The penalty is ``mean((grad . cav) ** 2)`` for ``rrc_loss="l2"`` or the
        mean absolute cosine similarity for ``"cosine"``; with ``cav_mode`` set
        the gradient is rearranged to ``(B*H*W, C)`` so the CAV lives in channel
        space. A NaN / inf total loss aborts the epoch.
        """

        best_model = copy.deepcopy(downstream_head)
        best_rrc_loss = sys.maxsize
        if feature_extractor is None:
            val_accuracies = self.get_stats(self.val_dataloader, downstream_head)
        else:
            feature_extractor.eval()
            val_accuracies = self.get_stats(
                self.val_dataloader, Sequential(feature_extractor, downstream_head)
            )
        best_val_group_acc = val_accuracies["avg_group_acc"]
        number_epochs_finetuned = 0

        downstream_head.train()
        self.train_dataloader.dataset.disable_class_restriction()

        optimizer = torch.optim.SGD(
            downstream_head.parameters(),
            lr=self.adaptor_config.training.learning_rate,
            momentum=0.95,
        )
        for epoch in range(self.adaptor_config.training.max_epochs):
            ce_losses = []
            rrc_losses = []
            accuracies = []

            with tqdm(self.train_dataloader) as pbar:
                for batch in pbar:
                    optimizer.zero_grad()

                    optimizer.zero_grad()
                    x = batch["x"].to(self.device)
                    y = batch["y"].to(self.device).squeeze()

                    representation = (
                        x if feature_extractor is None else feature_extractor(x)
                    ).requires_grad_()
                    prediction = downstream_head(representation)
                    prediction_filtered = self.get_gradient_target(prediction)

                    grad = torch.autograd.grad(
                        outputs=prediction_filtered,
                        inputs=representation,
                        create_graph=True,
                        retain_graph=True,
                        grad_outputs=torch.ones_like(prediction_filtered),
                    )[0]
                    if self.adaptor_config.mean_grad:
                        grad = grad.mean((2, 3), keepdim=True).expand_as(grad)

                    if self.adaptor_config.cav_mode is not None:
                        grad = (
                            grad.permute(1, 0, 2, 3).flatten(start_dim=1).permute(1, 0)
                        )
                    else:
                        grad = grad.flatten(start_dim=1)

                    if self.adaptor_config.rrc_loss == "l2":
                        rrc_loss = ((grad * cav).sum(1) ** 2).mean(0)
                    elif self.adaptor_config.rrc_loss == "cosine":
                        rrc_loss = (
                            torch.nn.functional.cosine_similarity(grad, cav)
                            .abs()
                            .mean(0)
                        )
                    ce_loss = torch.nn.functional.cross_entropy(
                        prediction, y.to(torch.long)
                    )
                    accuracy = (prediction.argmax(1) == y).float()
                    rrc_losses.append(rrc_loss.item())
                    ce_losses.append(ce_loss.item())
                    accuracies.append(accuracy.detach())

                    loss = ce_loss + lamb * rrc_loss.to(torch.float64)
                    if torch.isnan(loss) or torch.isinf(loss):
                        break

                    loss.backward()
                    optimizer.step()

                    pbar.set_description(
                        f"fine tuning epoch {epoch+1}/{self.adaptor_config.training.max_epochs}: accuracy={accuracy.mean()}, rrc_loss={rrc_loss}, ce_loss={ce_loss}"
                    )

            epoch_accuracy = torch.cat(accuracies).mean().item()
            epoch_ce_loss = torch.tensor(ce_losses).mean().item()
            epoch_rrc_loss = torch.tensor(rrc_losses).mean().item()
            if feature_extractor is None:
                val_accuracies = self.get_stats(self.val_dataloader, downstream_head)
            else:
                val_accuracies = self.get_stats(
                    self.val_dataloader, Sequential(feature_extractor, downstream_head)
                )

            if val_accuracies["avg_group_acc"] > best_val_group_acc or (
                epoch_rrc_loss < best_rrc_loss
                and val_accuracies["avg_group_acc"] == best_val_group_acc
            ):
                best_val_group_acc = val_accuracies["avg_group_acc"]
                best_rrc_loss = epoch_rrc_loss
                best_model = copy.deepcopy(downstream_head)
                number_epochs_finetuned = epoch + 1

            self.log_writer.add_scalar(
                model_name + "/val/emp_acc", val_accuracies["accuracy"], epoch
            )
            self.log_writer.add_scalar(
                model_name + "/val/avg_group_acc",
                val_accuracies["avg_group_acc"],
                epoch,
            )
            self.log_writer.add_scalar(
                model_name + "/val/worst_group_acc",
                val_accuracies["worst_group_acc"],
                epoch,
            )
            self.log_writer.add_scalar(
                model_name + "/train/accuracy", epoch_accuracy, epoch
            )
            self.log_writer.add_scalar(
                model_name + "/train/ce_loss", epoch_ce_loss, epoch
            )
            self.log_writer.add_scalar(
                model_name + "/train/rrc_loss", epoch_rrc_loss, epoch
            )

            _log.info(
                "%s",
                f"Epoch {epoch+1}: train_accuracy={epoch_accuracy}, val_accuracy={val_accuracies['accuracy']}, avg_group_acc={val_accuracies['avg_group_acc']}, worst_group_acc={val_accuracies['worst_group_acc']}, ce_loss={epoch_ce_loss}, rrc_loss={epoch_rrc_loss}",
            )

        self.log_writer.flush()
        if self.adaptor_config.attacked_class is not None:
            self.train_dataloader.dataset.enable_class_restriction(
                self.adaptor_config.attacked_class
            )

        return best_model, number_epochs_finetuned

    def get_gradient_target(self, prediction):
        """Reduce logits ``(B, K)`` to the scalar per sample that is differentiated.

        Selected by ``config.gradient_target``: ``"max"`` (top logit),
        ``"attacked_class"`` (that class' logit), ``"all"`` (sum of logits) or
        ``"all_random"`` (sum with random signs).

        Raises
        ------
        NotImplementedError
            For any other value.
        """
        if self.adaptor_config.gradient_target == "max":
            return prediction.max(1)[0]
        elif self.adaptor_config.gradient_target == "attacked_class":
            return prediction[:, self.adaptor_config.attacked_class]
        elif self.adaptor_config.gradient_target == "all":
            return prediction.sum(1)
        elif self.adaptor_config.gradient_target == "all_random":
            return (prediction * torch.sign(0.5 - torch.rand_like(prediction))).sum(1)
        else:
            raise NotImplementedError


def split_model(model: Module, split_at: int, device) -> (Module, Module):
    """Split a model into ``Sequential`` feature extractor and downstream head.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose (recursively flattened) children are split.
    split_at : int
        Index into the flattened child list; children before it form the
        feature extractor, the rest the head.
    device
        Device both parts are moved to.

    Returns
    -------
    (torch.nn.Module, torch.nn.Module)
        ``(feature_extractor, downstream_head)``. For torchvision ``ResNet``
        a ``Flatten`` is inserted before the final ``fc`` layer since the
        flattening is not a child module.
    """
    children_list = extract_all_children(model)[0]
    _log.info("%s", f"splitting model into {len(children_list)} children")

    # for i, node in enumerate(children_list):
    #     print(f"layer {i+1}: {node}")
    # exit()

    feature_extractor = torch.nn.Sequential(*children_list[:split_at])
    if isinstance(model, ResNet):
        downstream_head = torch.nn.Sequential(
            *children_list[split_at:-1],
            torch.nn.Flatten(start_dim=1),
            children_list[-1],
        )
    else:
        downstream_head = torch.nn.Sequential(*children_list[split_at:])

    return feature_extractor.to(device), downstream_head.to(device)


def extract_all_children(model: Module, prefix: str = "") -> (list[Module], list[str]):
    """Flatten nested ``nn.Sequential`` containers into a list of leaf children.

    Parameters
    ----------
    model : torch.nn.Module
        Model to walk with ``named_children``.
    prefix : str
        Dotted name prefix used for recursion.

    Returns
    -------
    (list of torch.nn.Module, list of str)
        Modules in forward order and their dotted names. Only ``Sequential``
        is descended into; other containers are returned as single children.
    """
    children = []
    children_names = []
    for name, child in model.named_children():
        if prefix:
            name = prefix + "." + name
        if isinstance(child, torch.nn.Sequential):
            grandchildren, grandchildren_names = extract_all_children(
                child, prefix=name
            )
            children.extend(grandchildren)
            children_names.extend(grandchildren_names)

        else:
            children.append(child)
            children_names.append(name)

    return children, children_names


def get_layer_name(model: Module, layer_index: int) -> str:
    """Return the dotted name of the child at 1-based ``layer_index``."""
    return extract_all_children(model)[1][layer_index - 1]


def calculate_cav(
    activations: torch.Tensor, annotations: torch.Tensor, projection_type: str
) -> torch.Tensor:
    """Dispatch to :func:`calculate_pcav` or :func:`calculate_svm_cav`.

    Parameters
    ----------
    activations : torch.Tensor
        Shape ``(N, D)``.
    annotations : torch.Tensor
        Shape ``(N,)`` with 0 / 1 artifact labels.
    projection_type : str
        ``"pcav"`` or ``"svm"``.

    Returns
    -------
    torch.Tensor
        Unit-norm CAV of shape ``(1, D)`` (pcav) or ``(D,)`` (svm).

    Raises
    ------
    NotImplementedError
        For other projection types.
    """
    if projection_type == "pcav":
        return calculate_pcav(activations, annotations)
    elif projection_type == "svm":
        return calculate_svm_cav(activations, annotations)
    else:
        raise NotImplementedError


def calculate_pcav(
    activations: torch.Tensor, annotations: torch.Tensor
) -> torch.Tensor:
    """Pattern CAV: covariance of activations with the +-1 artifact label.

    The labels are mapped to ``{-1, +1}``, both sides are centred and the CAV
    is ``cov(activations, labels) / var(labels)``, normalised to unit length.

    Parameters
    ----------
    activations : torch.Tensor
        Shape ``(N, D)``.
    annotations : torch.Tensor
        Shape ``(N,)`` with 0 / 1 labels (not modified in place).

    Returns
    -------
    torch.Tensor
        Shape ``(1, D)``.
    """
    actvs_centered = activations - activations.mean(dim=0)[None]

    annotations = annotations.clone()
    annotations[annotations == 0] = -1
    annotations = annotations.to(activations.dtype)
    annotations_centered = annotations - annotations.mean()

    covar = (actvs_centered * annotations_centered[:, None]).sum(dim=0) / (
        annotations.shape[0] - 1
    )
    vary = torch.sum(annotations_centered**2, dim=0) / (annotations.shape[0] - 1)
    w = (covar / vary)[None]

    cav = w / torch.sqrt((w**2).sum())
    _log.info("%s %s", "cav shape:", cav.shape)
    return cav


def calculate_svm_cav(activations, annotations) -> torch.Tensor:
    """CAV from the normal of a balanced ``LinearSVC`` (C=1, up to 10000 iters).

    Returns
    -------
    torch.Tensor
        Unit-norm weight vector of shape ``(D,)`` on CPU (float64).
    """
    activations = activations.cpu()
    annotations = annotations.cpu()
    model = LinearSVC(
        C=1.0, penalty="l2", max_iter=10000, class_weight="balanced", verbose=2
    )
    model.fit(activations, annotations)

    cav = torch.tensor(model.coef_[0])
    _log.info("%s %s", "cav shape:", cav.shape)
    return cav / ((cav**2).sum() ** 0.5).item()


class CavProjection(torch.nn.Module):
    """Layer that projects the CAV direction out of its input.

    Computes ``x - x cav cav^T + z`` where ``z`` is the projection of the mean
    non-artifact activation onto the CAV, followed by a ReLU. With a
    ``cav_mode`` the projection coefficient is computed on spatially pooled
    activations and broadcast over ``H, W``.

    Parameters
    ----------
    activations : torch.Tensor
        Shape ``(N, D)`` activations used for the clean mean.
    annotations : torch.Tensor
        Shape ``(N,)``; samples with label 0 define the clean mean.
    cav : torch.Tensor
        CAV, reshaped to ``(D, 1)`` and scaled by ``correction_strength``.
    cav_mode : str, optional
        None (flat input), ``"cavs_max"`` or ``"cavs_mean"``.
    correction_strength : float, default 1.0
        Multiplier of the CAV; values other than 1 give a partial correction.
    """

    def __init__(
        self,
        activations: torch.Tensor,
        annotations: torch.Tensor,
        cav: torch.Tensor,
        cav_mode: str = None,
        correction_strength: float = 1.0,
    ):
        super().__init__()
        self.correction_strength = correction_strength
        self.cav_mode = cav_mode

        self.cav = self.correction_strength * cav.reshape(-1, 1)
        z = torch.mean(activations[annotations == 0], dim=0, keepdim=True).to(
            device=cav.device, dtype=cav.dtype
        )
        self.z = z @ self.cav @ self.cav.T

    def forward(self, x):
        """Project the CAV direction out of ``x`` and apply ReLU.

        Raises
        ------
        NotImplementedError
            For an unknown ``cav_mode``.
        """

        out = x + 0
        if self.cav_mode is None:
            out = out.flatten(1)
            out = out - (out @ self.cav @ self.cav.T) + self.z
            out = out.reshape(x.shape)
        else:
            if self.cav_mode == "cavs_max":
                out = out.flatten(start_dim=2).max(2).values
            elif self.cav_mode == "cavs_mean":
                out = out.mean((2, 3))
            else:
                raise NotImplementedError
            out = (
                x
                - (out @ self.cav @ self.cav.T)[:, :, None, None]
                + self.z[:, :, None, None]
            )

        return torch.nn.functional.relu(out)


class SimpleProjection(torch.nn.Module):
    """Layer that subtracts the mean artifact minus non-artifact activation.

    Parameters
    ----------
    activations : torch.Tensor
        Shape ``(N, ...)``; flattened per sample.
    annotations : torch.Tensor
        Shape ``(N,)`` with 0 / 1 labels.
    **kwargs
        Ignored (accepts the :class:`CavProjection` keyword arguments).
    """

    def __init__(self, activations: torch.Tensor, annotations: torch.Tensor, **kwargs):
        super().__init__()
        activations = activations.flatten(start_dim=1)
        self.difference = activations[annotations == 1].mean(0) - activations[
            annotations == 0
        ].mean(0)

    def forward(self, x):
        """Subtract the stored difference vector, ReLU, restore ``x.shape``."""
        x_flat = x.flatten(start_dim=1)
        out = x_flat - self.difference
        out = torch.nn.functional.relu(out)
        return out.reshape(x.shape)


#: Memo of the CAV estimated for one layer: ``layer`` (int), ``cav``,
#: ``annotations`` and ``activations`` (tensors from
#: :meth:`ClArC.get_annotations_and_activations`).
CavCache = namedtuple("CavCache", ["layer", "cav", "annotations", "activations"])
