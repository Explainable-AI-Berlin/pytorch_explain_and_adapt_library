"""
Attribution (heatmap) explainer built on zennit.

Besides counterfactuals PEAL can explain a classifier with pixel-wise
attribution maps. ``LRPExplainer`` wraps zennit's ``IntegratedGradients``
attributor with an ``EpsilonGammaBox`` LRP composite and optional model
canonizers, and exposes the same ``explain_batch``/``run`` interface the
counterfactual explainers use, so it can be plugged into the visualization
tools (``peal.visualization.model_comparison``).
"""

import torch
import torchvision.utils
import os

from tqdm import tqdm
from zennit.attribution import IntegratedGradients
from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from zennit.image import imgify
from torchvision.transforms import ToTensor

from peal.architectures.predictors import get_predictor
from peal.architectures.interfaces import TaskConfig
from peal.data.dataset_factory import get_datasets
from peal.global_utils import load_yaml_config
from peal.log import get_logger

_log = get_logger(__name__)


CANONIZERS = {
    "vgg": VGGCanonizer,
    "resnet": ResNetCanonizer,
    "sequential_merge_batch_norm": SequentialMergeBatchNorm,
}


class LRPExplainer:
    """
    Heatmap explainer using zennit Integrated Gradients with an LRP composite.

    The explainer config (a yaml path or loaded config) is read for the keys
    ``predictor`` (fallback if none is passed), ``data_config`` (fallback to
    the predictor's data config), ``canonizers`` (list of keys into
    ``CANONIZERS``), ``composite_kwargs`` (forwarded to ``EpsilonGammaBox``),
    ``explanations_dir`` and ``max_samples``.

    Parameters
    ----------
    explainer_config : str or dict-like
        Path to the explainer yaml or an already loaded config.
    predictor : torch.nn.Module or str, optional
        Model or path to a model; resolved through ``get_predictor``.
    num_classes : int, optional
        Number of output classes, used to build one-hot targets.
    datasets : sequence, optional
        ``(train, val)`` datasets; when omitted they are created from the
        data config.

    Attributes
    ----------
    attributor : zennit.attribution.IntegratedGradients
        The attributor whose context manager is entered in ``explain_batch``.
    predictor_datasets : sequence
        The first two datasets (train, val); ``run`` iterates over the second.
    """

    def __init__(
        self, explainer_config, predictor=None, num_classes=None, datasets=None
    ):
        self.explainer_config = load_yaml_config(explainer_config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if predictor is None:
            predictor = explainer_config.predictor

        self.predictor, self.predictor_config = get_predictor(predictor, self.device)

        if not datasets is None:
            self.predictor_datasets = datasets

        else:
            if not self.explainer_config.data_config is None:
                data_config = self.explainer_config.data_config

            elif not self.predictor_config is None:
                data_config = self.predictor_config.data

            else:
                _log.info("%s", "No data config found!")
                raise ValueError

            if not self.predictor_config is None:
                task_config = TaskConfig(**self.predictor_config.task)

            else:
                task_config = None

            self.predictor_datasets = get_datasets(
                data_config, task_config=task_config
            )[:2]

        self.num_classes = num_classes
        self.explainer_config = load_yaml_config(explainer_config)
        self.device = "cuda" if next(self.predictor.parameters()).is_cuda else "cpu"

        #
        composite = EpsilonGammaBox(
            canonizers=[
                CANONIZERS[key]() for key in self.explainer_config.get("canonizers", [])
            ],
            **self.explainer_config.get("composite_kwargs", {}),
        )
        self.attributor = IntegratedGradients(model=self.predictor, composite=composite)

    def explain_batch(self, batch, labels):
        """
        Compute attribution heatmaps for a batch of images.

        Parameters
        ----------
        batch : torch.Tensor
            Images of shape ``(N, C, H, W)`` (a single ``(C, H, W)`` image
            works as well since the attributor broadcasts).
        labels : torch.Tensor or int
            Class index per image; turned into one-hot targets with
            ``num_classes`` entries.

        Returns
        -------
        heatmaps : torch.Tensor
            Shape ``(N, 3, H, W)``: per-pixel relevance (input-normalised,
            summed over channels, scaled to ``[0, 1]`` per image, squared
            and repeated over three channels) on CPU.
        overlays : torch.Tensor
            Shape ``(N, 3, H, W)``: zennit ``imgify`` renderings of the raw
            attributions with the ``wred`` colormap.
        predictions : torch.Tensor
            Shape ``(N,)``: argmax of the model output on CPU.
        """
        with self.attributor:
            predictions, attributions = self.attributor(
                batch.to(self.device),
                torch.eye(self.num_classes)[labels].to(self.device),
            )
            overlays_imgs = []
            # TODO is this the best solution?
            for i in range(attributions.shape[0]):
                overlays_imgs.append(
                    ToTensor()(
                        imgify(
                            attributions[i].detach().cpu(), cmap="wred", symmetric=True
                        )
                    )
                )

            overlays = torch.stack(overlays_imgs, 0)
            heatmaps = torch.abs(attributions.detach().cpu() / (batch + 0.0001)).sum(1)
            epsilon = 0.00000001
            heatmaps = (heatmaps + epsilon) / (
                torch.max(heatmaps.flatten(1), 1).values + epsilon
            ).unsqueeze(-1).unsqueeze(-1).tile([1] + list(heatmaps.shape[1:]))
            heatmaps = torch.square(heatmaps.unsqueeze(1).tile([1, 3, 1, 1]))
            return heatmaps, overlays, predictions.detach().cpu().argmax(-1)

    def run(self, *args, **kwargs):
        """
        Explain the validation split and write one image per sample.

        Iterates over ``predictor_datasets[1]`` (enabling hints if the dataset
        has them), predicts each sample, attributes the predicted class and
        saves ``explanation_<i>.png`` (input stacked with its heatmap) into
        ``explainer_config.explanations_dir``. Stops after
        ``explainer_config.max_samples`` samples when that is set.

        Returns
        -------
        dict
            Keys ``"heatmap"``, ``"x"`` and ``"prediction"`` holding the
            values of the last processed sample.
        """
        if not os.path.exists(self.explainer_config.explanations_dir):
            os.makedirs(self.explainer_config.explanations_dir)

        out_dict = {"heatmap": None, "x": None, "prediction": None}
        collage_idx = 0
        if self.predictor_datasets[1].config.has_hints:
            self.predictor_datasets[1].enable_hints()

        pbar = tqdm(total=self.explainer_config.max_samples)
        pbar.stored_values = {}
        pbar.stored_values["n_total"] = 0
        for idx in range(len(self.predictor_datasets[1])):
            if (
                not self.explainer_config.max_samples is None
                and collage_idx >= self.explainer_config.max_samples
            ):
                break

            x, y = self.predictor_datasets[1][idx]

            y_logits = self.predictor(x.unsqueeze(0).to(self.device))[0]
            y_pred = y_logits.argmax()
            heatmap, _, _ = self.explain_batch(x, y_pred)
            out_dict["heatmap"] = heatmap
            out_dict["x"] = x
            out_dict["prediction"] = y_pred
            torchvision.utils.save_image(
                torch.cat([x, heatmap], 0),
                os.path.join(
                    self.explainer_config.explanations_dir,
                    f"explanation_{collage_idx}.png",
                ),
            )

            pbar.stored_values["n_total"] += 1

        return out_dict
