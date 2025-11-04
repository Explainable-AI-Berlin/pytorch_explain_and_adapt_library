import torch
import json
import h5py
import torchvision.utils
import yaml
import os
import time
import multiprocessing
import numpy as np

from tqdm import tqdm
from os.path import dirname, relpath
from typing import List
from corelay.base import Param
from corelay.processor.base import Processor
from corelay.processor.affinity import SparseKNN
from corelay.processor.distance import SciPyPDist
from corelay.processor.flow import Sequential, Parallel
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.embedding import TSNEEmbedding, UMAPEmbedding, EigenDecomposition
from corelay.processor.clustering import (
    KMeans,
    DBSCAN,
    HDBSCAN,
    AgglomerativeClustering,
)
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


CANONIZERS = {
    "vgg": VGGCanonizer,
    "resnet": ResNetCanonizer,
    "sequential_merge_batch_norm": SequentialMergeBatchNorm,
}


class LRPExplainer:
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
                print("No data config found!")
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
        """ """
        with self.attributor:
            """X = torch.clone(batch).to(self.device)
            X.requires_grad = True
            X.retain_grad = True
            heatmaps = torch.abs(X.grad).sum(1)"""
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
        This function runs the explainer.
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
