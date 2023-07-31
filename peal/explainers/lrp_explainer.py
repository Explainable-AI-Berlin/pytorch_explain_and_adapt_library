import torch
import json
import h5py
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
from corelay.processor.clustering import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from zennit.attribution import IntegratedGradients
from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from zennit.image import imgify
from torchvision.transforms import ToTensor

from peal.utils import (
    load_yaml_config
)


CANONIZERS = {
    'vgg': VGGCanonizer,
    'resnet': ResNetCanonizer,
    'sequential_merge_batch_norm': SequentialMergeBatchNorm,
}


class LRPExplainer:
    def __init__(
            self,
            downstream_model,
            num_classes,
            explainer_config = '<PEAL_BASE>/configs/explainers/lrp_default.yaml',
    ):
        self.downstream_model = downstream_model
        self.num_classes = num_classes
        self.explainer_config = load_yaml_config(explainer_config)
        self.device = 'cuda' if next(self.downstream_model.parameters()).is_cuda else 'cpu'

        #
        composite = EpsilonGammaBox(
            canonizers=[CANONIZERS[key]() for key in self.explainer_config.get('canonizers', [])],
            **self.explainer_config.get('composite_kwargs', {})
        )
        self.attributor = IntegratedGradients(model=self.downstream_model, composite=composite)

    def explain_batch(self, batch, labels):
        '''

        '''
        with self.attributor:
            '''X = torch.clone(batch).to(self.device)
            X.requires_grad = True
            X.retain_grad = True
            heatmaps = torch.abs(X.grad).sum(1)'''
            predictions, attributions = self.attributor(
                batch.to(self.device),
                torch.eye(self.num_classes)[labels].to(self.device)
            )
            overlays_imgs = []
            # TODO is this the best solution?
            for i in range(attributions.shape[0]):
                overlays_imgs.append(ToTensor()(imgify(attributions[i].detach().cpu(), cmap='wred', symmetric=True)))

            overlays = torch.stack(overlays_imgs, 0)
            heatmaps = torch.abs(attributions.detach().cpu() / (batch + 0.0001)).sum(1)
            epsilon = 0.00000001
            heatmaps = (heatmaps + epsilon) / (torch.max(heatmaps.flatten(1), 1).values + epsilon).unsqueeze(-1).unsqueeze(-1).tile([1] + list(heatmaps.shape[1:]))
            heatmaps = torch.square(heatmaps.unsqueeze(1).tile([1,3,1,1]))
            return heatmaps, overlays, predictions.detach().cpu().argmax(-1)

