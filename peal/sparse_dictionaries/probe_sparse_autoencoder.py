"""Sparse dictionary made of the weight rows of a pretrained linear probe.

``ProbeSAE`` is the degenerate member of PEAL's sparse dictionary family: instead
of learning atoms from activations it takes the final ``fc`` layer of a trained
classifier checkpoint and exposes its class weight vectors as the components. It
lets the DiDAE / CFKD pipelines use the classifier's own decision directions as
candidate edit directions without any fitting step.
"""

from typing import Union

import torch

from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig


class ProbeSAEConfig(SparseDictionaryConfig):
    """Config of :class:`ProbeSAE`.

    Parameters
    ----------
    pretrained_model_path : str
        Path of a ``torch.save``-d Lightning predictor whose ``model.model.fc``
        is the linear layer whose weight rows become the dictionary atoms.
    n_components : int or None, default 2
        Number of components; informational only, the probe decides the count.
    sparse_dictionaries_type : str
        Registry key, fixed to ``"ProbeSAE"``.
    ending : str
        File ending the factory expects for saved dictionaries (``".npz"``).
    """

    pretrained_model_path: str
    n_components: Union[int, None] = 2
    sparse_dictionaries_type: str = "ProbeSAE"
    ending: str = ".npz"


class ProbeSAE(SparseDictionary):
    """Dictionary whose atoms are the rows of a classifier's last linear layer.

    Parameters
    ----------
    config : ProbeSAEConfig
        Points to the checkpoint; ``fc.weight`` is cloned into ``self.W`` with
        shape ``(n_classes, feature_dim)``, and ``self.mu`` is a zero mean vector
        of length ``feature_dim`` so callers treating the dictionary as centered
        get an identity offset.
    """

    def __init__(self, config):
        """Load the checkpoint on CPU and copy its ``fc`` weight matrix."""
        self.config = config
        model = torch.load(config.pretrained_model_path, map_location="cpu")
        self.W = model.model.model.fc.weight.data.clone()
        self.mu = torch.zeros(self.W.size(1))

    def fit(self, X):
        """No-op: the probe weights are fixed by the checkpoint."""

    def fit_from_dataloaders(self, dataloaders, feature_extractor):
        """No-op: the probe weights are fixed by the checkpoint."""

    def get_components(self):
        """Return the atoms as columns, i.e. ``W.t()`` of shape ``(D, n_classes)``."""
        return self.W.t()

    def save_on_disk(self, path):
        """Save ``{"W": self.W}`` with ``torch.save`` to ``path``."""
        torch.save(
            {
                "W": self.W,
            },
            path,
        )

    def load_from_disk(self, path):
        """Restore ``self.W`` from a file written by :meth:`save_on_disk`."""
        checkpoint = torch.load(path)
        self.W = checkpoint["W"]
