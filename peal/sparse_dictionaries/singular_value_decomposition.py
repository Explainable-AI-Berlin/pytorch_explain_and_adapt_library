"""Truncated SVD (PCA) as the simplest PEAL sparse dictionary.

The principal directions of a feature matrix serve as a dense, non-sparse
baseline dictionary that adaptors such as DiDAE can search for class-relevant
directions in the same way they search an SAE or MSAE dictionary.
"""

from typing import Union

import torch

from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig


class SVDDictionaryConfig(SparseDictionaryConfig):
    """Config for :class:`SVDDictionary`.

    Parameters
    ----------
    n_components : int or None, optional
        Number of singular vectors to keep; ``None`` keeps all. Default 10.
    sparse_dictionaries_type : str
        Factory key, fixed to ``"SVDDictionary"``.
    ending : str
        File ending used for the saved weights. Default ``".npz"``.
    """

    n_components: Union[int, None] = 10
    sparse_dictionaries_type: str = "SVDDictionary"
    ending: str = ".npz"


class SVDDictionary(SparseDictionary):
    """Dictionary whose components are the top right-singular vectors of the data.

    ``fit`` stores ``U``, ``S`` and ``Vt`` of the (already centred) feature
    matrix truncated to ``config.n_components``; ``fit_from_dataloaders``
    additionally records the feature mean ``mu`` that was subtracted.

    Parameters
    ----------
    config : SVDDictionaryConfig, optional
        Number of components and factory metadata.

    Attributes
    ----------
    U, S, Vt : torch.Tensor or None
        Truncated SVD factors, ``None`` until fitted.
    mu : torch.Tensor or None
        Feature mean subtracted before the SVD (only set by
        ``fit_from_dataloaders``).
    """

    def __init__(self, config=SVDDictionaryConfig()):
        """Store the config and leave the SVD factors unfitted (``None``)."""
        self.config = config
        self.U = None
        self.S = None
        self.Vt = None
        self.mu = None

    def fit(self, X):
        """Compute the truncated SVD of a feature matrix.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix of shape ``(n_samples, n_features)``; no centring
            is applied here.
        """
        # Perform SVD on the input data matrix X
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        if self.config.n_components is None:
            n_components = S.shape[0]

        else:
            n_components = self.config.n_components

        self.U = U[:, :n_components]
        self.S = S[:n_components]
        self.Vt = Vt[:n_components, :]

    def fit_from_dataloaders(self, dataloaders, feature_extractor):
        """Extract features for every batch, centre them and call ``fit``.

        Parameters
        ----------
        dataloaders : iterable of torch.utils.data.DataLoader
            Loaders whose first batch element is the input image; it is moved
            to CUDA before feature extraction.
        feature_extractor : callable
            Maps a batch of inputs to a ``(batch, n_features)`` tensor.
        """
        X_list = []
        # derive which device to use from feature extractor

        for dataloader in dataloaders:
            for batch in dataloader:
                X_list.append(feature_extractor(batch[0].cuda()).cpu())

        X = torch.cat(X_list, dim=0)
        self.mu = torch.mean(X, dim=0)
        self.fit(X - self.mu)

    def get_components(self):
        """Return the dictionary as a ``(n_features, n_components)`` matrix."""
        return self.Vt.transpose(0, 1)

    def save_on_disk(self, path):
        """Save ``U``, ``S``, ``Vt`` and ``mu`` with ``torch.save`` to ``path``."""
        torch.save({"U": self.U, "S": self.S, "Vt": self.Vt, "mu": self.mu}, path)

    def load_from_disk(self, path):
        """Restore the factors written by :meth:`save_on_disk` from ``path``."""
        checkpoint = torch.load(path)
        self.U = checkpoint["U"]
        self.S = checkpoint["S"]
        self.Vt = checkpoint["Vt"]
        self.mu = checkpoint["mu"]
