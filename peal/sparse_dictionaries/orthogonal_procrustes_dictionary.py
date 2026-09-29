"""Supervised concept dictionary fitted by orthogonal Procrustes analysis.

Unlike the unsupervised SAEs in this subpackage, this dictionary uses the
dataset labels: it finds the (optionally orthonormal) directions ``W`` in
feature space that best map centred activations onto the one-hot / multi-hot
targets, so each atom is the direction of one labelled attribute. DiDAE uses
it as the oracle-style baseline (``Procrustes-CFKD``) against which the
SAE-based direction search is compared.
"""

import copy
from typing import Union
import torch
import torch.nn.functional as F

from peal.architectures.interfaces import TaskConfig
from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig
from peal.log import get_logger

_log = get_logger(__name__)


class OrthogonalProcrustesDictionaryConfig(SparseDictionaryConfig):
    # n_components now effectively determines the Output Dimension (Number of Classes)
    """Config of ``OrthogonalProcrustesDictionary``.

    Parameters
    ----------
    n_components : int or None
        Number of output directions ``K``. ``None`` infers it from the targets
        (number of columns of a multi-hot ``y`` or ``max(y) + 1`` for indices).
    sparse_dictionaries_type : str
        Fixed to ``"OrthogonalProcrustesDictionary"`` for the factory.
    ending : str
        File ending of the saved weights.
    task : dict or None
        ``TaskConfig`` fields applied to the datasets while extracting features,
        so ``batch[1]`` yields the attributes to fit on.
    orthogonal : bool
        ``True`` solves the orthogonal Procrustes problem (one orthonormal set of
        atoms); ``False`` fits each attribute direction independently as the
        unit-norm class-mean-difference direction. See the attribute note below
        for why the free fit is preferable with many attributes.
    """
    n_components: Union[int, None] = 2
    sparse_dictionaries_type: str = "OrthogonalProcrustesDictionary"
    ending: str = ".npz"
    task: Union[type(None), dict] = None
    orthogonal: bool = True
    """False: fit every attribute direction on its own (the row of the
    cross-covariance Y^T X, i.e. the class-mean-difference direction, unit
    norm) instead of one orthonormal set. Orthonormality distorts the atoms
    once K is large: with 40 CelebA attributes the orthogonal Male atom has
    cosine 0.61 to the Male atom of the 2-attribute fit, and its edit flips
    the student 14% of the time where the 2-attribute atom flips 59%
    (2026-09-14). The 2-attribute fit is unaffected either way."""


class OrthogonalProcrustesDictionary(SparseDictionary):
    """Label-supervised dictionary of class-mean directions in feature space.

    ``fit`` solves ``min ||X W^T - Y||`` subject to ``W W^T = I`` via the SVD of
    the cross-covariance ``Y^T X`` (or, with ``config.orthogonal=False``, takes
    the normalised rows of that cross-covariance directly).
    ``fit_from_dataloaders`` extracts and centres the features first.

    Parameters
    ----------
    config : OrthogonalProcrustesDictionaryConfig
        Fit options; see the config class.

    Attributes
    ----------
    W_ortho : torch.Tensor or None
        Fitted directions of shape ``[K, D]`` (rows are unit atoms).
    mu : torch.Tensor or None
        Feature mean of shape ``[D]`` that was subtracted before fitting; edits
        project ``z - mu`` onto the atoms.
    """

    def __init__(self, config=OrthogonalProcrustesDictionaryConfig()):
        """Store the config; ``W_ortho`` and ``mu`` stay ``None`` until fitted."""
        self.config = config
        self.W_ortho = None  # This will hold the orthogonal weights (K, D)
        self.mu = None

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Solves the Orthogonal Procrustes problem:
        Find W such that ||XW^T - Y|| is minimized subject to W @ W^T = I.
        """
        # 1. Determine Output Dimension (K)
        _log.info("%s", "finding W!")
        if self.config.n_components is not None:
            K = self.config.n_components
        else:
            # Infer K from targets if not specified
            if y.dim() > 1:
                K = y.shape[1]
            else:
                K = int(y.max().item()) + 1

        # 2. Prepare Target Matrix Y
        # If y is a 1D tensor of class indices, convert to One-Hot
        if y.dim() == 1 or (y.dim() == 2 and y.shape[1] == 1):
            y_indices = y.long().view(-1)
            # Create One-Hot Float Matrix (N, K)
            Y = F.one_hot(y_indices, num_classes=K).float()
        else:
            # Assume y is already a matrix of vectors (N, K)
            Y = y.float()

        # Check Dimensions
        N, D = X.shape
        if D < K:
            raise ValueError(
                f"Feature dimension ({D}) must be >= Output dimension ({K}) for strict orthogonality."
            )

        # 3. Compute Cross-Covariance Matrix M = Y^T @ X
        # This captures the correlation between targets and features
        M = torch.matmul(Y.T, X)  # Shape: (K, D)

        if not getattr(self.config, "orthogonal", True):
            # Per-attribute directions, no orthogonality constraint: each row of
            # M is the (centred) mean of the samples carrying that attribute.
            self.W_ortho = M / (M.norm(dim=1, keepdim=True) + 1e-8)  # (K, D)
            _log.info(
                "%s", f"[Procrustes] fitted {K} free (non-orthogonal) unit directions"
            )
            return

        # 4. Perform SVD on the Correlation Matrix
        # We decompose the correlation to find the optimal rotation
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # 5. Compute the Optimal Orthogonal Weights
        # W = U @ Vh aligns X to Y best
        self.W_ortho = torch.matmul(U, Vh)  # Shape: (K, D)

    def fit_from_dataloaders(self, dataloaders, feature_extractor):
        """Extract features for every dataloader, centre them and call ``fit``.

        The dataset's ``task_config`` is temporarily replaced by
        ``TaskConfig(**config.task)`` (or ``None``) while iterating so the loader
        yields the attributes the dictionary should be supervised with, then
        restored. Features are collected on CPU; ``self.mu`` is set to their mean.

        Parameters
        ----------
        dataloaders : list of torch.utils.data.DataLoader
            Loaders yielding ``(inputs, targets)`` batches; all are concatenated.
        feature_extractor : torch.nn.Module
            Encoder mapping inputs to ``[N, D]`` features; put into eval mode and
            run without gradients. Its parameters' device is used for inference.
        """
        X_list = []
        y_list = []

        # Determine device from the feature extractor
        try:
            device = next(feature_extractor.parameters()).device

        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feature_extractor.eval()

        with torch.no_grad():
            for dataloader in dataloaders:
                task_config_buffer = copy.deepcopy(dataloader.dataset.task_config)
                task_config = (
                    TaskConfig(**self.config.task)
                    if self.config.task is not None
                    else None
                )
                _log.info("%s", "extracting, features!")
                dataloader.dataset.task_config = task_config
                for batch in dataloader:
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device)  # Get targets from batch[1]

                    features = feature_extractor(inputs)

                    # Move to CPU to save GPU memory during collection
                    X_list.append(features.cpu())
                    y_list.append(targets.cpu())

                dataloader.dataset.task_config = task_config_buffer

        # Concatenate all batches
        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)

        # Calculate Mean (Centroid) of features
        self.mu = torch.mean(X, dim=0)

        # Center the data before fitting (Critical for correct rotation)
        self.fit(X - self.mu, y)

    def get_components(self):
        # Returns the dictionary atoms (Weight vectors)
        # Transposed to (D, K) to match standard Linear layer shape or previous SVD output
        """Return the atoms as a ``[D, K]`` matrix (columns are directions).

        Returns
        -------
        torch.Tensor or None
            ``W_ortho`` transposed to match the ``[D, K]`` layout of the other
            dictionaries, or ``None`` before ``fit`` was called.
        """
        return self.W_ortho.t() if self.W_ortho is not None else None

    def save_on_disk(self, path):
        """Save ``W_ortho`` and ``mu`` with ``torch.save`` to ``path``."""
        torch.save({"W_ortho": self.W_ortho, "mu": self.mu}, path)

    def load_from_disk(self, path):
        """Restore ``W_ortho`` and ``mu`` from a file written by ``save_on_disk``."""
        checkpoint = torch.load(path)
        self.W_ortho = checkpoint["W_ortho"]
        self.mu = checkpoint["mu"]
