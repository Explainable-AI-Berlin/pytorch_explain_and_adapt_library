"""SpLICE concept dictionary over CLIP image embeddings.

SpLICE (Bhalla et al.) decomposes a CLIP image embedding into sparse
non-negative weights over a fixed vocabulary of text concepts, so no dictionary
training is needed. PEAL exposes it through the ``SparseDictionary`` interface
so that DiDAE can edit a CLIP-conditioned generator (Stable Diffusion, RAE)
along named concepts. The SpLiCE package is imported from
``peal/dependencies/SpLiCE`` by extending ``sys.path`` at import time.
"""

from typing import Union
import torch
import torch.nn.functional as F
from tqdm import tqdm

from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig

# Import SpLICE from the project dependency
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "dependencies", "SpLiCE")
)
import splice as splice_lib


class SpLICEDecompositionConfig(SparseDictionaryConfig):
    """Configuration for SpLICE-based sparse decomposition.

    Uses a pretrained CLIP model and SpLICE vocabulary to decompose
    image embeddings into sparse concept weights. No training required.

    Parameters
    ----------
    clip_model : str
        SpLICE model name of the CLIP backbone (``"clip:ViT-L/14"``); it must
        be the encoder the generator conditions on.
    vocabulary : str
        Name of the SpLICE concept vocabulary (``"laion"``).
    vocabulary_size : int
        Number of vocabulary entries to keep; ``-1`` uses the full vocabulary.
    l1_penalty : float
        LASSO sparsity penalty of the SpLICE solver.
    solver : str
        ``"skl"`` (scikit-learn) or ``"admm"``.
    ending : str
        File extension of the saved state.
    n_components : int or None
        Set from the vocabulary size once the SpLICE model is loaded.
    component_strings : list or None
        Concept names DiDAE should target when editing.
    opposite_component_strings : list or None
        Opposite concepts used for DDPM-inversion benchmarking.
    benchmark_ddpm_inversion : bool
        If True the explainer edits with text prompts instead of z_sem.
    """

    sparse_dictionaries_type: str = "SpLICEDecomposition"
    clip_model: str = "clip:ViT-L/14"  # CLIP backbone (must match SD's encoder)
    vocabulary: str = "laion"  # SpLICE vocabulary
    vocabulary_size: int = -1  # -1 = full vocabulary
    l1_penalty: float = 0.15  # Sparsity penalty for LASSO
    solver: str = "skl"  # Solver type: 'skl' or 'admm'
    ending: str = ".pt"
    n_components: Union[int, None] = None  # Set dynamically from vocabulary size
    component_strings: Union[list, None] = None  # Targeted concepts for DiDAE edits
    opposite_component_strings: Union[list, None] = (
        None  # Opposites for DDPM benchmarking
    )
    benchmark_ddpm_inversion: bool = (
        False  # If True, use text prompts instead of z_sem for edits
    )


class SpLICEDecomposition(SparseDictionary):
    """Sparse dictionary using SpLICE for concept decomposition.

    Wraps the SpLICE library to decompose CLIP image embeddings into
    sparse nonnegative weights over a vocabulary of text concepts.
    This serves as the disentangled dictionary for the DiDAE framework
    when using pretrained foundation models.

    No training is required — SpLICE uses a pretrained CLIP vocabulary
    dictionary and an L1-regularized solver at inference time. The only
    data-dependent step is computing the dataset-specific image mean
    via fit_from_dataloaders().
    """

    def __init__(self, config=SpLICEDecompositionConfig()):
        """Store the config; the SpLICE model itself is loaded lazily.

        Parameters
        ----------
        config : SpLICEDecompositionConfig
            Backbone, vocabulary and solver settings.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.splice_model = None
        self.image_mean = None
        # Lazy loading: we don't call _load_splice_model() here
        # to avoid double loading CLIP onto GPU if the generator
        # is about to inject its own model via set_clip_model().

    def _load_splice_model(self):
        """Load the SpLICE model with the configured CLIP backbone."""
        self.splice_model = splice_lib.load(
            name=self.config.clip_model,
            vocabulary=self.config.vocabulary,
            vocabulary_size=self.config.vocabulary_size,
            device=self.device,
            l1_penalty=self.config.l1_penalty,
            solver=self.config.solver,
            return_weights=True,
            return_cosine=True,
        )
        self.splice_model.eval()

        # Store original image mean from SpLICE (internet-scale)
        self._original_image_mean = self.splice_model.image_mean.clone()

        # Set n_components from the actual dictionary size
        self.config.n_components = self.splice_model.dictionary.shape[0]

    def set_clip_model(self, clip_model):
        """Inject an external CLIP model to ensure identity with the generator.

        Args:
            clip_model: The loaded CLIP model instance (must be ViT-L/14)
        """
        if self.splice_model is None:
            self._load_splice_model()

        # SpLICE holds the CLIP model in its 'clip' attribute
        # We replace it with the external one to guarantee weight identity
        self.splice_model.clip = clip_model.to(self.device)
        self.splice_model.clip.eval()

    def fit(self, X: torch.Tensor, y: torch.Tensor = None):
        """Set the dataset-specific image mean from precomputed embeddings.

        Parameters
        ----------
        X : torch.Tensor
            CLIP image embeddings ``[N, D]``; their mean replaces SpLICE's
            internet-scale ``image_mean``. Ignored when ``None`` or empty.
        y : torch.Tensor, optional
            Unused; accepted for interface compatibility.
        """
        if self.splice_model is None:
            self._load_splice_model()
        if X is not None and X.shape[0] > 0:
            self.image_mean = X.mean(dim=0)
            self.splice_model.image_mean = self.image_mean.to(self.device)

    def fit_from_dataloaders(self, dataloaders, feature_extractor):
        """Compute dataset-specific image mean using the CLIP encoder.
        This replaces SpLICE's generic internet-scale mean with a
        domain-specific mean, improving decomposition quality for
        the target dataset.

        Args:
            dataloaders: List of dataloaders providing (image, label) batches
            feature_extractor: The CLIP image encoder (nn.Module)
        """
        embeddings_list = []

        try:
            enc_device = next(feature_extractor.parameters()).device
        except StopIteration:
            enc_device = torch.device(self.device)

        feature_extractor.eval()

        with torch.no_grad():
            for dataloader in dataloaders:
                for batch in tqdm(dataloader, desc="Computing dataset CLIP mean"):
                    try:
                        inputs, _ = batch
                    except (ValueError, TypeError):
                        inputs = batch

                    inputs = inputs.to(enc_device)
                    features = feature_extractor(inputs)
                    # Normalize embeddings (CLIP convention)
                    features = F.normalize(features.float(), dim=1)
                    embeddings_list.append(features.cpu())

        all_embeddings = torch.cat(embeddings_list, dim=0)
        self.image_mean = all_embeddings.mean(dim=0)

        # Update the SpLICE model's image mean to use dataset-specific mean
        if self.splice_model is None:
            self._load_splice_model()
        self.splice_model.image_mean = self.image_mean.to(self.device)

    def decompose(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Sparse concept weights of CLIP image embeddings.

        The embeddings are L2-normalised, centred by the image mean,
        re-normalised and passed to ``splice_model.decompose``.

        Parameters
        ----------
        embeddings : torch.Tensor
            CLIP image embeddings ``[N, D]`` on any device.

        Returns
        -------
        torch.Tensor
            Non-negative weights ``[N, n_components]`` on ``self.device``.
        """
        if self.splice_model is None:
            self._load_splice_model()
        embeddings = embeddings.to(self.device).float()
        embeddings = F.normalize(embeddings, dim=1)
        centered = F.normalize(embeddings - self.splice_model.image_mean, dim=1)
        weights = self.splice_model.decompose(centered)
        return weights

    def recompose(self, weights: torch.Tensor) -> torch.Tensor:
        """Reconstruct image embeddings from concept weights.

        Parameters
        ----------
        weights : torch.Tensor
            Concept weights ``[N, n_components]`` as returned by
            :meth:`decompose`.

        Returns
        -------
        torch.Tensor
            Reconstructed (mean-added, normalised) embeddings ``[N, D]``.
        """
        if self.splice_model is None:
            self._load_splice_model()
        return self.splice_model.recompose_image(weights.to(self.device))

    def get_components(self):
        """The concept dictionary as a CPU tensor of shape ``[D, n_components]``.

        Returns
        -------
        torch.Tensor
            One column per vocabulary concept, i.e. the transposed SpLICE
            ``dictionary``.
        """
        if self.splice_model is None:
            self._load_splice_model()
        return self.splice_model.dictionary.T.cpu()

    def save_on_disk(self, path):
        """Save the SpLICE decomposition state (image mean + config)."""
        torch.save(
            {
                "image_mean": self.image_mean,
                "config_clip_model": self.config.clip_model,
                "config_vocabulary": self.config.vocabulary,
                "config_vocabulary_size": self.config.vocabulary_size,
                "config_l1_penalty": self.config.l1_penalty,
                "config_solver": self.config.solver,
            },
            path,
        )

    def load_from_disk(self, path):
        """Load saved SpLICE decomposition state."""
        checkpoint = torch.load(path, map_location="cpu")
        self.image_mean = checkpoint["image_mean"]

        # Reload SpLICE model if not already loaded
        if self.splice_model is None:
            self._load_splice_model()

        # Apply loaded image mean
        self.splice_model.image_mean = self.image_mean.to(self.device)

    def get_vocabulary(self):
        """Get the text vocabulary used by SpLICE.

        Returns:
            vocab: list of concept word strings
        """
        return splice_lib.get_vocabulary(
            self.config.vocabulary, self.config.vocabulary_size
        )
