import copy
from typing import Union
import torch
import torch.nn.functional as F
from tqdm import tqdm

from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig

# Import SpLICE from the project dependency
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dependencies', 'SpLiCE'))
import splice as splice_lib


class SpLICEDecompositionConfig(SparseDictionaryConfig):
    """Configuration for SpLICE-based sparse decomposition.
    
    Uses a pretrained CLIP model and SpLICE vocabulary to decompose
    image embeddings into sparse concept weights. No training required.
    """
    sparse_dictionaries_type: str = "SpLICEDecomposition"
    clip_model: str = "clip:ViT-L/14"       # CLIP backbone (must match SD's encoder)
    vocabulary: str = "laion"                 # SpLICE vocabulary
    vocabulary_size: int = -1                 # -1 = full vocabulary
    l1_penalty: float = 0.15                  # Sparsity penalty for LASSO
    solver: str = "skl"                       # Solver type: 'skl' or 'admm'
    ending: str = ".pt"
    n_components: Union[int, None] = None     # Set dynamically from vocabulary size


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
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.splice_model = None
        self.image_mean = None
        self._load_splice_model()

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
        """No-op — SpLICE uses a pretrained dictionary, not learned from data.
        
        If X is provided, computes the image mean from it for centering.
        """
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
        self.splice_model.image_mean = self.image_mean.to(self.device)

    def decompose(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decompose CLIP embeddings into sparse concept weights.
        
        Args:
            embeddings: (batch, clip_dim) normalized CLIP image embeddings
            
        Returns:
            weights: (batch, n_concepts) sparse weight vector
        """
        embeddings = embeddings.to(self.device).float()
        embeddings = F.normalize(embeddings, dim=1)
        centered = F.normalize(
            embeddings - self.splice_model.image_mean, dim=1
        )
        weights = self.splice_model.decompose(centered)
        return weights

    def recompose(self, weights: torch.Tensor) -> torch.Tensor:
        """Reconstruct dense CLIP embeddings from sparse weights.
        
        Args:
            weights: (batch, n_concepts) sparse weight vector
            
        Returns:
            recon: (batch, clip_dim) reconstructed CLIP embedding
        """
        return self.splice_model.recompose_image(weights.to(self.device))

    def get_components(self):
        """Return the concept dictionary matrix.
        
        Returns:
            components: (clip_dim, n_concepts) — transposed to match
            the convention used in OrthogonalProcrustesDictionary and
            DiffusionAutoencoder._calculate_z_counterfactuals()
        """
        # SpLICE dictionary is (n_concepts, clip_dim), transpose to (clip_dim, n_concepts)
        return self.splice_model.dictionary.T.cpu()

    def save_on_disk(self, path):
        """Save the SpLICE decomposition state (image mean + config)."""
        torch.save({
            'image_mean': self.image_mean,
            'config_clip_model': self.config.clip_model,
            'config_vocabulary': self.config.vocabulary,
            'config_vocabulary_size': self.config.vocabulary_size,
            'config_l1_penalty': self.config.l1_penalty,
            'config_solver': self.config.solver,
        }, path)

    def load_from_disk(self, path):
        """Load saved SpLICE decomposition state."""
        checkpoint = torch.load(path, map_location='cpu')
        self.image_mean = checkpoint['image_mean']
        
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
            self.config.vocabulary, 
            self.config.vocabulary_size
        )
