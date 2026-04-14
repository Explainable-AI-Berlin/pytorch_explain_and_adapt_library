import copy
from typing import Union
import torch
import torch.nn.functional as F

from peal.architectures.interfaces import TaskConfig
from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig

class OrthogonalProcrustesDictionaryConfig(SparseDictionaryConfig):
    # n_components now effectively determines the Output Dimension (Number of Classes)
    n_components: Union[int, None] = 2
    sparse_dictionaries_type: str = 'OrthogonalProcrustesDictionary'
    ending: str = '.npz'
    task: Union[type(None), dict] = None


class OrthogonalProcrustesDictionary(SparseDictionary):
    def __init__(self, config=OrthogonalProcrustesDictionaryConfig()):
        self.config = config
        self.W_ortho = None  # This will hold the orthogonal weights (K, D)
        self.mu = None

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Solves the Orthogonal Procrustes problem:
        Find W such that ||XW^T - Y|| is minimized subject to W @ W^T = I.
        """
        # 1. Determine Output Dimension (K)
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
            raise ValueError(f"Feature dimension ({D}) must be >= Output dimension ({K}) for strict orthogonality.")

        # 3. Compute Cross-Covariance Matrix M = Y^T @ X
        # This captures the correlation between targets and features
        M = torch.matmul(Y.T, X)  # Shape: (K, D)

        # 4. Perform SVD on the Correlation Matrix
        # We decompose the correlation to find the optimal rotation
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # 5. Compute the Optimal Orthogonal Weights
        # W = U @ Vh aligns X to Y best
        self.W_ortho = torch.matmul(U, Vh)  # Shape: (K, D)

    def fit_from_dataloaders(self, dataloaders, feature_extractor):
        X_list = []
        y_list = []

        # Determine device from the feature extractor
        try:
            device = next(feature_extractor.parameters()).device

        except:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        feature_extractor.eval()

        with torch.no_grad():
            for dataloader in dataloaders:
                task_config_buffer = copy.deepcopy(dataloader.dataset.task_config)
                task_config = TaskConfig(**self.config.task) if self.config.task is not None else None
                dataloader.dataset.task_config = task_config
                for batch in dataloader:
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device) # Get targets from batch[1]

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
        return self.W_ortho.t() if self.W_ortho is not None else None

    def save_on_disk(self, path):
        torch.save({
            'W_ortho': self.W_ortho,
            'mu': self.mu
        }, path)

    def load_from_disk(self, path):
        checkpoint = torch.load(path)
        self.W_ortho = checkpoint['W_ortho']
        self.mu = checkpoint['mu']