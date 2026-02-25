import os

from peal.data.datasets import SymbolicDataset
from peal.data.dataset_generators import CircleDatasetGenerator

class CircleDataset(SymbolicDataset):
    def __init__(self, mode, config, **kwargs):
        if not os.path.exists(config.dataset_path):
            # use the circle dataset generator
            circle_dataset_generator = CircleDatasetGenerator(config)
            circle_dataset_generator.generate_dataset()
        super(CircleDataset, self).__init__(mode=mode, config=config, **kwargs)
        self.hints_enabled = False
        self.idx_enabled = False

    def calculate_outlier_score(self, x):
        import torch
        outlier_scores = {"absolute" : torch.zeros(x.shape[0], device=x.device)}
        if hasattr(self, "reference_outlier_scores") and self.reference_outlier_scores is not None:
            outlier_scores["relative"] = outlier_scores["absolute"] / (
                self.reference_outlier_scores + 1e-8
            )
        else:
            outlier_scores["relative"] = outlier_scores["absolute"]
        return outlier_scores