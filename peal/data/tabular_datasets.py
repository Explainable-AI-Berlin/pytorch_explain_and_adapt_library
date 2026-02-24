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