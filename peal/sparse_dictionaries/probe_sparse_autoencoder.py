from typing import Union

import torch

from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig


class ProbeSAEConfig(SparseDictionaryConfig):
    pretrained_model_path: str
    n_components: Union[int, None] = 2
    sparse_dictionaries_type: str = 'ProbeSAE'
    ending: str = '.npz'

class ProbeSAE(SparseDictionary):
    def __init__(self, config):
        self.config = config
        model = torch.load(config.pretrained_model_path, map_location='cpu')
        self.W = model.model.model.fc.weight.data.clone()
        self.mu = torch.zeros(self.W.size(1))

    def fit(self, X):
        pass

    def fit_from_dataloaders(self, dataloaders, feature_extractor):
        pass

    def get_components(self):
        return self.W.t()

    def save_on_disk(self, path):
        torch.save({
            'W': self.W,
        }, path)

    def load_from_disk(self, path):
        checkpoint = torch.load(path)
        self.W = checkpoint['W']
