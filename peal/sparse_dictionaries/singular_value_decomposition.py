from typing import Union

import torch

from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig


class SVDDictionaryConfig(SparseDictionaryConfig):
    n_components: Union[int, None] = 10
    sparse_dictionaries_type: str = 'SVDDictionary'
    ending: str = '.npz'

class SVDDictionary(SparseDictionary):
    def __init__(self, config=SVDDictionaryConfig()):
        self.config = config
        self.U = None
        self.S = None
        self.Vt = None
        self.mu = None

    def fit(self, X):
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
        X_list = []
        # derive which device to use from feature extractor

        for dataloader in dataloaders:
            for batch in dataloader:
                X_list.append(feature_extractor(batch[0].cuda()).cpu())

        X = torch.cat(X_list, dim=0)
        self.mu = torch.mean(X, dim=0)
        self.fit(X - self.mu)

    def get_components(self):
        return self.Vt.transpose(0, 1)

    def save_on_disk(self, path):
        torch.save({
            'U': self.U,
            'S': self.S,
            'Vt': self.Vt,
            'mu': self.mu
        }, path)

    def load_from_disk(self, path):
        checkpoint = torch.load(path)
        self.U = checkpoint['U']
        self.S = checkpoint['S']
        self.Vt = checkpoint['Vt']
        self.mu = checkpoint['mu']
