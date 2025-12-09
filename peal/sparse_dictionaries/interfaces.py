from typing import Union

from pydantic import BaseModel


class SparseDictionaryConfig(BaseModel):
    n_components: Union[int, None]
    sparse_dictionary_type: str
    ending: str

class SparseDictionary:
    config : SparseDictionaryConfig
    def fit(self, X):
        raise NotImplementedError("Subclasses should implement this method.")

    def fit_from_dataloaders(self, dataloaders):
        raise NotImplementedError("Subclasses should implement this method.")

    def save_on_disk(self, path):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_from_disk(self, path):
        raise NotImplementedError("Subclasses should implement this method.")