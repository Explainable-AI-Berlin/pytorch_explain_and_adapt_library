class SparseDictionary:
    def fit(self, X):
        raise NotImplementedError("Subclasses should implement this method.")

    def transform(self, X):
        raise NotImplementedError("Subclasses should implement this method.")

    def fit_transform(self, X):
        raise NotImplementedError("Subclasses should implement this method.")

    def fit_from_dataloaders(self, dataloaders):
        raise NotImplementedError("Subclasses should implement this method.")

    def save_on_disk(self, path):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_from_disk(self, path):
        raise NotImplementedError("Subclasses should implement this method.")