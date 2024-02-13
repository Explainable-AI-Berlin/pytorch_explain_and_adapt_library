from torch import nn

class Generator(nn.Module):
    def sample_x(self, batch_size=1):
        """
        This function samples a batch of data samples from the generator.
        If not implemented, it will throw a NotImplementedError.
        """
        raise NotImplementedError

    def train_model(self):
        """
        This function trains the generator.
        If not implemented, it will throw a NotImplementedError.
        """
        raise NotImplementedError