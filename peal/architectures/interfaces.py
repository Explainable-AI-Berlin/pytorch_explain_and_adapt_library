from torch import nn

class LocalExplainable:
    def layer_wise_relevance_propagation(self, input_activation, relevance, layer_idx, fusion_layer = None):
        # In case the function is not overwritten the relevance will be propagated through without any changes
        return relevance


class Invertible:
    def invert(self):
        # In case the function is not overwritten the inverse is assumed to be replacable with the Identity function
        return nn.Identity()

class InvertibleGenerator(nn.Module):
	def encode(self, x):
		pass

	def decode(self, z):
		pass

	def sample_z(self):
		pass

	def log_prob_z(self, z):
		pass

	def sample_x(self):
		z = self.sample_z()
		return self.decode(z)

	def log_prob_x(self, x):
		z = self.encode(x)
		return self.log_prob_z(z)