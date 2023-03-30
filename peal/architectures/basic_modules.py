import torch

from zennit.layer import Sum
from torch.autograd import Variable
from torch import nn

class TwoPathNetwork(nn.Module):
    '''

    '''
    def __init__(self, network):
        '''

        '''
        super().__init__()
        self.network = network

    def forward(self, x):
        '''

        '''
        return self.network(x), x


class NoiseLayer(nn.Module):
    '''

    '''
    def forward(self, x):
        '''

        '''
        if self.training:
            return x + torch.randn(x.shape).to(x.device)

        else:
            return x


class Unsqueeze(nn.Module):
    def __init__(self, dims):
        super(Unsqueeze, self).__init__()
        self.dims = dims

    def forward(self, x):
        for dim in self.dims:
            x = torch.unsqueeze(x, dim)
        return x


class Squeeze(nn.Module):
    def __init__(self, dims):
        super(Squeeze, self).__init__()
        self.dims = dims

    def forward(self, x):
        for dim in self.dims:
            x = torch.squeeze(x, dim)
        return x


class Mean(nn.Module):
    def __init__(self, dims, keepdim = True, input_shape = None):
        super(Mean, self).__init__()
        self.dims = dims
        self.keepdim = keepdim
        self.input_shape = input_shape

    def forward(self, x):
        self.input_shape = [-1] + list(x.shape[1:])
        for dim in self.dims:
            x = torch.mean(x, dim, keepdim = self.keepdim)

        return x


class SkipConnection(nn.Module):
    def __init__(self, module, downsample=None):
        super(SkipConnection, self).__init__()
        self.module = module
        if not downsample is None:
            self.downsample = downsample

        else:
            self.downsample = nn.Identity()

        self.sum = Sum()

    def forward(self, x_in):
        x = self.module(x_in)
        x_in = self.downsample(x_in)
        out = torch.stack([x, x_in], dim=-1)
        out = self.sum(out)
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        inplanes: int
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.attention_head = nn.MultiheadAttention(
            embed_dim=inplanes + 2, num_heads=1, batch_first=True)

    def forward(self, x):
        identity = x
        #
        positional_encodings = []
        for i in range(2, len(x.shape)):
            positional_encodings.append(
                torch.arange(x.shape[i]).to(torch.float32))
            #
            positional_encodings[-1] = (positional_encodings[-1] -
                                        positional_encodings[-1].mean()) / positional_encodings[-1].var()
            # 1 x 1 x C x 1
            tile_shape = []
            for j in range(len(x.shape)):
                if i != j:
                    positional_encodings[-1] = positional_encodings[-1].unsqueeze(
                        j)
                    if 1 != j:
                        tile_shape.append(x.shape[j])

                    else:
                        tile_shape.append(1)

                else:
                    tile_shape.append(1)

            positional_encodings[-1] = torch.tile(
                positional_encodings[-1], tile_shape).to(x.device)

        #
        x = torch.cat([x] + positional_encodings, dim=1)
        x = torch.flatten(x, 2)
        x = torch.transpose(x, 1, 2)
        #
        x = self.attention_head(x, x, x)[0][:, :, :-len(positional_encodings)]
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, identity.shape)

        return x


class DimensionSwitchAttentionLayer(nn.Module):
    def __init__(self, output_size, num_hidden, num_positional_encodings):
        super().__init__()
        self.lookup_table = Variable(torch.randn(
            [output_size, num_hidden + num_positional_encodings]))
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=num_hidden, num_heads=1, batch_first=True)

    def forward(self, x):
        #
        positional_encodings = []
        for i in range(2, len(x.shape)):
            positional_encodings.append(
                torch.arange(x.shape[i]).to(torch.float32))
            #
            positional_encodings[-1] = (positional_encodings[-1] -
                                        positional_encodings[-1].mean()) / positional_encodings[-1].var()
            # 1 x 1 x C x 1
            tile_shape = []
            for j in range(len(x.shape)):
                if i != j:
                    positional_encodings[-1] = positional_encodings[-1].unsqueeze(
                        j)
                    if 1 != j:
                        tile_shape.append(x.shape[j])

                    else:
                        tile_shape.append(1)

                else:
                    tile_shape.append(1)

            positional_encodings[-1] = torch.tile(
                positional_encodings[-1], tile_shape).to(x.device)
        #
        x = torch.cat([x] + positional_encodings, dim=1)
        x = torch.flatten(x, 2)
        x = torch.transpose(x, 1, 2)
        query = torch.tile(self.lookup_table.to(
            x.device).unsqueeze(0), [x.shape[0], 1, 1])
        key = x
        value = x
        x = self.attention_layer(query, key, value)[0][:, :,:-len(positional_encodings)]
        x = x.transpose(1, 2)
        return x
