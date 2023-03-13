import torch

from torch import nn

from zennit.layer import Sum

class SkipConnection(nn.Module):
    def __init__(self, module, downsample = None):
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
        out =  torch.stack([x, x_in], dim=-1)
        out = self.sum(out)
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.attention_head = nn.MultiheadAttention(embed_dim = inplanes + 2, num_heads = 1, batch_first = True)

    def forward(self, x):
        identity = x
        #
        positional_encodings = []
        for i in range(2, len(x.shape)):
            positional_encodings.append(torch.arange(x.shape[i]).to(torch.float32))
            #
            positional_encodings[-1] = (positional_encodings[-1] - positional_encodings[-1].mean()) / positional_encodings[-1].var()
            # 1 x 1 x C x 1
            tile_shape = []
            for j in range(len(x.shape)):
                if i != j:
                    positional_encodings[-1] = positional_encodings[-1].unsqueeze(j)
                    if 1 != j:
                        tile_shape.append(x.shape[j])

                    else:
                        tile_shape.append(1)

                else:
                    tile_shape.append(1)

            positional_encodings[-1] = torch.tile(positional_encodings[-1], tile_shape).to(x.device)

        #
        x = torch.cat([x] + positional_encodings, dim = 1)
        x = torch.flatten(x, 2)
        x = torch.transpose(x, 1, 2)
        #
        x = self.attention_head(x, x, x)[0][:,:,:-len(positional_encodings)]
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, identity.shape)

        return x


class DimensionSwitchAttentionLayer(nn.Module):
    def __init__(self, output_size, num_hidden, num_positional_encodings):
        super().__init__()
        self.lookup_table = Variable(torch.randn([output_size, num_hidden + num_positional_encodings]))
        self.attention_layer = nn.MultiheadAttention(embed_dim = num_hidden, num_heads = 1, batch_first = True)

    def forward(self, x):
        #
        positional_encodings = []
        for i in range(2, len(x.shape)):
            positional_encodings.append(torch.arange(x.shape[i]).to(torch.float32))
            #
            positional_encodings[-1] = (positional_encodings[-1] - positional_encodings[-1].mean()) / positional_encodings[-1].var()
            # 1 x 1 x C x 1
            tile_shape = []
            for j in range(len(x.shape)):
                if i != j:
                    positional_encodings[-1] = positional_encodings[-1].unsqueeze(j)
                    if 1 != j:
                        tile_shape.append(x.shape[j])

                    else:
                        tile_shape.append(1)

                else:
                    tile_shape.append(1)

            positional_encodings[-1] = torch.tile(positional_encodings[-1], tile_shape).to(x.device)
        #
        x = torch.cat([x] + positional_encodings, dim = 1)
        x = torch.flatten(x, 2)
        x = torch.transpose(x, 1, 2)
        query = torch.tile(self.lookup_table.to(x.device).unsqueeze(0), [x.shape[0], 1, 1])
        key = x
        value = x
        x = self.attention_layer(query, key, value)[0][:,:,:-len(positional_encodings)]
        x = x.transpose(1, 2)
        return x

'''
class MultiLatent2VectorDecoder(nn.Module):
    def __init__(self, output_size, num_hidden):
        super().__init__()
        self.lookup_table = Variable(torch.randn([output_size, num_hidden]))
        self.attention_layer = nn.MultiheadAttention(embed_dim = num_hidden, num_heads = 1, batch_first = True)
        self.conv = nn.Conv1d(
            neuron_numbers[-1],
            1,
            1
        )

    def forward(self, x):
        x = torch.flatten(x, 2)
        x = torch.transpose(x, 1, 2)
        query = torch.tile(self.lookup_table.to(x.device).unsqueeze(0), [x.shape[0], 1, 1])
        key = x
        value = x
        x = self.attention_layer(query, key, value)[0]
        x = x.transpose(1, 2)
        #x = PgelDropout(self.dropout)(x)
        x = self.conv(x)
        x = x.squeeze(1)
        return x

class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.attention_head = nn.MultiheadAttention(embed_dim = inplanes + 2, num_heads = 1, batch_first = True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv = nn.Conv2d(
            inplanes,
            planes,
            1
        )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x
        #
        x_positions = torch.arange(x.shape[-2]).to(torch.float32)
        y_positions = torch.arange(x.shape[-1]).to(torch.float32)
        #
        x_positions = (x_positions - x_positions.mean()) / x_positions.var()
        y_positions = (y_positions - y_positions.mean()) / y_positions.var()
        #
        x_positions = x_positions.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
        y_positions = y_positions.unsqueeze(0).unsqueeze(1).unsqueeze(-2)
        #
        x_positions = torch.tile(x_positions, [x.shape[0], 1, 1, x.shape[-1]]).to(x.device)
        y_positions = torch.tile(y_positions, [x.shape[0], 1, x.shape[-2], 1]).to(x.device)
        #
        x = torch.cat([x, x_positions, y_positions], dim = 1)
        x = torch.flatten(x, 2)
        x = torch.transpose(x, 1, 2)
        #
        x = self.attention_head(x, x, x)[0][:,:,:-2]
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, identity.shape)
        x = self.bn1(x)
        x = x + identity
        identity = x

        x = self.conv(x)
        x = self.bn2(x)
        x = x + identity

        return x
'''