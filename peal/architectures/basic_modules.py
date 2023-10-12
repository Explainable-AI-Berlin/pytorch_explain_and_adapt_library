import torch
import math

from zennit.layer import Sum
from torch.autograd import Variable
from torch import nn


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OneHotEncoding(nn.Module):
    def __init__(self, num_classes):
        super(OneHotEncoding, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.nn.functional.one_hot(x, num_classes=self.num_classes)

        return x


class Unsqueeze(nn.Module):
    """
    _summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dims):
        """
        _summary_

        Args:
            dims (_type_): _description_
        """
        super(Unsqueeze, self).__init__()
        self.dims = dims

    def forward(self, x):
        """
        _summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        for dim in self.dims:
            x = torch.unsqueeze(x, dim)
        return x


class Squeeze(nn.Module):
    """
    _summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dims):
        """
        _summary_

        Args:
            dims (_type_): _description_
        """
        super(Squeeze, self).__init__()
        self.dims = dims

    def forward(self, x):
        """
        _summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        for dim in self.dims:
            x = torch.squeeze(x, dim)
        return x


class Mean(nn.Module):
    """
    _summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dims=None, keepdim=False, input_shape=None):
        """
        _summary_

        Args:
            dims (_type_): _description_
            keepdim (bool, optional): _description_. Defaults to True.
            input_shape (_type_, optional): _description_. Defaults to None.
        """
        super(Mean, self).__init__()
        self.dims = dims
        self.keepdim = keepdim
        self.input_shape = input_shape

    def forward(self, x):
        """
        _summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.input_shape = [-1] + list(x.shape[1:])
        if self.dims is None:
            dims = list(range(2, len(x.shape))[::-1])

        else:
            dims = self.dims

        for dim in dims:
            x = torch.mean(x, dim, keepdim=self.keepdim)

        return x


class SkipConnection(nn.Module):
    """
    _summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, module, downsample=None):
        """
        _summary_

        Args:
            module (_type_): _description_
            downsample (_type_, optional): _description_. Defaults to None.
        """
        super(SkipConnection, self).__init__()
        self.module = module
        if not downsample is None:
            self.downsample = downsample

        else:
            self.downsample = nn.Identity()

        self.sum = Sum()

    def forward(self, x_in):
        """
        _summary_

        Args:
            x_in (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.module(x_in)
        x_in = self.downsample(x_in)
        out = torch.stack([x, x_in], dim=-1)
        out = self.sum(out)
        return out


class SelfAttentionLayer(nn.Module):
    """
    _summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        inplanes: int,
        num_heads: int = 1,
        use_masking: bool = False,
    ) -> None:
        """
        _summary_

        Args:
            inplanes (int): _description_
            num_heads (int, optional): _description_. Defaults to 1.
            use_masking (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.attention_head = nn.MultiheadAttention(
            embed_dim=inplanes, num_heads=num_heads, batch_first=True
        )
        self.use_masking = use_masking

    def forward(self, x):
        """
        _summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_in = x
        # create an empty positional encoding matrix
        pos_enc = torch.zeros(x.shape[1], x.shape[-1])
        # calculate the position and dimension values for each element in the matrix
        pos = torch.arange(x.shape[1], dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, x.shape[-1], 2).float() * (-math.log(10000.0) / x.shape[-1])
        )
        # apply the sin/cos formula to each element in the matrix
        pos_enc[:, 0::2] = torch.sin(pos * div)
        pos_enc[:, 1::2] = torch.cos(pos * div)
        pos_enc = pos_enc.unsqueeze(0)

        x = x + pos_enc.to(x)

        #
        if self.use_masking:
            mask = torch.ones(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
            mask = torch.triu(mask, diagonal=1)
            x = self.attention_head(x, x, x, attn_mask=mask)[0]

        else:
            x = self.attention_head(x, x, x)[0]

        return x


class ImgSelfAttentionLayer(nn.Module):
    """
    _summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        inplanes: int,
        num_heads: int = 1,
        use_masking: bool = False,
    ) -> None:
        """
        _summary_

        Args:
            inplanes (int): _description_
            num_heads (int, optional): _description_. Defaults to 1.
            use_masking (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.attention_head = nn.MultiheadAttention(
            embed_dim=inplanes + 2, num_heads=num_heads, batch_first=True
        )
        self.use_masking = use_masking

    def forward(self, x):
        """
        _summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        identity = x
        #
        positional_encodings = []
        for i in range(2, len(x.shape)):
            positional_encodings.append(torch.arange(x.shape[i]).to(torch.float32))
            #
            positional_encodings[-1] = (
                positional_encodings[-1] - positional_encodings[-1].mean()
            ) / positional_encodings[-1].var()
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

            positional_encodings[-1] = torch.tile(
                positional_encodings[-1],
                tile_shape,
            ).to(x.device)

        #
        x = torch.cat([x] + positional_encodings, dim=1)
        x = torch.flatten(x, 2)
        x = torch.transpose(x, 1, 2)
        import pdb

        pdb.set_trace()

        #
        if self.use_masking:
            mask = torch.ones(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
            mask = torch.triu(mask, diagonal=1)
            x = self.attention_head(x, x, x, attn_mask=mask)

        else:
            x = self.attention_head(x, x, x)

        x = x[0][:, :, : -len(positional_encodings)]
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, identity.shape)

        return x


class DimensionSwitchAttentionLayer(nn.Module):
    """
    _summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, output_size, num_hidden, num_positional_encodings):
        """
        _summary_

        Args:
            output_size (_type_): _description_
            num_hidden (_type_): _description_
            num_positional_encodings (_type_): _description_
        """
        super().__init__()
        self.lookup_table = Variable(
            torch.randn([output_size, num_hidden + num_positional_encodings])
        )
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=num_hidden, num_heads=1, batch_first=True
        )

    def forward(self, x):
        """
        _summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        #
        positional_encodings = []
        for i in range(2, len(x.shape)):
            positional_encodings.append(torch.arange(x.shape[i]).to(torch.float32))
            #
            positional_encodings[-1] = (
                positional_encodings[-1] - positional_encodings[-1].mean()
            ) / positional_encodings[-1].var()
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

            positional_encodings[-1] = torch.tile(
                positional_encodings[-1], tile_shape
            ).to(x.device)
        #
        x = torch.cat([x] + positional_encodings, dim=1)
        x = torch.flatten(x, 2)
        x = torch.transpose(x, 1, 2)
        query = torch.tile(
            self.lookup_table.to(x.device).unsqueeze(0), [x.shape[0], 1, 1]
        )
        key = x
        value = x
        x = self.attention_layer(query, key, value)[0][
            :, :, : -len(positional_encodings)
        ]
        x = x.transpose(1, 2)
        return x
