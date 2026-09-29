"""Small ``nn.Module`` building blocks used by PEAL's configurable architectures.

The predictor and generator builders in :mod:`peal.architectures` assemble
networks from layer lists; these modules wrap tensor operations (transpose,
squeeze, mean, one-hot) so they can appear in ``nn.Sequential``, plus a
zennit-friendly residual block and a few attention layers with hand-rolled
positional encodings for sequence and image inputs.
"""

import torch
import math

from zennit.layer import Sum
from torch.autograd import Variable
from torch import nn


class Transpose(nn.Module):
    """Module form of ``x.transpose(dim1, dim2)``.

    Parameters
    ----------
    dim1, dim2 : int
        Dimensions to swap.
    """

    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        """Swap the two configured dimensions of ``x``."""
        return x.transpose(self.dim1, self.dim2)


class OneHotEncoding(nn.Module):
    """One-hot encode integer inputs that arrive as a 2D tensor.

    Parameters
    ----------
    num_classes : int
        Size of the one-hot axis appended by ``torch.nn.functional.one_hot``.
    """

    def __init__(self, num_classes):
        super(OneHotEncoding, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        """Return ``one_hot(x)`` if ``x`` is 2D, otherwise ``x`` unchanged."""
        if len(x.shape) == 2:
            x = torch.nn.functional.one_hot(x, num_classes=self.num_classes)

        return x


class Unsqueeze(nn.Module):
    """Insert singleton dimensions, one ``torch.unsqueeze`` per entry of ``dims``.

    Parameters
    ----------
    dims : list of int
        Dimensions to insert, applied in order (later entries see the already
        expanded shape).
    """

    def __init__(self, dims):
        super(Unsqueeze, self).__init__()
        self.dims = dims

    def forward(self, x):
        """Unsqueeze ``x`` at every configured dimension."""
        for dim in self.dims:
            x = torch.unsqueeze(x, dim)
        return x


class Squeeze(nn.Module):
    """Remove singleton dimensions, one ``torch.squeeze`` per entry of ``dims``.

    Parameters
    ----------
    dims : list of int
        Dimensions to squeeze, applied in order.
    """

    def __init__(self, dims):
        super(Squeeze, self).__init__()
        self.dims = dims

    def forward(self, x):
        """Squeeze ``x`` at every configured dimension."""
        for dim in self.dims:
            x = torch.squeeze(x, dim)
        return x


class Mean(nn.Module):
    """Average over a set of dimensions (global average pooling by default).

    Parameters
    ----------
    dims : list of int, optional
        Dimensions to reduce sequentially. If None, every dimension from 2
        upwards is reduced in reverse order, i.e. all spatial axes of an
        ``(N, C, ...)`` tensor.
    keepdim : bool, default False
        Passed to ``torch.mean``.
    input_shape : list, optional
        Initial value of the ``input_shape`` attribute; overwritten on every
        forward pass with ``[-1] + list(x.shape[1:])`` so that inverse
        (un-pooling) layers can look it up.
    """

    def __init__(self, dims=None, keepdim=False, input_shape=None):
        super(Mean, self).__init__()
        self.dims = dims
        self.keepdim = keepdim
        self.input_shape = input_shape

    def forward(self, x):
        """Record ``input_shape`` and reduce ``x`` over the configured dims."""
        self.input_shape = [-1] + list(x.shape[1:])
        if self.dims is None:
            dims = list(range(2, len(x.shape))[::-1])

        else:
            dims = self.dims

        for dim in dims:
            x = torch.mean(x, dim, keepdim=self.keepdim)

        return x


class SkipConnection(nn.Module):
    """Residual block ``module(x) + downsample(x)`` with a zennit ``Sum`` layer.

    The two branches are stacked along a new last axis and summed with
    ``zennit.layer.Sum`` instead of ``+`` so that LRP attribution rules can be
    attached to the addition.

    Parameters
    ----------
    module : nn.Module
        Main branch.
    downsample : nn.Module, optional
        Shortcut branch; ``nn.Identity`` when None. Its output must match the
        shape of ``module(x)``.
    """

    def __init__(self, module, downsample=None):
        super(SkipConnection, self).__init__()
        self.module = module
        if not downsample is None:
            self.downsample = downsample

        else:
            self.downsample = nn.Identity()

        self.sum = Sum()

    def forward(self, x_in):
        """Return the sum of the main branch and the shortcut branch."""
        x = self.module(x_in)
        x_in = self.downsample(x_in)
        out = torch.stack([x, x_in], dim=-1)
        out = self.sum(out)
        return out


class SelfAttentionLayer(nn.Module):
    """Multi-head self-attention over a ``(B, L, C)`` sequence.

    A sinusoidal positional encoding is computed from the sequence length and
    added to the input on every forward pass before attention is applied.

    Parameters
    ----------
    inplanes : int
        Feature size ``C`` (embedding dimension of the attention).
    num_heads : int, default 1
        Number of attention heads.
    use_masking : bool, default False
        If True a causal (upper-triangular) mask is applied so that position
        ``i`` only attends to positions ``<= i``.
    """

    def __init__(
        self,
        inplanes: int,
        num_heads: int = 1,
        use_masking: bool = False,
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.attention_head = nn.MultiheadAttention(
            embed_dim=inplanes, num_heads=num_heads, batch_first=True
        )
        self.use_masking = use_masking

    def forward(self, x):
        """Add sin/cos positional encodings and apply self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, L, C)``; ``C`` must be even for the encoding.

        Returns
        -------
        torch.Tensor
            Attention output of shape ``(B, L, C)``.
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
    """Self-attention over the spatial positions of an image feature map.

    Two extra channels holding normalised row / column coordinates are
    concatenated to the input as positional encoding, the map is flattened to a
    ``(B, H*W, C+2)`` sequence, attended, and reshaped back to ``(B, C, H, W)``.

    Parameters
    ----------
    inplanes : int
        Number of input channels ``C``; the attention embedding is ``C + 2``.
    num_heads : int, default 1
        Number of attention heads.
    use_masking : bool, default False
        Apply a causal mask over the flattened positions.
    """

    def __init__(
        self,
        inplanes: int,
        num_heads: int = 1,
        use_masking: bool = False,
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.attention_head = nn.MultiheadAttention(
            embed_dim=inplanes + 2, num_heads=num_heads, batch_first=True
        )
        self.use_masking = use_masking

    def forward(self, x):
        """Attend over spatial positions of ``x`` of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``; the coordinate channels are dropped again.
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
    """Cross-attention that maps a spatial feature map to ``output_size`` slots.

    A fixed random lookup table of ``output_size`` query vectors attends over the
    flattened, coordinate-augmented input positions, so the spatial axis is
    replaced by a learned-query axis of fixed length. The table is created as a
    ``torch.autograd.Variable`` rather than an ``nn.Parameter``, so it is neither
    trained nor stored in ``state_dict``.

    Parameters
    ----------
    output_size : int
        Number of query slots (length of the output sequence).
    num_hidden : int
        Embedding dimension of the attention layer.
    num_positional_encodings : int
        Number of coordinate channels the input will carry; added to the width
        of the lookup table.
    """

    def __init__(self, output_size, num_hidden, num_positional_encodings):
        super().__init__()
        self.lookup_table = Variable(
            torch.randn([output_size, num_hidden + num_positional_encodings])
        )
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=num_hidden, num_heads=1, batch_first=True
        )

    def forward(self, x):
        """Cross-attend the lookup queries over the positions of ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, C, *spatial)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_hidden - len(spatial), output_size)`` after the
            coordinate channels are removed and the axes transposed.
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
