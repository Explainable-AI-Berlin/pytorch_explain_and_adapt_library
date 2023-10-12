import torch

from torch import nn
from zennit.layer import Sum

from peal.architectures.module_blocks import VGGBlock, ResnetBlock, TransformerBlock
from peal.architectures.basic_modules import Squeeze, Mean, Transpose, OneHotEncoding
from peal.architectures.basic_modules import DimensionSwitchAttentionLayer


class Img2LatentEncoder(nn.Sequential):
    """
    The encoder takes an image and outputs a latent vector
    """

    def __init__(
        self,
        neuron_numbers,
        blocks_per_layer,
        block_type,
        input_channels,
        use_batchnorm,
        activation,
    ):
        """
        This is the encoder part of the Img2Latent architecture.

        Args:
            neuron_numbers (list): The number of neurons in each layer
            blocks_per_layer (int): The number of blocks in each layer
            block_type (int): The type of block to use
            input_channels (int): The number of channels in the input image
            use_batchnorm (bool): True if batchnorm should be used
            activation (function): The activation function to use
        """
        #
        layers = []
        #
        sublayers = []

        #
        if block_type == "resnet":
            block_factory = ResnetBlock
            layers.append(
                VGGBlock(
                    input_channels,
                    neuron_numbers[0],
                    stride=2,
                    activation=activation,
                    use_batchnorm=False,
                    receptive_field=7,
                )
            )

        elif block_type == "vgg":
            block_factory = VGGBlock
            layers.append(
                VGGBlock(
                    input_channels,
                    neuron_numbers[0],
                    stride=2,
                    activation=activation,
                    use_batchnorm=False,
                    receptive_field=7,
                )
            )

        #
        for i in range(len(neuron_numbers) - 1):
            #
            sublayers = []
            for j in range(blocks_per_layer - 1):
                sublayers.append(
                    block_factory(
                        neuron_numbers[i],
                        neuron_numbers[i],
                        stride=1,
                        activation=activation,
                    )
                )

            sublayers.append(
                block_factory(
                    neuron_numbers[i],
                    neuron_numbers[i + 1],
                    stride=2,
                    activation=activation,
                    use_batchnorm=use_batchnorm,
                )
            )

            layers.append(nn.Sequential(*sublayers))
        #
        super(Img2LatentEncoder, self).__init__(*layers)


class Sequence2LatentEncoder(nn.Sequential):
    """
    The encoder takes a sequence and outputs a latent vector
    """

    def __init__(
        self,
        num_blocks,
        embedding_dim,
        num_heads,
        input_channels,
        activation,
    ):
        """
        The encoder takes a sequence and outputs a latent vector

        Args:
            num_blocks (int): The number of transformer blocks
            embedding_dim (int): The dimension of the token embeddings
            num_heads (int): The number of heads in the multi-head attention
            input_channels (int): The number of channels in the input
            activation (nn.Module): The activation function to use
        """
        layers = []
        # layers.append(nn.Embedding(input_channels + 2, embedding_dim))
        layers.append(OneHotEncoding(input_channels))
        layers.append(Transpose(dims=[1, 2]))
        layers.append(nn.Conv1d(input_channels, embedding_dim))
        layers.append(Transpose(dims=[1, 2]))
        for i in range(num_blocks):
            layers.append(
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    activation,
                )
            )

        layers.append(Mean(dims=[1]))
        #
        super(Sequence2LatentEncoder, self).__init__(*layers)


class Vector2LatentEncoder(nn.Sequential):
    """
    The encoder takes an image and outputs a latent vector
    """

    def __init__(self, input_channels, activation, neuron_numbers=[]):
        """
        The encoder takes an image and outputs a latent vector

        Args:
            input_channels (int): The number of channels in the input
            activation (function): The activation function to use
            neuron_numbers (list, optional): The number of neurons in each layer. Defaults to []. If empty, the output is a single vector
        """
        layers = []
        neurons = [input_channels] + neuron_numbers
        for i in range(len(neurons) - 1):
            layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            layers.append(activation())
        #
        super(Vector2LatentEncoder, self).__init__(*layers)


class Latent2ImgDecoder(nn.Sequential):
    """
    The decoder takes a latent vector and outputs an image
    """

    def __init__(
        self,
        neuron_numbers,
        blocks_per_layer,
        block_type,
        output_size,
        use_batchnorm,
        activation,
    ):
        """
        The decoder takes a latent vector and outputs an image

        Args:
            neuron_numbers (list): The number of neurons in each layer
            blocks_per_layer (int): The number of blocks per layer
            block_type (int): The type of block to use
            output_size (int): The number of channels in the output
            use_batchnorm (bool): True if batchnorm should be used
            activation (int): The activation function to use
        """
        #
        layers = []
        if block_type == "resnet":
            print("not implemented")
            block_factory = None

        #
        for i in range(len(neuron_numbers) - 1):
            sublayers = []
            for j in range(blocks_per_layer - 1):
                sublayers.append(
                    block_factory(neuron_numbers[i], neuron_numbers[i], 1, activation)
                )

            sublayers.append(
                block_factory(
                    neuron_numbers[i],
                    neuron_numbers[i + 1],
                    2,
                    activation,
                    use_batchnorm,
                )
            )
            layers.append(nn.Sequential(*sublayers))
        #
        layers.append(nn.ConvTranspose2d(neuron_numbers[-1], output_size, 3, 2, 1, 1))
        #
        super().__init__(*layers)


class Latent2SequenceDecoder(nn.Module):
    """
    Decodes a latent vector into a sequence of vectors
    """

    def __init__(
        self,
        num_blocks,
        embedding_dim,
        num_heads,
        input_channels,
        activation,
        embedding,
        max_length,
    ):
        """
        The decoder takes a latent vector and outputs a sequence of vectors

        Args:
            num_blocks (int): The number of transformer blocks to use
            embedding_dim (int): The dimension of the latent space
            num_heads (int): The number of heads to use in the multi-head attention
            input_channels (int): The number of channels in the input
            activation (nn.Module): The activation function to use
            embedding (nn.Module): The embedding function to use
        """
        #
        layers = []
        for i in range(num_blocks):
            layers.append(
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    activation,
                    use_masking=True,
                )
            )

        layers.append(Transpose(1, 2))
        layers.append(nn.Conv1d(embedding_dim, input_channels, 1))
        layers.append(Transpose(1, 2))
        #
        super(Latent2SequenceDecoder, self).__init__()
        self.network = nn.Sequential(*layers)
        self.embedding = embedding
        self.unknown_token = self.embedding.num_embeddings - 1
        self.max_length = max_length

    def forward(self, z, max_length=None):
        """
        Forward pass

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        if isinstance(z, list):
            tokens_in = z[1]
            tokens_in = torch.cat(
                [
                    self.unknown_token * torch.ones([z[1].shape[0], 1]).to(tokens_in),
                    tokens_in[:, :-1],
                ],
                dim=1,
            )
            z = z[0]

        else:
            tokens_in = self.unknown_token * torch.ones(
                [z.shape[0], max_length if max_length is not None else self.max_length]
            )

        z = self.embedding(tokens_in) + z

        return self.network(z)


class Latent2VectorDecoder(nn.Sequential):
    """
    The decoder takes a latent vector and outputs a vector
    """

    def __init__(
        self,
        output_size,
        num_hidden_in,
        activation,
        dropout=False,
        latent_height=None,
        dimension_reduction=None,
        neuron_numbers=[],
    ):
        """
        The decoder takes a latent vector and outputs a vector

        Args:
            output_size (int): The number of channels in the output
            num_hidden_in (injt): The number of channels in the input
            activation (function): The activation function to use
            dropout (bool, optional): Whether to use dropout
            latent_height (int, optional): The latent height
            dimension_reduction (int, optional): The type of dimensionality reduction to use
            neuron_numbers (list, optional): The number of neurons in each layer
        """
        layers = {}
        if dimension_reduction == "mean":
            layers["dimensionality_reductor"] = Mean([-2, -1], keepdim=True)
            kernel_size = 1
            num_hidden = num_hidden_in

        elif dimension_reduction == "flatten":
            kernel_size = latent_height
            num_hidden = latent_height * num_hidden_in

        elif dimension_reduction == "sum":
            layers["dimensionality_reductor"] = Sum([-2, -1], keepdim=True)
            kernel_size = 1

        if dimension_reduction in ["mean", "sum"]:
            layers["layer2"] = nn.Conv2d(num_hidden, output_size, 1)
            layers["squeezer"] = Squeeze([-1, -1])

        elif dimension_reduction == ["flatten"]:
            if dropout > 0.0:
                layers["dropout1"] = nn.Dropout(dropout / 2)
            layers["layer1"] = nn.Conv2d(num_hidden_in, num_hidden, kernel_size)
            layers["activation1"] = activation()
            if dropout > 0.0:
                layers["dropout2"] = nn.Dropout(dropout)
            layers["layer2"] = nn.Conv2d(num_hidden, output_size, 1)
            layers["squeezer"] = Squeeze([-1, -1])

        elif dimension_reduction == "attention":
            layers["layer1"] = DimensionSwitchAttentionLayer(output_size, num_hidden, 2)
            layers["activation1"] = activation()
            if dropout > 0.0:
                layers["dropout2"] = nn.Dropout(dropout)
            layers["layer2"] = nn.Conv1d(num_hidden, 1, 1)
            layers["squeezer"] = Squeeze([-2])

        elif dimension_reduction is None:
            layers = {}
            neuron_numbers = [num_hidden_in] + neuron_numbers + [output_size]
            for i in range(len(neuron_numbers) - 1):
                layers["layer_" + str(i)] = nn.Linear(
                    neuron_numbers[i], neuron_numbers[i + 1]
                )
                if i < len(neuron_numbers) - 2:
                    layers["activation_" + str(i)] = activation()

        super(Latent2VectorDecoder, self).__init__(*layers.values())
