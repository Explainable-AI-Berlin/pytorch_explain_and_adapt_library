"""
Some key layers used for constructing a Capsule Network.

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/Proto-Caps`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from scipy.spatial.distance import cdist


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def uniform(tensor, bound = 10.):
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def compute_adjacency_matrix_images(coord, sigma=0.01):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(- dist / (sigma * np.pi) ** 2)
    A[np.diag_indices_from(A)] = 0
    return A

class Multi_Head_Graph_Pooling(nn.Module):
    def __init__(self, num_caps_types, map_size, n_caps, output_dim, add_loop=True, improved=False, bias=True):
        super(Multi_Head_Graph_Pooling, self).__init__()
        self.n_caps = n_caps
        self.num_caps_types = num_caps_types
        self.map_size = map_size
        self.output_dim = output_dim

        coord = np.zeros((map_size, map_size, 2))
        for i in range(map_size):
            for j in range(map_size):
                coord[i][j][0] = i + 1
                coord[i][j][1] = j + 1

        adj = torch.from_numpy(compute_adjacency_matrix_images(coord)).float()

        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not improved else 2

        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj_buffer = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        self.register_buffer('adj', adj_buffer)

        self.weight = nn.Parameter(torch.Tensor(output_dim, n_caps))
        self.bias = nn.Parameter(torch.Tensor(n_caps))

        uniform(self.weight)
        zeros(self.bias)

    def forward(self, u_predict):
        x = u_predict.view(len(u_predict) * self.num_caps_types, self.map_size * self.map_size, -1)
        # print(x.shape)
        # print(self.weight.shape)
        s = torch.matmul(x, self.weight)
        s = torch.matmul(self.adj, s)
        s = s + self.bias

        s = torch.softmax(s, dim=1)
        saliency_map = torch.mean(s, 0)
        saliency_map = saliency_map.view(self.map_size, self.map_size, self.n_caps)

        x = torch.matmul(s.transpose(1, 2), x)

        u_predict = x.view(len(u_predict), -1, self.n_caps, self.output_dim)

        v = u_predict.sum(dim=1) / u_predict.size()[2]
        v = squash(v)
        return v, saliency_map

class GraphCapsule(nn.Module):
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, aggregation_module):
        super(GraphCapsule, self).__init__()
        self.in_dim_caps = in_dim_caps
        self.in_num_caps = in_num_caps
        self.out_dim_caps = out_dim_caps
        self.out_num_caps = out_num_caps

        self.weights = nn.Parameter(torch.Tensor(self.in_num_caps, in_dim_caps, out_dim_caps))

        self.aggregation_module = aggregation_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_num_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)

        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.in_num_caps, self.out_dim_caps)

        v, saliency_map = self.aggregation_module(u_predict)
        return v, saliency_map

def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of capsules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3, activation_fn="softmax"):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
        self.activation_fn = activation_fn

    def forward(self, x):
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            if self.activation_fn == "softmax":
                c = F.softmax(b, dim=1)
            elif self.activation_fn == "sigmoid":
                c = torch.sigmoid(b)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)
        return torch.squeeze(outputs, dim=-2)


class PrimaryCapsule(nn.Module):
    """
    Apply Conv with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """

    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, threeD, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        if threeD:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv(x)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        return squash(outputs)
