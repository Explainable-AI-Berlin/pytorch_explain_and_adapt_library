import torch
import copy
import numpy as np
import torch.nn as nn
import code # code.interact(local=dict(globals(), **locals()))

from torch.autograd import Variable

from peal.architectures.interfaces import (
    LocalExplainable,
    Invertible
)

class PgelSequential(nn.Sequential, LocalExplainable, Invertible):
    '''

    '''
    def layer_wise_relevance_propagation(self, input_activation, relevance, layer_idx, fusion_layer = None):
        '''

        '''
        layer_idx += self.num_submodules()
        activations = [input_activation]
        #
        for child_module in self.children():
            activations.append(child_module(activations[-1]))

        relevances = [relevance]
        #
        inverted_order_children = list(self.children())[::-1]
        fusion_layer = None
        for child_idx, child_module in enumerate(inverted_order_children):
            #
            if isinstance(child_module, PgelBatchNorm):
                fusion_layer = child_module
            else:
                relevances.append(child_module.layer_wise_relevance_propagation(activations[-2 - child_idx], relevances[-1], layer_idx, fusion_layer))
                fusion_layer = None

            #
            if isinstance(child_module, PgelSequential):
                layer_idx -= child_module.num_submodules()

            else:
                layer_idx -= 1

        return relevances[-1]


    def num_submodules(self):
        '''

        '''
        n = 0
        for child_module in self.children():
            if isinstance(child_module, PgelSequential):
                n += child_module.num_submodules()
            else:
                n += 1
        return n


    def invert(self):
        '''

        '''
        layers = []
        for submodule in list(self.children())[::-1]:
            layers.append(submodule.invert())
        return PgelSequential(*layers)


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


class PgelConv2d(nn.Conv2d, LocalExplainable):
    '''

    '''
    def layer_wise_relevance_propagation(self, input_activation, relevance, layer_idx, fusion_layer = None):
        '''

        '''
        #
        input_activation = Variable(input_activation, requires_grad = True)

        #
        if layer_idx <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
        if 17 <= layer_idx <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
        if layer_idx >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

        #
        layer = copy.deepcopy(self)
        # conv(x) = weight1 * x + bias1
        # bn(x) = x * weight2 + bias2
        # bn(conv(x)) = weight2 * (weight1 * x + bias1) + bias2 = weight1 * weight2 * x + weight2 * bias1 + bias2
        if not fusion_layer is None:
            layer.weight.data = torch.tile(
                fusion_layer.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                [1, layer.weight.shape[1], layer.weight.shape[2], layer.weight.shape[3]]
            ) * layer.weight
            layer.bias.data = fusion_layer.weight * layer.bias + fusion_layer.bias

        layer.weight = nn.Parameter(rho(layer.weight))
        layer.bias = nn.Parameter(rho(layer.bias))
        # step 1
        z = incr(layer.forward(input_activation))
        # step 2
        s = (relevance / z).data
        (z*s).sum().backward()
        # step 3
        c = input_activation.grad
        # step 4
        return (input_activation*c).data

    def invert(self):
        '''
        TODO still to implement - should be implemented with a transposed convolution!
        '''
        pseudoinvers_matrix = torch.linalg.pinv(torch.transpose(self.weight, 0, 1))
        fc_inverted = nn.Linear(pseudoinvers_matrix.shape[0], pseudoinvers_matrix.shape[1])
        fc_inverted.weight.data = torch.transpose(pseudoinvers_matrix, 0, 1)
        fc_inverted.bias.data = -1 * torch.tensordot(self.bias, pseudoinvers_matrix, dims = 1)
        return fc_inverted #fc_inverted


class PgelConv1d(nn.Conv1d, LocalExplainable):
    pass


class PgelLinear(nn.Linear, LocalExplainable, Invertible):
    '''

    '''
    def layer_wise_relevance_propagation(self, input_activation, relevance, layer_idx, fusion_layer = None):
        '''

        '''
        #
        input_activation = Variable(input_activation, requires_grad = True)

        #
        if layer_idx <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
        if 17 <= layer_idx <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
        if layer_idx >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

        #
        layer = copy.deepcopy(self)
        # conv(x) = weight1 * x + bias1
        # bn(x) = x * weight2 + bias2
        # bn(conv(x)) = weight2 * (weight1 * x + bias1) + bias2 = weight1 * weight2 * x + weight2 * bias1 + bias2
        if not fusion_layer is None:
            layer.weight.data = torch.tile(
                fusion_layer.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                [1, layer.weight.shape[1], layer.weight.shape[2], layer.weight.shape[3]]
            ) * layer.weight
            layer.bias.data = fusion_layer.weight * layer.bias + fusion_layer.bias

        layer.weight = nn.Parameter(rho(layer.weight))
        layer.bias = nn.Parameter(rho(layer.bias))
        # step 1
        z = incr(layer.forward(input_activation))
        # step 2
        s = (relevance / z).data
        (z*s).sum().backward()
        # step 3
        c = input_activation.grad
        # step 4
        return (input_activation*c).data

    def invert(self):
        '''

        '''
        pseudoinvers_matrix = torch.linalg.pinv(torch.transpose(self.weight, 0, 1))
        fc_inverted = nn.Linear(pseudoinvers_matrix.shape[0], pseudoinvers_matrix.shape[1])
        fc_inverted.weight.data = torch.transpose(pseudoinvers_matrix, 0, 1)
        fc_inverted.bias.data = -1 * torch.tensordot(self.bias, pseudoinvers_matrix, dims = 1)
        return fc_inverted #fc_inverted


class PgelLeakyReLU(nn.LeakyReLU, LocalExplainable, Invertible):
    '''
    '''
    def __init__(self, negative_slope = 0.01, positive_slope = 1.0):
        super(PgelLeakyReLU, self).__init__(negative_slope = negative_slope)
        self.positive_slope = positive_slope
        self.negative_slope = negative_slope

    def forward(self, x):
        x = self.positive_slope * super().forward(x)
        return x

    def invert(self):
        return PgelLeakyReLU(1 / self.negative_slope, 1 / self.positive_slope)


class PgelBatchNorm(nn.BatchNorm2d, LocalExplainable):
    pass


class Unsqueeze(nn.Module, LocalExplainable, Invertible):
    def __init__(self, dims):
        super(Unsqueeze, self).__init__()
        self.dims = dims

    def forward(self, x):
        for dim in self.dims:
            x = torch.unsqueeze(x, dim)
        return x

    def layer_wise_relevance_propagation(self, input_activation, relevance, layer_idx, fusion_layer = None):
        for dim in self.dims[::-1]:
            relevance = relevance.squeeze(dim)
        return relevance

    def invert(self):
        return Squeeze(self.dims[::-1])


class Squeeze(nn.Module, LocalExplainable, Invertible):
    def __init__(self, dims):
        super(Squeeze, self).__init__()
        self.dims = dims

    def forward(self, x):
        for dim in self.dims:
            x = torch.squeeze(x, dim)
        return x

    def layer_wise_relevance_propagation(self, input_activation, relevance, layer_idx, fusion_layer = None):
        for dim in self.dims[::-1]:
            relevance = relevance.unsqueeze(dim)
        return relevance

    def invert(self):
        return Unsqueeze(self.dims[::-1])


class Flatten(nn.Module, LocalExplainable, Invertible):
    def __init__(self, keepdim, input_shape = None):
        super(Flatten, self).__init__()
        self.keepdim = keepdim
        self.input_shape = input_shape

    def forward(self, x):
        self.input_shape = [-1] + list(x.shape[1:])
        if self.keepdim:
            return torch.reshape(x, [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        else:
            return torch.flatten(x, 1)

    def layer_wise_relevance_propagation(self, input_activation, relevance, layer_idx, fusion_layer = None):
        return torch.reshape(relevance, input_activation.shape)

    def invert(self):
        assert not self.input_shape is None, 'Input shape must either be given or at least one forward pass needs to be done!'
        return Reshape(self.input_shape)


class Reshape(nn.Module, LocalExplainable, Invertible):
    def __init__(self, output_shape, input_shape = None):
        super(Reshape, self).__init__()
        self.output_shape = output_shape
        self.input_shape = input_shape

    def forward(self, x):
        self.input_shape = [-1] + list(x.shape[1:])
        return torch.reshape(x, self.output_shape)

    def layer_wise_relevance_propagation(self, input_activation, relevance, layer_idx, fusion_layer = None):
        return torch.reshape(relevance, input_activation.shape)

    def invert(self):
        assert not self.input_shape is None, 'Input shape must either be given or at least one forward pass needs to be done!'
        return Reshape(self.input_shape)


class Distribute(nn.Module, LocalExplainable, Invertible):
    '''
    TODO: Careful, this class is not fully working yet!
    '''
    def __init__(self, dims, output_shape, create_dim):
        super(Distribute, self).__init__()
        self.dims = dims
        self.output_shape = output_shape
        self.create_dim = create_dim

    def forward(self, x):
        for dim in self.dims:
            if self.create_dim:
                x = torch.unsqueeze(x, dim)
            x = torch.tile()
            x = torch.matmul(weight)
        return x

    def layer_wise_relevance_propagation(self, input_activation, relevance, layer_idx, fusion_layer = None):
        input_activation = Variable(input_activation, requires_grad = True)        

        #
        if layer_idx <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
        if 17 <= layer_idx <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
        if layer_idx >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

        # step 1
        z = incr(self.forward(input_activation))
        # step 2
        s = (relevance / z).data
        (z*s).sum().backward()
        # step 3
        c = input_activation.grad
        # step 4
        return (input_activation*c).data

    def invert(self):
        return Mean()


class Mean(nn.Module, LocalExplainable, Invertible):
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

    def layer_wise_relevance_propagation(self, input_activation, relevance, layer_idx, fusion_layer = None):
        input_activation = Variable(input_activation, requires_grad = True)        

        #
        if layer_idx <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
        if 17 <= layer_idx <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
        if layer_idx >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

        # step 1
        z = incr(self.forward(input_activation))
        # step 2
        s = (relevance / z).data
        (z*s).sum().backward()
        # step 3
        c = input_activation.grad
        # step 4
        return (input_activation*c).data

    def invert(self):
        assert not self.input_shape is None, 'Input shape must either be given or at least one forward pass needs to be done!'
        return Distribute(self.dims[::-1], self.input_shape, not self.keepdim)


class Sum(nn.Module, LocalExplainable):
    def __init__(self, dims, keepdim = True):
        super(Sum, self).__init__()
        self.dims = dims
        self.keepdim = keepdim

    def forward(self, x):
        for dim in self.dims:
            x = torch.sum(x, dim, keepdim = self.keepdim)
        return x

    def layer_wise_relevance_propagation(self, input_activation, relevance, layer_idx, fusion_layer = None):
        input_activation = Variable(input_activation, requires_grad = True)        

        #
        if layer_idx <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
        if 17 <= layer_idx <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
        if layer_idx >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

        # step 1
        z = incr(self.forward(input_activation))
        # step 2
        s = (relevance / z).data
        (z*s).sum().backward()
        # step 3
        c = input_activation.grad
        # step 4
        return (input_activation*c).data


class PgelDropout(nn.Dropout, LocalExplainable):
    pass


class PgelReLU(nn.ReLU, LocalExplainable):
    pass


class PgelIdentity(nn.Identity, LocalExplainable):
    pass