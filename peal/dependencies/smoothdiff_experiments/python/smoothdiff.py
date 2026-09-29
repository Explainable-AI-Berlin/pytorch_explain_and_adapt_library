import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

SUPPORTED_LAYERS = {
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Linear,
    nn.Dropout,
}


class SmoothReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, collect_stats, smooth_backward, grad_local_summed, n_samples):
        if collect_stats:
            grad_local_summed += x > 0
            n_samples += 1
        ctx.smooth_backward = smooth_backward
        if smooth_backward:
            ctx.n_samples = n_samples
            ctx.save_for_backward(grad_local_summed)
        else:
            ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.smooth_backward:
            assert ctx.n_samples > 0
            (grad_local_summed,) = ctx.saved_tensors
            smooth_grad = grad_local_summed / ctx.n_samples.item()
            return grad_output * smooth_grad, None, None, None, None
        else:
            (x,) = ctx.saved_tensors
            return grad_output * (x > 0), None, None, None, None


class SmoothMaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        collect_stats,
        smooth_backward,
        grad_local_unfolded_summed,
        n_samples,
        kernel_size,
        stride,
        padding,
        dilation,
    ):
        assert (
            x.ndim == 4
        ), "Input must be 4D (N, C, H, W) (for torch.nn.functional.unfold)"
        assert (
            type(kernel_size) is int and type(stride) is int
        ), "kernel_size and stride must be equal across dimensions (for torch.nn.functional.unfold)"
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.smooth_backward = smooth_backward
        ctx.input_shape = x.shape

        # Unfold input into patches
        unfolded = F.unfold(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        unfolded = unfolded.view(x.size(0), x.size(1), kernel_size**2, -1)

        # Max pooling over patches
        max_vals, max_indices = unfolded.max(dim=2, keepdims=True)

        with torch.no_grad():
            # Compute mask of max values
            # in the edge case where multiple values in patch are equal to max this can diverge from MaxPool2d
            grad_local_unfolded = unfolded == max_vals

            if grad_local_unfolded_summed is None:
                grad_local_unfolded_summed = 0
            if collect_stats:
                grad_local_unfolded_summed += grad_local_unfolded
                n_samples += 1

        if smooth_backward:
            # these determine smooth local gradient:
            ctx.n_samples = n_samples
            ctx.save_for_backward(grad_local_unfolded_summed)
        else:
            # the one-hot vectors are the local derivative:
            ctx.save_for_backward(grad_local_unfolded)

        def calc_output_size(input_size, kernel_size, stride, padding, dilation):
            return (
                input_size + 2 * padding - dilation * (kernel_size - 1) - 1
            ) // stride + 1

        output_h = calc_output_size(x.size(2), kernel_size, stride, padding, dilation)
        output_w = calc_output_size(x.size(3), kernel_size, stride, padding, dilation)

        return max_vals.view(x.size(0), x.size(1), output_h, output_w)

    @staticmethod
    def backward(ctx, grad_output):
        N, C, H, W = ctx.input_shape

        if ctx.smooth_backward:
            assert ctx.n_samples > 0
            (grad_local_unfolded_summed,) = ctx.saved_tensors
            grad_local_unfolded = grad_local_unfolded_summed / ctx.n_samples.item()
        else:
            (grad_local_unfolded,) = ctx.saved_tensors

        grad_input_unfolded = (
            grad_output.flatten(start_dim=2).unsqueeze(2) * grad_local_unfolded
        )
        grad_input_unfolded = grad_input_unfolded.reshape(
            N, C * ctx.kernel_size**2, -1
        )  # (N, C*k², L)
        grad_input = F.fold(
            grad_input_unfolded,
            output_size=(H, W),
            kernel_size=ctx.kernel_size,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
        )

        return grad_input, None, None, None, None, None, None, None, None


class SmoothDiffLayer(nn.Module):
    def __init__(self, collect_stats=False, smooth_backward=False):
        super().__init__()
        self.collect_stats = collect_stats
        self.smooth_backward = smooth_backward
        self.register_buffer("grad_local_sum", None)
        self.register_buffer("n_samples", torch.tensor(0))

    def reset_stats(self):
        self.grad_local_summed = None
        self.n_samples.zero_()

    def forward(self, x):
        raise NotImplementedError("Must be implemented in subclass")


class SmoothReLU(SmoothDiffLayer):
    def forward(self, x):
        if self.grad_local_summed is None:
            # gradients are stored in the form of one sample:
            self.grad_local_summed = torch.zeros_like(x)

        return SmoothReLUFunction.apply(
            x,
            self.collect_stats,
            self.smooth_backward,
            self.grad_local_summed,
            self.n_samples,
        )


class SmoothMaxPool2d(SmoothDiffLayer):
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        if self.grad_local_summed is None:
            # gradients are stored in the form of one *unfolded* sample:
            unfolded = F.unfold(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            unfolded = unfolded.view(x.size(0), x.size(1), self.kernel_size**2, -1)
            self.grad_local_summed = torch.zeros_like(unfolded)
        return SmoothMaxPool2dFunction.apply(
            x,
            self.collect_stats,
            self.smooth_backward,
            self.grad_local_summed,
            self.n_samples,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )


def set_smoothdiff_layer_mode(model, collect_stats=None, smooth_backward=None):
    for module in model.modules():
        if isinstance(module, SmoothDiffLayer):
            if collect_stats is not None:
                module.collect_stats = collect_stats
                if collect_stats:
                    module.reset_stats()
            if smooth_backward is not None:
                module.smooth_backward = smooth_backward


def smooth_layer(l):
    if isinstance(l, torch.nn.ReLU):
        return SmoothReLU()
    if isinstance(l, torch.nn.MaxPool2d):
        return SmoothMaxPool2d(
            kernel_size=l.kernel_size,
            stride=l.stride,
            padding=l.padding,
            dilation=l.dilation,
        )
    return l


## SmoothDiff model preparation utilities


def check_supported_layers(model: nn.Module):
    """
    Check whether all layers in a PyTorch model are supported by SmoothDiff.
    Raise an error if any unsupported layer is found.
    """

    def _check_module(module, name="", is_root=False):
        has_children = len(list(module.children())) > 0

        # Only check leaf modules (no children) and skip the root
        if not is_root and not has_children:
            module_type = type(module)
            if module_type not in SUPPORTED_LAYERS:
                raise ValueError(
                    f"Unsupported layer type found: '{module_type.__name__}' at '{name if name else 'root'}'.\nPlease open a feature request at https://github.com/adrhill/smoothdiff-experiments/issues"
                )

        # Recursively check all child modules
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            _check_module(child_module, full_name, is_root=False)

    _check_module(model, is_root=True)


def replace_nonlinear_layers(model):
    """
    Recursively replace ReLU and MaxPool2d layers in any PyTorch model.
    """
    # Create a copy of the model to avoid modifying the original
    check_supported_layers(model)
    model_copy = copy.deepcopy(model)

    def _replace_in_module(module):
        for name, child in module.named_children():
            # First, recursively process children
            _replace_in_module(child)

            # Then replace the current child if it matches our criteria
            new_layer = smooth_layer(child)
            if new_layer is not child:  # Only replace if it actually changed
                setattr(module, name, new_layer)

    _replace_in_module(model_copy)
    return model_copy


def reset_statisticcs(model):
    for module in model.modules():
        if isinstance(module, SmoothDiffLayer):
            print("reset states")
            module.reset_stats()
