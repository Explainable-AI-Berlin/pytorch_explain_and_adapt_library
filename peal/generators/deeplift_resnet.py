"""ResNet variants with one ``nn.ReLU`` instance per activation site.

Captum's DeepLift hooks every ReLU module and requires each to be used
exactly once per forward pass; torchvision's ResNet reuses one ``relu``
inside every block, which breaks the attribution. These classes mirror
torchvision's ``BasicBlock`` / ``Bottleneck`` / ``ResNet`` with distinct
non-inplace ReLUs so that DeepLift can attribute through them. They are used
by the Stable Diffusion 3 generator when it rebuilds a predictor for
attribution-guided sampling (``DeepLiftResNet18`` ... ``152``).
"""

# Add these class definitions to your file or a separate module
from torch import nn
import torch


class DeepLiftBasicBlock(nn.Module):
    """BasicBlock with separate ReLU instances for DeepLift compatibility"""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        """Build the two 3x3 conv layers of a basic residual block.

        Parameters
        ----------
        inplanes : int
            Input channels.
        planes : int
            Output channels of both convolutions.
        stride : int
            Stride of the first convolution.
        downsample : nn.Module or None
            Projection applied to the identity path when shapes differ.
        groups, base_width : int
            Must stay at ``1`` / ``64``; other values raise ``ValueError``.
        dilation : int
            Must be ``1``; larger values raise ``NotImplementedError``.
        norm_layer : callable or None
            Normalisation layer factory, ``nn.BatchNorm2d`` by default.
        """
        super(DeepLiftBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
            bias=False,
            dilation=1,
        )
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=False)  # First ReLU - unique instance
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
            dilation=1,
        )
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=False)  # Second ReLU - unique instance
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Residual forward pass with a distinct ReLU after each stage."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)  # First ReLU

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)  # Second ReLU

        return out


class DeepLiftBottleneck(nn.Module):
    """Bottleneck with separate ReLU instances for DeepLift compatibility"""

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        """Build the 1x1 -> 3x3 -> 1x1 convolutions of a bottleneck block.

        Parameters
        ----------
        inplanes : int
            Input channels.
        planes : int
            Bottleneck width before the ``expansion`` factor (output has
            ``planes * 4`` channels).
        stride : int
            Stride of the 3x3 convolution.
        downsample : nn.Module or None
            Projection applied to the identity path when shapes differ.
        groups : int
            Groups of the 3x3 convolution.
        base_width : int
            Scales the inner width as ``planes * base_width / 64``.
        dilation : int
            Dilation (and padding) of the 3x3 convolution.
        norm_layer : callable or None
            Normalisation layer factory, ``nn.BatchNorm2d`` by default.
        """
        super(DeepLiftBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=False)  # First ReLU

        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=False)  # Second ReLU

        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=False)  # Third ReLU

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Residual forward pass with a distinct ReLU after each stage."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)  # First ReLU

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)  # Second ReLU

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)  # Third ReLU

        return out


class DeepLiftResNet(nn.Module):
    """Base ResNet class with unique ReLU instances for DeepLift compatibility"""

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        """Assemble the stem, four residual stages, pooling and the linear head.

        Parameters
        ----------
        block : type
            ``DeepLiftBasicBlock`` or ``DeepLiftBottleneck``.
        layers : list of int
            Number of blocks in each of the four stages.
        num_classes : int
            Output width of the final linear layer.
        zero_init_residual : bool
            Accepted for signature compatibility with torchvision; not applied.
        groups, width_per_group : int
            Passed to every block as ``groups`` / ``base_width``.
        replace_stride_with_dilation : list of bool or None
            Per stage 2-4, replace the stride-2 downsampling by dilation.
        norm_layer : callable or None
            Normalisation layer factory, ``nn.BatchNorm2d`` by default.
        """
        super(DeepLiftResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)  # Initial ReLU - unique instance
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Stack ``blocks`` residual blocks, adding a downsample path if needed."""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """Standard ResNet forward pass returning class logits ``[B, num_classes]``."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def DeepLiftResNet18(num_classes=1000):
    """DeepLift-compatible ResNet-18 (basic blocks ``[2, 2, 2, 2]``).

    Parameters
    ----------
    num_classes : int
        Output width of the classifier head.

    Returns
    -------
    DeepLiftResNet
        A randomly initialised model.
    """
    return DeepLiftResNet(DeepLiftBasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def DeepLiftResNet34(num_classes=1000):
    """DeepLift-compatible ResNet-34 (basic blocks ``[3, 4, 6, 3]``).

    Parameters
    ----------
    num_classes : int
        Output width of the classifier head.

    Returns
    -------
    DeepLiftResNet
        A randomly initialised model.
    """
    return DeepLiftResNet(DeepLiftBasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def DeepLiftResNet50(num_classes=1000):
    """DeepLift-compatible ResNet-50 (bottlenecks ``[3, 4, 6, 3]``).

    Parameters
    ----------
    num_classes : int
        Output width of the classifier head.

    Returns
    -------
    DeepLiftResNet
        A randomly initialised model.
    """
    return DeepLiftResNet(DeepLiftBottleneck, [3, 4, 6, 3], num_classes=num_classes)


def DeepLiftResNet101(num_classes=1000):
    """DeepLift-compatible ResNet-101 (bottlenecks ``[3, 4, 23, 3]``).

    Parameters
    ----------
    num_classes : int
        Output width of the classifier head.

    Returns
    -------
    DeepLiftResNet
        A randomly initialised model.
    """
    return DeepLiftResNet(DeepLiftBottleneck, [3, 4, 23, 3], num_classes=num_classes)


def DeepLiftResNet152(num_classes=1000):
    """DeepLift-compatible ResNet-152 (bottlenecks ``[3, 8, 36, 3]``).

    Parameters
    ----------
    num_classes : int
        Output width of the classifier head.

    Returns
    -------
    DeepLiftResNet
        A randomly initialised model.
    """
    return DeepLiftResNet(DeepLiftBottleneck, [3, 8, 36, 3], num_classes=num_classes)
