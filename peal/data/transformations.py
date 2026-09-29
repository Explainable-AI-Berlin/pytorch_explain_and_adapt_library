"""Callable image transformations used by PEAL datasets and generators.

The classes here are plugged into ``torchvision.transforms.Compose``
pipelines built by the datasets: geometric augmentations (``RandomRotation``,
``RandomResizeCropPad``, ``Padding``, ``CircularCut``), channel handling
(``SetChannels``), invertible normalization (``Normalization`` and its no-op
twin ``IdentityNormalization``, both exposing ``invert`` so tensors can be
mapped back to image space) and ``DiffusionAugmentation``, which augments an
image by encoding and re-decoding it with a PEAL generator.
"""

import torch
import torchvision
import random

# import imgaug.augmenters as iaa
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

from peal._optional import require
from peal.generators.generator_factory import get_generator


class CircularCut(object):
    """
    Black out everything outside the ellipse inscribed in a PIL image.

    A white ellipse spanning the full image is rasterized with pygame and
    combined with the input by an element-wise minimum, so pixels outside
    the ellipse become black while the inside is unchanged.
    """

    def __call__(self, sample):
        """
        Apply the elliptical cut.

        Parameters
        ----------
        sample : PIL.Image.Image
            RGB input image.

        Returns
        -------
        PIL.Image.Image
            Image with the corners outside the inscribed ellipse set to black.

        Raises
        ------
        ImportError
            If the optional ``pygame`` dependency is not installed.
        """
        pygame = require("pygame", "pygame", "rasterizing the elliptical mask")

        sample_np = np.array(sample)
        background_color = (0, 0, 0)
        (width, height) = (sample_np.shape[1], sample_np.shape[0])
        screen = pygame.Surface((width, height))
        screen.fill(background_color)
        pygame.draw.ellipse(screen, (255, 255, 255), pygame.Rect(0, 0, width, height))
        overlay = Image.frombytes(
            "RGB", (width, height), pygame.image.tostring(screen, "RGB")
        )
        overlay_np = np.array(overlay)
        img_cut = np.minimum(overlay_np, sample_np)
        return Image.fromarray(img_cut)


class Padding(object):
    """
    Symmetrically pad a ``[C, H, W]`` tensor up to a fixed spatial size.

    Parameters
    ----------
    input_size : tuple of int
        Target ``(height, width)``. The input must not be larger than this.
    """

    def __init__(self, input_size):
        """Store the target size."""
        self.input_size = input_size

    def __call__(self, sample):
        """
        Pad ``sample`` to ``input_size`` with grey (128) borders.

        Parameters
        ----------
        sample : torch.Tensor
            Image of shape ``[C, H, W]``.

        Returns
        -------
        torch.Tensor
            Image of shape ``[C, input_size[0], input_size[1]]``; odd size
            differences put the extra pixel on the bottom/right.
        """
        dif_x = self.input_size[0] - sample.shape[1]
        dif_y = self.input_size[1] - sample.shape[2]
        padding = transforms.Pad(
            [
                int(dif_y / 2),
                int(dif_x / 2),
                int(dif_y / 2) + dif_y % 2,
                int(dif_x / 2) + dif_x % 2,
            ],
            fill=128,
        )
        return padding(sample)


class RandomRotation(object):
    """
    Rotate a tensor image by a random integer angle.

    The angle drawn for the most recent call is kept in ``last_theta``
    (degrees) so that a paired transform, e.g. for a mask, can reuse it.

    Parameters
    ----------
    min_rotation : int, optional
        Smallest angle in degrees. Defaults to -180.
    max_rotation : int, optional
        Largest angle in degrees. Defaults to 180.
    """

    def __init__(self, min_rotation=-180, max_rotation=180):
        """Store the angle range and initialize ``last_theta``."""
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.last_theta = 0.0

    def __call__(self, sample):
        """
        Rotate ``sample`` by a uniformly drawn angle, filling with 0.5.

        Parameters
        ----------
        sample : torch.Tensor
            Image of shape ``[C, H, W]``.

        Returns
        -------
        torch.Tensor
            Rotated image of the same shape.
        """
        # TODO: fix this
        theta = random.randint(self.min_rotation, self.max_rotation)
        self.last_theta = theta
        sample = torchvision.transforms.functional.rotate(sample, theta, fill=0.5)
        # rotation = iaa.Rotate(theta)
        # sample = rotation.augment_image(sample.numpy().transpose([1, 2, 0]))
        # self.last_theta = theta / 180 * math.pi
        # return torch.tensor(sample.transpose([2, 0, 1]))
        return sample


class Normalization(object):
    """
    Per-channel ``(x - mean) / std`` normalization with an exact inverse.

    ``mean`` and ``std`` are stored as ``[C, 1, 1]`` tensors so the
    transform broadcasts over ``[C, H, W]`` images and ``[B, C, H, W]``
    batches alike and moves to the input's device on every call.

    Parameters
    ----------
    mean : sequence of float
        Per-channel means.
    std : sequence of float
        Per-channel standard deviations.
    """

    def __init__(self, mean, std):
        """Store mean and std as broadcastable ``[C, 1, 1]`` tensors."""
        self.mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1)

    def __call__(self, sample):
        """
        Normalize ``sample``.

        Parameters
        ----------
        sample : torch.Tensor
            Image ``[C, H, W]`` or batch ``[B, C, H, W]``.

        Returns
        -------
        torch.Tensor
            ``(sample - mean) / std`` with the same shape.
        """
        transform = (sample - self.mean.to(sample.device)) / self.std.to(sample.device)

        return transform

    def invert(self, batch):
        """
        Undo the normalization.

        Parameters
        ----------
        batch : torch.Tensor
            Normalized image or batch.

        Returns
        -------
        torch.Tensor
            ``batch * std + mean`` with the same shape.
        """
        return batch * self.std.to(batch.device) + self.mean.to(batch.device)


class SetChannels(object):
    """
    Convert a tensor image to a fixed number of channels.

    Multi-channel inputs are averaged to one channel when ``channels == 1``;
    single-channel inputs are tiled when ``channels > 1``; anything else is
    returned unchanged (no other conversions are attempted).

    Parameters
    ----------
    channels : int
        Desired number of channels.
    """

    def __init__(self, channels):
        """Store the desired channel count."""
        self.channels = channels

    def __call__(self, sample):
        """
        Adjust the channel count of ``sample``.

        Parameters
        ----------
        sample : torch.Tensor
            Image of shape ``[C, H, W]``.

        Returns
        -------
        torch.Tensor
            Image of shape ``[channels, H, W]`` when a conversion applied,
            otherwise ``sample`` itself.
        """
        if self.channels == 1 and sample.shape[0] > 1:
            return torch.mean(sample, 0, keepdim=True)

        elif self.channels > 1 and sample.shape[0] == 1:
            return torch.tile(sample, [self.channels, 1, 1])

        else:
            return sample


class IdentityNormalization(object):
    """
    No-op stand-in for ``Normalization``.

    Used where a dataset needs an object with ``__call__`` and ``invert`` but
    its images are already in the value range the model expects.
    """

    def __call__(self, sample):
        """Return ``sample`` unchanged."""
        return sample

    def invert(self, batch):
        """Return ``batch`` unchanged."""
        return batch


class RandomResizeCropPad(object):
    """
    Random zoom augmentation that keeps the spatial size fixed.

    The image is resized by a factor drawn uniformly from ``scale_range``,
    then padded with 0.5 (zoom out) and center-cropped (zoom in) back to the
    original ``H x W``.

    Parameters
    ----------
    scale_range : tuple of float, optional
        ``(min, max)`` of the uniform scale factor. Defaults to (0.8, 1.2).
    """

    def __init__(self, scale_range=(0.8, 1.2)):
        """Store the scale range."""
        self.scale_range = scale_range

    def __call__(self, img):
        """
        Zoom ``img`` in or out and restore its original size.

        Parameters
        ----------
        img : torch.Tensor
            Image of shape ``[..., H, W]``.

        Returns
        -------
        torch.Tensor
            Image of the same shape.
        """
        # TODO this has to be applicable also for the masks later!
        # Randomly select scale factor
        output_size = img.shape[-2:]
        scale_factor = torch.FloatTensor(1).uniform_(*self.scale_range).item()

        # Determine resized dimensions
        resized_height = int(output_size[0] * scale_factor)
        resized_width = int(output_size[1] * scale_factor)
        new_size = (resized_width, resized_height)

        # Resize image
        img = transforms.functional.resize(img, new_size)

        # Determine cropping/padding parameters
        pad_left = max(0, (output_size[0] - resized_width) // 2)
        pad_top = max(0, (output_size[1] - resized_height) // 2)
        pad_right = max(0, output_size[0] - resized_width - pad_left)
        pad_bottom = max(0, output_size[1] - resized_height - pad_top)

        # Apply crop/pad
        img = transforms.functional.pad(
            img, (pad_left, pad_top, pad_right, pad_bottom), fill=0.5
        )

        # Perform center crop
        img = transforms.functional.center_crop(img, output_size)

        return img


class DiffusionAugmentation(object):
    """
    Augment images by a stochastic encode/decode round trip through a generator.

    The generator (built with ``get_generator`` and moved to CUDA if
    available) encodes the image up to ``sampling_time_fraction`` of the
    diffusion process with ``stochastic="fully"`` and decodes it again
    stochastically, yielding a plausible variation of the input. Value
    ranges are converted via the dataset's ``project_to_pytorch_default`` /
    ``project_from_pytorch_default`` on both sides.

    Parameters
    ----------
    generator : str or dict or GeneratorInterface
        Generator specification accepted by ``get_generator``.
    sampling_time_fraction : float
        Fraction of the diffusion time the image is noised to.
    num_discretization_steps : int
        Number of sampler steps for encoding and decoding.
    dataset : PealDataset, optional
        Dataset whose value range the input lives in. If ``None`` the input
        is assumed to already be in the pytorch default range.
    """

    def __init__(
        self, generator, sampling_time_fraction, num_discretization_steps, dataset=None
    ):
        """Build the generator on the best available device and store the knobs."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = get_generator(generator).to(self.device)
        self.sampling_time_fraction = sampling_time_fraction
        self.num_discretization_steps = num_discretization_steps
        self.dataset = dataset

    def __call__(self, img):
        """
        Return a diffusion-resampled variant of ``img``.

        Parameters
        ----------
        img : torch.Tensor
            Image ``[C, H, W]`` or batch ``[B, C, H, W]`` in the value range of
            ``dataset`` (or the pytorch default range if no dataset is set).

        Returns
        -------
        torch.Tensor
            Reconstructed image(s) with the same shape and value range. Note
            that the result stays on the generator's device.
        """
        if len(img.shape) == 3:
            img_in = img.unsqueeze(0)
            was_unsqueezed = True

        else:
            img_in = img
            was_unsqueezed = False

        device_buffer = img_in.device
        if not self.dataset is None:
            img_in = self.dataset.project_to_pytorch_default(img_in)

        with torch.no_grad():
            img_in = self.generator.dataset.project_from_pytorch_default(img_in).to(
                self.device
            )
            z = self.generator.encode(
                img_in,
                self.sampling_time_fraction,
                num_steps=self.num_discretization_steps,
                stochastic="fully",
            )
            img_reconstructed = self.generator.decode(
                z,
                self.sampling_time_fraction,
                num_steps=self.num_discretization_steps,
                stochastic=True,
            )

        img_reconstructed.to(device_buffer)
        img_reconstructed = self.generator.dataset.project_to_pytorch_default(
            img_reconstructed
        )
        if not self.dataset is None:
            img_reconstructed = self.dataset.project_from_pytorch_default(
                img_reconstructed
            )

        if was_unsqueezed:
            img_reconstructed = img_reconstructed.squeeze(0)

        return img_reconstructed
