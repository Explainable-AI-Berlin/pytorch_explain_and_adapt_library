import torch
import torchvision
import pygame
import random
import math

# import imgaug.augmenters as iaa
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

from peal.generators.generator_factory import get_generator


class CircularCut(object):
    """ """

    def __call__(self, sample):
        """ """
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
    """ """

    def __init__(self, input_size):
        """ """
        self.input_size = input_size

    def __call__(self, sample):
        """ """
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
    """ """

    def __init__(self, min_rotation=-180, max_rotation=180):
        """ """
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.last_theta = 0.0

    def __call__(self, sample):
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
    """ """

    def __init__(self, mean, std):
        """ """
        self.mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1)

    def __call__(self, sample):
        """ """
        return (sample - self.mean.to(sample.device)) / self.std.to(sample.device)

    def invert(self, batch):
        """ """
        return batch * self.std.to(batch.device) + self.mean.to(batch.device)


class SetChannels(object):
    """ """

    def __init__(self, channels):
        """ """
        self.channels = channels

    def __call__(self, sample):
        """ """
        if self.channels == 1 and sample.shape[0] > 1:
            return torch.mean(sample, 0, keepdim=True)

        elif self.channels > 1 and sample.shape[0] == 1:
            return torch.tile(sample, [self.channels, 1, 1])

        else:
            return sample


class IdentityNormalization(object):
    """ """

    def __call__(self, sample):
        """ """
        return sample

    def invert(self, batch):
        """ """
        return batch


class RandomResizeCropPad(object):
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, img):
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
    def __init__(
        self, generator, sampling_time_fraction, num_discretization_steps, dataset=None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = get_generator(generator).to(self.device)
        self.sampling_time_fraction = sampling_time_fraction
        self.num_discretization_steps = num_discretization_steps
        self.dataset = dataset

    def __call__(self, img):
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
