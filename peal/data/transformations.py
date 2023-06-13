import torch
import torchvision
import pygame
import random
import math
import imgaug.augmenters as iaa
import torchvision.transforms as transforms
import numpy as np

from PIL import Image


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
        img_follicle = np.minimum(overlay_np, sample_np)
        return Image.fromarray(img_follicle)


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
            ]
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
        theta = random.randint(self.min_rotation, self.max_rotation)
        rotation = iaa.Rotate(theta)
        sample = rotation.augment_image(sample.numpy().transpose([1, 2, 0]))
        self.last_theta = theta / 180 * math.pi
        return torch.tensor(sample.transpose([2, 0, 1]))


class Normalization(object):
    """ """

    def __init__(self, mean, std):
        """ """
        self.mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1)

    def __call__(self, sample):
        """ """
        return (sample - self.mean) / self.std

    def invert(self, batch):
        """ """
        return batch * self.std + self.mean


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
