import unittest
import numpy as np
import torch
import os

from peal.visualization.image_grid import make_image_grid


class TestMakeImageGrid(unittest.TestCase):
    def test_sample1(self):
        checkboxes = {"bla": [True, True, False], "blub": [False, True, False]}
        blue_img = np.zeros([64, 64, 3])
        blue_img[:, :, 2] = 1.0
        red_img = np.zeros([64, 64, 3])
        red_img[:, :, 0] = 1.0
        green_img = np.zeros([64, 64, 3])
        green_img[:, :, 1] = 1.0
        tensor_one = torch.tensor(
            np.stack([blue_img, red_img, green_img], axis=0).transpose(0, 3, 1, 2)
        )
        tensor_two = torch.tensor(
            np.stack([red_img, green_img, blue_img], axis=0).transpose(0, 3, 1, 2)
        )
        tensor_three = torch.tensor(
            np.stack([green_img, blue_img, red_img], axis=0).transpose(0, 3, 1, 2)
        )
        images = {
            "tensor1": [tensor_one, ["a", "b", "c"]],
            "tensor2": [[tensor_two, tensor_three], ["d", "e", "f"]],
        }
        grid = make_image_grid(checkboxes, images, 64)
        if not os.path.exists(os.path.join("tests", "outputs")):
            os.makedirs(os.path.join("tests", "outputs"))

        grid.save(os.path.join("tests", "outputs", "test.png"))
        grid_np = np.array(grid)
        self.assertEqual(grid_np[0, 0, 0], 255)
