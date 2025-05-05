"""
These test primarily assure the proper functionality of the get_group_stats function in LossComputer
"""

import unittest
import os
import sys

import torch
import numpy as np
from torch import nn

from torch.utils.data import Dataset

from peal.dependencies.group_dro.loss import LossComputer

class DummyConfig():
    def __init__(self, confounding_factors):
        self.confounding_factors = confounding_factors

class DummyDataset(Dataset):
    def __init__(self, config, attributes, keys, data):
        self.config = config
        self.attributes = attributes
        self.keys = keys
        self.data = data


class TestGetGroupStats(unittest.TestCase):

    def setUp(self):
        self.attributes = ["a", "b", "c", "d", "e"]
        self.keys = [str(i) for i in range(10)]
        self.data = {
            '0': torch.tensor([0,0,0,0,0]).float(),
            '1': torch.tensor([0,0,1,0,0]).float(),
            '2': torch.tensor([0,1,0,2,0]).float(),
            '3': torch.tensor([0,1,1,2,0]).float(),
            '4': torch.tensor([0,1,0,2,0]).float(),
            '5': torch.tensor([1,0,1,0,0]).float(),
            '6': torch.tensor([1,0,0,0,0]).float(),
            '7': torch.tensor([1,0,1,0,0]).float(),
            '8': torch.tensor([1,0,0,0,0]).float(),
            '9': torch.tensor([1,1,1,2,0]).float(),
        }


    def test_calculation_0(self):
        config = DummyConfig(["a", "b"])
        dataset = DummyDataset(config, self.attributes, self.keys, self.data)

        loss = LossComputer(nn.MSELoss(),
                            False,
                            dataset)

        n_groups, group_counts, group_frac = loss.get_group_stats(dataset)

        self.assertEqual(n_groups, 4)
        self.assertEqual(group_counts[0], 2)
        self.assertEqual(group_counts[1], 4)
        self.assertEqual(group_counts[2], 3)
        self.assertEqual(group_counts[3], 1)
        self.assertEqual(group_frac[0], 0.2)
        self.assertEqual(group_frac[1], 0.4)
        self.assertEqual(group_frac[2], 0.3)
        self.assertEqual(group_frac[3], 0.1)


    def test_calculation_1(self):
        config = DummyConfig(["a", "b", "c"])
        dataset = DummyDataset(config, self.attributes, self.keys, self.data)

        loss = LossComputer(nn.MSELoss(),
                            False,
                            dataset)

        n_groups, group_counts, group_frac = loss.get_group_stats(dataset)

        self.assertEqual(n_groups, 8)
        self.assertEqual(group_counts[0], 1)
        self.assertEqual(group_counts[1], 2)
        self.assertEqual(group_counts[2], 2)
        self.assertEqual(group_counts[3], 0)
        self.assertEqual(group_counts[4], 1)
        self.assertEqual(group_counts[5], 2)
        self.assertEqual(group_counts[6], 1)
        self.assertEqual(group_counts[7], 1)

    def test_calculation_2(self):
        config = DummyConfig(["a", "d"])
        dataset = DummyDataset(config, self.attributes, self.keys, self.data)

        loss = LossComputer(nn.MSELoss(),
                            False,
                            dataset)

        n_groups, group_counts, group_frac = loss.get_group_stats(dataset)

        self.assertEqual(n_groups, 6)
        self.assertEqual(group_counts[0], 2)
        self.assertEqual(group_counts[1], 4)
        self.assertEqual(group_counts[2], 0)
        self.assertEqual(group_counts[3], 0)
        self.assertEqual(group_counts[4], 3)
        self.assertEqual(group_counts[5], 1)

if __name__=="__main__":
    unittest.main()