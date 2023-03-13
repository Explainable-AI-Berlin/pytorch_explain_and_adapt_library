import unittest
import torch
import numpy as np

from torch.utils.data import (
    Dataset,
    DataLoader
)

from peal.data.dataloaders import (
    DataStack,
    DataloaderMixer
)


class DummyDataset(Dataset):
    def __init__(self, elements):
        self.elements = elements
    
    def __len__(self):
        return len(self.elements)
    
    def __getitem__(self, idx):
        return self.elements[idx]


class TestDataStack(unittest.TestCase):
    def test_sample1(self):
        datastack = DataStack(DummyDataset([[1, 0], [2, 0], [3, 1], [4, 0], [5, 1]]), 2)
        self.assertEqual(datastack.pop(0)[0], 1)
        self.assertEqual(datastack.pop(1)[0], 3)
        self.assertEqual(datastack.pop(0)[0], 2)
        self.assertEqual(datastack.pop(1)[0], 5)


class TestDataloaderMixer(unittest.TestCase):
    def test_priorities1(self):
        dataloader = DataloaderMixer(
            {'iterations_per_episode': 10000},
            DataLoader(DummyDataset([1]))
        )
        dataloader.append(DataLoader(DummyDataset([2])), priority=9)
        value_list = []
        for values in dataloader:
            value_list.append(values[0])
        
        self.assertTrue(value_list.count(2) > 8000)
    
    def test_priorities2(self):
        dataloader = DataloaderMixer(
            {'iterations_per_episode': 10000},
            DataLoader(DummyDataset(list(np.ones([9], dtype=np.int))))
        )
        dataloader.append(DataLoader(DummyDataset([2])), priority=9)
        value_list = []
        for values in dataloader:
            value_list.append(values[0])
        
        self.assertTrue(value_list.count(1) < 6000)
        self.assertTrue(value_list.count(1) > 4000)