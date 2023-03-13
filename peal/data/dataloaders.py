import json
import torch
import random
import os
import torchvision
import code
import pygame
import cv2 as cv
import math
import numpy as np
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
import sklearn
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader

from peal.data.datasets import (
    get_datasets,
    GlowDatasetWrapper
)

class DataStack:
    def __init__(self, datasource, num_classes):
        self.datasource = datasource
        if isinstance(datasource, torch.utils.data.Dataset):
            self.dataset = datasource
            self.current_idx = 0

        else:
            self.dataset = datasource.dataset

        self.num_classes = num_classes
        self.data = []
        for idx in range(num_classes):
            self.data.append([])

        self.fill_stack()

    def fill_stack(self):
        while np.min(list(map(lambda x: len(x), self.data))) == 0:
            if isinstance(self.datasource, torch.utils.data.Dataset):
                X, y = self.dataset.__getitem__(self.current_idx)
                if hasattr(self.dataset, 'hints_enabled') and self.dataset.hints_enabled:
                    y_index = y[0]

                else:
                    y_index = y

                self.data[int(y_index)].append([X,y])
                self.current_idx = (self.current_idx + 1) % self.dataset.__len__()

            else:
                X, y = next(iter(self.datasource))
                if hasattr(self.dataset, 'hints_enabled') and self.dataset.hints_enabled:
                    for i in range(X.shape[0]):
                        self.data[y[0][i]].append([X[i], (y[0][i], y[1][i])])

                else:
                    for i in range(X.shape[0]):
                        self.data[y[i]].append([X[i], y[i]])

    def pop(self, class_idx):
        sample = self.data[class_idx].pop(0)
        self.fill_stack()
        return sample
    
    def reset(self):
        self.data = []
        for idx in range(self.num_classes):
            self.data.append([])

        self.fill_stack()


class DataIterator:
   ''' Iterator class '''
   def __init__(self, dataloader):
       self.dataloader = dataloader
       # member variable to keep track of current index
       self._index = 0

   def __next__(self):
       if self._index < self.dataloader.train_config['iterations_per_episode']:
           self._index +=1
           return self.dataloader.sample()
           
       # End of Iteration
       raise StopIteration


class DataloaderMixer(DataLoader):
    def __init__(self, train_config, initial_dataloader):
        self.train_config = train_config
        self.dataloaders = [initial_dataloader]
        self.batch_size = initial_dataloader.batch_size
        self.priorities = None
        self.dataset = initial_dataloader.dataset # TODO kind of hacky
        self.iterators = [iter(self.dataloaders[0])]

    def append(self, dataloader, priority = 1):
        self.dataloaders.append(dataloader)
        self.iterators.append(iter(self.dataloaders[-1]))
        self.priorities = np.zeros(len(self.dataloaders))
        for i in range(len(self.dataloaders)):
            self.priorities[i] = self.dataloaders[i].dataset.__len__()

        self.priorities[-1] *= priority
        self.priorities = self.priorities / self.priorities.sum()

    def __iter__(self):
        return DataIterator(self)

    def sample(self):
        if not self.priorities is None:
            idx = int(np.random.multinomial(1, self.priorities).argmax())

        else:
            idx = 0

        item = next(self.iterators[idx], 'STOP')
        if isinstance(item, str) and item == 'STOP':
            self.iterators[idx] = iter(self.dataloaders[idx])
            item = next(self.iterators[idx])

        return item

    def reset(self):
        for i in range(len(self.dataloaders)):
            self.dataloaders[i] = DataLoader(self.dataloaders[i].dataset, batch_size = self.dataloaders[i].batch_size)
            self.iterators[i] = iter(self.dataloaders[i])

    def __len__(self):
        length = 0
        for dataloader in self.dataloaders:
            length += len(dataloader.dataset)

        return length


def get_dataloader(dataset, training_config, mode, task_config = None, batch_size = None):
    dataset.task_config = task_config
    if batch_size is None:
        dataloader = DataLoader(dataset, batch_size = training_config[mode + '_batch_size'])

    else:
        dataloader = DataLoader(dataset, batch_size = batch_size)

    if mode == 'train' and 'iterations_per_episode' in training_config.keys():
        dataloader = DataloaderMixer(training_config, dataloader)

    return dataloader


def create_class_ordered_batch(dataset, config):
    if 'output_size' in config['task'].keys():
        output_size = config['task']['output_size']

    else:
        output_size = config['data']['output_size']
        
    datastack = DataStack(dataset, output_size)

    test_X = []
    test_y = []
    for i in range(output_size):
        test_X.append(datastack.pop(i)[0])
        test_X.append(datastack.pop(i)[0])
        test_y.append(i)
        test_y.append(i)

    test_X = torch.stack(test_X)
    test_y = torch.tensor(test_y)

    return test_X, test_y


def create_dataloaders_from_datasource(datasource, config, enable_hints = False, gigabyte_vram = None):
    '''

    '''
    if 'base_batch_size' in config['training'].keys():
        multiplier = float(np.prod(config['training']['assumed_input_size']) / np.prod(config['data']['input_size']))
        if not gigabyte_vram is None and 'gigabyte_vram' in config['training'].keys():
            multiplier = multiplier * (gigabyte_vram / config['training']['gigabyte_vram'])
        
        batch_size_adapted = int(config['training']['base_batch_size'] * multiplier)
        if config['training']['train_batch_size'] == -1:  config['training']['train_batch_size'] = batch_size_adapted
        if config['training']['val_batch_size'] == -1:  config['training']['val_batch_size'] = batch_size_adapted
        if config['training']['test_batch_size'] == -1:  config['training']['test_batch_size'] = batch_size_adapted

    if (isinstance(datasource, tuple) or isinstance(datasource, list)) and isinstance(datasource[0], DataLoader):
        train_dataloader = datasource[0]

        val_dataloader = datasource[1]
        if len(datasource) == 2:
            test_dataloader = val_dataloader

        else:
            test_dataloader = datasource[2]

    else:
        if isinstance(datasource, str):
            dataset_train, dataset_val, dataset_test = get_datasets(
                config = config['data'],
                base_dir = datasource
            )

        elif isinstance(datasource[0], torch.utils.data.Dataset):
            if len(datasource) == 2:
                dataset_train, dataset_val = datasource
                dataset_test = dataset_val

            else:
                dataset_train, dataset_val, dataset_test = datasource

        if 'n_bits' in config['architecture'].keys():
            dataset_train = GlowDatasetWrapper(dataset_train, config['architecture']['n_bits'])
            dataset_val = GlowDatasetWrapper(dataset_val, config['architecture']['n_bits'])
            dataset_test = GlowDatasetWrapper(dataset_test, config['architecture']['n_bits'])

        elif enable_hints:
            dataset_train.enable_hints()

        train_dataloader = get_dataloader(
            dataset = dataset_train,
            training_config = config['training'],
            mode = 'train',
            task_config = config['task']
        )
        val_dataloader = get_dataloader(
            dataset = dataset_val,
            training_config = config['training'],
            mode = 'val',
            task_config = config['task']
        )
        test_dataloader = get_dataloader(
            dataset = dataset_test,
            training_config = config['training'],
            mode = 'test',
            task_config = config['task']
        )

    # TODO this seems quite hacky and could cause problems when combining multiclass dataset with SegmentationMask teacher
    if 'config' in train_dataloader.dataset.__dict__.keys() and train_dataloader.dataset.config['output_type'] != 'multiclass':
        # TODO sanity check or warning
        config['data'] = train_dataloader.dataset.config

    return train_dataloader, val_dataloader, test_dataloader