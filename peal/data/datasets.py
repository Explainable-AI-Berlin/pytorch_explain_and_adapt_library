import torch
import random
import os
import copy
import numpy as np
import torchvision.transforms as transforms

from torchvision.transforms import ToTensor
from PIL import Image

from peal.utils import load_yaml_config
from peal.data.transformations import (
    CircularCut,
    Padding,
    RandomRotation,
    Normalization,
    IdentityNormalization,
    SetChannels
)

class Image2ClassDataset(torch.utils.data.Dataset):
    """Shape Attribute dataset."""

    def __init__(self, root_dir, mode, config, transform=ToTensor(), task_config = None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        if 'has_hints' in self.config.keys() and self.config['has_hints']:
            self.root_dir = os.path.join(root_dir, 'imgs')
            self.mask_dir = os.path.join(root_dir, 'masks')
            self.all_urls = []
            self.urls_with_hints = []
        
        else:
            self.root_dir = root_dir
        
        self.hints_enabled = False
        self.task_config = task_config
        self.transform = transform
        self.urls = []
        self.idx_to_name = os.listdir(self.root_dir)
        '''if 'output_size' in self.config.keys():
            self.idx_to_name = self.idx_to_name[:self.config['output_size']]'''

        self.idx_to_name.sort()
        for label_str in self.idx_to_name:
            files = os.listdir(os.path.join(self.root_dir, label_str))
            '''if self.config['num_samples'] != 'None' and self.config['output_size'] in self.config.keys():
                files = files[:int(self.config['num_samples'] / self.config['output_size'])]'''

            files.sort()
            for file in files:
                self.urls.append((label_str, file))                

        random.seed(0)
        random.shuffle(self.urls)

        if mode == 'train':
            self.urls = self.urls[:int(config['split'][0] * len(self.urls))]

        elif mode == 'val':
            self.urls = self.urls[int(config['split'][0] * len(self.urls)):int(config['split'][1] * len(self.urls))]

        elif mode == 'test':
            self.urls = self.urls[int(config['split'][1] * len(self.urls)):]

        if 'has_hints' in self.config.keys() and self.config['has_hints']:
            self.all_urls = copy.deepcopy(self.urls)
            for label_str, file in self.all_urls:
                if os.path.exists(os.path.join(self.mask_dir, file)):
                    self.urls_with_hints.append((label_str, file))

    def class_idx_to_name(self, class_idx):
        if self.config['dataset_name'] == 'imagenet':
            return imagenet_map[class_idx]

        else:
            return self.idx_to_name[class_idx]

    def enable_hints(self):
        self.urls = copy.deepcopy(self.urls_with_hints)
        self.hints_enabled = True

    def disable_hints(self):
        self.urls = copy.deepcopy(self.all_urls)
        self.hints_enabled = False

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        label_str, file = self.urls[idx]

        img = Image.open(os.path.join(self.root_dir, label_str, file))
        state = torch.get_rng_state()
        img = self.transform(img)

        if img.shape[0] == 1 and self.config['input_size'][0] != 1:
            img = torch.tile(img, [self.config['input_size'][0], 1, 1])

        #label = torch.zeros([len(self.idx_to_name)], dtype=torch.float32)
        #label[self.idx_to_name.index(label_str)] = 1.0
        label = torch.tensor(self.idx_to_name.index(label_str))
        
        if not self.hints_enabled:
            return img, label
        
        else:
            # TODO how to apply same randomized transformation?
            mask = Image.open(os.path.join(self.mask_dir, file))
            torch.set_rng_state(state)
            mask = self.transform(mask)
            return img, (label, mask)


class Image2MixedDataset(torch.utils.data.Dataset):
    """
    The celeba dataset.
    """

    def __init__(self, root_dir, mode, config, transform=ToTensor(), task_config = None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.config = config
        self.transform = transform
        self.task_config = task_config
        self.hints_enabled = False
        raw_data = open(os.path.join(self.root_dir, 'data.csv'), 'r').read().split('\n')
        self.attributes = raw_data[0].split(',')[1:]
        raw_data = raw_data[1:]
        #raw_data = raw_data[:self.self.config['num_samples']]
        self.data = {}
        if not len(self.config['confounding_factors']) == 0:
            selection_idx1 = self.attributes.index(self.config['confounding_factors'][0])
            selection_idx2 = self.attributes.index(self.config['confounding_factors'][1])
            n_attribute_confounding = np.array([[0,0],[0,0]])
            max_attribute_confounding = np.array([[0,0],[0,0]])
            max_attribute_confounding[0][0] = int(self.config['num_samples'] * self.config['confounder_probability'] * 0.5)
            max_attribute_confounding[1][0] = int(self.config['num_samples'] * round(1 - self.config['confounder_probability'], 2) * 0.5)
            max_attribute_confounding[0][1] = int(self.config['num_samples'] * round(1 - self.config['confounder_probability'], 2) * 0.5)
            max_attribute_confounding[1][1] = int(self.config['num_samples'] * self.config['confounder_probability'] * 0.5)
            keys = [[[], []],[[], []]]

        for line in raw_data:
            instance_attributes = line.split(',')
            instance_attributes_int = list(map(lambda x: int(float(x)), instance_attributes[1:]))
            instances_tensor = torch.tensor(instance_attributes_int)
            if len(self.config['confounding_factors']) == 0:
                self.data[instance_attributes[0]] = torch.maximum(torch.zeros_like(instances_tensor), instances_tensor)

            else:
                attribute = instance_attributes_int[selection_idx1]
                confounder = instance_attributes_int[selection_idx2]
                if n_attribute_confounding[attribute][confounder] < max_attribute_confounding[attribute][confounder]:
                    self.data[instance_attributes[0]] = torch.maximum(torch.zeros_like(instances_tensor), instances_tensor)
                    keys[attribute][confounder].append(instance_attributes[0])
                    n_attribute_confounding[attribute][confounder] += 1

                if np.sum(n_attribute_confounding == max_attribute_confounding) == 4:
                    break

        if not len(self.config['confounding_factors']) == 0:
            assert np.sum(n_attribute_confounding == max_attribute_confounding) == 4, 'something went wrong with filling up the attributes'
            assert np.sum(n_attribute_confounding) == self.config['num_samples'], 'wrong number of samples!'
            assert len(keys[0][0]) + len(keys[0][1]) + len(keys[1][0]) + len(keys[1][1]) == self.config['num_samples'], 'wrong number of keys!'
            if mode == 'train':
                self.keys = keys[0][0][:int(len(keys[0][0]) * self.config['split'][0])]
                self.keys += keys[0][1][:int(len(keys[0][1]) * self.config['split'][0])]
                self.keys += keys[1][0][:int(len(keys[1][0]) * self.config['split'][0])]
                self.keys += keys[1][1][:int(len(keys[1][1]) * self.config['split'][0])]
                random.shuffle(self.keys)

            elif mode == 'val':
                self.keys = keys[0][0][int(len(keys[0][0]) * self.config['split'][0]):int(len(keys[0][0]) * self.config['split'][1])]
                self.keys += keys[0][1][int(len(keys[0][1]) * self.config['split'][0]):int(len(keys[0][1]) * self.config['split'][1])]
                self.keys += keys[1][0][int(len(keys[1][0]) * self.config['split'][0]):int(len(keys[1][0]) * self.config['split'][1])]
                self.keys += keys[1][1][int(len(keys[1][1]) * self.config['split'][0]):int(len(keys[1][1]) * self.config['split'][1])]
                random.shuffle(self.keys)

            elif mode == 'test':
                self.keys = keys[0][0][int(len(keys[0][0]) * self.config['split'][1]):]
                self.keys += keys[0][1][int(len(keys[0][1]) * self.config['split'][1]):]
                self.keys += keys[1][0][int(len(keys[1][0]) * self.config['split'][1]):]
                self.keys += keys[1][1][int(len(keys[1][1]) * self.config['split'][1]):]
                random.shuffle(self.keys)

            else:
                self.keys = keys[0][0] + keys[0][1] + keys[1][0] + keys[1][1]

        else:
            self.keys = list(self.data.keys())
            if mode == 'train':
                self.keys = self.keys[:int(len(self.keys) * self.config['split'][0])]

            elif mode == 'val':
                self.keys = self.keys[int(len(self.keys) * self.config['split'][0]):int(len(self.keys) * self.config['split'][1])]

            elif mode == 'test':
                self.keys = self.keys[int(len(self.keys) * self.config['split'][1]):]

            else:
                self.keys = keys

    def __len__(self):
        return len(self.keys)

    def enable_hints(self):
        self.hints_enabled = True

    def disable_hints(self):
        self.hints_enabled = False        

    def __getitem__(self, idx):
        name = self.keys[idx]

        img = Image.open(os.path.join(self.root_dir, 'imgs', name))
        #code.interact(local=dict(globals(), **locals()))
        state = torch.get_rng_state()
        img_tensor = self.transform(img)

        targets = self.data[name]

        if not self.task_config is None and not len(self.task_config['selection']) == 0:
            target = torch.zeros([len(self.task_config['selection'])])
            for idx, selection in enumerate(self.task_config['selection']):
                target[idx] = targets[self.attributes.index(selection)]

        else:
            target = torch.tensor(targets[:self.config['output_size']], dtype = torch.float32)

        if not self.task_config is None and 'ce' in self.task_config['criterions']:
            assert target.shape[0] == 1, 'output shape inacceptable for singleclass classification'
            target = torch.tensor(target[0], dtype=torch.int64)

        if not self.hints_enabled:
            return img_tensor, target

        else:
            mask = Image.open(os.path.join(self.root_dir, 'masks', name))
            torch.set_rng_state(state)
            mask_tensor = self.transform(mask)
            return img_tensor, (target, mask_tensor)


class GlowDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, n_bits):
        self.base_dataset = base_dataset
        self.n_bits = n_bits
        self.n_bins = 2.0 ** n_bits

    def __len__(self):
        return self.base_dataset.__len__()

    def __getitem__(self, idx):
        image, label = self.base_dataset.__getitem__(idx)
        image = image * 255
        image = torch.floor(image / 2 ** (8 - self.n_bits))
        image = image / self.n_bins - 0.5
        image = image + torch.rand_like(image) / self.n_bins
        return image, label

    def project_to_pytorch_default(self, image):
        '''
        This function maps processed image back to human visible image
        '''
        return image + 0.5


def get_datasets(config, base_dir):
    '''

    '''
    config = load_yaml_config(config)

    #
    transform_list_train = []
    transform_list_test = []
    #
    if 'circular_cut' in config['invariances']:
        transform_list_train.append(CircularCut())
        transform_list_test.append(CircularCut())

    #
    transform_list_train.append(ToTensor())
    transform_list_test.append(ToTensor())
    
    #
    if 'crop_size' in config.keys():
        transform_list_train.append(Padding(config['crop_size'][1:]))
        transform_list_test.append(Padding(config['crop_size'][1:]))

    #
    if 'rotation' in config['invariances']: transform_list_train.append(RandomRotation())
    if 'hflipping' in config['invariances']: transform_list_train.append(transforms.RandomHorizontalFlip(p=0.5))
    if 'vflipping' in config['invariances']: transform_list_train.append(transforms.RandomVerticalFlip(p=0.5))

    #
    if 'crop_size' in config.keys():
        transform_list_train.append(transforms.RandomCrop(config['crop_size'][1:]))
        transform_list_test.append(transforms.CenterCrop(config['crop_size'][1:]))

    transform_list_train.append(transforms.Resize(config['input_size'][1:]))
    transform_list_test.append(transforms.Resize(config['input_size'][1:]))

    transform_list_train.append(SetChannels(config['input_size'][0]))
    transform_list_test.append(SetChannels(config['input_size'][0]))

    #
    transform_train = transforms.Compose(transform_list_train)
    transform_test = transforms.Compose(transform_list_test)

    if config['input_type'] == 'image' and config['output_type'] == 'singleclass': dataset = Image2ClassDataset
    elif config['input_type'] == 'image' and config['output_type'] in ['multiclass', 'mixed']: dataset = Image2MixedDataset
    else: raise ValueError(config['dataset_name'] + " is not valid data set name.")

    #
    if config['use_normalization']:
        stats_dataset = dataset(base_dir, 'train', config, transform_test)
        samples = []
        for idx in range(stats_dataset.__len__()):
            samples.append(stats_dataset.__getitem__(idx)[0])

        samples = torch.stack(samples)
        config['normalization'].append(list(torch.mean(samples, [0,2,3]).numpy()))
        config['normalization'].append(list(torch.std(samples, [0,2,3]).numpy()))

        #
        normalization = Normalization(config['normalization'][0], config['normalization'][1])

    else:
        normalization = IdentityNormalization()

    transform_train = transforms.Compose([transform_train, normalization])
    transform_test = transforms.Compose([transform_test, normalization])

    train_data = dataset(base_dir, 'train', config, transform_train)
    val_data = dataset(base_dir, 'val', config, transform_train)
    test_data = dataset(base_dir, 'test', config, transform_test)

    # this is kind of dirty
    train_data.normalization = normalization
    val_data.normalization = normalization
    test_data.normalization = normalization
    
    return train_data, val_data, test_data