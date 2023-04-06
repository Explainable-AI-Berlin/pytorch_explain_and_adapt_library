import torch
import random
import os
import json
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
    SetChannels,
)

class PealDataset(torch.utils.data.Dataset):
    def generate_contrastive_collage(batch_in, counterfactual):
        return torch.zeros([3,64,64])
    
    def serialize_dataset(output_dir, x_list, y_list, sample_names = None):
        pass
    
    def project_to_pytorch_default(x):
        return x
    
    def project_from_pytorch_default(x):
        return x


def get_datasets(config, base_dir):
    '''
    _summary_

    Args:
        config (_type_): _description_
        base_dir (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    '''
    config = load_yaml_config(config)

    #
    transform_list_train = []
    transform_list_test = []
    #
    if config["input_type"] == "image" and "circular_cut" in config["invariances"]:
        transform_list_train.append(CircularCut())
        transform_list_test.append(CircularCut())

    #
    transform_list_train.append(ToTensor())
    transform_list_test.append(ToTensor())

    #
    if config["input_type"] == "image":
        if "crop_size" in config.keys():
            transform_list_train.append(Padding(config["crop_size"][1:]))
            transform_list_test.append(Padding(config["crop_size"][1:]))

        #
        if "rotation" in config["invariances"]:
            transform_list_train.append(RandomRotation())
        if "hflipping" in config["invariances"]:
            transform_list_train.append(transforms.RandomHorizontalFlip(p=0.5))
        if "vflipping" in config["invariances"]:
            transform_list_train.append(transforms.RandomVerticalFlip(p=0.5))

        #
        if "crop_size" in config.keys():
            transform_list_train.append(
                transforms.RandomCrop(config["crop_size"][1:]))
            transform_list_test.append(
                transforms.CenterCrop(config["crop_size"][1:]))

        transform_list_train.append(
            transforms.Resize(config["input_size"][1:]))
        transform_list_test.append(transforms.Resize(config["input_size"][1:]))

        transform_list_train.append(SetChannels(config["input_size"][0]))
        transform_list_test.append(SetChannels(config["input_size"][0]))

    #
    transform_train = transforms.Compose(transform_list_train)
    transform_test = transforms.Compose(transform_list_test)

    if config["input_type"] == "image" and config["output_type"] == "singleclass":
        dataset = Image2ClassDataset

    elif config["input_type"] == "image" and config["output_type"] in [
        "multiclass",
        "mixed",
    ]:
        dataset = Image2MixedDataset

    elif config["input_type"] == "symbolic" and config["output_type"] in [
        "multiclass",
        "mixed",
    ]:
        dataset = SymbolicDataset

    elif config["input_type"] == "sequence" and config["output_type"] in [
        "singleclass",
    ]:
        dataset = SequenceDataset

    else:
        raise ValueError("input_type: " + config["input_type"] + ", output_type: " +
                         config["output_type"] + " combination is not supported!")

    #
    if config["input_type"] == "image" and config["use_normalization"]:
        stats_dataset = dataset(base_dir, "train", config, transform_test)
        samples = []
        for idx in range(stats_dataset.__len__()):
            samples.append(stats_dataset.__getitem__(idx)[0])

        samples = torch.stack(samples)
        config["normalization"].append(
            list(torch.mean(samples, [0, 2, 3]).numpy()))
        config["normalization"].append(
            list(torch.std(samples, [0, 2, 3]).numpy()))

        #
        normalization = Normalization(
            config["normalization"][0], config["normalization"][1]
        )

    else:
        normalization = IdentityNormalization()

    transform_train = transforms.Compose([transform_train, normalization])
    transform_test = transforms.Compose([transform_test, normalization])

    train_data = dataset(base_dir, "train", config, transform_train)
    val_data = dataset(base_dir, "val", config, transform_train)
    test_data = dataset(base_dir, "test", config, transform_test)

    # this is kind of dirty
    train_data.normalization = normalization
    val_data.normalization = normalization
    test_data.normalization = normalization

    return train_data, val_data, test_data


def parse_json(data_dir, config, mode):
    '''
    _summary_

    Args:
        data_dir (_type_): _description_
        config (_type_): _description_
        mode (_type_): _description_

    Returns:
        _type_: _description_
    '''
    with open(data_dir, "r") as f:
        raw_data = json.load(f)

    if config['known_confounder']:
        def extract_instances_tensor_confounder(idx, line):
            key = str(idx)
            instances_tensor = torch.tensor(line["values"])
            attribute = line["target"]
            confounder = line["has_confounder"]
            instances_tensor = torch.cat(
                [instances_tensor, torch.tensor([attribute])]
            )
            return key, instances_tensor, attribute, confounder

        return process_confounder_data_controlled(
            raw_data=raw_data.values(),
            config=config,
            mode=mode,
            extract_instances_tensor=extract_instances_tensor_confounder,
        )


def parse_csv(data_dir, config, mode, key_type="idx"):
    '''
    _summary_

    Args:
        data_dir (_type_): _description_
        config (_type_): _description_
        mode (_type_): _description_
        key_type (str, optional): _description_. Defaults to "idx".

    Returns:
        _type_: _description_
    '''
    raw_data = open(data_dir, "r").read().split("\n")
    attributes = raw_data[0].split(",")
    if key_type == "name":
        attributes = attributes[1:]

    raw_data = raw_data[1:]

    def extract_instances_tensor(idx, line):
        instance_attributes = line.split(",")

        if key_type == "idx":
            key = str(idx)

        elif key_type == "name":
            key = instance_attributes[0]
            instance_attributes = instance_attributes[1:]

        instance_attributes_int = list(
            map(
                lambda x: float(x),
                instance_attributes,
            )
        )
        instances_tensor = torch.tensor(instance_attributes_int)
        return key, instances_tensor

    if not len(config["confounding_factors"]) == 0:
        def extract_instances_tensor_confounder(idx, line):
            selection_idx1 = attributes.index(config["confounding_factors"][0])
            selection_idx2 = attributes.index(config["confounding_factors"][1])
            key, instances_tensor = extract_instances_tensor(
                idx,
                line
            )
            attribute = int(instances_tensor[selection_idx1])
            confounder = int(instances_tensor[selection_idx2])
            return key, instances_tensor, attribute, confounder

        data, keys_out = process_confounder_data_controlled(
            raw_data=raw_data,
            config=config,
            mode=mode,
            extract_instances_tensor=extract_instances_tensor_confounder,
        )

    else:
        data = {}
        for idx, line in enumerate(raw_data):
            key, instances_tensor = extract_instances_tensor(
                idx,
                line
            )
            data[key] = torch.maximum(
                torch.zeros_like(instances_tensor),
                instances_tensor,
            )

        keys = list(data.keys())
        if mode == "train":
            keys_out = keys[: int(len(keys) * config["split"][0])]

        elif mode == "val":
            keys_out = keys[
                int(len(keys) * config["split"][0]): int(
                    len(keys) * config["split"][1]
                )
            ]

        elif mode == "test":
            keys_out = keys[int(len(keys) * config["split"][1]):]

        else:
            keys_out = keys

    return attributes, data, keys_out


def process_confounder_data_controlled(raw_data, config, mode, extract_instances_tensor):
    '''
    _summary_

    Args:
        raw_data (_type_): _description_
        config (_type_): _description_
        mode (_type_): _description_
        extract_instances_tensor (_type_): _description_

    Returns:
        _type_: _description_
    '''
    data = {}
    n_attribute_confounding = np.array([[0, 0], [0, 0]])
    max_attribute_confounding = np.array([[0, 0], [0, 0]])
    max_attribute_confounding[0][0] = int(
        config["num_samples"] * config["confounder_probability"] * 0.5
    )
    max_attribute_confounding[1][0] = int(
        config["num_samples"] *
        round(1 - config["confounder_probability"], 2) * 0.5
    )
    max_attribute_confounding[0][1] = int(
        config["num_samples"] *
        round(1 - config["confounder_probability"], 2) * 0.5
    )
    max_attribute_confounding[1][1] = int(
        config["num_samples"] * config["confounder_probability"] * 0.5
    )
    keys = [[[], []], [[], []]]

    for idx, line in enumerate(raw_data):
        key, instances_tensor, attribute, confounder = extract_instances_tensor(
            idx=idx,
            line=line
        )
        if (
            n_attribute_confounding[attribute][confounder]
            < max_attribute_confounding[attribute][confounder]
        ):
            data[key] = torch.maximum(
                torch.zeros_like(instances_tensor),
                instances_tensor,
            )
            keys[attribute][confounder].append(key)
            n_attribute_confounding[attribute][confounder] += 1

        if np.sum(n_attribute_confounding == max_attribute_confounding) == 4:
            break

    assert (
        np.sum(n_attribute_confounding == max_attribute_confounding) == 4
    ), "something went wrong with filling up the attributes"
    assert (
        np.sum(n_attribute_confounding) == config["num_samples"]
    ), "wrong number of samples!"
    assert (
        len(keys[0][0]) + len(keys[0][1]) +
        len(keys[1][0]) + len(keys[1][1])
        == config["num_samples"]
    ), "wrong number of keys!"
    if mode == "train":
        keys_out = keys[0][0][: int(len(keys[0][0]) * config["split"][0])]
        keys_out += keys[0][1][: int(len(keys[0][1]) * config["split"][0])]
        keys_out += keys[1][0][: int(len(keys[1][0]) * config["split"][0])]
        keys_out += keys[1][1][: int(len(keys[1][1]) * config["split"][0])]
        random.shuffle(keys_out)

    elif mode == "val":
        keys_out = keys[0][0][
            int(len(keys[0][0]) * config["split"][0]): int(
                len(keys[0][0]) * config["split"][1]
            )
        ]
        keys_out += keys[0][1][
            int(len(keys[0][1]) * config["split"][0]): int(
                len(keys[0][1]) * config["split"][1]
            )
        ]
        keys_out += keys[1][0][
            int(len(keys[1][0]) * config["split"][0]): int(
                len(keys[1][0]) * config["split"][1]
            )
        ]
        keys_out += keys[1][1][
            int(len(keys[1][1]) * config["split"][0]): int(
                len(keys[1][1]) * config["split"][1]
            )
        ]
        random.shuffle(keys_out)

    elif mode == "test":
        keys_out = keys[0][0][int(len(keys[0][0]) * config["split"][1]):]
        keys_out += keys[0][1][int(len(keys[0][1]) * config["split"][1]):]
        keys_out += keys[1][0][int(len(keys[1][0]) * config["split"][1]):]
        keys_out += keys[1][1][int(len(keys[1][1]) * config["split"][1]):]
        random.shuffle(keys_out)

    else:
        keys_out = keys[0][0] + keys[0][1] + keys[1][0] + keys[1][1]
        random.shuffle(keys_out)

    return data, keys_out


class ImageDataset(torch.utils.data.Dataset):
    pass


class SequenceDataset(torch.utils.data.Dataset):
    """Sequence dataset."""

    def __init__(self, data_dir, mode, config, transform=ToTensor(), task_config=None):
        '''
        Initialize the dataset.

        Args:
            data_dir (Path): The path to the data directory.
            mode (str): The mode of the dataset. Can be "train", "val", "test" or "all".
            config (obj): The data config object.
            transform (torchvision.transform, optional): The transform to apply to the data. Defaults to ToTensor().
            task_config (obj, optional): The task config object. Defaults to None.
        '''
        self.config = config
        self.transform = transform
        self.task_config = task_config
        self.data, self.keys = parse_json(
            data_dir,
            config,
            mode
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        name = self.keys[idx]

        data = torch.tensor(self.data[name], dtype=torch.int64)

        x = data[:-1]
        # pad with EOS tokens
        x = torch.cat([
            x,
            self.config['input_size'][1] * torch.ones(
                [self.config['input_size'][0] - x.shape[0]], dtype=torch.int64
            )
        ])
        y = data[-1]

        return x, y


class SymbolicDataset(torch.utils.data.Dataset):
    """Symbolic dataset."""

    def __init__(self, data_dir, mode, config, transform=ToTensor(), task_config=None):
        '''
        Initialize the dataset.

        Args:
            data_dir (Path): The path to the data directory.
            mode (str): The mode of the dataset. Can be "train", "val", "test" or "all".
            config (obj): The data config object.
            transform (torchvision.transform, optional): The transform to apply to the data. Defaults to ToTensor().
            task_config (obj, optional): The task config object. Defaults to None.
        '''
        self.config = config
        self.transform = transform
        self.task_config = task_config
        self.attributes, self.data, self.keys = parse_csv(
            data_dir,
            config,
            mode
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        name = self.keys[idx]

        data = torch.tensor(self.data[name], dtype=torch.float32)

        if (
            not self.task_config is None
            and "x_selection" in self.task_config.keys()
            and not len(self.task_config["x_selection"]) == 0
        ):
            x = torch.zeros(
                [len(self.task_config["selection"])], dtype=torch.float32)
            for idx, selection in enumerate(self.task_config["selection"]):
                x[idx] = x[self.attributes.index(selection)]

        else:
            x = data[:-1].to(torch.float32)

        if (
            not self.task_config is None
            and "y_selection" in self.task_config.keys()
            and not len(self.task_config["y_selection"]) == 0
        ):
            y = torch.zeros([len(self.task_config["y_selection"])])
            for idx, selection in enumerate(self.task_config["y_selection"]):
                y[idx] = y[self.attributes.index(selection)]

        else:
            y = data[-1]

        return x, y


class Image2MixedDataset(ImageDataset):
    """
    The celeba dataset.
    """

    def __init__(self, root_dir, mode, config, transform=ToTensor(), task_config=None):
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
        data_dir = os.path.join(root_dir, "data.csv")
        self.attributes, self.data, self.keys = parse_csv(
            data_dir, config, mode)

    def __len__(self):
        return len(self.keys)

    def enable_hints(self):
        self.hints_enabled = True

    def disable_hints(self):
        self.hints_enabled = False

    def __getitem__(self, idx):
        name = self.keys[idx]

        img = Image.open(os.path.join(self.root_dir, "imgs", name))
        # code.interact(local=dict(globals(), **locals()))
        state = torch.get_rng_state()
        img_tensor = self.transform(img)

        targets = self.data[name]

        if not self.task_config is None and not len(self.task_config["selection"]) == 0:
            target = torch.zeros([len(self.task_config["selection"])])
            for idx, selection in enumerate(self.task_config["selection"]):
                target[idx] = targets[self.attributes.index(selection)]

        else:
            target = torch.tensor(
                targets[: self.config["output_size"]], dtype=torch.float32
            )

        if not self.task_config is None and "ce" in self.task_config["criterions"]:
            assert (
                target.shape[0] == 1
            ), "output shape inacceptable for singleclass classification"
            target = torch.tensor(target[0], dtype=torch.int64)

        if not self.hints_enabled:
            return img_tensor, target

        else:
            mask = Image.open(os.path.join(self.root_dir, "masks", name))
            torch.set_rng_state(state)
            mask_tensor = self.transform(mask)
            return img_tensor, (target, mask_tensor)


class GlowDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, n_bits):
        self.base_dataset = base_dataset
        self.n_bits = n_bits
        self.n_bins = 2.0**n_bits

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
        """
        This function maps processed image back to human visible image
        """
        return image + 0.5


class VAEDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return self.base_dataset.__len__()

    def __getitem__(self, idx):
        x, y = self.base_dataset.__getitem__(idx)
        return x, x


class Image2ClassDataset(ImageDataset):
    """Shape Attribute dataset."""

    def __init__(self, root_dir, mode, config, transform=ToTensor(), task_config=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        if "has_hints" in self.config.keys() and self.config["has_hints"]:
            self.root_dir = os.path.join(root_dir, "imgs")
            self.mask_dir = os.path.join(root_dir, "masks")
            self.all_urls = []
            self.urls_with_hints = []

        else:
            self.root_dir = root_dir

        self.hints_enabled = False
        self.task_config = task_config
        self.transform = transform
        self.urls = []
        self.idx_to_name = os.listdir(self.root_dir)

        self.idx_to_name.sort()
        for label_str in self.idx_to_name:
            files = os.listdir(os.path.join(self.root_dir, label_str))
            files.sort()
            for file in files:
                self.urls.append((label_str, file))

        random.seed(0)
        random.shuffle(self.urls)

        if mode == "train":
            self.urls = self.urls[: int(config["split"][0] * len(self.urls))]

        elif mode == "val":
            self.urls = self.urls[
                int(config["split"][0] * len(self.urls)): int(
                    config["split"][1] * len(self.urls)
                )
            ]

        elif mode == "test":
            self.urls = self.urls[int(config["split"][1] * len(self.urls)):]

        if "has_hints" in self.config.keys() and self.config["has_hints"]:
            self.all_urls = copy.deepcopy(self.urls)
            for label_str, file in self.all_urls:
                if os.path.exists(os.path.join(self.mask_dir, file)):
                    self.urls_with_hints.append((label_str, file))

    def class_idx_to_name(self, class_idx):
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

        if img.shape[0] == 1 and self.config["input_size"][0] != 1:
            img = torch.tile(img, [self.config["input_size"][0], 1, 1])

        # label = torch.zeros([len(self.idx_to_name)], dtype=torch.float32)
        # label[self.idx_to_name.index(label_str)] = 1.0
        label = torch.tensor(self.idx_to_name.index(label_str))

        if not self.hints_enabled:
            return img, label

        else:
            # TODO how to apply same randomized transformation?
            mask = Image.open(os.path.join(self.mask_dir, file))
            torch.set_rng_state(state)
            mask = self.transform(mask)
            return img, (label, mask)
