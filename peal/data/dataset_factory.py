import torch

from torchvision import transforms
from torchvision.transforms import ToTensor

from peal.utils import load_yaml_config
from peal.data.transformations import (
    CircularCut,
    Padding,
    RandomRotation,
    Normalization,
    IdentityNormalization,
    SetChannels,
)
from peal.data.datasets import (
    Image2MixedDataset,
    SequenceDataset,
    Image2ClassDataset,
    SymbolicDataset,
)


def get_datasets(config, base_dir, task_config=None, return_dict=False):
    """
    This function is used to get the datasets for training, validation and testing.

    Args:
        config (_type_): _description_
        base_dir (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
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
            transform_list_train.append(transforms.RandomCrop(config["crop_size"][1:]))
            transform_list_test.append(transforms.CenterCrop(config["crop_size"][1:]))

        transform_list_train.append(transforms.Resize(config["input_size"][1:]))
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
        "singleclass",
    ]:
        dataset = SymbolicDataset

    elif config["input_type"] == "sequence" and config["output_type"] in [
        "singleclass",
    ]:
        dataset = SequenceDataset

    else:
        raise ValueError(
            "input_type: "
            + config["input_type"]
            + ", output_type: "
            + config["output_type"]
            + " combination is not supported!"
        )

    #
    if config["input_type"] == "image" and config["use_normalization"]:
        stats_dataset = dataset(base_dir, "train", config, transform_test)
        samples = []
        for idx in range(stats_dataset.__len__()):
            samples.append(stats_dataset.__getitem__(idx)[0])

        samples = torch.stack(samples)
        config["normalization"].append(list(torch.mean(samples, [0, 2, 3]).numpy()))
        config["normalization"].append(list(torch.std(samples, [0, 2, 3]).numpy()))

        #
        normalization = Normalization(
            config["normalization"][0], config["normalization"][1]
        )

    else:
        normalization = IdentityNormalization()

    transform_train = transforms.Compose([transform_train, normalization])
    transform_test = transforms.Compose([transform_test, normalization])

    train_data = dataset(
        base_dir, "train", config, transform_train, return_dict=return_dict
    )
    val_data = dataset(
        base_dir, "val", config, transform_train, return_dict=return_dict
    )
    test_data = dataset(
        base_dir, "test", config, transform_test, return_dict=return_dict
    )

    # this is kind of dirty
    train_data.normalization = normalization
    val_data.normalization = normalization
    test_data.normalization = normalization

    # this is kind of dirty
    train_data.task_config = task_config
    val_data.task_config = task_config
    test_data.task_config = task_config

    return train_data, val_data, test_data
