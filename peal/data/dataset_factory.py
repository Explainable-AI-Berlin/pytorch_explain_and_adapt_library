import torch
import os

from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import v2

from peal.architectures.interfaces import TaskConfig
from peal.global_utils import (
    load_yaml_config,
    get_project_resource_dir,
    find_subclasses,
)
from peal.data.transformations import (
    CircularCut,
    Padding,
    RandomRotation,
    Normalization,
    IdentityNormalization,
    SetChannels,
    RandomResizeCropPad,
)
from peal.data.datasets import (
    Image2MixedDataset,
    Image2ClassDataset,
    SymbolicDataset,
)
from peal.data.interfaces import PealDataset, DataConfig


def get_datasets(
    config: DataConfig,
    base_dir: str = None,
    task_config: TaskConfig = None,
    return_dict: bool = False,
    test_config: DataConfig = None,
    data_dir: str = None,
):
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

    if base_dir is None:
        base_dir = config.dataset_path

    if test_config is None:
        test_config = config

    #
    transform_list_train = []
    transform_list_validation = []
    transform_list_test = []
    #
    if config.input_type == "image" and "circular_cut" in config.invariances:
        transform_list_train.append(CircularCut())
        transform_list_validation.append(CircularCut())

    if test_config.input_type == "image" and "circular_cut" in test_config.invariances:
        transform_list_test.append(CircularCut())

    #
    transform_list_train.append(ToTensor())
    transform_list_validation.append(ToTensor())
    transform_list_test.append(ToTensor())

    #
    if config.input_type == "image":
        if not config.crop_size is None:
            transform_list_train.append(Padding(config.crop_size[1:]))
            transform_list_validation.append(Padding(config.crop_size[1:]))

        if not test_config.crop_size is None:
            transform_list_test.append(Padding(test_config.crop_size[1:]))

        #
        if "rotation" in config.invariances:
            transform_list_train.append(RandomRotation())

        if "rotation10" in config.invariances:
            transform_list_train.append(RandomRotation(-10, 10))

        if "hflipping" in config.invariances:
            transform_list_train.append(transforms.RandomHorizontalFlip(p=0.5))

        if "vflipping" in config.invariances:
            transform_list_train.append(transforms.RandomVerticalFlip(p=0.5))

        if "random_resize10" in config.invariances:
            transform_list_train.append(RandomResizeCropPad((0.9, 1.1)))

        if "random_resize20" in config.invariances:
            transform_list_train.append(RandomResizeCropPad((0.8, 1.2)))

        if "random_resize50" in config.invariances:
            transform_list_train.append(RandomResizeCropPad((0.2, 1.5)))

        if "color_jitter" in config.invariances:
            transform_list_train.append(
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                )
            )

        if "sharpness" in config.invariances:
            transform_list_train.append(transforms.RandomAdjustSharpness(0.3))

        if "blur" in config.invariances:
            transform_list_train.append(v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)))

        #
        if not config.crop_size is None:
            transform_list_train.append(transforms.RandomCrop(config.crop_size[1:]))
            transform_list_validation.append(
                transforms.CenterCrop(config.crop_size[1:])
            )

        if not test_config.crop_size is None:
            transform_list_test.append(transforms.CenterCrop(test_config.crop_size[1:]))

        transform_list_train.append(transforms.Resize(config.input_size[1:]))
        transform_list_validation.append(transforms.Resize(config.input_size[1:]))
        transform_list_test.append(transforms.Resize(test_config.input_size[1:]))

        transform_list_train.append(SetChannels(config.input_size[0]))
        transform_list_validation.append(SetChannels(config.input_size[0]))
        transform_list_test.append(SetChannels(test_config.input_size[0]))

        if config.diffusion_augmented:
            from peal.data.transformations import DiffusionAugmentation
            transform_list_train.append(DiffusionAugmentation(
                config.generator,
                config.sampling_time_fraction,
                config.num_discretization_steps
            ))

    #
    transform_train = transforms.Compose(transform_list_train)
    transform_validation = transforms.Compose(transform_list_validation)
    transform_test = transforms.Compose(transform_list_test)

    dataset_class_list = find_subclasses(
        PealDataset,
        os.path.join(get_project_resource_dir(), "peal", "data"),
    )
    dataset_class_dict = {
        dataset_class.__name__: dataset_class for dataset_class in dataset_class_list
    }
    if config.dataset_class in dataset_class_dict.keys():
        dataset = dataset_class_dict[config.dataset_class]

    elif config.input_type == "image" and config.output_type == "singleclass":
        dataset = Image2ClassDataset

    elif config.input_type == "image" and config.output_type in [
        "multiclass",
        "mixed",
    ]:
        dataset = Image2MixedDataset

    elif config.input_type == "symbolic" and config.output_type in [
        "multiclass",
        "mixed",
        "singleclass",
    ]:
        dataset = SymbolicDataset

    else:
        raise ValueError(
            "input_type: "
            + test_config.input_type
            + ", output_type: "
            + test_config.output_type
            + " combination is not supported!"
        )

    #
    if config.input_type == "image" and not config.normalization is None:
        if len(config.normalization) == 0:
            stats_dataset = dataset(base_dir, "train", config, transform_test)
            samples = []
            for idx in range(stats_dataset.__len__()):
                samples.append(stats_dataset.__getitem__(idx)[0])

            samples = torch.stack(samples)
            config.normalization.append(list(torch.mean(samples, [0, 2, 3]).numpy()))
            config.normalization.append(list(torch.std(samples, [0, 2, 3]).numpy()))

        #
        normalization = Normalization(config.normalization[0], config.normalization[1])

    else:
        normalization = IdentityNormalization()

    transform_train = transforms.Compose([transform_train, normalization])
    transform_validation = transforms.Compose([transform_validation, normalization])
    transform_test = transforms.Compose([transform_test, normalization])

    train_data = dataset(
        root_dir=base_dir,
        mode="train",
        config=config,
        transform=transform_train,
        return_dict=return_dict,
        data_dir=data_dir,
    )
    val_data = dataset(
        root_dir=base_dir,
        mode="val",
        config=config,
        transform=transform_validation,
        return_dict=return_dict,
        data_dir=data_dir,
    )
    # TODO this is super dirty!!!
    if len(test_config.split) == 2 and test_config.split[1] == 1.0:
        test_data = val_data

    else:
        test_data = dataset(
            mode="test",
            config=test_config,
            transform=transform_test,
            return_dict=return_dict,
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
