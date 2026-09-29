"""Build the train / validation / test datasets described by a ``DataConfig``.

This is the entry point every PEAL training script, adaptor and explainer uses to
turn a data yaml into ``PealDataset`` instances. It assembles the torchvision
transform pipelines (invariance augmentations for training, deterministic
resizing and cropping for evaluation), resolves the dataset class from
``config.dataset_class`` or the input/output type, computes dataset statistics
for normalization when requested, and attaches the normalization and task
config to the returned datasets.
"""

import torch
import os

from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import v2

from peal.architectures.interfaces import TaskConfig
from peal.registry import lookup, UnknownComponentError
from peal.global_utils import (
    load_yaml_config,
    get_project_resource_dir,
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
    """Instantiate the train, validation and test datasets of a data config.

    Parameters
    ----------
    config : DataConfig or str
        Data config object or path of its yaml (resolved by
        ``load_yaml_config``). Consumed keys: ``dataset_path``, ``input_type``,
        ``output_type``, ``dataset_class``, ``invariances``, ``crop_size``,
        ``input_size``, ``normalization``, ``split``, ``diffusion_augmented``,
        ``batch_wise_augmentation`` and the diffusion augmentation keys
        ``generator``, ``sampling_time_fraction``, ``num_discretization_steps``.
    base_dir : str, optional
        Root directory of the dataset; defaults to ``config.dataset_path``.
    task_config : TaskConfig, optional
        Stored on every returned dataset as ``task_config``.
    return_dict : bool, default False
        Passed through to the dataset constructors.
    test_config : DataConfig or str, optional
        Separate config for the test split; defaults to ``config``.
    data_dir : str, optional
        Passed through to the train / val dataset constructors.

    Returns
    -------
    tuple of PealDataset
        ``(train_data, val_data, test_data)``. When ``test_config.split`` is
        ``[x, 1.0]`` the validation dataset object is reused as test set. Each
        dataset gets ``normalization`` and ``task_config`` attributes.

    Raises
    ------
    ValueError
        If ``dataset_class`` is unknown and the ``input_type`` / ``output_type``
        combination has no default dataset class.

    Notes
    -----
    Augmentations are selected by name from ``config.invariances``:
    ``circular_cut``, ``rotation``, ``rotation10``, ``hflipping``,
    ``vflipping``, ``random_resize10/20/50``, ``color_jitter``, ``sharpness``,
    ``blur``, ``crop``, ``horizontalflip``. An empty ``normalization`` list is
    filled in place with the per-channel mean and std of the training split.
    """
    config = load_yaml_config(config)
    if base_dir is None:
        base_dir = config.dataset_path

    if test_config is None:
        test_config = config
    else:
        test_config = load_yaml_config(test_config, DataConfig)

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
            transform_list_train.append(
                v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))
            )
        if "crop" in config.invariances:
            transform_list_train.append(
                transforms.RandomCrop(
                    224, scale=(0.7, 1.0), ratio=(0.75, 1.3333), interpolation=2
                )
            )
        if "horizontalflip" in config.invariances:
            transform_list_train.append(transforms.RandomHorizontalFlip())

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

        if config.diffusion_augmented and not config.batch_wise_augmentation:
            from peal.data.transformations import DiffusionAugmentation

            transform_list_train.append(
                DiffusionAugmentation(
                    config.generator,
                    config.sampling_time_fraction,
                    config.num_discretization_steps,
                )
            )

    #
    transform_train = transforms.Compose(transform_list_train)
    transform_validation = transforms.Compose(transform_list_validation)
    transform_test = transforms.Compose(transform_list_test)

    dataset = None
    if config.dataset_class:
        try:
            dataset = lookup(
                "datasets",
                config.dataset_class,
                base_class=PealDataset,
                scan_dir=os.path.join(get_project_resource_dir(), "peal", "data"),
            )
        except UnknownComponentError:
            # Historical behaviour: an unknown dataset_class falls through to
            # the input/output-type defaults below rather than failing here.
            dataset = None
    if dataset is not None:
        pass

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
    if config.diffusion_augmented and config.batch_wise_augmentation:
        from peal.data.transformations import DiffusionAugmentation

        train_data.diffusion_augmentation = DiffusionAugmentation(
            config.generator,
            config.sampling_time_fraction,
            config.num_discretization_steps,
            train_data,
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
