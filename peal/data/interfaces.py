from typing import Union

import torch
from pydantic import BaseModel, PositiveInt

from torchvision import transforms

from peal.generators.interfaces import Generator




class PealDataset(torch.utils.data.Dataset):
    """
    This is the base class for all datasets in PEAL. It is a wrapper around

    Args:
        torch.utils.data.Dataset (nn.Module): The parent class for all datasets in PEAL
    """

    def generate_contrastive_collage(
        self,
        x_list: list,
        x_counterfactual_list: list,
        y_target_list: list,
        y_source_list: list,
        y_list: list,
        y_target_start_confidence_list: list,
        y_target_end_confidence_list: list,
        base_path: str,
        start_idx: int,
        y_counterfactual_teacher_list=None,
        y_original_teacher_list=None,
        feedback_list=None,
        **kwargs: dict,
    ):
        """
        This function generates a collage of the input and the counterfactual

        Args:
            batch_in (torch.tensor): The input batch
            counterfactual (torch.tensor): The counterfactual batch

        Returns:
            torch.tensor: The collage
        """
        return torch.zeros([3, 64, 64])

    def serialize_dataset(self, output_dir, x_list, y_list, sample_names=None):
        """
        This function serializes the dataset to a given directory

        Args:
            output_dir (Path): The output directory
            x_list (list): The list of inputs
            y_list (list): The list of labels
            sample_names (list, optional): The list of sample names. Defaults to None.
        """
        pass

    def project_to_pytorch_default(self, x):
        """
        This function maps processed data sample back to pytorch default format

        Args:
            x (torch.tensor): The data sample in the processed format

        Returns:
            torch.tensor: The data sample in the pytorch default format
        """
        if hasattr(self, "normalization"):
            x = self.normalization.invert(x)

        return x

    def project_from_pytorch_default(self, x):
        """
        This function maps pytorch default image to the processed format

        Args:
            x (torch.tensor): The data sample in the pytorch default format

        Returns:
            torch.tensor: The data sample in the processed format
        """
        if hasattr(self, "normalization"):
            x = self.normalization(x)

        if list(x.shape[-3:]) != self.config.input_size:
            x = transforms.Resize(self.config.input_size[1:])(x)

        return x

    def track_generator_performance(self, generator: Generator, batch_size=1):
        """
        This function tracks the performance of the generator

        Args:
            generator (Generator): The generator
        """
        return {}

    def distribution_distance(self, x_list):
        pass

    def pair_wise_distance(self, x1, x2):
        pass

    def variance(self, x_list):
        pass

    def flip_rate(self, y_list, y_counterfactual_list):
        pass


class DataConfig(BaseModel):
    """
    This class defines the config of a dataset.
    """
    """
    The type of config. This is necessary to find config class from yaml config
    """
    config_name: str = "DataConfig"
    """
    The input type of the data:
    Options: ['image', 'sequence', 'symbolic']
    """
    input_type: str = 'image'
    """
    The output type of the data.
    Options: ['singleclass', 'multiclass', 'continuous', 'mixed']
    'mixed' is a hybrid between binary multiclass classification and continuous and
    requires 'output_split' to be set
    """
    output_type: str = 'singleclass'
    """
    The input size of data.
    For images: [Channels, Height, Width]
    For sequences: [MaxLength, NumTokens]
    For symbolic: [NumVariables]
    """
    input_size: list[PositiveInt] = [3, 128, 128]
    """
    The output size of the model.
    For singleclass: [NumClasses]
    For multiclass: [NumBinaryClasses]
    For continuous: [NumVariables]
    For mixed: [NumBinaryClasses + NumVariables]
    """
    output_size: list[PositiveInt] = [2]
    """
    The path to the dataset.
    """
    dataset_path: Union[type(None), str] = None
    """
    The path to the dataset origin this dataset got derived from.
    """
    dataset_origin_path: Union[type(None), str] = None
    """
    The number of samples in the dataset.
    Sometimes important when executing specific experiments.
    """
    num_samples: Union[type(None), int] = None
    """
    The name of the dataset.
    Only necessary to tell dataset factory which customized dataset class to use
    """
    dataset_class: Union[type(None), str] = None
    """
    The split between train, validation and test set.
    """
    split: list[float] = [0.8, 0.9]
    """
    Whether the dataset contains spatial annotations where the true feature is.
    """
    has_hints: Union[type(None), bool] = False
    """
    The applied normalization.
    Options: ['mean0std1']
    """
    normalization: Union[type(None), list] = None
    """
    A list of the invariances exploited for data augmentation:
    Options: ['hflipping', 'vflipping', 'rotation', 'circlecut']
    """
    invariances: list[str] = []
    """
    The number of binary multiclass variables in the mixed setting.
    Has to be smaller than output_size.
    """
    output_split: Union[type(None), int] = None
    """
    The way how to downsize an sample if required.
    Options: ['Downsample', 'RandomCrop', 'CenterCrop']
    """
    downsize: Union[type(None), str] = None
    """
    A pair of known confounding factors, one usually being the target.
    This knowledge helps for controlled sampling of confounders for experiments.
    """
    confounding_factors: Union[type(None), list[str]] = []
    """
    The correlation strength of the target and the confounding variable.
    """
    confounder_probability: Union[type(None), float] = None
    """
    alternative to confounder_probability, specify individual group sizes, e.g. [0.25, 0.25, 0.25, 0.25]
    """
    full_confounder_config: Union[type(None), list[float]] = None
    """
    The ratio of the classes in the dataset.
    """
    class_ratios: Union[type(None), list] = None
    """
    The seed the dataset was generated with.
    Only relevant for generated datasets!
    """
    seed: Union[type(None), int] = 0
    """
    The label noise of a generated dataset.
    Necessary to mimic real dataset behauviour and avoid trivial non-robust solutions.
    """
    label_noise: Union[type(None), float] = None
    """
    Whether to set negative values to zero.
    """
    set_negative_to_zero: bool = False
    """
    The delimiter used for the csv file.
    """
    delimiter: Union[type(None), str] = None
    """
    The number of classes in the dataset.
    """
    crop_size: Union[type(None), int] = None
    """
    The path of the original dataset.
    """
    dataset_origin_path: Union[type(None), str] = None
    """
    The path of the original dataset.
    """
    inverse: Union[type(None), str] = None
    """
    Where the label path is relative to the dataset path.
    """
    label_rel_path: str = "data.csv"
    """
    The name of the folder where the images are stored.
    The header of the column in the underlying csv either has to be called like this as well or the path to the images
    has to be in the first column.
    """
    x_selection: str = "imgs"
    """
    Whether to load all datasets into the RAM or not. Careful with big datasets!
    """
    in_memory: bool = False
    """
    Path to spray label csv file. When set, use spray labels instead of true confounder labels.
    Samples without a spray label will be dropped
    """
    spray_label_file: str = None
    """
    Whether to re-balance group sizes after dropping samples without a spray label
    """
    spray_groups_balanced: bool = False
    spray_group_sizes: list[int] = None

