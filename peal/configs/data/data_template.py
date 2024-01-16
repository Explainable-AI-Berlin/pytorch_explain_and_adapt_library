from datetime import datetime
from pydantic import BaseModel, PositiveInt
from typing import Union


class DataConfig(BaseModel):
    """
    The input type of the data:
    Options: ['image', 'sequence', 'symbolic']
    """

    input_type: str  # TODO give options?
    """
    The output type of the data.
    Options: ['singleclass', 'multiclass', 'continuous', 'mixed']
    'mixed' is a hybrid between binary multiclass classification and continuous and
    requires 'output_split' to be set
    """
    output_type: str
    """
    The input size of data.
    For images: [Channels, Height, Width]
    For sequences: [MaxLength, NumTokens]
    For symbolic: [NumVariables]
    """
    input_size: list[PositiveInt]
    """
    The output size of the model.
    For singleclass: [NumClasses]
    For multiclass: [NumBinaryClasses]
    For continuous: [NumVariables]
    For mixed: [NumBinaryClasses + NumVariables]
    """
    output_size: list[PositiveInt]
    """
    The path to the dataset.
    """
    dataset_path: Union[type(None), str] = None
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
    confounding_factors: list[str] = []
    """
    The correlation strength of the target and the confounding variable.
    """
    confounder_probability: Union[type(None), float] = None
    """
    The ratio of the classes in the dataset.
    """
    class_ratios: Union[type(None), list] = None
    """
    The seed the dataset was generated with.
    Only relevant for generated datasets!
    """
    seed: Union[type(None), int] = None
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
    The type of confounder present in the dataset.
    """
    confounding: Union[type(None), str] = None
    """
    The path of the original dataset.
    """
    dataset_origin_path: Union[type(None), str] = None
    """
    The name of the class.
    """
    __name__: str = "peal.DataConfig"
