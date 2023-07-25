from datetime import datetime
from pydantic import BaseModel, PositiveInt

class DataConfig(BaseModel):
    '''
    The input type of the data:
    Options: ['image', 'sequence', 'symbolic']
    '''
    input_type: str # TODO give options?
    '''
    The output type of the data.
    Options: ['singleclass', 'multiclass', 'continuous', 'mixed']
    'mixed' is a hybrid between binary multiclass classification and continuous and
    requires 'output_split' to be set
    '''
    output_type: str
    '''
    The input size of data.
    For images: [Channels, Height, Width]
    For sequences: [MaxLength, NumTokens]
    For symbolic: [NumVariables]
    '''
    input_size: list[int]
    '''
    The output size of the model.
    For singleclass: [NumClasses]
    For multiclass: [NumBinaryClasses]
    For continuous: [NumVariables]
    For mixed: [NumBinaryClasses + NumVariables]
    '''
    output_size: PositiveInt
    '''
    The number of samples in the dataset.
    Sometimes important when executing specific experiments.
    '''
    num_samples: int = None
    '''
    The name of the dataset.
    Only necessary to tell dataset factory which customized dataset class to use
    '''
    name: str = None
    '''
    The split between train, validation and test set.
    '''
    split: list[float] = [0.8, 0.9]
    '''
    Whether the dataset contains spatial annotations where the true feature is.
    '''
    has_hints: bool = False
    '''
    The applied normalization.
    Options: ['mean0std1']
    '''
    normalization: str = None
    '''
    A list of the invariances exploited for data augmentation:
    Options: ['hflipping', 'vflipping', 'rotation', 'circlecut']
    '''
    invariances: list[str] = None
    '''
    The number of binary multiclass variables in the mixed setting.
    Has to be smaller than output_size.
    '''
    output_split: int = None
    '''
    The way how to downsize an sample if required.
    Options: ['Downsample', 'RandomCrop', 'CenterCrop']
    '''
    downsize: str = None
    '''
    A pair of known confounding factors, one usually being the target.
    This knowledge helps for controlled sampling of confounders for experiments.
    '''
    confounding_factors: list[str] = None
    '''
    The correlation strength of the target and the confounding variable.
    '''
    confounder_probability: float = None
    '''
    The seed the dataset was generated with.
    Only relevant for generated datasets!
    '''
    seed: int = None
    '''
    The label noise of a generated dataset.
    Necessary to mimic real dataset behauviour and avoid trivial non-robust solutions.
    '''
    label_noise: float = None
