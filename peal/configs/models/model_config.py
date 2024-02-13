import types
import os

from pydantic import BaseModel, PositiveInt
from typing import Union

from peal.configs.data.data_config import DataConfig
from peal.configs.training.training_template import TrainingConfig
from peal.configs.architectures.architecture_template import ArchitectureConfig


class TaskConfig:
    """
    A dict of critirion names (that have to be implemented in peal.training.criterions)
    mapped to the weight.
    Like this the loss function can be post_hoc attached without changing the code.
    """

    criterions: dict = {'ce' : 1.0, 'l1' : 10000.0, 'l2' : 1.0}
    """
    The output_type that either can just be the output_type of the dataset or could be some
    possible subtype.
    E.g. when having a binary multiclass dataset one could use as task binary single class
    classification for one of the output variables.
    """
    output_type: str = "singleclass"
    """
    The output_channels that can be at most the output_channels of the dataset, but if a subtask is chosen
    the output_channels has also be adapted accordingly
    """
    output_channels: PositiveInt = 2
    """
    Gives the option to select a subset of the input variables. Only works for symbolic data.
    """
    x_selection: list[str] = None
    """
    Gives the option to select a subset of the output_variables.
    Can be used e.g. to transform binary multiclass into the subtask of predicting one of the
    binary variables with single class classification.
    """
    y_selection: list[str] = None
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
    """
    The name of the class.
    """
    __name__: str = "peal.TaskConfig"

    def __init__(
        self,
        criterions: dict = None,
        output_type: str = None,
        output_channels: PositiveInt = None,
        x_selection: list[str] = None,
        y_selection: list[str] = None,
        **kwargs
    ):
        self.criterions = criterions if criterions is not None else self.criterions
        self.output_type = output_type if output_type is not None else self.output_type
        self.output_channels = output_channels if output_channels is not None else self.output_channels
        self.x_selection = x_selection if x_selection is not None else self.x_selection
        self.y_selection = y_selection if y_selection is not None else self.y_selection
        self.kwargs = kwargs


class ModelConfig:
    """
    The config template for a model.
    """

    """
    The config of the architecture of the model.
    """
    architecture: Union[ArchitectureConfig, types.SimpleNamespace]
    """
    The config of the training of the model.
    """
    training: TrainingConfig
    """
    The config of the task the model shall solve.
    """
    task: TaskConfig
    """
    The config of the data used for training the model.
    """
    data: DataConfig = None
    """
    The name of the model.
    """
    model_path: str = "peal_runs/predictor1"
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
    """
    A flag that indicates if the model is loaded from a checkpoint.
    """
    is_loaded: bool = False
    """
    The name of the class.
    """
    __name__: str = "peal.ModelConfig"
    """
    The name of the class.
    """
    model_type: str = "discriminator"
    """
    The name of the class.
    """
    base_path: str = None

    def __init__(
        self,
        architecture: Union[dict, ArchitectureConfig],
        training: Union[dict, TrainingConfig],
        task: Union[dict, TaskConfig],
        data: Union[dict, DataConfig] = None,
        model_path: str = None,
        model_type: str = None,
        **kwargs
    ):
        if isinstance(architecture, ArchitectureConfig):
            self.architecture = architecture

        self.architecture = ArchitectureConfig(**architecture)

        self.training = (
            training
            if isinstance(training, TrainingConfig)
            else TrainingConfig(**training)
        )
        self.task = task if isinstance(task, TaskConfig) else TaskConfig(**task)
        if isinstance(data, DataConfig):
            self.data = data

        elif data is None:
            self.data = None

        else:
            self.data = DataConfig(**data)

        if not model_path is None:
            self.model_path = model_path

        if not model_type is None:
            self.model_type = model_type

        self.kwargs = kwargs
