from pydantic import BaseModel, PositiveInt

from peal.configs.data.template import DataConfig
from peal.configs.training.template import TrainingConfig
from peal.configs.architectures.template import ArchitectureConfig

class TaskConfig(BaseModel):
    '''
    A dict of critirion names (that have to be implemented in peal.training.criterions)
    mapped to the weight.
    Like this the loss function can be post_hoc attached without changing the code.
    '''
    criterions : dict
    '''
    The output_type that either can just be the output_type of the dataset or could be some
    possible subtype.
    E.g. when having a binary multiclass dataset one could use as task binary single class
    classification for one of the output variables.
    '''
    output_type : str
    '''
    The output_size that can be at most the output_size of the dataset, but if a subtask is chosen
    the output_size has also be adapted accordingly
    '''
    output_size : PositiveInt
    '''
    Gives the option to select a subset of the input variables. Only works for symbolic data.
    '''
    x_selection : list[str]
    '''
    Gives the option to select a subset of the output_variables.
    Can be used e.g. to transform binary multiclass into the subtask of predicting one of the
    binary variables with single class classification.
    '''
    y_selection: list[str]
    '''
    A dict containing all variables that could not be given with the current config structure
    '''
    kwargs : dict = {}

class ModelConfig(BaseModel):
    '''
    The config of the data used for training the model.
    '''
    data : DataConfig
    '''
    The config of the architecture of the model.
    '''
    architecture : ArchitectureConfig
    '''
    The config of the training of the model.
    '''
    training : TrainingConfig
    '''
    The config of the task the model shall solve.
    '''
    task : TaskConfig
    '''
    A dict containing all variables that could not be given with the current config structure
    '''
    kwargs : dict = {}

