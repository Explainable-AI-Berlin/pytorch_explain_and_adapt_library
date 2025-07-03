import types
from typing import Union

from pydantic import BaseModel, PositiveInt

from peal.architectures.interfaces import TaskConfig, ArchitectureConfig
from peal.data.interfaces import DataConfig


class TrainingConfig(BaseModel):
    """
    The maxmimum number of epochs that the model is trained.
    """

    """
    The learning rate the model is trained with.
    """
    max_epochs: PositiveInt = 15
    """
    The learning rate the model is trained with.
    """
    learning_rate: float = 0.0001
    """
    The dropout rate the model is trained with.
    """
    dropout: float = 0.5
    """
    Logs how many steps the model was trained with already.
    """
    global_train_step: int = 0
    """
    Logs how many steps the model was validated for.
    """
    global_validation_step: int = 0
    """
    The current epoch of the model training.
    """
    epoch: int = -1
    """
    The optimizer used for training the model.
    """
    optimizer: str = "Adam"
    """
    The train batch size. Can either be set manually or be left empty and calculated by adaptive batch_size.
    """
    train_batch_size: PositiveInt = 1
    """
    The val batch size. Can either be set manually or be left empty and calculated by adaptive batch_size.
    """
    val_batch_size: PositiveInt = 1
    """
    The test batch size. Can either be set manually or be left empty and calculated by adaptive batch_size.
    """
    test_batch_size: PositiveInt = 1
    """
    The number of iterations per episode when using DataloaderMixer.
    If it is not set, the DataloaderMixer is not used and the value implicitly becomes
    dataset_size / batch_size
    """
    steps_per_epoch: Union[type(None), PositiveInt] = None
    concatenate_batches: bool = True
    adv_training: bool = False
    input_noise_std: float = 0.1
    num_noise_vec: int = 1
    no_grad_attack: bool = False
    attack_epsilon: float = 1.0
    attack_num_steps: int = 5
    train_on_test: bool = False
    class_balanced: bool = False
    use_mixup: bool = False
    mixup_alpha: float = 1.0
    label_smoothing: float = 0.0
    early_stopping_goal: str = "average_accuracy"


class PredictorConfig(BaseModel):
    """
    The config template for a model.
    """

    """
    The config of the training of the model.
    """
    training: TrainingConfig
    """
    The config of the task the model shall solve.
    """
    task: TaskConfig
    """
    The config of the architecture of the model.
    """
    architecture: Union[ArchitectureConfig, str, type(None)] = None
    """
    The config of the data used for training the model.
    """
    data: Union[DataConfig, type(None)] = None
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
    model_type: str = "discriminator"
    """
    The name of the class.
    """
    base_path: Union[str, type(None)] = None
    """
    The seed that is used to ensure reproducability of results.
    """
    seed: int = 0
    """
    Where to distill from if used for distillation. Could either be done from the dataset or from the model.
    """
    distill_from: str = "predictor"
    weights_path: Union[str, type(None)] = None
    continue_training: bool = False
    tracking_level: int = 4
    only_last_layer: bool = False
