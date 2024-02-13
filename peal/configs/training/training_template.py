from pydantic import BaseModel, PositiveInt

from typing import Union

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
    The train batch size. Can either be set manually or be left empty and calulated by adaptive batch_size.
    """
    train_batch_size: PositiveInt = 1
    """
    The val batch size. Can either be set manually or be left empty and calulated by adaptive batch_size.
    """
    val_batch_size: PositiveInt = 1
    """
    The test batch size. Can either be set manually or be left empty and calulated by adaptive batch_size.
    """
    test_batch_size: PositiveInt = 1
    """
    The number of iterations per episode when using DataloaderMixer.
    If it is not set, the DataloaderMixer is not used and the value implicitly becomes
    dataset_size / batch_size
    """
    steps_per_epoch: Union[type(None), PositiveInt] = None
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
