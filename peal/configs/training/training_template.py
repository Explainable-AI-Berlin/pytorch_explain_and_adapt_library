from pydantic import BaseModel, PositiveInt

class TrainingConfig(BaseModel):
    '''
    The maxmimum number of epochs that the model is trained.
    '''
    max_epochs : PositiveInt
    '''
    The learning rate the model is trained with.
    '''
    learning_rate : float = 0.0001
    '''
    Logs how many steps the model was trained with already.
    '''
    global_train_step : int = 0
    '''
    Logs how many steps the model was validated for.
    '''
    global_validation_step : int = 0
    '''
    The current epoch of the model training.
    '''
    epoch: int = -1
    '''
    The reference batch size when automatically adapting the batch_size to the vram
    '''
    base_batch_size : PositiveInt = None
    '''
    A dict containing all variables that could not be given with the current config structure
    '''
    kwargs : dict = {}
    '''
    The optimizer used for training the model.
    '''
    optimizer : str = 'Adam'
    '''
    The reference vram of the gpu when using adaptive batch_size.
    '''
    gigabyte_vram : float = None
    '''
    The reference input size when using adaptive batch_size.
    '''
    assumed_input_size : list[PositiveInt] = None
    '''
    The train batch size. Can either be set manually or be left empty and calulated by adaptive batch_size.
    '''
    train_batch_size : PositiveInt = None
    '''
    The val batch size. Can either be set manually or be left empty and calulated by adaptive batch_size.
    '''
    val_batch_size : PositiveInt = None
    '''
    The test batch size. Can either be set manually or be left empty and calulated by adaptive batch_size.
    '''
    test_batch_size : PositiveInt = None
    '''
    The number of iterations per episode when using DataloaderMixer.
    If it is not set, the DataloaderMixer is not used and the value implicitly becomes
    dataset_size / batch_size
    '''
    steps_per_epoch : PositiveInt = None
