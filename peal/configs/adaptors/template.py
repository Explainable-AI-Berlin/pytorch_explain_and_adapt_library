from datetime import datetime
from pydantic import BaseModel, PositiveInt

from peal.configs.data.template import DataConfig
from peal.configs.training.template import TrainingConfig
from peal.configs.architectures.template import ArchitectureConfig
from peal.configs.models.template import TaskConfig
from peal.configs.explainers.template import ExplainerConfig

class AdaptorConfig(BaseModel):
    '''
    TODO: The minimum number of samples used for finetuning in every iteration.
    The actual number could be higher since not for every sample a counterfactual can be found
    and processing is done in batches.
    '''
    max_train_samples : PositiveInt
    '''
    The maximum number of validation samples that are used for tracking stats every iteration.
    '''
    max_validation_samples : PositiveInt
    '''
    The number of finetune iterations when executing the adaptor.
    '''
    finetune_iterations : PositiveInt
    '''
    The config of the task the student model shall solve.
    '''
    task : TaskConfig
    '''
    The config of the counterfactual explainer that is used.
    '''
    explainer : ExplainerConfig
    '''
    The config of the trainer used for finetuning the student model.
    '''
    training : TrainingConfig
    '''
    The config of the data used to create the counterfactuals from.
    '''
    data : DataConfig
    '''
    Logging of the current finetune iteration
    '''
    current_iteration : PositiveInt = 0
    '''
    Whether to continue training from the current student model or start training from scratch
    again.
    '''
    continuous_learning : bool = False
    '''
    Whether to select sample for counterfactual creation the model is not that confident about.
    '''
    min_start_target_percentile : float = 0.0
    '''
    Whether to draw samples for counterfactual creation according to the error matrix or not.
    Makes particular sense in the multiclass setting where some classes might be in very
    different modes and one only wants to restrict to connected modes.
    '''
    use_confusion_matrix : bool = False
    '''
    Whether to replace the model every iteration or not.
    '''
    replace_model : bool = True
    '''
    Logging of the Feedback Accuracy.
    '''
    fa_1sided_prime : float = 0.0
    '''
    Whether to directly replace the model or wait one iteration.
    The latter sometimes makes sense if the model strategy at some point can't be detected anymore
    by the counterfactual explainer.
    '''
    replacement_strategy : str = 'delayed'
    '''
    The config of the architecture of the student model.
    '''
    architecture : ArchitectureConfig = None
    '''
    Defines whether the created counterfactuals are oversampled in relation to their number
    while finetuning.
    The mixing ratio defines how often is sampled from the base data distribution and how often
    from the counterfactual data distribution, e.g. with 0.5 both would be sampled equally often.
    '''
    mixing_ratio : float = None
    '''
    What batch_size is used for creating the counterfactuals?
    '''
    batch_size : PositiveInt = None
    '''
    The number of batches per iteration used for training.
    Can be calculated automatical from batch_size and the number of samples per iteration
    '''
    num_batches : PositiveInt = None
    '''
    The reference batch size when automatically adapting the batch_size to the vram
    '''
    base_batch_size : PositiveInt = None
    '''
    The reference vram of the gpu when using adaptive batch_size.
    '''
    gigabyte_vram : float = None
    '''
    The reference input size when using adaptive batch_size.
    '''
    assumed_input_size : list[PositiveInt] = None
    '''
    A dict containing all variables that could not be given with the current config structure
    '''
    kwargs : dict = {}
