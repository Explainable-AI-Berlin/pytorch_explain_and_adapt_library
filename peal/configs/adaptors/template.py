from pydantic import BaseModel, PositiveInt
from typing import Union

from peal.configs.data.template import DataConfig
from peal.configs.training.template import TrainingConfig
from peal.configs.architectures.template import ArchitectureConfig
from peal.configs.models.template import TaskConfig
from peal.configs.explainers.template import ExplainerConfig


class AdaptorConfig:
    """
    The config template for an adaptor.
    """
    """
    The minimum number of samples used for finetuning in every iteration.
    The actual number could be higher since not for every sample a counterfactual can be found
    and processing is done in batches.
    """
    min_train_samples: PositiveInt
    """
    The maximum number of validation samples that are used for tracking stats every iteration.
    """
    max_validation_samples: PositiveInt
    """
    The number of finetune iterations when executing the adaptor.
    """
    finetune_iterations: PositiveInt
    """
    The config of the task the student model shall solve.
    """
    task: TaskConfig
    """
    The config of the counterfactual explainer that is used.
    """
    explainer: ExplainerConfig
    """
    The config of the trainer used for finetuning the student model.
    """
    training: TrainingConfig
    """
    The config of the data used to create the counterfactuals from.
    """
    data: DataConfig
    """
    Logging of the current finetune iteration
    """
    current_iteration: PositiveInt = 0
    """
    Whether to continue training from the current student model or start training from scratch
    again.
    """
    continuous_learning: bool = False
    """
    Whether to select sample for counterfactual creation the model is not that confident about.
    """
    min_start_target_percentile: float = 0.0
    """
    Whether to draw samples for counterfactual creation according to the error matrix or not.
    Makes particular sense in the multiclass setting where some classes might be in very
    different modes and one only wants to restrict to connected modes.
    """
    use_confusion_matrix: bool = False
    """
    Whether to replace the model every iteration or not.
    """
    replace_model: bool = True
    """
    Logging of the Feedback Accuracy.
    """
    fa_1sided_prime: float = 0.0
    """
    Whether to directly replace the model or wait one iteration.
    The latter sometimes makes sense if the model strategy at some point can't be detected anymore
    by the counterfactual explainer.
    """
    replacement_strategy: str = "delayed"
    """
    Defines whether the created counterfactuals are oversampled in relation to their number
    while finetuning.
    The mixing ratio defines how often is sampled from the base data distribution and how often
    from the counterfactual data distribution, e.g. with 0.5 both would be sampled equally often.
    """
    mixing_ratio: float = None
    """
    What batch_size is used for creating the counterfactuals?
    """
    batch_size: PositiveInt = None
    """
    The number of batches per iteration used for training.
    Can be calculated automatical from batch_size and the number of samples per iteration
    """
    num_batches: PositiveInt = None
    """
    The reference batch size when automatically adapting the batch_size to the vram
    """
    base_batch_size: PositiveInt = None
    """
    The reference vram of the gpu when using adaptive batch_size.
    """
    gigabyte_vram: float = None
    """
    The reference input size when using adaptive batch_size.
    """
    assumed_input_size: list[PositiveInt] = None
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
    """
    The name of the class.
    """
    __name__ : str = 'peal.AdaptorConfig'

    def __init__(
        self,
        training: Union[dict, TrainingConfig],
        task: Union[dict, TaskConfig],
        explainer: Union[dict, ExplainerConfig],
        data: Union[dict, DataConfig] = None,
        batch_size: PositiveInt = None,
        num_batches: PositiveInt = None,
        base_batch_size: PositiveInt = None,
        gigabyte_vram: float = None,
        assumed_input_size: list[PositiveInt] = None,
        replace_model: bool = True,
        continuous_learning: bool = False,
        mixing_ratio: float = None,
        min_start_target_percentile: float = 0.0,
        use_confusion_matrix: bool = False,
        replacement_strategy: str = "delayed",
        min_train_samples: PositiveInt = None,
        max_validation_samples: PositiveInt = None,
        finetune_iterations: PositiveInt = None,
        current_iteration: PositiveInt = 0,
        **kwargs,
    ):
        """
        The config template for an adaptor.
        Args:
            training: The config of the trainer used for finetuning the student model.
            task: The config of the task the student model shall solve.
            explainer: The config of the counterfactual explainer that is used.
            data: The config of the data used to create the counterfactuals from.
            batch_size: What batch_size is used for creating the counterfactuals?
            num_batches:    The number of batches per iteration used for training.
            base_batch_size: The reference batch size when automatically adapting the batch_size to the vram
            gigabyte_vram: The reference vram of the gpu when using adaptive batch_size.
            assumed_input_size: The reference input size when using adaptive batch_size.
            replace_model: Whether to replace the model every iteration or not.
            continuous_learning: Whether to continue training from the current student model or start training from scratch
            mixing_ratio: Defines whether the created counterfactuals are oversampled in relation to their number
            min_start_target_percentile: Whether to select sample for counterfactual creation the model is not that confident about.
            use_confusion_matrix: Whether to draw samples for counterfactual creation according to the error matrix or not.
            replacement_strategy: Whether to directly replace the model or wait one iteration.
            min_train_samples: The minimum number of samples used for finetuning in every iteration.
            max_validation_samples: The maximum number of validation samples that are used for tracking stats every iteration.
            finetune_iterations: The number of finetune iterations when executing the adaptor.
            current_iteration: Logging of the current finetune iteration
            **kwargs: A dict containing all variables that could not be given with the current config structure
        """
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
            DataConfig(**data)

        if isinstance(explainer, ExplainerConfig):
            self.explainer = explainer

        else:
            self.explainer = ExplainerConfig(**explainer)

        self.batch_size = batch_size
        self.num_batches = num_batches
        self.base_batch_size = base_batch_size
        self.gigabyte_vram = gigabyte_vram
        self.assumed_input_size = assumed_input_size
        self.replace_model = replace_model
        self.continuous_learning = continuous_learning
        self.mixing_ratio = mixing_ratio
        self.min_start_target_percentile = min_start_target_percentile
        self.use_confusion_matrix = use_confusion_matrix
        self.replacement_strategy = replacement_strategy
        self.min_train_samples = min_train_samples
        self.max_validation_samples = max_validation_samples
        self.finetune_iterations = finetune_iterations
        self.current_iteration = current_iteration
        self.kwargs = kwargs
