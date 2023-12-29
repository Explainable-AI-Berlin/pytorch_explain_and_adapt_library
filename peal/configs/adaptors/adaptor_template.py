from pydantic import BaseModel, PositiveInt
from typing import Union

from peal.configs.data.data_template import DataConfig
from peal.configs.training.training_template import TrainingConfig
from peal.configs.architectures.architecture_template import ArchitectureConfig
from peal.configs.models.model_template import TaskConfig
from peal.configs.explainers.explainer_template import ExplainerConfig


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
    data: DataConfig = None
    """
    The config of the test data used evaluate the real progress on.
    """
    test_data: DataConfig = None
    """
    The path of the student used.
    """
    student: str = None
    """
    The type of teacher used.
    """
    teacher: str = "human@8000"
    """
    The path of the generator used.
    """
    generator: str = None
    """
    The base directory where the run of CFKD is stored.
    """
    base_dir: str = "peal_runs/cfkd"
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
    The attribution threshold.
    """
    attribution_threshold: float = 2.0
    """
    What batch_size is used for creating the counterfactuals?
    """
    batch_size: PositiveInt = None
    """
    The number validation runs used for evaluating CFKD.
    """
    validation_runs: PositiveInt = 3
    """
    The reference batch size when automatically adapting the batch_size to the vram
    """
    calculate_group_accuracies: bool = True
    """
    The reference vram of the gpu when using adaptive batch_size.
    """
    gigabyte_vram: float = None
    """
    The reference input size when using adaptive batch_size.
    """
    assumed_input_size: list[PositiveInt] = None
    """
    Whether to overwrite the logs and intermediate results.
    """
    overwrite: bool = True
    """
    Whether to overwrite the logs and intermediate results.
    """
    use_visualization: bool = False
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
    """
    The name of the class.
    """
    __name__: str = "peal.AdaptorConfig"

    def __init__(
        self,
        training: Union[dict, TrainingConfig],
        task: Union[dict, TaskConfig],
        explainer: Union[dict, ExplainerConfig],
        data: Union[dict, DataConfig] = None,
        test_data: Union[dict, DataConfig] = None,
        student: str = None,
        teacher: str = None,
        generator: str = None,
        base_dir: str = None,
        batch_size: PositiveInt = None,
        validation_runs: PositiveInt = None,
        calculate_group_accuracies: bool = None,
        gigabyte_vram: float = None,
        assumed_input_size: list[PositiveInt] = None,
        replace_model: bool = True,
        continuous_learning: bool = False,
        attribution_threshold: float = None,
        min_start_target_percentile: float = 0.0,
        use_confusion_matrix: bool = False,
        replacement_strategy: str = "delayed",
        min_train_samples: PositiveInt = None,
        max_validation_samples: PositiveInt = None,
        finetune_iterations: PositiveInt = None,
        current_iteration: PositiveInt = None,
        overwrite: bool = None,
        use_visualization: bool = None,
        **kwargs,
    ):
        """
        The config template for an adaptor.
        Args:
            training: The config of the trainer used for finetuning the student model.
            task: The config of the task the student model shall solve.
            explainer: The config of the counterfactual explainer that is used.
            data: The config of the data used to create the counterfactuals from.
            testdata: The config of the test data used evaluate the real progress on.
            student: The type of student used.
            teacher: The type of teacher used.
            generator: The type of generator used.
            base_dir: The base directory where the run of CFKD is stored.
            batch_size: What batch_size is used for creating the counterfactuals?
            validation_runs:    The number of batches per iteration used for training.
            calculate_group_accuracies: The reference batch size when automatically adapting the batch_size to the vram
            gigabyte_vram: The reference vram of the gpu when using adaptive batch_size.
            assumed_input_size: The reference input size when using adaptive batch_size.
            replace_model: Whether to replace the model every iteration or not.
            continuous_learning: Whether to continue training from the current student model or start training from scratch
            attribution_threshold: Defines whether the created counterfactuals are oversampled in relation to their number
            min_start_target_percentile: Whether to select sample for counterfactual creation the model is not that confident about.
            use_confusion_matrix: Whether to draw samples for counterfactual creation according to the error matrix or not.
            replacement_strategy: Whether to directly replace the model or wait one iteration.
            min_train_samples: The minimum number of samples used for finetuning in every iteration.
            max_validation_samples: The maximum number of validation samples that are used for tracking stats every iteration.
            finetune_iterations: The number of finetune iterations when executing the adaptor.
            current_iteration: Logging of the current finetune iteration
            overwrite: Whether to overwrite the logs and intermediate results.
            use_visualization: Whether to visualize the results.
            **kwargs: A dict containing all variables that could not be given with the current config structure
        """
        self.training = (
            training
            if isinstance(training, TrainingConfig)
            else TrainingConfig(**training)
        )
        self.task = task if isinstance(task, TaskConfig) else TaskConfig(**task)
        if not data is None:
            if isinstance(data, DataConfig):
                self.data = data

            else:
                self.data = DataConfig(**data)

        if not test_data is None:
            if isinstance(test_data, DataConfig):
                self.test_data = test_data

            else:
                self.test_data = DataConfig(**test_data)

        if isinstance(explainer, ExplainerConfig):
            self.explainer = explainer

        else:
            self.explainer = ExplainerConfig(**explainer)

        self.student = student if not student is None else self.student
        self.teacher = teacher if not teacher is None else self.teacher
        self.generator = generator if not generator is None else self.generator
        self.base_dir = base_dir if not base_dir is None else self.base_dir
        self.batch_size = batch_size if not batch_size is None else self.batch_size
        self.validation_runs = validation_runs if not validation_runs is None else self.validation_runs
        self.calculate_group_accuracies = (
            calculate_group_accuracies if not calculate_group_accuracies is None else self.calculate_group_accuracies
        )
        self.gigabyte_vram = (
            gigabyte_vram if not gigabyte_vram is None else self.gigabyte_vram
        )
        self.assumed_input_size = (
            assumed_input_size
            if not assumed_input_size is None
            else self.assumed_input_size
        )
        self.replace_model = (
            replace_model if not replace_model is None else self.replace_model
        )
        self.continuous_learning = (
            continuous_learning
            if not continuous_learning is None
            else self.continuous_learning
        )
        self.attribution_threshold = (
            attribution_threshold if not attribution_threshold is None else self.attribution_threshold
        )
        self.min_start_target_percentile = (
            min_start_target_percentile
            if not min_start_target_percentile is None
            else self.min_start_target_percentile
        )
        self.use_confusion_matrix = (
            use_confusion_matrix
            if not use_confusion_matrix is None
            else self.use_confusion_matrix
        )
        self.replacement_strategy = (
            replacement_strategy
            if not replacement_strategy is None
            else self.replacement_strategy
        )
        self.min_train_samples = (
            min_train_samples
            if not min_train_samples is None
            else self.min_train_samples
        )
        self.max_validation_samples = (
            max_validation_samples
            if not max_validation_samples is None
            else self.max_validation_samples
        )
        self.finetune_iterations = (
            finetune_iterations
            if not finetune_iterations is None
            else self.finetune_iterations
        )
        self.current_iteration = (
            current_iteration
            if not current_iteration is None
            else self.current_iteration
        )
        self.overwrite = overwrite if not overwrite is None else self.overwrite
        self.use_visualization = use_visualization if not use_visualization is None else self.use_visualization
        self.kwargs = kwargs
