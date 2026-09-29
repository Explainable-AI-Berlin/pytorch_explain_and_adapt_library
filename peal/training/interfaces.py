"""Pydantic configs of a predictor run: how it is trained and what it is.

``TrainingConfig`` carries the optimisation hyper-parameters and the
augmentation / robustness switches ``ModelTrainer`` reads, plus the running
counters (``epoch``, ``global_train_step``) that the trainer updates in place
so a checkpoint can be resumed. ``PredictorConfig`` is the top-level schema of
a ``predictor.yaml``: training, task, architecture and data configs, the run
directory and the checkpoint / distillation switches.
"""

from typing import Union

from pydantic import BaseModel, PositiveInt

from peal.architectures.interfaces import TaskConfig, ArchitectureConfig
from peal.data.interfaces import DataConfig
from peal.generators.interfaces import GeneratorConfig


class TrainingConfig(BaseModel):
    """Hyper-parameters and switches of one predictor training run.

    Besides ``max_epochs``, ``learning_rate``, ``dropout``, ``optimizer`` and the
    three batch sizes, the config holds the ``ModelTrainer`` switches for
    adversarial training (``adv_training``, ``attack_epsilon``,
    ``attack_num_steps``, ``input_noise_std``), mixup and label smoothing,
    class balancing, diffusion-based augmentation (``diffusion_augmented`` with
    ``sampling_time_fraction`` / ``num_discretization_steps``) and the
    ``early_stopping_goal`` metric the best checkpoint is picked by.
    ``steps_per_epoch`` enables the ``DataloaderMixer`` episode length;
    ``epoch``, ``global_train_step`` and ``global_validation_step`` are counters
    the trainer and logger increment during training.
    """

    max_epochs: PositiveInt = 15
    """
    The learning rate the model is trained with.
    """
    learning_rate: float = 0.0001
    """
    The learning rate the model is trained with.
    """
    dropout: float = 0.5
    """
    The dropout rate the model is trained with.
    """
    num_workers: int = 0
    """
    DataLoader worker processes for the loaders built from this config. 0 keeps
    loading in the training process (the historical behaviour); see
    ``peal.data.dataloaders.resolve_num_workers`` for the ``$PEAL_NUM_WORKERS``
    override and the reason the default is conservative.
    """
    global_train_step: int = 0
    """
    Logs how many steps the model was trained with already.
    """
    global_validation_step: int = 0
    """
    Logs how many steps the model was validated for.
    """
    epoch: int = -1
    """
    The current epoch of the model training.
    """
    optimizer: str = "Adam"
    """
    The optimizer used for training the model.
    """
    train_batch_size: PositiveInt = 1
    """
    The train batch size. Can either be set manually or be left empty and calculated by adaptive batch_size.
    """
    val_batch_size: PositiveInt = 1
    """
    The val batch size. Can either be set manually or be left empty and calculated by adaptive batch_size.
    """
    test_batch_size: PositiveInt = 1
    """
    The test batch size. Can either be set manually or be left empty and calculated by adaptive batch_size.
    """
    steps_per_epoch: Union[type(None), PositiveInt] = None
    """
    The number of iterations per episode when using DataloaderMixer.
    If it is not set, the DataloaderMixer is not used and the value implicitly becomes
    dataset_size / batch_size
    """
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
    diffusion_augmented: bool = False
    sampling_time_fraction: float = 0.3
    num_discretization_steps: int = 20
    regulization_level: float = 1.3


class PredictorConfig(BaseModel):
    """
    The config template for a model.
    """

    training: TrainingConfig
    """
    The config of the training of the model.
    """
    task: TaskConfig
    """
    The config of the task the model shall solve.
    """
    architecture: Union[ArchitectureConfig, str, type(None)] = None
    """
    The config of the architecture of the model.
    """
    data: Union[DataConfig, type(None)] = None
    """
    The config of the data used for training the model.
    """
    model_path: str = "peal_runs/predictor1"
    """
    The name of the model.
    """
    kwargs: dict = {}
    """
    A dict containing all variables that could not be given with the current config structure
    """
    is_loaded: bool = False
    """
    A flag that indicates if the model is loaded from a checkpoint.
    """
    model_type: str = "discriminator"
    """
    The name of the class.
    """
    base_path: Union[str, type(None)] = None
    """
    The name of the class.
    """
    seed: int = 0
    """
    The seed that is used to ensure reproducibility of results.
    """
    distill_from: str = "predictor"
    """
    Where to distill from if used for distillation. Could either be done from the dataset or from the model.
    """
    weights_path: Union[str, type(None)] = None
    continue_training: bool = False
    tracking_level: int = 4
    only_last_layer: bool = False
    generator: Union[type(None), GeneratorConfig] = None
    base_model: Union[type(None), str] = None
