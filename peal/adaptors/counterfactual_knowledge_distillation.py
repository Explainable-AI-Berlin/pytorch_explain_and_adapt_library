import os
from datetime import datetime

import torch
import copy
import shutil
import torchvision
import numpy as np
import platform

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Union
from torch import nn
from types import SimpleNamespace
from pydantic import PositiveInt
from typing import Union

from peal.global_utils import (
    load_yaml_config,
    save_yaml_config,
)
from peal.training.loggers import log_images_to_writer
from peal.data.dataloaders import (
    DataStack,
    DataloaderMixer,
    create_dataloaders_from_datasource,
)
from peal.training.trainers import ModelTrainer, calculate_test_accuracy
from peal.explainers.counterfactual_explainer import (
    CounterfactualExplainer,
    PerfectFalseCounterfactualConfig,
)
from peal.visualization.model_comparison import (
    create_comparison,
)
from peal.teachers.segmentation_mask_teacher import SegmentationMaskTeacher
from peal.data.datasets import ImageDataset, Image2MixedDataset
from peal.teachers.teacher_factory import get_teacher
from peal.teachers.interfaces import TeacherInterface
from peal.generators.interfaces import InvertibleGenerator
from peal.generators.generator_factory import get_generator
from peal.training.training_utils import (
    calculate_validation_statistics,
)
from peal.data.datasets import DataConfig
from peal.generators.interfaces import GeneratorConfig
from peal.training.trainers import TrainingConfig, TaskConfig
from peal.explainers.interfaces import ExplainerConfig
from peal.explainers.counterfactual_explainer import ACEConfig
from peal.adaptors.interfaces import AdaptorConfig, Adaptor
from peal.global_utils import get_config_model


class CFKDConfig(AdaptorConfig):
    """
    The config template for an running the CFKD adaptor.
    """

    """
    The config template for an adaptor.
    """
    adaptor_type: str = "CFKD"
    """
    The minimum number of samples used for finetuning in every iteration.
    The actual number could be higher since not for every sample a counterfactual can be found
    and processing is done in batches.
    """
    min_train_samples: PositiveInt = 500
    """
    The maximum number of validation samples that are used for tracking stats every iteration.
    """
    max_validation_samples: PositiveInt = 100
    """
    The maximum number of test batches.
    """
    max_test_batches: Union[type(None), PositiveInt] = None
    """
    The number of finetune iterations when executing the adaptor.
    """
    finetune_iterations: PositiveInt = 2
    """
    The config of the task the student model shall solve.
    """
    task: TaskConfig = TaskConfig()
    """
    The config of the counterfactual explainer that is used.
    """
    explainer: ExplainerConfig = ACEConfig()
    """
    The config of the trainer used for finetuning the student model.
    """
    training: TrainingConfig = TrainingConfig()
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
    continuous_learning: str = "deep_feature_reweighting"
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
    best_feedback_accuracy: float = 0.0
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
    If use_confusion_matrix is deativated always use an even batch_size!!!
    """
    batch_size: PositiveInt = 2
    """
    The number validation runs used for evaluating CFKD.
    """
    validation_runs: PositiveInt = 1
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
    What level of tracking is used.
    """
    tracking_level: int = 0
    """
    What level of tracking is used.
    """
    counterfactual_type: str = "1sided"
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
        training: Union[dict, TrainingConfig] = None,
        task: Union[dict, TaskConfig] = None,
        explainer: Union[dict, ExplainerConfig] = None,
        data: Union[dict, DataConfig] = None,
        test_data: Union[dict, DataConfig] = None,
        generator: Union[dict, GeneratorConfig] = None,
        student: str = None,
        teacher: str = None,
        base_dir: str = None,
        batch_size: PositiveInt = None,
        validation_runs: PositiveInt = None,
        calculate_group_accuracies: bool = None,
        gigabyte_vram: float = None,
        assumed_input_size: list[PositiveInt] = None,
        replace_model: bool = None,
        continuous_learning: bool = None,
        attribution_threshold: float = None,
        min_start_target_percentile: float = None,
        use_confusion_matrix: bool = None,
        replacement_strategy: str = None,
        min_train_samples: PositiveInt = None,
        max_validation_samples: PositiveInt = None,
        finetune_iterations: PositiveInt = None,
        current_iteration: PositiveInt = None,
        overwrite: bool = None,
        use_visualization: bool = None,
        tracking_level: int = None,
        counterfactual_type: str = None,
        max_test_batches: Union[type(None), PositiveInt] = None,
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
        if not training is None:
            self.training = (
                training
                if isinstance(training, TrainingConfig)
                else TrainingConfig(**training)
            )

        if not task is None:
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

        elif isinstance(explainer, dict):
            explainer_config_model = get_config_model(explainer)
            self.explainer = explainer_config_model(**explainer)

        if isinstance(generator, GeneratorConfig):
            self.generator = generator

        elif isinstance(generator, dict):
            generator_config_model = get_config_model(generator)
            self.generator = generator_config_model(**generator)
            self.generator.full_args = generator

        self.student = student if not student is None else self.student
        self.teacher = teacher if not teacher is None else self.teacher
        self.base_dir = base_dir if not base_dir is None else self.base_dir
        self.batch_size = batch_size if not batch_size is None else self.batch_size
        self.validation_runs = (
            validation_runs if not validation_runs is None else self.validation_runs
        )
        self.calculate_group_accuracies = (
            calculate_group_accuracies
            if not calculate_group_accuracies is None
            else self.calculate_group_accuracies
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
            attribution_threshold
            if not attribution_threshold is None
            else self.attribution_threshold
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
        self.use_visualization = (
            use_visualization
            if not use_visualization is None
            else self.use_visualization
        )
        self.max_test_batches = (
            max_test_batches if not max_test_batches is None else self.max_test_batches
        )
        self.tracking_level = (
            tracking_level if not tracking_level is None else self.tracking_level
        )
        self.counterfactual_type = (
            counterfactual_type
            if not counterfactual_type is None
            else self.counterfactual_type
        )
        if not self.use_confusion_matrix:
            assert (
                self.batch_size % 2 == 0
            ), "Batch size must be even when using deterministic CFKD!"
        self.kwargs = kwargs


class CounterfactualKnowledgeDistillation(Adaptor):
    """
    This class implements the counterfactual knowledge distillation approach.
    """

    def __init__(
        self,
        student: nn.Module = None,
        datasource: Union[list, tuple] = None,
        output_size: int = None,
        generator: Union[InvertibleGenerator, Path, str] = None,
        base_dir: Union[str, Path] = None,
        teacher: Union[str, TeacherInterface] = None,
        adaptor_config: Union[
            dict, str, Path, AdaptorConfig
        ] = "<PEAL_BASE>/configs/adaptors/symbolic_cfkd.yaml",
        gigabyte_vram: float = None,
        overwrite: bool = None,
    ):
        """
        This is the constructor for the CounterfactualKnowledgeDistillation class.

        Args:
            student (nn.Module): The student model that is improved with CFKD
            datasource (Union): The datasource that is used for training
            output_size (int, optional): The output size of the student model. Defaults to None.
            generator (Union, optional):
                The generator that is used for CFKD.
                Defaults to "<PEAL_BASE>/configs/models/default_generator.yaml".
            base_dir (Union, optional):
                The base directory for the run.
                Defaults to os.path.join("peal_runs", "counterfactual_knowledge_distillation").
            teacher (Union, optional): The teacher that is used for CFKD. Defaults to "human@8000".
            adaptor_config (Union, optional):
                The config for the adaptor.
                Defaults to "<PEAL_BASE>/configs/adaptors/symbolic_cfkd.yaml".
            gigabyte_vram (float, optional): The amount of vram in gigabytes. Defaults to None.
            overwrite (bool, optional):
                The flag that indicates whether the run should be overwritten. Defaults to False.
            visualization (callable, optional):
                The visualization function that is used for the run. Defaults to lambda x: x.
        """
        # TODO make sure to use seeds everywhere!
        self.adaptor_config = load_yaml_config(adaptor_config, AdaptorConfig)
        self.base_dir = (
            base_dir if not base_dir is None else self.adaptor_config.base_dir
        )
        #
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if student is None:
            student = torch.load(self.adaptor_config.student, map_location=self.device)

        self.original_student = student
        self.original_student.eval()
        self.overwrite = (
            overwrite if not overwrite is None else self.adaptor_config.overwrite
        )
        self.adaptor_config.overwrite = False
        self.student = copy.deepcopy(student)
        self.student.eval()
        teacher = teacher if not teacher is None else self.adaptor_config.teacher

        """self.output_size = integrate_data_config_into_adaptor_config(
            self.adaptor_config, datasource, output_size
        )"""

        """set_adaptive_batch_size(
            self.adaptor_config, gigabyte_vram, self.adaptor_config.min_train_samples
        )"""

        #
        # kind of dirty, but also very confusing if not done this way since validation batches are fed directly
        # into the explainer and thereby potentially causing VRAM overflows otherwise
        self.adaptor_config.training.val_batch_size = self.adaptor_config.batch_size
        (
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        ) = create_dataloaders_from_datasource(
            datasource=datasource,
            config=self.adaptor_config,
            test_config=self.adaptor_config.test_data,
            enable_hints=bool(teacher == "SegmentationMask"),
        )
        self.dataloaders_val = [self.val_dataloader, None]
        self.adaptor_config.data = self.train_dataloader.dataset.config

        #
        self.generator = get_generator(
            generator=(
                generator if not generator is None else self.adaptor_config.generator
            ),
            device=self.device,
            predictor_dataset=self.val_dataloader.dataset,
        )

        self.output_size = (
            self.adaptor_config.task.output_channels
            if self.adaptor_config.task.output_channels is not None
            else self.adaptor_config.data.output_size[0]
        )

        #
        self.teacher = get_teacher(
            teacher=teacher,
            output_size=self.output_size,
            adaptor_config=self.adaptor_config,
            dataset=self.val_dataloader.dataset,
            device=self.device,
            tracking_level=self.adaptor_config.tracking_level,
        )
        if self.adaptor_config.training.steps_per_epoch is None:
            self.adaptor_config.training.steps_per_epoch = (
                len(self.train_dataloader.dataset)
                // self.adaptor_config.training.train_batch_size
            )

        self.dataloader_mixer = DataloaderMixer(
            self.adaptor_config.training, self.train_dataloader
        )
        self.datastack = DataStack(
            self.train_dataloader.dataset,
            self.output_size,
            transform=self.val_dataloader.dataset.transform,
        )

        if teacher[:5] == "human":
            assert self.adaptor_config.tracking_level > 0, "Tracking level too low!"

        self.explainer = CounterfactualExplainer(
            predictor=self.student,
            generator=self.generator,
            input_type=self.adaptor_config.data.input_type,
            explainer_config=self.adaptor_config.explainer,
            datasets=[self.train_dataloader.dataset, self.val_dataloader.dataset],
            tracking_level=self.adaptor_config.tracking_level,
        )
        self.logits_to_prediction = lambda logits: logits.argmax(-1)
        self.tracked_keys = [
            "x_counterfactual_list",
            "y_source_list",
            "y_target_list",
            "y_target_end_confidence_list",
            "x_list",
            "y_list",
            "x_attribution_list",
        ]

        if self.adaptor_config.tracking_level > 0:
            self.tracked_keys.extend(
                [
                    "y_target_start_confidence_list",
                    "z_difference_list",
                    "collage_path_list",
                ]
            )

        if teacher == "SegmentationMask":
            self.tracked_keys.append("hint_list")
            self.train_dataloader.dataset.enable_hints()
            self.val_dataloader.dataset.enable_hints()

        if isinstance(
            self.explainer.explainer_config, PerfectFalseCounterfactualConfig
        ):
            self.tracked_keys.append("idx_list")
            self.train_dataloader.dataset.enable_idx()
            self.val_dataloader.dataset.enable_idx()
            self.test_dataloader.dataset.enable_idx()

        self.data_config = copy.deepcopy(self.adaptor_config)
        self.data_config.data.split = [1.0, 1.0]
        self.data_config.data.confounding_factors = []
        self.data_config.data.confounder_probability = None
        self.data_config.data.output_type = "singleclass"
        self.data_config.data.output_size = self.train_dataloader.dataset.output_size
        self.data_config.data.delimiter = ","
        self.data_config.data.num_samples = self.adaptor_config.min_train_samples
        self.data_config.data.dataset_class = None
        self.validation_data_config = copy.deepcopy(self.data_config)
        self.validation_data_config.data.num_samples = (
            self.adaptor_config.max_validation_samples
        )
        self.validation_data_config.data.split = [0.0, 1.0]

    def initialize_run(self):
        if self.overwrite:
            # move from self.base_dir to self.base_dir + "_old_" + {date}_{timestamp}
            if os.path.exists(self.base_dir):
                shutil.move(
                    self.base_dir,
                    self.base_dir
                    + "_old_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S"),
                )

        if not os.path.exists(
            os.path.join(self.base_dir, "0", "validation_tracked_values.npz")
        ):
            assert self.adaptor_config.current_iteration == 0
            print("Create base_dir in: " + str(self.base_dir))
            Path(self.base_dir).mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(os.path.join(self.base_dir, "logs"))
            log_images_to_writer(self.train_dataloader, writer, "train0")
            log_images_to_writer(self.val_dataloader, writer, "validation0")
            log_images_to_writer(self.test_dataloader, writer, "test")

            test_accuracy = calculate_test_accuracy(
                self.student,
                self.test_dataloader,
                self.device,
                self.adaptor_config.calculate_group_accuracies,
                self.adaptor_config.max_test_batches,
            )

            if (
                isinstance(self.val_dataloader.dataset, ImageDataset)
                and self.adaptor_config.use_visualization
            ):
                generator_sample = self.generator.sample_x(
                    batch_size=self.adaptor_config.batch_size
                )
                if not generator_sample is None:
                    torchvision.utils.save_image(
                        generator_sample,
                        os.path.join(self.base_dir, "generator_sample.png"),
                        normalize=True,
                    )

                    # TODO move this back!!!
                    generator_performance = (
                        self.val_dataloader.dataset.track_generator_performance(
                            self.generator, batch_size=self.adaptor_config.batch_size
                        )
                    )
                    print("Generator performance: " + str(generator_performance))

                    self.adaptor_config.generator_performance = generator_performance

            save_yaml_config(
                self.adaptor_config, os.path.join(self.base_dir, "config.yaml")
            )

            with open(os.path.join(self.base_dir, "platform.txt"), "w") as f:
                f.write(platform.node())

            if self.adaptor_config.calculate_group_accuracies:
                (
                    test_accuracy,
                    group_accuracies,
                    group_distribution,
                    worst_group_accuracy,
                ) = test_accuracy
                for idx in range(len(group_accuracies)):
                    writer.add_scalar(
                        "test_group_accuracy_" + str(idx),
                        group_accuracies[idx],
                        self.adaptor_config.current_iteration,
                    )
                    writer.add_scalar(
                        "test_group_distribution_" + str(idx),
                        group_distribution[idx],
                        self.adaptor_config.current_iteration,
                    )

                writer.add_scalar(
                    "test_worst_group_accuracy",
                    worst_group_accuracy,
                    self.adaptor_config.current_iteration,
                )
                print("group_accuracies: " + str(group_accuracies))
                print("group_distribution: " + str(group_distribution))
                print("worst_group_accuracy: " + str(worst_group_accuracy))

            writer.add_scalar(
                "test_accuracy", test_accuracy, self.adaptor_config.current_iteration
            )
            validation_stats = self.retrieve_validation_stats(finetune_iteration=0)
            for key in validation_stats.keys():
                if isinstance(validation_stats[key], float):
                    writer.add_scalar(
                        "validation_" + key,
                        validation_stats[key],
                        self.adaptor_config.current_iteration,
                    )

            if self.output_size == 2 and self.adaptor_config.use_visualization:
                print("visualize progress!!!")
                self.visualize_progress(
                    [os.path.join(self.base_dir, "visualization.png")]
                )

            self.adaptor_config.test_accuracies = [test_accuracy]

        else:
            with open(os.path.join(self.base_dir, "platform.txt"), "w") as f:
                f.write(platform.node())

            writer = SummaryWriter(os.path.join(self.base_dir, "logs"))
            validation_stats = self.retrieve_validation_stats(
                finetune_iteration=self.adaptor_config.current_iteration
            )
            self.dataloader_mixer = DataloaderMixer(
                self.adaptor_config.training, self.train_dataloader
            )

            for i in range(1, self.adaptor_config.current_iteration + 1):
                dataset_dir = os.path.join(self.base_dir, str(i), "train_dataset")
                # TODO how to recover mixing_ratio?
                self.dataloader_mixer = self.add_dataset_to_dataloader_mixer(
                    dataloader_old=self.dataloader_mixer,
                    dataset_path=dataset_dir,
                    mixing_ratio=0.5,
                    writer=writer,
                )

            if self.adaptor_config.current_iteration > 0:
                self.student = torch.load(
                    os.path.join(self.adaptor_config.base_dir, "model.cpl"),
                    map_location=self.device,
                )

        return validation_stats, writer

    def get_batch(
        self,
        error_matrix: torch.Tensor = None,
    ):
        x_batch = []
        y_source_batch = []
        y_target_batch = []
        y_batch = []
        y_target_start_confidence_batch = []
        hint_batch = []
        idx_batch = []
        sample_idx = 0
        torch.manual_seed(torch.seed())
        if self.adaptor_config.use_confusion_matrix:
            error_distribution = torch.distributions.Categorical(error_matrix)

        else:
            cm_idx = 1

        while not sample_idx >= self.adaptor_config.batch_size:
            if self.adaptor_config.use_confusion_matrix:
                cm_idx = error_distribution.sample()

            # TODO verify that this is actually balancing itself!
            y_source = int(cm_idx / self.output_size)
            y_target = int(cm_idx % self.output_size)
            while y_source == y_target:
                cm_idx = (cm_idx + 1) % (self.output_size**2)
                y_source = int(cm_idx / self.output_size)
                y_target = int(cm_idx % self.output_size)

            cm_idx = (cm_idx + 1) % (self.output_size**2)
            x, y = self.datastack.pop(int(y_source))

            if isinstance(self.teacher, SegmentationMaskTeacher) or isinstance(
                self.explainer.explainer_config, PerfectFalseCounterfactualConfig
            ):
                y_res = y[1:]
                y = y[0]
                if isinstance(self.teacher, SegmentationMaskTeacher):
                    hint = y_res[0]

                if isinstance(
                    self.explainer.explainer_config, PerfectFalseCounterfactualConfig
                ):
                    idx = y_res[-1]

            logits = (
                self.student(x.to(self.device).unsqueeze(0)).squeeze(0).detach().cpu()
            )
            y_target_start_confidence = torch.nn.Softmax()(logits)[y_target]
            prediction = self.logits_to_prediction(logits)
            if (
                not self.adaptor_config.counterfactual_type == "1sided"
                or prediction == y == y_source
            ):
                x_batch.append(x)
                y_source_batch.append(y_source)
                y_target_batch.append(torch.tensor(y_target))
                y_batch.append(y)
                y_target_start_confidence_batch.append(y_target_start_confidence)
                if isinstance(self.teacher, SegmentationMaskTeacher):
                    hint_batch.append(hint)

                else:
                    hint_batch.append(torch.zeros_like(x))

                if isinstance(
                    self.explainer.explainer_config, PerfectFalseCounterfactualConfig
                ):
                    idx_batch.append(idx)

                else:
                    idx_batch.append(0)

                sample_idx += 1
                print([int(y), y_source, y_target, y_target_start_confidence])

        x_batch = torch.stack(x_batch)
        y_target_batch = torch.stack(y_target_batch)
        return {
            "x_list": x_batch,
            "y_list": y_batch,
            "y_target_list": y_target_batch,
            "y_source_list": y_source_batch,
            "y_target_start_confidence_list": y_target_start_confidence_batch,
            "hint_list": hint_batch,
            "idx_list": idx_batch,
        }

    def generate_x_counterfactual_list(
        self,
        error_matrix,
        confidence_score_stats,
        finetune_iteration,
        tracked_keys,
    ):
        self.datastack.reset()

        collage_base_path = os.path.join(
            self.base_dir, str(finetune_iteration), "collages"
        )
        if os.path.exists(collage_base_path):
            # move from self.base_dir to self.base_dir + "_old_" + {date}_{timestamp}
            shutil.move(
                collage_base_path,
                collage_base_path
                + "_old_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        Path(collage_base_path).mkdir(parents=True, exist_ok=True)

        tracked_values = {key: [] for key in tracked_keys}

        continue_collecting = True
        if hasattr(self.adaptor_config.explainer, "y_target_goal_confidence"):
            acceptance_threshold = (
                self.adaptor_config.explainer.y_target_goal_confidence
            )

        else:
            acceptance_threshold = 0.51

        pbar = tqdm(
            total=int(
                self.adaptor_config.min_train_samples / self.adaptor_config.batch_size
                + 0.99
            )
            * (
                self.adaptor_config.explainer.gradient_steps
                if hasattr(self.adaptor_config.explainer, "gradient_steps")
                else 1
            )
        )
        pbar.stored_values = {}
        pbar.stored_values["n_total"] = 0
        while continue_collecting:
            num_batches_per_iteration = int(
                1
                + (
                    self.adaptor_config.min_train_samples
                    - len(list(tracked_values.values())[0])
                )
                / self.adaptor_config.batch_size
            )
            for i in range(num_batches_per_iteration):
                batch = self.get_batch(error_matrix)
                values = self.explainer.explain_batch(
                    batch=batch,
                    base_path=collage_base_path,
                    start_idx=len(list(tracked_values.values())[0]),
                    y_target_goal_confidence_in=acceptance_threshold,
                    remove_below_threshold=True,
                    pbar=pbar,
                    mode="Training",
                    explainer_path=os.path.join(
                        self.base_dir, str(finetune_iteration - 1)
                    ),
                )
                for key in tracked_keys:
                    tracked_values[key].extend(values[key])

                pbar.stored_values["n_valid"] = (
                    str(len(list(tracked_values.values())[0]))
                    + "/"
                    + str(self.adaptor_config.min_train_samples)
                )
                pbar.stored_values["th"] = acceptance_threshold
                pbar.stored_values["n_total"] += self.adaptor_config.batch_size
                pbar.stored_values["fr"] = (
                    len(list(tracked_values.values())[0])
                    / pbar.stored_values["n_total"]
                )

            if (
                len(list(tracked_values.values())[0])
                < self.adaptor_config.min_train_samples
            ):
                if (
                    acceptance_threshold == 0.51
                    and len(list(tracked_values.values())[0]) == 0
                ):
                    continue_collecting = False

                elif (
                    len(list(values.values())[0])
                    < self.adaptor_config.min_train_samples / 2
                ):
                    acceptance_threshold = float(
                        np.maximum(0.51, acceptance_threshold - 0.1)
                    )

            else:
                continue_collecting = False

        pbar.close()
        return tracked_values

    def retrieve_counterfactual_list(self, validation_stats, finetune_iteration):
        tracked_values_path = os.path.join(
            self.base_dir, str(finetune_iteration), "tracked_values.npz"
        )
        if self.overwrite or not os.path.exists(tracked_values_path):
            tracked_values = self.generate_x_counterfactual_list(
                error_matrix=validation_stats["error_matrix"],
                confidence_score_stats=validation_stats["confidence_score_stats"],
                finetune_iteration=finetune_iteration,
                tracked_keys=self.tracked_keys,
            )

            if len(list(tracked_values.values())[0]) == 0:
                return tracked_values

            if self.adaptor_config.tracking_level > 0:
                with open(
                    tracked_values_path,
                    "wb",
                ) as f:
                    tracked_values_file = {}
                    for key in self.tracked_keys:
                        if isinstance(tracked_values[key][0], torch.Tensor):
                            tracked_values_file[key] = torch.stack(
                                tracked_values[key], dim=0
                            ).numpy()

                        elif isinstance(tracked_values[key][0], int) or isinstance(
                            tracked_values[key][0], float
                        ):
                            tracked_values_file[key] = np.array(tracked_values[key])

                    np.savez(f, **tracked_values_file)

        else:
            with open(
                tracked_values_path,
                "rb",
            ) as f:
                tracked_values = {}
                tracked_values_file = np.load(f, allow_pickle=True)
                for key in tracked_values_file.keys():
                    tracked_values[key] = list(torch.tensor(tracked_values_file[key]))

        collage_path_list = os.listdir(
            os.path.join(self.base_dir, str(finetune_iteration), "collages")
        )
        tracked_values["collage_path_list"] = list(
            map(
                lambda x: os.path.join(
                    self.base_dir, str(finetune_iteration), "collages", x
                ),
                collage_path_list,
            )
        )

        return tracked_values

    def retrieve_feedback(self, tracked_values, finetune_iteration, mode):
        if self.overwrite or not os.path.exists(
            os.path.join(self.base_dir, str(finetune_iteration), mode + "_feedback.txt")
        ):
            feedback = self.teacher.get_feedback(
                base_dir=os.path.join(
                    self.base_dir, str(finetune_iteration), mode + "_teacher"
                ),
                student=self.student,
                **tracked_values,
            )

            os.makedirs(
                os.path.join(self.base_dir, str(finetune_iteration)), exist_ok=True
            )
            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), mode + "_feedback.txt"
                ),
                "w",
            ) as f:
                f.write("\n".join(feedback))

        else:
            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), mode + "_feedback.txt"
                ),
                "r",
            ) as f:
                feedback = f.read().split("\n")

        # TODO this is not correct for calculating training stats.
        if mode == "validation":
            num_samples = min(
                self.adaptor_config.max_validation_samples,
                len(self.val_dataloader.dataset),
            )

        else:
            num_samples = -1

        flip_rate = (
            len(
                list(
                    filter(
                        lambda x: x >= 0.51,
                        tracked_values["y_target_end_confidence_list"],
                    )
                )
            )
            / num_samples
        )
        ood_rate = len(list(filter(lambda sample: sample == "ood", feedback))) / len(
            feedback
        )

        """num_true_2sided = len(list(filter(lambda sample: sample == "true", feedback)))
        num_false_2sided = len(list(filter(lambda sample: sample == "false", feedback)))
        if num_true_2sided == 0:
            fa_2sided = 0

        else:
            fa_2sided = num_true_2sided / (num_true_2sided + num_false_2sided)"""

        num_true_1sided = len(
            list(
                filter(
                    lambda x: x[1] == "true"
                    and tracked_values["y_list"][x[0]]
                    == tracked_values["y_source_list"][x[0]],
                    enumerate(feedback),
                )
            )
        )
        num_false_1sided = len(
            list(
                filter(
                    lambda x: x[1] == "false"
                    and tracked_values["y_list"][x[0]]
                    == tracked_values["y_source_list"][x[0]],
                    enumerate(feedback),
                )
            )
        )
        if num_true_1sided + num_false_1sided > 0:
            fa_1sided = num_true_1sided / (num_true_1sided + num_false_1sided)

        else:
            fa_1sided = -1

        fa_absolute = num_true_1sided / num_samples

        feedback_stats = {
            "flip_rate": flip_rate,
            "ood_rate": ood_rate,
            "feedback_accuracy": fa_1sided,
        }

        return feedback, feedback_stats

    def create_dataset(
        self,
        x_counterfactual_list,
        feedback,
        y_source_list,
        y_target_list,
        y_list,
        finetune_iteration,
        config,
        mode="",
        **args,
    ):
        assert (
            len(x_counterfactual_list)
            == len(feedback)
            == len(y_source_list)
            == len(y_target_list)
        ), "missmatch in list lengths"

        dataset_dir = os.path.join(
            self.base_dir, str(finetune_iteration), mode + "_dataset"
        )
        if os.path.exists(dataset_dir):
            return dataset_dir
        #
        x_list = []
        y_counterfactual_list = []
        sample_names = []
        for sample_idx in range(len(feedback)):
            if feedback[sample_idx] == "ood":
                continue

            elif feedback[sample_idx] == "true":
                """sample_name = (
                    "true_"
                    + str(int(y_source_list[sample_idx]))
                    + "_to_"
                    + str(int(y_target_list[sample_idx]))
                    + "_"
                    + str(sample_idx)
                )
                x_list.append(x_counterfactual_list[sample_idx])
                y_list.append(int(y_target_list[sample_idx]))
                sample_names.append(sample_name)
                sample_idx += 1"""
                continue

            elif feedback[sample_idx] == "false":
                prediction = self.student(
                    x_counterfactual_list[sample_idx].unsqueeze(0).to(self.device)
                ).argmax()
                if prediction == int(y_source_list[sample_idx]):
                    print("This can not be a false counterfactual!")
                    continue

                sample_name = (
                    "false_"
                    + str(int(y_source_list[sample_idx]))
                    + "_to_"
                    + str(int(y_target_list[sample_idx]))
                    + "_"
                    + str(sample_idx)
                )
                x_list.append(x_counterfactual_list[sample_idx])
                # y_list.append(int(y_source_list[sample_idx]))
                y_counterfactual_list.append(int(y_source_list[sample_idx]))

                sample_names.append(sample_name)
                sample_idx += 1

        self.train_dataloader.dataset.serialize_dataset(
            output_dir=dataset_dir,
            x_list=x_list,
            y_list=y_counterfactual_list,
            sample_names=sample_names,
            classifier=self.student,
        )
        return dataset_dir

    def add_dataset_to_dataloader_mixer(
        self, dataloader_old, dataset_path, mixing_ratio, writer, finetune_iteration
    ):
        #
        dataloader, _, _ = create_dataloaders_from_datasource(
            config=self.data_config,
            datasource=dataset_path,
        )
        log_images_to_writer(dataloader, writer, "train_" + str(finetune_iteration))
        dataloader = DataloaderMixer(self.adaptor_config.training, dataloader)
        dataloader.append(dataloader_old, mixing_ratio=1 - mixing_ratio)
        dataloader.return_src = True
        return dataloader

    def finetune_student(self, finetune_iteration, dataset_path, writer):
        if self.overwrite or not os.path.exists(
            os.path.join(
                self.base_dir,
                str(finetune_iteration),
                "finetuned_model",
                "model.cpl",
            )
        ):
            if os.path.exists(
                os.path.join(self.base_dir, str(finetune_iteration), "finetuned_model")
            ):
                # move from self.base_dir to self.base_dir + "_old_" + {date}_{timestamp}
                shutil.move(
                    os.path.join(self.base_dir, str(finetune_iteration), "finetuned_model"),
                    os.path.join(
                        self.base_dir,
                        str(finetune_iteration),
                        "finetuned_model_old_"
                        + datetime.now().strftime("%Y%m%d_%H%M%S"),
                    ),
                )

            Path(
                os.path.join(self.base_dir, str(finetune_iteration), "finetuned_model")
            ).mkdir(parents=True, exist_ok=True)

            #
            val_dataset_path = os.path.join(
                self.base_dir, str(finetune_iteration), "validation_dataset"
            )
            _, dataloader_val, _ = create_dataloaders_from_datasource(
                config=self.validation_data_config,
                datasource=val_dataset_path,
            )
            self.dataloaders_val[1] = dataloader_val
            log_images_to_writer(
                dataloader_val, writer, "validation_" + str(finetune_iteration)
            )

            #
            mixing_ratio = min(0.5, 1 - self.feedback_accuracy)
            writer.add_scalar("mixing_ratio", mixing_ratio, finetune_iteration)
            if not hasattr(self, "dataloader_mixer"):
                self.dataloader_mixer = DataloaderMixer(
                    self.adaptor_config.training, self.train_dataloader
                )

            self.dataloader_mixer = self.add_dataset_to_dataloader_mixer(
                dataloader_old=self.dataloader_mixer,
                dataset_path=dataset_path,
                mixing_ratio=mixing_ratio,
                writer=writer,
                finetune_iteration=finetune_iteration,
            )

            y_list_dataset = [
                self.dataloader_mixer.dataloaders[0].dataset[idx][1]
                for idx in range(len(self.dataloader_mixer.dataloaders[0].dataset))
            ]
            for c in range(self.output_size):
                writer.add_scalar(
                    "class_ratio_" + str(c),
                    np.sum((torch.tensor(y_list_dataset) == c).numpy())
                    / len(y_list_dataset),
                    finetune_iteration,
                )

            if self.adaptor_config.continuous_learning == "retrain":

                def weight_reset(m):
                    reset_parameters = getattr(m, "reset_parameters", None)
                    if callable(reset_parameters):
                        m.reset_parameters()

                self.student.apply(weight_reset)

            finetune_trainer = ModelTrainer(
                config=copy.deepcopy(self.adaptor_config),
                model=self.student,
                datasource=(self.dataloader_mixer, self.dataloaders_val),
                model_path=os.path.join(
                    self.base_dir, str(finetune_iteration), "finetuned_model"
                ),
                val_dataloader_weights=[
                    1 - mixing_ratio,
                    mixing_ratio,
                ],
                only_last_layer=self.adaptor_config.continuous_learning
                == "deep_feature_reweighting",
            )
            if isinstance(self.teacher, SegmentationMaskTeacher):
                self.train_dataloader.dataset.disable_hints()
                self.val_dataloader.dataset.disable_hints()

            if isinstance(
                self.explainer.explainer_config, PerfectFalseCounterfactualConfig
            ):
                self.train_dataloader.dataset.disable_idx()
                self.val_dataloader.dataset.disable_idx()

            finetune_trainer.fit(continue_training=True)

            if isinstance(self.teacher, SegmentationMaskTeacher):
                self.train_dataloader.dataset.enable_hints()
                self.val_dataloader.dataset.enable_hints()

            if isinstance(
                self.explainer.explainer_config, PerfectFalseCounterfactualConfig
            ):
                self.train_dataloader.dataset.enable_idx()
                self.val_dataloader.dataset.enable_idx()

        self.student = torch.load(
            os.path.join(
                self.base_dir,
                str(finetune_iteration),
                "finetuned_model",
                "model.cpl",
            ),
            map_location=self.device,
        )

    def visualize_progress(self, paths):
        task_config_buffer = copy.deepcopy(self.test_dataloader.dataset.task_config)
        # TODO use canonic explainer config!!
        criterions = {}
        if (
            isinstance(self.test_dataloader.dataset, Image2MixedDataset)
            and "Confounder" in self.test_dataloader.dataset.attributes
        ):
            self.test_dataloader.dataset.task_config = SimpleNamespace(
                **{
                    "y_selection": None,
                    "criterions": [],
                }
            )
            criterions["class"] = lambda X, y: int(
                y[
                    self.test_dataloader.dataset.attributes.index(
                        task_config_buffer.y_selection[0]
                    )
                ]
            )
            criterions["confounder"] = lambda X, y: int(
                y[self.test_dataloader.dataset.attributes.index("Confounder")]
            )
            criterions["uncorrected"] = lambda X, y: int(
                self.original_student(X.unsqueeze(0).to(self.device))
                .squeeze(0)
                .cpu()
                .argmax()
            )
            criterions["cfkd"] = lambda X, y: int(
                self.student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
            )

        else:
            criterions["class"] = lambda X, y: int(y)
            criterions["uncorrected"] = lambda X, y: int(
                self.original_student(X.unsqueeze(0).to(self.device))
                .squeeze(0)
                .cpu()
                .argmax()
            )
            criterions["cfkd"] = lambda X, y: int(
                self.student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
            )

        checkbox_dict = {
            "class": torch.tensor([0, 0, 0, 1, 1, 1]),
            "confounder": torch.tensor([1, 1, 1, 0, 0, 0]),
            "uncorrected": torch.tensor([1, 1, 1, 0, 0, 0]),
            "cfkd": torch.tensor([0, 0, 0, 1, 1, 1]),
        }
        # TODO introduce column for teacher
        explainer_config = copy.deepcopy(self.explainer.explainer_config)
        for attribute in self.explainer.explainer_config.__dict__.items():
            if isinstance(attribute[1], list) and len(attribute[1]) == 2:
                setattr(
                    explainer_config,
                    attribute[0],
                    0.5 * (attribute[1][1] + attribute[1][0]),
                )

        tracking_level_buffer = self.explainer.tracking_level
        self.explainer.tracking_level = 0
        img_success = create_comparison(
            explainer=self.explainer,
            dataset=self.test_dataloader.dataset,
            criterions=criterions,
            columns={
                "Counterfactual\nExplanation": [
                    "cf",
                    self.original_student,
                    "uncorrected",
                    os.path.join(self.adaptor_config.base_dir, "0"),
                ],
                "CFKD\ncorrected": [
                    "cf",
                    self.student,
                    "cfkd",
                    os.path.join(
                        self.adaptor_config.base_dir,
                        str(self.adaptor_config.current_iteration),
                    ),
                ],
            },
            score_reference_idx=1,
            device=self.device,
            checkbox_dict_in=checkbox_dict,
            batch_size=self.adaptor_config.batch_size,
            max_samples=100,
        )
        for path in paths:
            img_success.save(path.replace(".png", "_success.png"))
            print("Saved: " + path.replace(".png", "_success.png"))

        img = create_comparison(
            explainer=self.explainer,
            dataset=self.test_dataloader.dataset,
            criterions=criterions,
            columns={
                "Counterfactual\nExplanation": [
                    "cf",
                    self.original_student,
                    "uncorrected",
                    os.path.join(self.adaptor_config.base_dir, "0"),
                ],
                "CFKD\ncorrected": [
                    "cf",
                    self.student,
                    "cfkd",
                    os.path.join(
                        self.adaptor_config.base_dir,
                        str(self.adaptor_config.current_iteration),
                    ),
                ],
            },
            score_reference_idx=1,
            device=self.device,
            batch_size=self.adaptor_config.batch_size,
            max_samples=100,
        )
        self.explainer.predictor = self.student
        self.explainer.tracking_level = tracking_level_buffer

        for path in paths:
            img.save(path)
            print("Saved: " + path)

        self.test_dataloader.dataset.task_config = task_config_buffer
        return img

    def retrieve_validation_stats(self, finetune_iteration):
        if not self.overwrite and os.path.exists(
            os.path.join(self.base_dir, str(finetune_iteration), "validation_stats.npz")
        ):
            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), "validation_stats.npz"
                ),
                "rb",
            ) as f:
                validation_stats = {}
                validation_tracked_file = np.load(f, allow_pickle=True)
                for key in validation_tracked_file.keys():
                    validation_stats[key] = torch.tensor(validation_tracked_file[key])

        validation_values_path = os.path.join(
            self.base_dir, str(finetune_iteration), "validation_tracked_values.npz"
        )
        if self.overwrite or not os.path.exists(validation_values_path):
            x_list_collection = []
            x_counterfactual_collection = []
            y_confidence_list = []
            original_explainer_config = copy.deepcopy(self.explainer.explainer_config)
            validation_tracked_values = None
            validation_stats = []
            for i in range(self.adaptor_config.validation_runs):
                self.explainer.explainer_config = copy.deepcopy(
                    original_explainer_config
                )
                if self.adaptor_config.validation_runs > 1:
                    for attribute in self.explainer.explainer_config.__dict__.items():
                        if isinstance(attribute[1], list) and len(attribute[1]) == 2:
                            if i == 0:
                                effective_idx = 0

                            else:
                                effective_idx = (
                                    i
                                    / (self.adaptor_config.validation_runs - 1)
                                    * (attribute[1][1] - attribute[1][0])
                                )

                            setattr(
                                self.explainer.explainer_config,
                                attribute[0],
                                attribute[1][0] + effective_idx,
                            )

                (
                    validation_tracked_values_current,
                    validation_stats_current,
                ) = calculate_validation_statistics(
                    model=self.student,
                    dataloader=self.dataloaders_val[0],
                    tracked_keys=self.tracked_keys,
                    base_path=os.path.join(
                        self.base_dir,
                        str(finetune_iteration),
                        "validation_collages" + str(i),
                    ),
                    output_size=self.output_size,
                    explainer=self.explainer,
                    device=self.device,
                    logits_to_prediction=self.logits_to_prediction,
                    use_confusion_matrix=self.adaptor_config.use_confusion_matrix,
                    max_validation_samples=self.adaptor_config.max_validation_samples,
                    min_start_target_percentile=self.adaptor_config.min_start_target_percentile,
                )
                # torch.nn.functional.softmax(self.student(validation_tracked_values_current['x_counterfactual_list'][i]
                # .unsqueeze(0).to('cuda')).squeeze(0))[validation_tracked_values_current['y_target_list'][i]]
                if validation_tracked_values is None:
                    validation_tracked_values = validation_tracked_values_current

                else:
                    for key in validation_tracked_values.keys():
                        validation_tracked_values[key].extend(
                            validation_tracked_values_current[key]
                        )

                validation_stats.append(validation_stats_current)

                if self.adaptor_config.validation_runs > 1:
                    x_list_collection.append(
                        copy.deepcopy(validation_tracked_values_current["x_list"])
                    )
                    x_counterfactual_collection.append(
                        copy.deepcopy(
                            validation_tracked_values_current["x_counterfactual_list"]
                        )
                    )
                    y_confidence_list.append(
                        copy.deepcopy(
                            validation_tracked_values_current[
                                "y_target_end_confidence_list"
                            ]
                        )
                    )

            validation_stats = {
                key: torch.mean(
                    torch.stack(
                        [
                            torch.tensor(validation_stats_current[key])
                            for validation_stats_current in validation_stats
                        ]
                    ),
                    dim=0,
                )
                for key in validation_stats[0].keys()
            }
            self.explainer.explainer_config = original_explainer_config
            if self.adaptor_config.validation_runs > 1:
                self.datastack.dataset._initialize_performance_metrics()
                validation_stats[
                    "distance_to_manifold"
                ] = self.datastack.dataset.distribution_distance(
                    x_counterfactual_collection
                )
                validation_stats[
                    "pairwise_distance"
                ] = self.datastack.dataset.pair_wise_distance(
                    x_list_collection, x_counterfactual_collection
                )
                validation_stats["diversity"] = self.datastack.dataset.variance(
                    x_counterfactual_collection
                )
                validation_stats["validity"] = self.datastack.dataset.flip_rate(
                    y_confidence_list
                )

            if self.adaptor_config.tracking_level > 0:
                os.makedirs(
                    os.path.join(self.base_dir, str(finetune_iteration)), exist_ok=True
                )
                with open(
                    validation_values_path,
                    "wb",
                ) as f:
                    tracked_values_file = {}
                    for key in self.tracked_keys:
                        if isinstance(validation_tracked_values[key][0], torch.Tensor):
                            tracked_values_file[key] = torch.stack(
                                validation_tracked_values[key], dim=0
                            ).numpy()

                        elif isinstance(
                            validation_tracked_values[key][0], int
                        ) or isinstance(validation_tracked_values[key][0], float):
                            tracked_values_file[key] = np.array(
                                validation_tracked_values[key]
                            )

                    np.savez(f, **tracked_values_file)

                with open(
                    os.path.join(
                        self.base_dir,
                        str(finetune_iteration),
                        "validation_prestats.npz",
                    ),
                    "wb",
                ) as f:
                    validation_stats_file = {}
                    for key in validation_stats.keys():
                        if isinstance(validation_stats[key], torch.Tensor):
                            validation_stats_file[key] = validation_stats[key].numpy()

                        elif isinstance(validation_stats[key], int) or isinstance(
                            validation_stats[key], float
                        ):
                            validation_stats_file[key] = np.array(validation_stats[key])

                    np.savez(f, **validation_stats_file)

        else:
            # TODO think about this again
            if self.adaptor_config.tracking_level > 0:
                with open(
                    validation_values_path,
                    "rb",
                ) as f:
                    validation_tracked_values = {}
                    validation_tracked_value_file = np.load(f, allow_pickle=True)
                    for key in validation_tracked_value_file.keys():
                        validation_tracked_values[key] = list(
                            torch.tensor(validation_tracked_value_file[key])
                        )

                with open(
                    os.path.join(
                        self.base_dir,
                        str(finetune_iteration),
                        "validation_prestats.npz",
                    ),
                    "rb",
                ) as f:
                    validation_stats = {}
                    validation_tracked_file = np.load(f, allow_pickle=True)
                    for key in validation_tracked_file.keys():
                        validation_stats[key] = torch.tensor(
                            validation_tracked_file[key]
                        )

            get_collage_path = lambda x: os.path.join(
                self.base_dir, str(finetune_iteration), "validation_collages" + str(x)
            )
            idx = 0
            collage_path_list = []
            while os.path.exists(get_collage_path(idx)):
                collage_path_list.extend(os.listdir(get_collage_path(idx)))
                idx += 1

            validation_tracked_values["collage_path_list"] = list(
                map(
                    lambda x: os.path.join(
                        self.base_dir,
                        str(finetune_iteration),
                        "validation_collages",
                        x,
                    ),
                    collage_path_list,
                )
            )

        validation_feedback, validation_feedback_stats = self.retrieve_feedback(
            tracked_values=validation_tracked_values,
            finetune_iteration=finetune_iteration,
            mode="validation",
        )

        for key in validation_feedback_stats.keys():
            validation_stats[key] = validation_feedback_stats[key]

        if self.adaptor_config.tracking_level > 0:
            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), "validation_stats.npz"
                ),
                "wb",
            ) as f:
                validation_stats_file = {}
                for key in validation_stats.keys():
                    if isinstance(validation_stats[key], torch.Tensor):
                        validation_stats_file[key] = validation_stats[key].numpy()

                    elif isinstance(validation_stats[key], int) or isinstance(
                        validation_stats[key], float
                    ):
                        validation_stats_file[key] = np.array(validation_stats[key])

                np.savez(f, **validation_stats_file)

        self.create_dataset(
            feedback=validation_feedback,
            finetune_iteration=finetune_iteration + 1,
            mode="validation",
            config=self.validation_data_config,
            **validation_tracked_values,
        )

        return validation_stats

    def run(self):
        """
        Run the counterfactual knowledge distillation
        """
        print("Adaptor Config: " + str(self.adaptor_config))
        validation_stats, writer = self.initialize_run()
        self.feedback_accuracy = validation_stats["feedback_accuracy"]

        # iterate over the finetune iterations
        for finetune_iteration in range(
            self.adaptor_config.current_iteration + 1,
            self.adaptor_config.finetune_iterations + 1,
        ):
            tracked_values = self.retrieve_counterfactual_list(
                validation_stats=validation_stats, finetune_iteration=finetune_iteration
            )
            if (
                len(list(tracked_values.values())[0])
                < self.adaptor_config.min_train_samples
            ):
                print("No counterfactuals could be found anymore!")
                open(os.path.join(self.base_dir, "warning.txt"), "w").write(
                    "No x_counterfactual_list could be found anymore in iteration "
                    + str(finetune_iteration)
                    + "!"
                )
                return self.student

            feedback, feedback_stats = self.retrieve_feedback(
                tracked_values=tracked_values,
                finetune_iteration=finetune_iteration,
                mode="train",
            )
            for key in validation_stats.keys():
                if isinstance(validation_stats[key], float):
                    writer.add_scalar(
                        "train_" + key,
                        validation_stats[key],
                        finetune_iteration,
                    )

            dataset_path = self.create_dataset(
                feedback=feedback,
                finetune_iteration=finetune_iteration,
                mode="train",
                config=self.data_config,
                **tracked_values,
            )
            self.finetune_student(
                finetune_iteration=finetune_iteration,
                dataset_path=dataset_path,
                writer=writer,
            )

            test_accuracy = calculate_test_accuracy(
                self.student,
                self.test_dataloader,
                self.device,
                self.adaptor_config.calculate_group_accuracies,
                self.adaptor_config.max_test_batches,
            )
            if self.adaptor_config.calculate_group_accuracies:
                (
                    test_accuracy,
                    group_accuracies,
                    group_distribution,
                    worst_group_accuracy,
                ) = test_accuracy
                for idx in range(len(group_accuracies)):
                    writer.add_scalar(
                        "test_group_accuracy_" + str(idx),
                        group_accuracies[idx],
                        finetune_iteration,
                    )
                    writer.add_scalar(
                        "test_group_distribution_" + str(idx),
                        group_distribution[idx],
                        finetune_iteration,
                    )

                writer.add_scalar(
                    "test_worst_group_accuracy",
                    worst_group_accuracy,
                    finetune_iteration,
                )
                print("group_accuracies: " + str(group_accuracies))

            writer.add_scalar("test_accuracy", test_accuracy, finetune_iteration)
            print("test_accuracy: " + str(test_accuracy))
            validation_stats = self.retrieve_validation_stats(
                finetune_iteration=finetune_iteration
            )
            for key in validation_stats.keys():
                if isinstance(validation_stats[key], float):
                    writer.add_scalar(
                        "validation_" + key,
                        validation_stats[key],
                        finetune_iteration,
                    )

            self.feedback_accuracy = validation_stats["feedback_accuracy"]

            if self.output_size == 2 and self.adaptor_config.use_visualization:
                self.visualize_progress(
                    [
                        os.path.join(
                            self.base_dir, str(finetune_iteration), "visualization.png"
                        ),
                        os.path.join(self.base_dir, "visualization.png"),
                    ]
                )

            if (
                self.adaptor_config.replacement_strategy == "delayed"
                and self.adaptor_config.replace_model
            ):
                torch.save(self.student, os.path.join(self.base_dir, "model.cpl"))

            if (
                validation_stats["feedback_accuracy"]
                > self.adaptor_config.best_feedback_accuracy
            ):
                # self.adaptor_config["fa_1sided_prime"] = validation_stats["fa_1sided"]
                self.adaptor_config.best_feedback_accuracy = validation_stats[
                    "feedback_accuracy"
                ]
                if self.adaptor_config.replacement_strategy == "direct":
                    torch.save(self.student, os.path.join(self.base_dir, "model.cpl"))

                self.adaptor_config.replace_model = True

            else:
                self.adaptor_config.replace_model = False

            self.adaptor_config.current_iteration = (
                self.adaptor_config.current_iteration + 1
            )
            save_yaml_config(
                self.adaptor_config, os.path.join(self.base_dir, "config.yaml")
            )

        return self.student
