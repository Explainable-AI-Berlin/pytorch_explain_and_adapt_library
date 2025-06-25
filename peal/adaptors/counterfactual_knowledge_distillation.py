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
from torch import nn
from types import SimpleNamespace
from pydantic import PositiveInt
from typing import Union

from peal.architectures.predictors import TorchvisionModel, get_predictor
from peal.global_utils import load_yaml_config, save_yaml_config, cprint
from peal.training.loggers import log_images_to_writer
from peal.data.dataloaders import (
    DataStack,
    DataloaderMixer,
    create_dataloaders_from_datasource,
    WeightedDataloaderList,
)
from peal.training.trainers import (
    ModelTrainer,
    calculate_test_accuracy,
    distill_predictor,
)
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
from peal.data.interfaces import DataConfig
from peal.generators.interfaces import GeneratorConfig
from peal.training.interfaces import TrainingConfig, PredictorConfig
from peal.architectures.interfaces import TaskConfig
from peal.explainers.interfaces import ExplainerConfig
from peal.explainers.counterfactual_explainer import SCEConfig
from peal.adaptors.interfaces import AdaptorConfig, Adaptor


class CFKDConfig(AdaptorConfig):
    """
    The config template for an running the CFKD adaptor.
    """

    """
    The adaptor_type for CFKDConfig has to be CFKD to find the CFKDConfig class when loading from a yaml file.
    """
    adaptor_type: str = "CFKD"
    """
    The minimum number of samples used for finetuning in every iteration.
    The actual number could be higher since not for every sample a counterfactual can be found
    and processing is done in batches.
    """
    min_train_samples: PositiveInt = 800
    """
    The maximum number of validation samples that are used for tracking stats every iteration.
    """
    max_validation_samples: PositiveInt = 200
    """
    The maximum number of test batches.
    If set to None the test will be done on the full test set.
    """
    max_test_batches: Union[type(None), PositiveInt] = None
    """
    The number of finetune iterations when executing the adaptor.
    If set to 0 only the explanation and no adaption is done.
    """
    finetune_iterations: int = 1
    """
    The config of the task the student model shall solve.
    """
    task: Union[TaskConfig, type(None)] = None
    """
    The config of the counterfactual explainer that is used.
    All parameters regarding paths, where the generator is from etc in there are overwritten by CFKD and only
    used if the information is not available for CFKD
    """
    explainer: ExplainerConfig = SCEConfig()
    """
    The config of the training used for finetuning the student model.
    If not set student config can be used.
    """
    training: Union[TrainingConfig, type(None)] = TrainingConfig()
    """
    The config of the data used to create the counterfactuals from.
    """
    data: DataConfig = None
    """
    The config of the test data used evaluate the real progress on.
    Often this data has a distribution shift compared to the training data or comes from a totally different data source.
    Hence the option to give its own config.
    If set to None the normal data config is taken.
    """
    test_data: DataConfig = None
    """
    The path of the student used.
    Can be either the path to a PyTorch or an onnx model directly or the path to a predictor config or a PredictorConfig
    object.
    """
    student: Union[PredictorConfig, str, type(None)] = None
    """
    The type of teacher used.
    """
    teacher: str = "cluster@8000"
    """
    The config of the generator used.
    This value will be overwritten if Generator is given via constructor directly.
    If the Generator is not given via constructor and this value is set to None explainer config is searched for
    generator config.
    """
    generator: Union[GeneratorConfig, type(None)] = None
    """
    The base directory where the run of CFKD is stored.
    All the visualzations and the caching is stored here.
    """
    base_dir: str = "peal_runs/cfkd"
    """
    Logging of the current finetune iteration
    """
    current_iteration: int = 0
    """
    Whether to continue training from the current student model or start training from scratch
    again. Can e.g. be "retrain", which retrains model on original data and counterfactuals from scratch,
    "finetune", which starts of at the weights of the uncorrected student or "deep_feature_reweighting", which
    only finetunes the last layer of the the uncorrected student.
    """
    continuous_learning: str = "finetune"
    """
    Whether to draw samples for counterfactual creation according to the error matrix or not.
    Makes particular sense in the multiclass setting where some classes might be in very
    different modes and one only wants to restrict to connected modes.
    """
    use_confusion_matrix: bool = False
    """
    Logging of the Feedback Accuracy.
    """
    best_feedback_accuracy: float = 0.0
    """
    The attribution threshold when using the SegmentationMask teacher.
    Setting it to 0.0 means that every counterfactual that did on average bigger changes inside than outside the mask
    is considered a True counterfactual and every counterfactual that does not is considered a false counterfactual.
    If it is set higher the bar for True Counterfactual is set higher and if is set lower the bar is set lower as well.
    """
    attribution_threshold: float = 0.0
    """
    What batch_size is used for creating the counterfactuals?
    """
    batch_size: PositiveInt = 1
    """
    The number validation runs used for evaluating CFKD.
    """
    validation_runs: PositiveInt = 1
    """
    Whether to calculate group accuracies or not. This can only be done if confounding factors are known.
    """
    calculate_group_accuracies: bool = False
    """
    Whether to overwrite the logs and cache intermediate results.
    If overwrite is set to False cached results are loaded. If CFKDConfig is stored as yaml on disk overwrite is 
    automatically set to False so that CFKD can be continued at the last cached result.
    Using this feature dramatically improves ability to debug!
    """
    overwrite: bool = True
    """
    How aggressively to change the model based on the counterfactual samples. 0 -> No change, 1 -> Full change
    """
    mixing_ratio: float = 0.5
    """"
    A list of the feedback accuracies.
    """
    feedback_accuracies: list = []
    """
    What type of counterfactuals are valid. 1sided means that we can only start from samples with correct prediction,
    2sided also allows that we start from samples with wrong orignal prediction.
    """
    counterfactual_type: str = "1sided"
    """
    Whether to always give feedback directly after creating validation counterfactuals or whether to wait until
    the next train feedback shall be given as well (which means less interruptions!)
    """
    lazy_feedback: bool = True
    """
    The path of the last finetuned model
    """
    model_path: str = ""
    """
    Dummy field to be able to use it as a model config!
    """
    is_loaded: bool = False
    """
    The performance of the generative model measured e.g. in FID score.
    """
    generator_performance: dict = {}
    """
    The restriction to interesting counterfactual transitions.
    Helpful in the case of datasets with a lot of classes and heavy modes like ImageNet.
    """
    transition_restrictions: Union[list, type(None)] = None
    """
    The clustering strategy used by the counterfactual explainer
    """
    clustering_strategy: Union[str, type(None)] = None


class CFKD(Adaptor):
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
        adaptor_config: Union[dict, str, Path, AdaptorConfig] = "<PEAL_BASE>/configs/adaptors/symbolic_cfkd.yaml",
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
                Defaults to "<PEAL_BASE>/configs/predictors/default_generator.yaml".
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
        self.adaptor_config = load_yaml_config(adaptor_config, AdaptorConfig)
        if self.adaptor_config.test_data is None:
            self.adaptor_config.test_data = self.adaptor_config.data

        self.adaptor_config.data.in_memory = self.adaptor_config.in_memory
        self.adaptor_config.test_data.in_memory = self.adaptor_config.in_memory
        '''assert (
            self.adaptor_config.finetune_iterations == 0 or self.adaptor_config.batch_size % 2 == 0
        ), "only even batch sizes are supported so far for CFKD finetuning!"'''
        self.adaptor_config.explainer.tracking_level = self.adaptor_config.tracking_level
        self.adaptor_config.explainer.transition_restrictions = self.adaptor_config.transition_restrictions
        if not self.adaptor_config.clustering_strategy is None:
            self.adaptor_config.explainer.clustering_strategy = self.adaptor_config.clustering_strategy

        self.base_dir = base_dir if not base_dir is None else self.adaptor_config.base_dir
        #
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if student is None:
            # student = torch.load(self.adaptor_config.student, map_location=self.device)
            student, student_config = get_predictor(self.adaptor_config.student, device=self.device)

        self.overwrite = overwrite if not overwrite is None else self.adaptor_config.overwrite
        self.adaptor_config.overwrite = False
        self.original_student = student
        if isinstance(student, torch.nn.Module):
            self.original_student.eval()
            self.student = copy.deepcopy(student)
            self.student.eval()

        else:
            self.student = student

        teacher = teacher if not teacher is None else self.adaptor_config.teacher

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
        self.joint_validation_dataloader = WeightedDataloaderList([self.val_dataloader])
        self.adaptor_config.data = self.train_dataloader.dataset.config

        #
        explainer_config = load_yaml_config(self.adaptor_config.explainer)
        if hasattr(explainer_config, "num_discretization_steps") and hasattr(
            explainer_config, "sampling_time_fraction"
        ):
            timestep_respacing = int(
                1.0 / explainer_config.sampling_time_fraction * explainer_config.num_discretization_steps
            )

        elif hasattr(explainer_config, "timestep_respacing") and not explainer_config.timestep_respacing is None:
            timestep_respacing = explainer_config.timestep_respacing

        else:
            timestep_respacing = None

        self.generator = get_generator(
            generator=(generator if not generator is None else self.adaptor_config.generator),
            device=self.device,
            predictor_dataset=self.val_dataloader.dataset,
            timestep_respacing=timestep_respacing,
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
                len(self.train_dataloader.dataset) // self.adaptor_config.training.train_batch_size
            )

        self.dataloader_mixer = DataloaderMixer(self.adaptor_config.training, self.train_dataloader)
        self.datastack = DataStack(
            self.dataloader_mixer,
            self.output_size,
            transform=self.val_dataloader.dataset.transform,
        )

        if teacher[:5] == "human":
            assert self.adaptor_config.tracking_level >= 4, "Tracking level too low!"

        self.explainer = CounterfactualExplainer(
            predictor=self.student,
            generator=self.generator,
            input_type=self.adaptor_config.data.input_type,
            explainer_config=self.adaptor_config.explainer,
            datasource=[self.dataloader_mixer, self.joint_validation_dataloader],
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
            "y_target_start_confidence_list",
        ]

        if self.adaptor_config.tracking_level >= 4:
            self.tracked_keys.extend(
                [
                    "z_difference_list",
                    "collage_path_list",
                ]
            )

        # teacher == "SegmentationMask" or self.adaptor_config.tracking_level > 0:
        if self.adaptor_config.data.has_hints:
            self.hints_enabled = True
            self.tracked_keys.append("hint_list")
            self.train_dataloader.dataset.enable_hints()
            self.val_dataloader.dataset.enable_hints()

        else:
            self.hints_enabled = False

        if isinstance(self.explainer.explainer_config, PerfectFalseCounterfactualConfig):
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
        self.data_config.data.x_selection = "imgs"
        self.data_config.data.num_samples = self.adaptor_config.min_train_samples
        self.data_config.data.dataset_class = None
        self.validation_data_config = copy.deepcopy(self.data_config)
        self.validation_data_config.data.x_selection = "imgs"
        self.validation_data_config.data.num_samples = self.adaptor_config.max_validation_samples
        self.validation_data_config.data.split = [0.0, 1.0]

    def initialize_run(self):
        cprint("initialize run!!!", self.adaptor_config.tracking_level, 2)
        if self.overwrite:
            # move from self.base_dir to self.base_dir + "_old_" + {date}_{timestamp}
            if os.path.exists(self.base_dir):
                shutil.move(
                    self.base_dir,
                    self.base_dir + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                )

        if not os.path.exists(os.path.join(self.base_dir, "0")):
            os.makedirs(os.path.join(self.base_dir, "0"))

        boundary_path = os.path.join(self.base_dir, "0", "decision_boundary.png")
        if (
            self.adaptor_config.tracking_level >= 4
            and not os.path.exists(boundary_path)
            and hasattr(
                self.joint_validation_dataloader.dataloaders[0].dataset,
                "visualize_decision_boundary",
            )
            and not os.path.exists(boundary_path)
        ):
            self.joint_validation_dataloader.dataloaders[0].dataset.visualize_decision_boundary(
                self.student,
                self.adaptor_config.training.test_batch_size,
                self.device,
                boundary_path,
                temperature=self.adaptor_config.explainer.temperature,
                train_dataloader=self.dataloader_mixer,
                val_dataloaders=self.joint_validation_dataloader,
            )

        log_dir = os.path.join(self.base_dir, "logs")
        if not os.path.exists(log_dir):
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir)

            hints_enabled_buffer = self.val_dataloader.dataset.hints_enabled
            if hints_enabled_buffer:
                self.val_dataloader.dataset.disable_hints()

            val_accuracy = calculate_test_accuracy(
                self.student,
                self.val_dataloader,
                self.device,
                False,
                self.adaptor_config.max_test_batches,
                tracking_level=self.adaptor_config.tracking_level,
            )
            cprint("val_accuracy: ", self.adaptor_config.tracking_level, 2)
            writer.add_scalar("val_accuracy", val_accuracy, self.adaptor_config.current_iteration)
            if hints_enabled_buffer:
                self.val_dataloader.dataset.enable_hints()

            test_accuracy = calculate_test_accuracy(
                self.student,
                self.test_dataloader,
                self.device,
                self.adaptor_config.calculate_group_accuracies,
                self.adaptor_config.max_test_batches,
                tracking_level=self.adaptor_config.tracking_level,
            )

            if self.adaptor_config.calculate_group_accuracies:
                (
                    test_accuracy,
                    group_accuracies,
                    group_distribution,
                    groups,
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
                cprint(
                    "group_accuracies: " + str(group_accuracies),
                    self.adaptor_config.tracking_level,
                    2,
                )
                cprint(
                    "group_distribution: " + str(group_distribution),
                    self.adaptor_config.tracking_level,
                    2,
                )
                cprint(
                    "group_numbers: " + str(groups),
                    self.adaptor_config.tracking_level,
                    2,
                )
                cprint(
                    "worst_group_accuracy: " + str(worst_group_accuracy),
                    self.adaptor_config.tracking_level,
                    2,
                )
                avg_group_accuracy = np.mean(group_accuracies)
                cprint(
                    "avg_group_accuracy: " + str(avg_group_accuracy),
                    self.adaptor_config.tracking_level,
                    2,
                )
                writer.add_scalar(
                    "test_avg_group_accuracy",
                    avg_group_accuracy,
                    self.adaptor_config.current_iteration,
                )

            writer.add_scalar("test_accuracy", test_accuracy, self.adaptor_config.current_iteration)
            cprint("log sample batches!", self.adaptor_config.tracking_level, 2)
            log_images_to_writer(self.train_dataloader, writer, "train0")
            log_images_to_writer(self.val_dataloader, writer, "validation0")
            log_images_to_writer(self.test_dataloader, writer, "test")
            cprint("log sample batches done!", self.adaptor_config.tracking_level, 2)

            if isinstance(self.val_dataloader.dataset, ImageDataset) and self.adaptor_config.tracking_level >= 4:
                cprint("visualizing sample!!!", self.adaptor_config.tracking_level, 2)
                generator_sample = self.generator.sample_x()
                if not generator_sample is None:
                    torchvision.utils.save_image(
                        generator_sample,
                        os.path.join(self.base_dir, "generator_sample.png"),
                        normalize=True,
                        nrow=int(np.sqrt(generator_sample.shape[0])),
                    )
                    cprint("sample visualized!", self.adaptor_config.tracking_level, 2)

                    # TODO move this back!!!
                    generator_performance = self.val_dataloader.dataset.track_generator_performance(generator_sample)
                    cprint(
                        "Generator performance: " + str(generator_performance),
                        self.adaptor_config.tracking_level,
                        2,
                    )
                    writer.add_scalar(
                        "generator_fid",
                        generator_performance["fid"],
                        self.adaptor_config.current_iteration,
                    )
                    self.adaptor_config.generator_performance = generator_performance

                else:
                    # TODO why was a pdb here?
                    cprint(
                        "log sample batches done!",
                        self.adaptor_config.tracking_level,
                        2,
                    )

            else:
                cprint("no visualization!!!", self.adaptor_config.tracking_level, 2)

        else:
            writer = SummaryWriter(log_dir)

        if not os.path.exists(os.path.join(self.base_dir, "0", "validation_tracked_values.npz")):
            assert self.adaptor_config.current_iteration == 0

            save_yaml_config(self.adaptor_config, os.path.join(self.base_dir, "config.yaml"))

            with open(os.path.join(self.base_dir, "platform.txt"), "w") as f:
                f.write(platform.node())

            cprint(
                "start generating validation stats!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            (
                validation_tracked_values,
                validation_stats,
            ) = self.retrieve_validation_prestats(finetune_iteration=0)
            for key in validation_stats.keys():
                if isinstance(validation_stats[key], float):
                    writer.add_scalar(
                        "validation_" + key,
                        validation_stats[key],
                        self.adaptor_config.current_iteration,
                    )

            cprint("validation stats generated!!!", self.adaptor_config.tracking_level, 2)

        else:
            with open(os.path.join(self.base_dir, "platform.txt"), "w") as f:
                f.write(platform.node())

            cprint(
                "start loading validation stats!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            validation_stats_existed = os.path.exists(
                os.path.join(
                    self.base_dir,
                    str(max(0, self.adaptor_config.current_iteration - 1)),
                    "validation_stats.npz",
                )
            )
            (
                validation_tracked_values,
                validation_prestats,
            ) = self.retrieve_validation_prestats(finetune_iteration=max(0, self.adaptor_config.current_iteration - 1))
            validation_stats = self.retrieve_validation_stats(
                finetune_iteration=self.adaptor_config.current_iteration,
                validation_tracked_values=validation_tracked_values,
                validation_prestats=validation_prestats,
            )
            if not validation_stats_existed:
                for key in validation_stats.keys():
                    if isinstance(validation_stats[key], float):
                        writer.add_scalar(
                            "validation_" + key,
                            validation_stats[key],
                            self.adaptor_config.current_iteration,
                        )

            cprint(
                "Create dataloader mixer and add counterfactual datasets!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            self.dataloader_mixer = DataloaderMixer(self.adaptor_config.training, self.train_dataloader)

            for i in range(1, self.adaptor_config.current_iteration + 1):
                dataset_dir = os.path.join(self.base_dir, str(i), "train_dataset")
                self.dataloader_mixer = self.add_dataset_to_dataloader_mixer(
                    dataloader_old=self.dataloader_mixer,
                    dataset_path=dataset_dir,
                    mixing_ratio=self.adaptor_config.mixing_ratio,
                    writer=writer,
                    finetune_iteration=i,
                )
                cprint(
                    "counterfactual dataset " + str(i) + " added!!!",
                    self.adaptor_config.tracking_level,
                    2,
                )

            self.datastack = DataStack(
                self.dataloader_mixer,
                self.output_size,
                transform=self.val_dataloader.dataset.transform,
            )

            if self.adaptor_config.current_iteration > 0:
                cprint(
                    "load already updated student model!!!",
                    self.adaptor_config.tracking_level,
                    2,
                )
                self.student = torch.load(
                    os.path.join(self.adaptor_config.base_dir, "model.cpl"),
                    map_location=self.device,
                )
                self.explainer.predictor = self.student

        visualization_path = os.path.join(self.base_dir, "visualization.png")
        if self.output_size == 2 and self.adaptor_config.tracking_level >= 6 and not os.path.exists(visualization_path):
            cprint("visualize progress!!!", self.adaptor_config.tracking_level, 2)
            self.visualize_progress([visualization_path])
            cprint("Visualization done!!!", self.adaptor_config.tracking_level, 2)

        cprint("initialization done!!!", self.adaptor_config.tracking_level, 2)
        return validation_stats, validation_tracked_values, writer

    def get_batch(
        self,
        error_matrix: torch.Tensor = None,
        cm_idx_in: int = 0,
    ):
        x_batch = []
        y_source_batch = []
        y_target_batch = []
        y_batch = []
        y_target_start_confidence_batch = []
        hint_batch = []
        idx_batch = []
        sample_idx = 0
        cm_idx = cm_idx_in
        print("cm_idx_in", cm_idx_in)
        print("cm_idx_in", cm_idx_in)
        print("cm_idx_in", cm_idx_in)
        print("cm_idx_in", cm_idx_in)
        print("cm_idx_in", cm_idx_in)
        torch.manual_seed(torch.seed())
        if self.adaptor_config.use_confusion_matrix:
            error_distribution = torch.distributions.Categorical(error_matrix)

        while not sample_idx >= self.adaptor_config.batch_size:
            if self.adaptor_config.use_confusion_matrix:
                cm_idx = error_distribution.sample()

            # TODO verify that this is actually balancing itself!
            y_source = int(cm_idx / self.output_size)
            y_target = int(cm_idx % self.output_size)
            x, y = self.datastack.pop(int(y_source))

            if self.hints_enabled:
                y_res = y[1:]
                y = y[0]
                if isinstance(self.teacher, SegmentationMaskTeacher):
                    hint = y_res[0]

                if isinstance(self.explainer.explainer_config, PerfectFalseCounterfactualConfig):
                    idx = y_res[-1]

            elif isinstance(self.explainer.explainer_config, PerfectFalseCounterfactualConfig):
                idx = y[-1]
                y = y[0]

            logits = self.student(x.to(self.device).unsqueeze(0)).squeeze(0).detach().cpu()
            y_target_start_confidence = torch.nn.Softmax()(logits / self.explainer.explainer_config.temperature)[
                y_target
            ]
            prediction = self.logits_to_prediction(logits)
            if not self.adaptor_config.counterfactual_type == "1sided" or prediction == y == y_source:
                x_batch.append(x)
                y_source_batch.append(y_source)
                y_target_batch.append(torch.tensor(y_target))
                y_batch.append(y)
                y_target_start_confidence_batch.append(y_target_start_confidence)
                if isinstance(self.teacher, SegmentationMaskTeacher):
                    hint_batch.append(hint)

                else:
                    hint_batch.append(torch.zeros_like(x))

                if isinstance(self.explainer.explainer_config, PerfectFalseCounterfactualConfig):
                    idx_batch.append(idx)

                else:
                    idx_batch.append(0)

                sample_idx += 1

            else:
                pass


            while y_source == y_target:
                cm_idx = (cm_idx + 1) % (self.output_size**2)
                y_source = int(cm_idx / self.output_size)
                y_target = int(cm_idx % self.output_size)

            cm_idx = (cm_idx + 1) % (self.output_size**2)

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
        cprint("generate x counterfactual list!", self.adaptor_config.tracking_level, 2)
        self.datastack.reset()

        collage_base_path = os.path.join(self.base_dir, str(finetune_iteration), "collages")
        if os.path.exists(collage_base_path):
            # move from self.base_dir to self.base_dir + "_old_" + {date}_{timestamp}
            shutil.move(
                collage_base_path,
                collage_base_path + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            )

        Path(collage_base_path).mkdir(parents=True, exist_ok=True)

        tracked_values = {key: [] for key in tracked_keys}

        continue_collecting = True
        acceptance_threshold = 0.51

        cprint(
            "Start generating x counterfactual list!!!",
            self.adaptor_config.tracking_level,
            2,
        )
        pbar = tqdm(
            total=int(self.adaptor_config.min_train_samples / self.adaptor_config.batch_size + 0.99)
            * (
                self.adaptor_config.explainer.gradient_steps
                if hasattr(self.adaptor_config.explainer, "gradient_steps")
                else 1
            )
        )
        pbar.stored_values = {}
        pbar.stored_values["n_total"] = 0
        remaining_sample_number = self.adaptor_config.min_train_samples
        while continue_collecting:
            num_batches_per_iteration = int(1 + remaining_sample_number / self.adaptor_config.batch_size)
            if len(list(tracked_values.values())[0]) >= self.adaptor_config.min_train_samples:
                break

            for i in range(num_batches_per_iteration):
                batch = self.get_batch(error_matrix, cm_idx_in=i % 2)
                values = self.explainer.explain_batch(
                    batch=batch,
                    base_path=collage_base_path,
                    start_idx=len(list(tracked_values.values())[0]),
                    pbar=pbar,
                    mode="Training",
                    explainer_path=os.path.join(self.base_dir, str(finetune_iteration - 1)),
                )
                for key in tracked_keys:
                    tracked_values[key].extend(values[key])

                pbar.stored_values["n_valid"] = (
                    str(len(list(tracked_values.values())[0])) + "/" + str(self.adaptor_config.min_train_samples)
                )
                pbar.stored_values["th"] = acceptance_threshold
                pbar.stored_values["n_total"] += self.adaptor_config.batch_size
                pbar.stored_values["fr"] = len(list(tracked_values.values())[0]) / pbar.stored_values["n_total"]
                remaining_sample_number = self.adaptor_config.min_train_samples - len(list(tracked_values.values())[0])

                if remaining_sample_number <= 0:
                    break

            else:
                continue_collecting = False

        cprint("x counterfactual list generated!!!", self.adaptor_config.tracking_level, 2)
        pbar.close()
        return tracked_values

    def retrieve_counterfactual_list(self, validation_stats, finetune_iteration):
        tracked_values_path = os.path.join(self.base_dir, str(finetune_iteration), "tracked_values.npz")
        if self.overwrite or not os.path.exists(tracked_values_path):
            cprint(
                "Start generating tracked values!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            tracked_values = self.generate_x_counterfactual_list(
                error_matrix=validation_stats["error_matrix"],
                confidence_score_stats=validation_stats["confidence_score_stats"],
                finetune_iteration=finetune_iteration,
                tracked_keys=self.tracked_keys,
            )

            if self.adaptor_config.explainer.use_clustering and not hasattr(tracked_values, "cluster0"):
                tracked_values = self.explainer.cluster_explanations(
                    tracked_values,
                    self.adaptor_config.batch_size,
                    self.adaptor_config.explainer.num_attempts * self.adaptor_config.parallel_attempts,
                )

            if len(list(tracked_values.values())[0]) == 0:
                return tracked_values

            if self.adaptor_config.tracking_level >= 3:
                with open(
                    tracked_values_path,
                    "wb",
                ) as f:
                    tracked_values_file = {}
                    for key in tracked_values.keys():
                        if isinstance(tracked_values[key][0], torch.Tensor):
                            tracked_values_file[key] = torch.stack(tracked_values[key], dim=0).numpy()

                        elif isinstance(tracked_values[key][0], int) or isinstance(tracked_values[key][0], float):
                            tracked_values_file[key] = np.array(tracked_values[key])

                    np.savez(f, **tracked_values_file)

        else:
            with open(
                tracked_values_path,
                "rb",
            ) as f:
                cprint("Load tracked values!!!", self.adaptor_config.tracking_level, 2)
                tracked_values = {}
                tracked_values_file = np.load(f, allow_pickle=True)
                for key in tracked_values_file.keys():
                    tracked_values[key] = list(torch.tensor(tracked_values_file[key]))

        cprint("Create collage path list!!!", self.adaptor_config.tracking_level, 2)
        collage_path_list = os.listdir(os.path.join(self.base_dir, str(finetune_iteration), "collages"))
        collage_path_list.sort()
        collage_path_list = list(filter(lambda x: x[-4:] == ".png", collage_path_list))
        tracked_values["collage_path_list"] = list(
            map(
                lambda x: os.path.join(self.base_dir, str(finetune_iteration), "collages", x),
                collage_path_list,
            )
        )

        return tracked_values

    def retrieve_feedback(self, tracked_values, finetune_iteration, mode):
        if self.overwrite or not os.path.exists(
            os.path.join(self.base_dir, str(finetune_iteration), mode + "_feedback.txt")
        ):
            cprint("retrieve feedback!", self.adaptor_config.tracking_level, 2)
            feedback = self.teacher.get_feedback(
                base_dir=os.path.join(self.base_dir, str(finetune_iteration), mode + "_teacher"),
                student=self.student,
                num_clusters=self.adaptor_config.explainer.num_attempts,
                mode=mode,
                **tracked_values,
            )

            os.makedirs(os.path.join(self.base_dir, str(finetune_iteration)), exist_ok=True)
            with open(
                os.path.join(self.base_dir, str(finetune_iteration), mode + "_feedback.txt"),
                "w",
            ) as f:
                f.write("\n".join(feedback))

        else:
            cprint("load feedback!", self.adaptor_config.tracking_level, 2)
            with open(
                os.path.join(self.base_dir, str(finetune_iteration), mode + "_feedback.txt"),
                "r",
            ) as f:
                feedback = f.read().split("\n")

        return feedback

    def calculate_feedback_stats(self, tracked_values, feedback, finetune_iteration):
        num_samples = len(tracked_values["y_list"])

        # TODO this seems like a bug!
        num_samples = int(len(feedback) / self.adaptor_config.explainer.num_attempts)
        feedback = feedback[:num_samples]
        # this is not always a perfect match, because some explanations might be filtered out
        if finetune_iteration == 0:
            flip_rate_reference = max(num_samples, self.adaptor_config.max_validation_samples)

        else:
            flip_rate_reference = num_samples

        flip_rate = (
            len(
                list(
                    filter(
                        lambda x: x >= 0.51,
                        tracked_values["y_target_end_confidence_list"][:num_samples],
                    )
                )
            )
            / flip_rate_reference
        )
        ood_rate = len(list(filter(lambda sample: sample == "ood", feedback))) / num_samples

        num_true_1sided = len(
            list(
                filter(
                    lambda x: x[1] == "true"
                    and tracked_values["y_list"][x[0]] == tracked_values["y_source_list"][x[0]],
                    enumerate(feedback),
                )
            )
        )
        num_false_1sided = len(
            list(
                filter(
                    lambda x: x[1] == "false"
                    and tracked_values["y_list"][x[0]] == tracked_values["y_source_list"][x[0]],
                    enumerate(feedback),
                )
            )
        )
        if num_true_1sided + num_false_1sided > 0:
            fa_1sided = num_true_1sided / (num_true_1sided + num_false_1sided)

        else:
            fa_1sided = -1

        feedback_stats = {
            "flip_rate": flip_rate,
            "ood_rate": ood_rate,
            "feedback_accuracy": fa_1sided,
        }
        cprint("flip_rate: " + str(flip_rate), self.adaptor_config.tracking_level, 2)

        if self.adaptor_config.calculate_explainer_stats:
            # this is only for scientific experiments and could also be sourced out into another file!
            # distill into equivalent model
            predictor_distillation = load_yaml_config(
                "<PEAL_BASE>/configs/predictors/simple_distillation.yaml",
                PredictorConfig,
            )
            distillation_path = os.path.join(self.base_dir, str(finetune_iteration), "distilled_predictor")
            distilled_predictor_final = os.path.join(distillation_path, "distilled_predictor", "model.cpl")
            if not os.path.exists(distilled_predictor_final):
                distilled_predictor = distill_predictor(
                    predictor_distillation,
                    distillation_path,
                    self.student,
                    [self.train_dataloader.dataset, self.val_dataloader.dataset],
                    replace_with_activation="leakysoftplus",
                    tracking_level=self.adaptor_config.tracking_level,
                )

            else:
                distilled_predictor = torch.load(distilled_predictor_final, map_location=self.device)

            # add y_target_end_confidence_distilled_list
            tracked_values["y_target_end_confidence_distilled_list"] = []
            for idx in range(len(tracked_values["x_counterfactual_list"])):
                x = tracked_values["x_counterfactual_list"][idx]
                y = tracked_values["y_target_list"][idx]
                y_target_end_confidence = (
                    distilled_predictor(x.to(self.device).unsqueeze(0)).squeeze(0).detach().cpu()[y]
                )
                tracked_values["y_target_end_confidence_distilled_list"].append(y_target_end_confidence)

            # calculate distilled flip rate
            flipped_samples = list(
                filter(
                    lambda x: x > 0.5,
                    tracked_values["y_target_end_confidence_distilled_list"][:num_samples],
                )
            )
            flip_rate_distilled = len(flipped_samples) / flip_rate_reference
            feedback_stats["flip_rate_distilled"] = float(flip_rate_distilled)
            cprint(
                "flip_rate_distilled: " + str(flip_rate_distilled),
                self.adaptor_config.tracking_level,
                2,
            )
            num_true_1sided_distilled = len(
                list(
                    filter(
                        lambda idx: feedback[idx] == "true"
                        and tracked_values["y_target_end_confidence_distilled_list"][idx] > 0.5,
                        range(num_samples),
                    )
                )
            )
            num_false_1sided_distilled = len(
                list(
                    filter(
                        lambda idx: feedback[idx] == "false"
                        and tracked_values["y_target_end_confidence_distilled_list"][idx] > 0.5,
                        range(num_samples),
                    )
                )
            )
            if num_true_1sided_distilled + num_false_1sided_distilled > 0:
                fa_1sided_distilled = num_true_1sided_distilled / (
                    num_true_1sided_distilled + num_false_1sided_distilled
                )

            else:
                fa_1sided_distilled = 0.0

            feedback_stats["feedback_accuracy_distilled"] = float(fa_1sided_distilled)
            cprint(
                "feedback_accuracy_distilled: " + str(fa_1sided_distilled),
                self.adaptor_config.tracking_level,
                2,
            )
            tracked_stats = self.explainer.calculate_latent_difference_stats(tracked_values)
            for key in tracked_stats.keys():
                feedback_stats[key] = tracked_stats[key]

        return feedback_stats

    def create_dataset(
        self,
        x_counterfactual_list,
        feedback,
        y_source_list,
        y_target_list,
        finetune_iteration,
        hint_list=None,
        mode="",
        **kwargs,
    ):
        if not (len(x_counterfactual_list) == len(feedback) == len(y_source_list) == len(y_target_list)):
            print("missmatch in list lengths while dataset creation!")
            if self.adaptor_config.tracking_level >= 5:
                import pdb

                pdb.set_trace()

            else:
                raise Exception("missmatch in list lengths while dataset creation!")

        dataset_dir = os.path.join(self.base_dir, str(finetune_iteration), mode + "_dataset")
        if os.path.exists(dataset_dir):
            return dataset_dir
        #
        x_list = []
        hint_list_dataset = []
        y_counterfactual_list = []
        sample_names = []
        for sample_idx in range(len(feedback)):
            """if feedback[sample_idx] == "true":
            sample_name = (
                "true_"
                + str(int(y_source_list[sample_idx]))
                + "_to_"
                + str(int(y_target_list[sample_idx]))
                + "_"
                + str(sample_idx)
            )
            x_list.append(x_counterfactual_list[sample_idx])
            if not hint_list is None:
                hint_list_dataset.append(hint_list[sample_idx])

            y_counterfactual_list.append(int(y_target_list[sample_idx]))
            sample_names.append(sample_name)
            """

            if feedback[sample_idx] == "false":
                sample_name = (
                    "false_"
                    + str(int(y_source_list[sample_idx]))
                    + "_to_"
                    + str(int(y_target_list[sample_idx]))
                    + "_"
                    + str(sample_idx)
                )
                x_list.append(x_counterfactual_list[sample_idx])
                if not hint_list is None:
                    hint_list_dataset.append(hint_list[sample_idx])

                y_counterfactual_list.append(int(y_source_list[sample_idx]))

                sample_names.append(sample_name)

        self.train_dataloader.dataset.serialize_dataset(
            output_dir=dataset_dir,
            x_list=x_list,
            y_list=y_counterfactual_list,
            hint_list=hint_list_dataset,
            sample_names=sample_names,
            classifier=self.student,
        )
        return dataset_dir

    def add_dataset_to_dataloader_mixer(self, dataloader_old, dataset_path, mixing_ratio, writer, finetune_iteration):
        # TODO adapt batch size so that it matches!
        dataloader, _, _ = create_dataloaders_from_datasource(
            config=self.data_config,
            datasource=dataset_path,
        )
        # import pdb; pdb.set_trace()
        log_images_to_writer(dataloader, writer, "train_" + str(finetune_iteration))
        dataloader = DataloaderMixer(self.adaptor_config.training, dataloader)
        # mixing ratio has to be flipped because in fact the old dataloader is the one appended
        dataloader.append(dataloader_old, weight_added_dataloader=1 - mixing_ratio)
        dataloader.return_src_internal = True
        if self.hints_enabled:
            dataloader.enable_hints()

        return dataloader

    def finetune_student(self, finetune_iteration, dataset_path, writer):
        #
        val_dataset_path = os.path.join(self.base_dir, str(finetune_iteration), "validation_dataset")
        _, dataloader_val, _ = create_dataloaders_from_datasource(
            config=self.validation_data_config,
            datasource=val_dataset_path,
        )
        if (
            not isinstance(dataloader_val, torch.utils.data.DataLoader)
            or len(dataloader_val.dataset) < 2 * self.adaptor_config.training.val_batch_size
        ):
            open(
                os.path.join(
                    self.adaptor_config.base_dir,
                    "error_iteration_" + str(finetune_iteration) + ".txt",
                ),
                "w",
            ).write("dataloader_val in " + str(finetune_iteration) + " is too empty!")
            return

        self.joint_validation_dataloader.append(dataloader_val)
        log_images_to_writer(dataloader_val, writer, "validation_" + str(finetune_iteration))

        #
        if not hasattr(self, "dataloader_mixer"):
            self.dataloader_mixer = DataloaderMixer(self.adaptor_config.training, self.train_dataloader)

        self.dataloader_mixer = self.add_dataset_to_dataloader_mixer(
            dataloader_old=self.dataloader_mixer,
            dataset_path=dataset_path,
            mixing_ratio=self.adaptor_config.mixing_ratio,
            writer=writer,
            finetune_iteration=finetune_iteration,
        )
        self.datastack = DataStack(
            self.dataloader_mixer,
            self.output_size,
            transform=self.val_dataloader.dataset.transform,
        )

        if self.overwrite or not os.path.exists(
            os.path.join(
                self.base_dir,
                str(finetune_iteration),
                "finetuned_model",
                "model.cpl",
            )
        ):
            if os.path.exists(os.path.join(self.base_dir, str(finetune_iteration), "finetuned_model")):
                shutil.move(
                    os.path.join(self.base_dir, str(finetune_iteration), "finetuned_model"),
                    os.path.join(
                        self.base_dir,
                        str(finetune_iteration),
                        "finetuned_model_old_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                    ),
                )

            Path(os.path.join(self.base_dir, str(finetune_iteration), "finetuned_model")).mkdir(
                parents=True, exist_ok=True
            )

            if self.adaptor_config.continuous_learning == "retrain":
                # TODO this should be changed!
                self.student = TorchvisionModel("resnet18", 2)

            finetune_trainer = ModelTrainer(
                config=copy.deepcopy(self.adaptor_config),
                model=self.student,
                datasource=(self.dataloader_mixer, self.joint_validation_dataloader),
                model_path=os.path.join(self.base_dir, str(finetune_iteration), "finetuned_model"),
                only_last_layer=self.adaptor_config.continuous_learning == "deep_feature_reweighting",
            )
            if self.hints_enabled:
                self.dataloader_mixer.disable_hints()
                for val_dataloader in self.joint_validation_dataloader.dataloaders:
                    val_dataloader.dataset.disable_hints()

            if isinstance(self.explainer.explainer_config, PerfectFalseCounterfactualConfig):
                self.dataloader_mixer.disable_idx()
                for val_dataloader in self.joint_validation_dataloader.dataloaders:
                    val_dataloader.dataset.disable_idx()

            finetune_trainer.fit(continue_training=True)  # bool(self.adaptor_config.continuous_learning != "retrain"))

            if self.hints_enabled:
                self.dataloader_mixer.enable_hints()
                for val_dataloader in self.joint_validation_dataloader.dataloaders:
                    val_dataloader.dataset.enable_hints()

            if isinstance(self.explainer.explainer_config, PerfectFalseCounterfactualConfig):
                self.dataloader_mixer.enable_idx()
                for val_dataloader in self.joint_validation_dataloader.dataloaders:
                    val_dataloader.dataset.enable_idx()

        self.student = torch.load(
            os.path.join(
                self.base_dir,
                str(finetune_iteration),
                "finetuned_model",
                "model.cpl",
            ),
            map_location=self.device,
        )
        self.explainer.predictor = self.student
        self.explainer.predictor_datasources = [
            self.dataloader_mixer,
            self.joint_validation_dataloader,
        ]

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
                y[self.test_dataloader.dataset.attributes.index(task_config_buffer.y_selection[0])]
            )
            criterions["confounder"] = lambda X, y: int(y[self.test_dataloader.dataset.attributes.index("Confounder")])
            criterions["uncorrected"] = lambda X, y: int(
                self.original_student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
            )
            criterions["cfkd"] = lambda X, y: int(
                self.student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
            )

        else:
            criterions["class"] = lambda X, y: int(y)
            criterions["uncorrected"] = lambda X, y: int(
                self.original_student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
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
        self.explainer.tracking_level = 0.5
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
            cprint(
                "Saved: " + path.replace(".png", "_success.png"),
                self.adaptor_config.tracking_level,
                2,
            )

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
            cprint("Saved: " + path, self.adaptor_config.tracking_level, 2)

        self.test_dataloader.dataset.task_config = task_config_buffer
        return img

    def retrieve_validation_prestats(self, finetune_iteration):
        validation_values_path = os.path.join(self.base_dir, str(finetune_iteration), "validation_tracked_values.npz")
        if self.overwrite or not os.path.exists(validation_values_path):
            cprint(
                "calculate validation tracked values from scratch!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            x_list_collection = []
            x_counterfactual_collection = []
            y_confidence_list = []
            original_explainer_config = copy.deepcopy(self.explainer.explainer_config)
            validation_tracked_values = None
            validation_stats = []
            for i in range(self.adaptor_config.validation_runs):
                cprint("Validation run: " + str(i), self.adaptor_config.tracking_level, 2)
                self.explainer.explainer_config = copy.deepcopy(original_explainer_config)
                if self.adaptor_config.validation_runs > 1:
                    for attribute in self.explainer.explainer_config.__dict__.items():
                        if isinstance(attribute[1], list) and len(attribute[1]) == 2:
                            if i == 0:
                                effective_idx = 0

                            else:
                                effective_idx = (
                                    i / (self.adaptor_config.validation_runs - 1) * (attribute[1][1] - attribute[1][0])
                                )

                            setattr(
                                self.explainer.explainer_config,
                                attribute[0],
                                attribute[1][0] + effective_idx,
                            )

                validation_collages_base_path = os.path.join(
                    self.base_dir,
                    str(finetune_iteration),
                    "validation_collages" + str(i),
                )
                (
                    validation_tracked_values_current,
                    validation_stats_current,
                ) = calculate_validation_statistics(
                    model=self.student,
                    dataloaders=self.joint_validation_dataloader.dataloaders,
                    tracked_keys=self.tracked_keys,
                    base_path=validation_collages_base_path,
                    output_size=self.output_size,
                    explainer=self.explainer,
                    device=self.device,
                    logits_to_prediction=self.logits_to_prediction,
                    use_confusion_matrix=self.adaptor_config.use_confusion_matrix,
                    max_validation_samples=self.adaptor_config.max_validation_samples,
                )
                # torch.nn.functional.softmax(
                # self.student(validation_tracked_values_current['x_counterfactual_list'][i]
                # .unsqueeze(0).to('cuda')).squeeze(0))[validation_tracked_values_current['y_target_list'][i]]
                if validation_tracked_values is None:
                    validation_tracked_values = validation_tracked_values_current

                else:
                    for key in validation_tracked_values.keys():
                        validation_tracked_values[key].extend(validation_tracked_values_current[key])

                validation_stats.append(validation_stats_current)

                if self.adaptor_config.validation_runs > 1:
                    x_list_collection.append(copy.deepcopy(validation_tracked_values_current["x_list"]))
                    x_counterfactual_collection.append(
                        copy.deepcopy(validation_tracked_values_current["x_counterfactual_list"])
                    )
                    y_confidence_list.append(
                        copy.deepcopy(validation_tracked_values_current["y_target_end_confidence_list"])
                    )

            validation_stats = {
                key: torch.mean(
                    torch.stack(
                        [torch.tensor(validation_stats_current[key]) for validation_stats_current in validation_stats]
                    ),
                    dim=0,
                )
                for key in validation_stats[0].keys()
            }
            self.explainer.explainer_config = original_explainer_config

            if self.adaptor_config.tracking_level >= 3:
                os.makedirs(os.path.join(self.base_dir, str(finetune_iteration)), exist_ok=True)
                with open(
                    validation_values_path,
                    "wb",
                ) as f:
                    tracked_values_file = {}
                    for key in self.tracked_keys:
                        if isinstance(validation_tracked_values[key][0], torch.Tensor):
                            tracked_values_file[key] = torch.stack(validation_tracked_values[key], dim=0).numpy()

                        elif isinstance(validation_tracked_values[key][0], int) or isinstance(
                            validation_tracked_values[key][0], float
                        ):
                            tracked_values_file[key] = np.array(validation_tracked_values[key])

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

                        elif isinstance(validation_stats[key], int) or isinstance(validation_stats[key], float):
                            validation_stats_file[key] = np.array(validation_stats[key])

                    np.savez(f, **validation_stats_file)

        else:
            # TODO think about this again
            if self.adaptor_config.tracking_level > 0:
                cprint(
                    "load validation tracked values!!!",
                    self.adaptor_config.tracking_level,
                    2,
                )
                with open(
                    validation_values_path,
                    "rb",
                ) as f:
                    validation_tracked_values = {}
                    validation_tracked_value_file = np.load(f, allow_pickle=True)
                    for key in validation_tracked_value_file.keys():
                        validation_tracked_values[key] = list(torch.tensor(validation_tracked_value_file[key]))

                cprint("load validation prestats!!!", self.adaptor_config.tracking_level, 2)
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
                        validation_stats[key] = torch.tensor(validation_tracked_file[key])

            if "collage_path_list" in self.tracked_keys:
                cprint(
                    "recreate validation collage path!",
                    self.adaptor_config.tracking_level,
                    2,
                )
                get_collage_path = lambda x: os.path.join(
                    self.base_dir,
                    str(finetune_iteration),
                    "validation_collages" + str(x),
                )
                idx = 0
                collage_path_list = []
                while os.path.exists(get_collage_path(idx)):
                    collage_paths = os.listdir(get_collage_path(idx))
                    collage_paths.sort()
                    collage_paths = list(filter(lambda x: x[-4:] == ".png", collage_paths))
                    collage_path_list.extend(collage_paths)
                    idx += 1
                    # TODO this is a bug, but currently not used
                    if idx == 1:
                        break

                validation_tracked_values["collage_path_list"] = list(
                    map(
                        lambda x: os.path.join(
                            self.base_dir,
                            str(finetune_iteration),
                            "validation_collages0",
                            x,
                        ),
                        collage_path_list,
                    )
                )

        if self.adaptor_config.explainer.use_clustering:
            validation_cluster_values_path = os.path.join(
                self.base_dir,
                str(finetune_iteration),
                "validation_tracked_cluster_values.npz",
            )
            """
            TODO loading collage paths does not work yet...
            if os.path.exists(validation_cluster_values_path):
                cprint(
                    "load clustered counterfactual explanations!",
                    self.adaptor_config.tracking_level,
                    2,
                )
                with open(
                    validation_cluster_values_path,
                    "rb",
                ) as f:
                    validation_tracked_values = {}
                    validation_tracked_value_file = np.load(f, allow_pickle=True)
                    for key in validation_tracked_value_file.keys():
                        validation_tracked_values[key] = list(
                            torch.tensor(validation_tracked_value_file[key])
                        )

            else:
            """
            cprint(
                "cluster counterfactual explanations!",
                self.adaptor_config.tracking_level,
                2,
            )
            validation_tracked_values = self.explainer.cluster_explanations(
                validation_tracked_values,
                self.adaptor_config.batch_size,
                self.adaptor_config.explainer.num_attempts,
            )
            if self.adaptor_config.tracking_level >= 3:
                with open(
                    validation_cluster_values_path,
                    "wb",
                ) as f:
                    tracked_values_file = {}
                    for key in validation_tracked_values.keys():
                        if isinstance(validation_tracked_values[key][0], torch.Tensor):
                            tracked_values_file[key] = torch.stack(validation_tracked_values[key], dim=0).numpy()

                        elif isinstance(validation_tracked_values[key][0], int) or isinstance(
                            validation_tracked_values[key][0], float
                        ):
                            tracked_values_file[key] = np.array(validation_tracked_values[key])

                    np.savez(f, **tracked_values_file)

        if self.adaptor_config.tracking_level >= 4 and hasattr(
            self.joint_validation_dataloader.dataloaders[0].dataset,
            "global_counterfactual_visualization",
        ):
            self.joint_validation_dataloader.dataloaders[0].dataset.global_counterfactual_visualization(
                os.path.join(
                    self.base_dir,
                    str(finetune_iteration),
                    "val_counterfactuals_global.png",
                ),
                validation_tracked_values["x_list"],
                validation_tracked_values["x_counterfactual_list"],
                validation_tracked_values["y_target_start_confidence_list"],
                validation_tracked_values["y_target_end_confidence_list"],
                validation_tracked_values["y_target_list"],
                validation_tracked_values["hint_list"],
            )
            cprint(
                "global counterfactual visualization saved!!!",
                self.adaptor_config.tracking_level,
                2,
            )

        return validation_tracked_values, validation_stats

    def retrieve_validation_stats(self, finetune_iteration, validation_tracked_values, validation_prestats):
        if not self.overwrite and os.path.exists(
            os.path.join(self.base_dir, str(finetune_iteration), "validation_stats.npz")
        ):
            cprint(
                "load already completed validation stats!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            with open(
                os.path.join(self.base_dir, str(finetune_iteration), "validation_stats.npz"),
                "rb",
            ) as f:
                validation_stats = {}
                validation_tracked_file = np.load(f, allow_pickle=True)
                for key in validation_tracked_file.keys():
                    validation_stats[key] = torch.tensor(validation_tracked_file[key])

            cprint("validation stats loaded!!!", self.adaptor_config.tracking_level, 2)
            return validation_stats

        validation_stats = validation_prestats

        validation_feedback = self.retrieve_feedback(
            tracked_values=validation_tracked_values,
            finetune_iteration=finetune_iteration,
            mode="validation",
        )

        validation_feedback_stats = self.calculate_feedback_stats(
            tracked_values=validation_tracked_values,
            feedback=validation_feedback,
            finetune_iteration=finetune_iteration,
        )

        self.create_dataset(
            feedback=validation_feedback,
            finetune_iteration=finetune_iteration + 1,
            mode="validation",
            config=self.validation_data_config,
            **validation_tracked_values,
        )

        for key in validation_feedback_stats.keys():
            validation_stats[key] = validation_feedback_stats[key]

        if self.adaptor_config.tracking_level >= 3:
            with open(
                os.path.join(self.base_dir, str(finetune_iteration), "validation_stats.npz"),
                "wb",
            ) as f:
                validation_stats_file = {}
                for key in validation_stats.keys():
                    if isinstance(validation_stats[key], torch.Tensor):
                        validation_stats_file[key] = validation_stats[key].numpy()

                    elif isinstance(validation_stats[key], int) or isinstance(validation_stats[key], float):
                        validation_stats_file[key] = np.array(validation_stats[key])

                np.savez(f, **validation_stats_file)

        return validation_stats

    def run(self):
        """
        Run the counterfactual knowledge distillation
        """
        cprint(
            "Adaptor Config: " + str(self.adaptor_config),
            self.adaptor_config.tracking_level,
            4,
        )
        validation_prestats, validation_tracked_values, writer = self.initialize_run()

        # iterate over the finetune iterations
        for finetune_iteration in range(
            self.adaptor_config.current_iteration + 1,
            self.adaptor_config.finetune_iterations + 1,
        ):
            cprint(
                "Start retrieving training counterfactuals for iteration " + str(finetune_iteration),
                self.adaptor_config.tracking_level,
                2,
            )
            tracked_values = self.retrieve_counterfactual_list(
                validation_stats=validation_prestats,
                finetune_iteration=finetune_iteration,
            )

            feedback = self.retrieve_feedback(
                tracked_values=tracked_values,
                finetune_iteration=finetune_iteration,
                mode="train",
            )
            validation_stats = self.retrieve_validation_stats(
                finetune_iteration=finetune_iteration - 1,
                validation_prestats=validation_prestats,
                validation_tracked_values=validation_tracked_values,
            )
            for key in validation_stats.keys():
                if isinstance(validation_stats[key], float):
                    writer.add_scalar(
                        "validation_" + key,
                        validation_stats[key],
                        finetune_iteration,
                    )

            self.adaptor_config.feedback_accuracies.append(validation_stats["feedback_accuracy"])

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

            hints_enabled_buffer = self.val_dataloader.dataset.hints_enabled
            if hints_enabled_buffer:
                self.val_dataloader.dataset.disable_hints()

            val_accuracy = calculate_test_accuracy(
                self.student,
                self.val_dataloader,
                self.device,
                False,
                self.adaptor_config.max_test_batches,
                tracking_level=self.adaptor_config.tracking_level,
            )
            cprint(
                "val_accuracy: " + str(val_accuracy),
                self.adaptor_config.tracking_level,
                2,
            )
            writer.add_scalar("val_accuracy", val_accuracy, finetune_iteration)
            if hints_enabled_buffer:
                self.val_dataloader.dataset.enable_hints()

            test_accuracy = calculate_test_accuracy(
                self.student,
                self.test_dataloader,
                self.device,
                self.adaptor_config.calculate_group_accuracies,
                self.adaptor_config.max_test_batches,
                tracking_level=self.adaptor_config.tracking_level,
            )
            if self.adaptor_config.calculate_group_accuracies:
                (
                    test_accuracy,
                    group_accuracies,
                    group_distribution,
                    groups,
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
                cprint(
                    "group_accuracies: " + str(group_accuracies),
                    self.adaptor_config.tracking_level,
                    2,
                )
                cprint(
                    "group_distribution: " + str(group_distribution),
                    self.adaptor_config.tracking_level,
                    2,
                )
                cprint("group_sizes: " + str(groups), self.adaptor_config.tracking_level, 2)
                cprint(
                    "worst_group_accuracy: " + str(worst_group_accuracy),
                    self.adaptor_config.tracking_level,
                    2,
                )
                avg_group_accuracy = np.mean(group_accuracies)
                cprint(
                    "avg_group_accuracy: " + str(avg_group_accuracy),
                    self.adaptor_config.tracking_level,
                    2,
                )
                writer.add_scalar("test_avg_group_accuracy", avg_group_accuracy, finetune_iteration)

            writer.add_scalar("test_accuracy", test_accuracy, finetune_iteration)
            cprint(
                "test_accuracy: " + str(test_accuracy),
                self.adaptor_config.tracking_level,
                2,
            )
            cprint(
                "Start to retrieve validation stats",
                self.adaptor_config.tracking_level,
                2,
            )

            decision_boundary_path = os.path.join(self.base_dir, str(finetune_iteration), "decision_boundary.png")
            if (
                hasattr(
                    self.joint_validation_dataloader.dataloaders[0].dataset,
                    "visualize_decision_boundary",
                )
                and not os.path.exists(decision_boundary_path)
                and self.adaptor_config.tracking_level >= 4
            ):
                self.joint_validation_dataloader.dataloaders[0].dataset.visualize_decision_boundary(
                    self.student,
                    self.adaptor_config.training.test_batch_size,
                    self.device,
                    decision_boundary_path,
                    temperature=self.adaptor_config.explainer.temperature,
                    train_dataloader=self.dataloader_mixer,
                    val_dataloaders=self.joint_validation_dataloader,
                )

            (
                validation_tracked_values,
                validation_prestats,
            ) = self.retrieve_validation_prestats(finetune_iteration=finetune_iteration)

            visualization_path = os.path.join(self.base_dir, str(finetune_iteration), "visualization.png")
            if (
                self.output_size == 2
                and self.adaptor_config.tracking_level >= 6
                and not os.path.exists(visualization_path)
            ):
                self.visualize_progress(
                    [
                        visualization_path,
                        os.path.join(self.base_dir, "visualization.png"),
                    ]
                )

            torch.save(self.student, os.path.join(self.base_dir, "model.cpl"))

            self.adaptor_config.current_iteration = self.adaptor_config.current_iteration + 1
            save_yaml_config(self.adaptor_config, os.path.join(self.base_dir, "config.yaml"))

        validation_stats = self.retrieve_validation_stats(
            finetune_iteration=self.adaptor_config.current_iteration,
            validation_prestats=validation_prestats,
            validation_tracked_values=validation_tracked_values,
        )
        for key in validation_stats.keys():
            if isinstance(validation_stats[key], float):
                writer.add_scalar(
                    "validation_" + key,
                    validation_stats[key],
                    self.adaptor_config.current_iteration,
                )

        self.adaptor_config.feedback_accuracies.append(validation_stats["feedback_accuracy"])

        return self.student
