import os
import torch
import copy
import yaml
import shutil
import numpy as np
import platform
import sys

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Union
from torch import nn

from peal.utils import load_yaml_config, set_adaptive_batch_size
from peal.data.dataloaders import (
    DataStack,
    DataloaderMixer,
    create_dataloaders_from_datasource,
)
from peal.training.trainers import ModelTrainer, calculate_test_accuracy
from peal.explainers.counterfactual_explainer import CounterfactualExplainer
from peal.data.dataset_interfaces import PealDataset
from peal.visualization.model_comparison import create_comparison
from peal.teachers.segmentation_mask_teacher import SegmentationMaskTeacher
from peal.teachers.teacher_factory import get_teacher
from peal.teachers.teacher_interface import TeacherInterface
from peal.generators.interfaces import InvertibleGenerator
from peal.generators.generator_factory import get_generator
from peal.adaptors.adaptor_utils import integrate_data_config_into_adaptor_config
from peal.training.training_utils import (
    calculate_validation_statistics,
)


class CounterfactualKnowledgeDistillation:
    """
    This class implements the counterfactual knowledge distillation approach.
    """

    def __init__(
        self,
        student: nn.Module,
        datasource: Union[list, tuple],
        output_size: int = None,
        generator: Union[
            InvertibleGenerator, Path, str
        ] = "$PEAL/configs/models/default_generator.yaml",
        base_dir: Union[str, Path] = os.path.join(
            "peal_runs", "counterfactual_knowledge_distillation"
        ),
        teacher: Union[str, TeacherInterface] = "human@8000",
        adaptor_config: Union[
            dict, str, Path
        ] = "$PEAL/configs/adaptors/counterfactual_knowledge_distillation_default.yaml",
        gigabyte_vram: float = None,
        overwrite: bool = False,
        visualization: callable = lambda x: x,
    ):
        """
        This is the constructor for the CounterfactualKnowledgeDistillation class.

        Args:
            student (nn.Module): The student model that is improved with CFKD
            datasource (Union): The datasource that is used for training
            output_size (int, optional): The output size of the student model. Defaults to None.
            generator (Union, optional):
                The generator that is used for CFKD.
                Defaults to "$PEAL/configs/models/default_generator.yaml".
            base_dir (Union, optional):
                The base directory for the run.
                Defaults to os.path.join("peal_runs", "counterfactual_knowledge_distillation").
            teacher (Union, optional): The teacher that is used for CFKD. Defaults to "human@8000".
            adaptor_config (Union, optional):
                The config for the adaptor.
                Defaults to "$PEAL/configs/adaptors/counterfactual_knowledge_distillation_default.yaml".
            gigabyte_vram (float, optional): The amount of vram in gigabytes. Defaults to None.
            overwrite (bool, optional):
                The flag that indicates whether the run should be overwritten. Defaults to False.
            visualization (callable, optional):
                The visualization function that is used for the run. Defaults to lambda x: x.
        """
        # TODO make sure to use seeds everywhere!
        self.base_dir = base_dir
        #
        self.original_student = student
        self.original_student.eval()
        self.device = (
            "cuda" if next(self.original_student.parameters()).is_cuda else "cpu"
        )
        self.overwrite = overwrite
        # either copy or load the student
        if self.overwrite or not os.path.exists(
            os.path.join(self.base_dir, "config.yaml")
        ):
            self.adaptor_config = load_yaml_config(adaptor_config)
            self.student = copy.deepcopy(student)

        else:
            self.adaptor_config = load_yaml_config(
                os.path.join(self.base_dir, "config.yaml")
            )
            if os.path.exists(os.path.join(self.base_dir, "model.cpl")):
                self.student = torch.load(
                    os.path.join(self.base_dir, "model.cpl"), map_location=self.device
                )

            else:
                self.student = copy.deepcopy(student)

        self.student.eval()

        self.output_size = integrate_data_config_into_adaptor_config(
            self.adaptor_config, datasource, output_size
        )

        set_adaptive_batch_size(
            self.adaptor_config, gigabyte_vram, self.adaptor_config["max_train_samples"]
        )

        #
        self.enable_hints = bool(teacher == "SegmentationMask")
        self.adaptor_config["data"]["has_hint"] = self.enable_hints
        (
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        ) = create_dataloaders_from_datasource(
            datasource=datasource,
            config=self.adaptor_config,
            enable_hints=self.enable_hints,
            gigabyte_vram=gigabyte_vram,
        )
        self.dataloaders_val = [self.val_dataloader, None]

        # in case the used dataloader has a non-default data normalization it is assumed
        # the inverse function of this normalization is attribute of the underlying dataset
        if hasattr(self.train_dataloader.dataset, "project_to_pytorch_default"):
            self.project_to_pytorch_default = (
                self.train_dataloader.dataset.project_to_pytorch_default
            )

        else:
            self.project_to_pytorch_default = lambda x: x

        #
        self.generator = get_generator(
            generator=generator,
            data_config=self.adaptor_config["data"],
            train_dataloader=self.train_dataloader,
            dataloaders_val=self.dataloaders_val,
            base_dir=base_dir,
            gigabyte_vram=gigabyte_vram,
            device=self.device,
        )

        #
        self.teacher = get_teacher(
            teacher=teacher,
            output_size=self.output_size,
            adaptor_config=self.adaptor_config,
        )

        self.dataloader_mixer = DataloaderMixer(
            self.adaptor_config["training"], self.train_dataloader
        )
        self.datastack = DataStack(self.train_dataloader, self.output_size)
        self.explainer = CounterfactualExplainer(
            downstream_model=self.student,
            generator=self.generator,
            input_type=self.adaptor_config["data"]["input_type"],
            explainer_config=self.adaptor_config["explainer"],
            dataset=self.val_dataloader.dataset,
        )
        self.logits_to_prediction = lambda logits: logits.argmax(-1)
        self.use_visualization = visualization
        self.tracked_keys = [
            "x_list",
            "x_counterfactual_list",
            "x_attribution_list",
            "y_list",
            "y_source_list",
            "y_target_list",
            "y_target_start_confidence_list",
            "y_target_end_confidence_list",
            "z_difference_list",
            "collage_path_list",
            # "hint_list",
        ]
        self.data_config = copy.deepcopy(self.adaptor_config)
        self.data_config["data"]["split"] = [1.0, 1.0]
        self.data_config["data"]["confounding_factors"] = []
        self.data_config["data"]["confounder_probability"] = None
        self.data_config["data"]["known_confounder"] = False
        self.data_config["data"]["output_type"] = "singleclass"
        self.data_config["data"]["output_size"] = self.train_dataloader.dataset.output_size
        self.data_config["data"]["delimiter"] = ","
        self.data_config["data"]["num_samples"] = self.adaptor_config[
            "max_train_samples"
        ]
        self.validation_data_config = copy.deepcopy(self.data_config)
        self.validation_data_config["data"]["num_samples"] = self.adaptor_config[
            "max_validation_samples"
        ]
        self.validation_data_config["data"]["split"] = [1.0, 1.0]

    def get_batch(
        self,
        error_matrix: torch.Tensor,
        confidence_score_stats: torch.tensor,
    ):
        x_batch = []
        y_source_batch = []
        y_target_batch = []
        y_batch = []
        y_target_start_confidence_batch = []
        hint_batch = []
        sample_idx = 0
        error_distribution = torch.distributions.Categorical(error_matrix)
        while not sample_idx >= self.adaptor_config["batch_size"]:
            cm_idx = error_distribution.sample()
            # TODO verify that this is actually balancing itself!
            y_source = int(cm_idx / self.output_size)
            y_target = int(cm_idx % self.output_size)
            x, y = self.datastack.pop(int(y_source))
            """
            TODO should be done with context manager
            if isinstance(self.teacher, SegmentationMaskTeacher):
                y, hint = y
            """

            logits = (
                self.student(x.to(self.device).unsqueeze(0)).squeeze(0).detach().cpu()
            )
            y_target_start_confidence = torch.nn.Softmax()(logits)[y_target]
            prediction = self.logits_to_prediction(logits)
            if (
                prediction == y == y_source
                and y_target_start_confidence
                > confidence_score_stats[y_source][y_target]
            ):
                x_batch.append(x)
                y_source_batch.append(y_source)
                y_target_batch.append(torch.tensor(y_target))
                y_batch.append(y)
                y_target_start_confidence_batch.append(y_target_start_confidence)
                hint_batch.append(torch.zeros_like(x))
                sample_idx += 1

        x_batch = torch.stack(x_batch)
        y_target_batch = torch.stack(y_target_batch)
        return {
            "x_list": x_batch,
            "y_list": y_batch,
            "y_target_list": y_target_batch,
            "y_source_list": y_source_batch,
            "y_target_start_confidence_list": y_target_start_confidence_batch,
            "hint_list": hint_batch,
        }

    def generate_x_counterfactual_list(
        self,
        error_matrix,
        confidence_score_stats,
        finetune_iteration,
        tracked_keys,
    ):
        # TODO this should be done with a context manager
        """if isinstance(self.teacher, SegmentationMaskTeacher):
        for dataloader in self.datastack.datasource.dataloaders:
            dataloader.dataset.enable_hints()"""

        self.datastack.datasource.reset()
        self.datastack.reset()

        collage_base_path = os.path.join(
            self.base_dir, str(finetune_iteration), "collages"
        )
        shutil.rmtree(
            collage_base_path,
            ignore_errors=True,
        )
        Path(collage_base_path).mkdir(parents=True, exist_ok=True)

        tracked_values = {key: [] for key in tracked_keys}

        continue_collecting = True
        acceptance_threshold = self.adaptor_config["explainer"][
            "y_target_goal_confidence"
        ]

        pbar = tqdm(
            total=int(
                self.adaptor_config["max_train_samples"]
                / self.adaptor_config["batch_size"]
                + 0.99
            )
            * self.adaptor_config["explainer"]["gradient_steps"],
        )
        pbar.stored_values = {}
        pbar.stored_values["n_total"] = 0
        while continue_collecting:
            num_batches_per_iteration = int(
                1
                + (
                    self.adaptor_config["max_train_samples"]
                    - len(list(tracked_values.values())[0])
                )
                / self.adaptor_config["batch_size"]
            )
            for i in range(num_batches_per_iteration):
                batch = self.get_batch(error_matrix, confidence_score_stats)
                values = self.explainer.explain_batch(
                    batch=batch,
                    base_path=collage_base_path,
                    start_idx=len(list(tracked_values.values())[0]),
                    y_target_goal_confidence_in=acceptance_threshold,
                    remove_below_threshold=True,
                    pbar=pbar,
                    mode="Training",
                )
                for key in tracked_keys:
                    tracked_values[key].extend(values[key])

                pbar.stored_values["n_valid"] = (
                    str(len(list(tracked_values.values())[0]))
                    + "/"
                    + str(self.adaptor_config["max_train_samples"])
                )
                pbar.stored_values["th"] = acceptance_threshold
                pbar.stored_values["n_total"] += self.adaptor_config["batch_size"]
                pbar.stored_values["fr"] = (
                    len(list(tracked_values.values())[0])
                    / pbar.stored_values["n_total"]
                )

            if (
                len(list(tracked_values.values())[0])
                < self.adaptor_config["max_train_samples"]
            ):
                if (
                    acceptance_threshold == 0.51
                    and len(list(tracked_values.values())[0]) == 0
                ):
                    continue_collecting = False

                elif (
                    len(list(values.values())[0])
                    < self.adaptor_config["max_train_samples"] / 2
                ):
                    acceptance_threshold = float(
                        np.maximum(0.51, acceptance_threshold - 0.1)
                    )

            else:
                continue_collecting = False

        """if isinstance(self.teacher, SegmentationMaskTeacher):
            for dataloader in self.datastack.datasource.dataloaders:
                dataloader.dataset.disable_hint()

            self.datastack.datasource.reset()"""
        pbar.close()
        return tracked_values

    def retrieve_validation_stats(self, finetune_iteration):
        if os.path.exists(
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

                return validation_stats

        validation_values_path = os.path.join(
            self.base_dir, str(finetune_iteration), "validation_tracked_values.npz"
        )
        if not os.path.exists(validation_values_path):
            (
                validation_tracked_values,
                validation_stats,
            ) = calculate_validation_statistics(
                model=self.student,
                dataloader=self.dataloaders_val[0],
                tracked_keys=self.tracked_keys,
                base_path=os.path.join(
                    self.base_dir, str(finetune_iteration), "validation_collages"
                ),
                output_size=self.output_size,
                explainer=self.explainer,
                device=self.device,
                logits_to_prediction=self.logits_to_prediction,
                use_confusion_matrix=self.adaptor_config["use_confusion_matrix"],
                max_validation_samples=self.adaptor_config["max_validation_samples"],
                min_start_target_percentile=self.adaptor_config[
                    "min_start_target_percentile"
                ],
            )
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

        else:
            # TODO think about this again
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

            collage_path_lists = os.listdir(
                os.path.join(
                    self.base_dir, str(finetune_iteration), "validation_collages"
                )
            )
            validation_tracked_values["collage_path_lists"] = list(
                map(
                    lambda x: os.path.join(
                        self.base_dir,
                        str(finetune_iteration),
                        "validation_collages",
                        x,
                    ),
                    collage_path_lists,
                )
            )

        validation_feedback, validation_feedback_stats = self.retrieve_feedback(
            tracked_values=validation_tracked_values,
            finetune_iteration=finetune_iteration,
            mode="validation",
        )

        for key in validation_feedback_stats.keys():
            validation_stats[key] = validation_feedback_stats[key]

        self.create_dataset(
            feedback=validation_feedback,
            finetune_iteration=finetune_iteration + 1,
            mode="validation",
            config=self.validation_data_config,
            **validation_tracked_values,
        )

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

        return validation_stats

    def initialize_run(self):
        if self.overwrite or not os.path.exists(
            os.path.join(self.base_dir, "0", "validation_tracked_values.npz")
        ):
            assert self.adaptor_config["current_iteration"] == 0
            print("Create base_dir in: " + str(self.base_dir))
            shutil.rmtree(self.base_dir, ignore_errors=True)
            Path(self.base_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(self.base_dir, "config.yaml"), "w") as file:
                yaml.dump(self.adaptor_config, file)

            with open(os.path.join(self.base_dir, "platform.txt"), "w") as f:
                f.write(platform.node())

            validation_stats = self.retrieve_validation_stats(finetune_iteration=0)

            """if self.output_size == 2 and self.use_visualization:
                self.visualize_progress(
                    [os.path.join(self.base_dir, "visualization.png")]
                )"""

            test_accuracy = calculate_test_accuracy(
                self.student, self.test_dataloader, self.device
            )
            self.adaptor_config["test_accuracies"] = [test_accuracy]

        else:
            with open(os.path.join(self.base_dir, "platform.txt"), "w") as f:
                f.write(platform.node())

            validation_stats = self.retrieve_validation_stats(finetune_iteration=0)

            # test_accuracy = self.adaptor_config["test_accuracies"][-1]
            test_accuracy = -1.0

        return validation_stats, test_accuracy

    def retrieve_counterfactual_list(self, validation_stats, finetune_iteration):
        tracked_values_path = os.path.join(
            self.base_dir, str(finetune_iteration), "tracked_values.npz"
        )
        if not os.path.exists(tracked_values_path):
            tracked_values = self.generate_x_counterfactual_list(
                error_matrix=validation_stats["error_matrix"],
                confidence_score_stats=validation_stats["confidence_score_stats"],
                finetune_iteration=finetune_iteration,
                tracked_keys=self.tracked_keys,
            )

            if len(list(tracked_values.values())[0]) == 0:
                return tracked_values

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

                collage_path_lists = os.listdir(
                    os.path.join(self.base_dir, str(finetune_iteration), "collages")
                )
                tracked_values["collage_path_lists"] = list(
                    map(
                        lambda x: os.path.join(
                            self.base_dir, str(finetune_iteration), "collages", x
                        ),
                        collage_path_lists,
                    )
                )

        return tracked_values

    def retrieve_feedback(self, tracked_values, finetune_iteration, mode):
        if not os.path.exists(
            os.path.join(self.base_dir, str(finetune_iteration), mode + "_feedback.txt")
        ):
            feedback = self.teacher.get_feedback(
                base_dir=os.path.join(
                    self.base_dir, str(finetune_iteration), "teacher"
                ),
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
        num_samples = min(
            self.adaptor_config["max_" + mode + "_samples"],
            len(self.val_dataloader.dataset),
        )
        counterfactual_rate = len(list(tracked_values.values())[0]) / num_samples
        ood_rate = len(list(filter(lambda sample: sample == "ood", feedback))) / len(
            feedback
        )

        num_true_2sided = len(list(filter(lambda sample: sample == "true", feedback)))
        num_false_2sided = len(list(filter(lambda sample: sample == "false", feedback)))
        fa_2sided = num_true_2sided / (num_true_2sided + num_false_2sided)

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
        fa_1sided = num_true_1sided / (num_true_1sided + num_false_1sided)
        fa_absolute = num_true_1sided / num_samples

        feedback_stats = {
            "counterfactual_rate": counterfactual_rate,
            "ood_rate": ood_rate,
            "fa_2sided": fa_2sided,
            "fa_1sided": fa_1sided,
            "fa_absolute": fa_absolute,
        }

        return feedback, feedback_stats

    def create_dataset(
        self,
        x_counterfactual_list,
        feedback,
        y_source_list,
        y_target_list,
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
        #
        sample_idx = 0
        x_list = []
        y_list = []
        sample_names = []
        for sample_idx in range(len(feedback)):
            if feedback[sample_idx] == "ood":
                continue

            elif feedback[sample_idx] == "true":
                sample_name = (
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
                sample_idx += 1

            elif feedback[sample_idx] == "false":
                sample_name = (
                    "false_"
                    + str(int(y_source_list[sample_idx]))
                    + "_to_"
                    + str(int(y_target_list[sample_idx]))
                    + "_"
                    + str(sample_idx)
                )
                x_list.append(x_counterfactual_list[sample_idx])
                y_list.append(int(y_source_list[sample_idx]))
                sample_names.append(sample_name)
                sample_idx += 1

        self.train_dataloader.dataset.serialize_dataset(
            output_dir=dataset_dir,
            x_list=x_list,
            y_list=y_list,
            sample_names=sample_names,
        )
        return dataset_dir

    def finetune_student(self, finetune_iteration, dataset_path):
        if not os.path.exists(
            os.path.join(
                self.base_dir,
                str(finetune_iteration),
                "finetuned_model",
                "model.cpl",
            )
        ):
            shutil.rmtree(
                os.path.join(self.base_dir, str(finetune_iteration), "finetuned_model"),
                ignore_errors=True,
            )
            Path(
                os.path.join(self.base_dir, str(finetune_iteration), "finetuned_model")
            ).mkdir(parents=True, exist_ok=True)
            #
            dataloader, _, _ = create_dataloaders_from_datasource(
                dataset_path, self.data_config
            )

            #
            val_dataset_path = os.path.join(
                self.base_dir, str(finetune_iteration), "validation_dataset"
            )
            _, dataloader_val, _ = create_dataloaders_from_datasource(
                val_dataset_path,
                self.validation_data_config,  # TODO: why dataset_path???
            )

            #
            priority = (
                (1 / (1 - self.adaptor_config["mixing_ratio"]))
                * self.adaptor_config["mixing_ratio"]
                * len(self.dataloader_mixer)
                / len(dataloader.dataset)
            )
            self.dataloader_mixer.append(dataloader, priority=priority)
            assert (
                abs(
                    self.dataloader_mixer.priorities[-1]
                    - self.adaptor_config["mixing_ratio"]
                )
                < 0.01
            ), "priorities do not match! " + str(self.dataloader_mixer.priorities)
            self.dataloaders_val[1] = dataloader_val

            if not self.adaptor_config["continuos_learning"]:

                def weight_reset(m):
                    reset_parameters = getattr(m, "reset_parameters", None)
                    if callable(reset_parameters):
                        m.reset_parameters()

                self.student.apply(weight_reset)

            finetune_trainer = ModelTrainer(
                config=copy.deepcopy(self.adaptor_config),
                model=self.student,
                datasource=(self.dataloader_mixer, self.dataloaders_val),
                model_name="finetuned_model",
                base_dir=os.path.join(self.base_dir, str(finetune_iteration)),
                val_dataloader_weights=[
                    1 - self.adaptor_config["mixing_ratio"],
                    self.adaptor_config["mixing_ratio"],
                ],
            )
            finetune_trainer.fit(continue_training=True)

        else:
            self.student = torch.load(
                os.path.join(
                    self.base_dir,
                    str(finetune_iteration),
                    "finetuned_model",
                    "model.cpl",
                ),
                map_location=self.device,
            )

    def run(self):
        """
        Run the counterfactual knowledge distillation
        """
        print("Adaptor Config: " + str(self.adaptor_config))
        validation_stats, test_accuracy = self.initialize_run()
        writer = SummaryWriter(os.path.join(self.base_dir, "logs"))

        for key in validation_stats.keys():
            if isinstance(validation_stats[key], float):
                writer.add_scalar(
                    key, validation_stats[key], self.adaptor_config["current_iteration"]
                )

        writer.add_scalar(
            "test_accuracy", test_accuracy, self.adaptor_config["current_iteration"]
        )

        # iterate over the finetune iterations
        for finetune_iteration in range(
            self.adaptor_config["current_iteration"] + 1,
            self.adaptor_config["finetune_iterations"] + 1,
        ):
            tracked_values = self.retrieve_counterfactual_list(
                validation_stats=validation_stats, finetune_iteration=finetune_iteration
            )
            if (
                len(list(tracked_values.values())[0])
                < self.adaptor_config["max_train_samples"]
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
            )
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

            test_accuracy = calculate_test_accuracy(
                self.student, self.test_dataloader, self.device
            )
            writer.add_scalar("test_accuracy", test_accuracy, finetune_iteration)

            """if self.output_size == 2 and self.use_visualization:
                self.visualize_progress(
                    [
                        os.path.join(
                            self.base_dir, str(finetune_iteration), "visualization.png"
                        ),
                        os.path.join(self.base_dir, "visualization.png"),
                    ]
                )"""

            if (
                self.adaptor_config["replacement_strategy"] == "delayed"
                and self.adaptor_config["replace_model"]
            ):
                torch.save(self.student, os.path.join(self.base_dir, "model.cpl"))

            if validation_stats["fa_1sided"] > self.adaptor_config["fa_1sided_prime"]:
                self.adaptor_config["fa_1sided_prime"] = validation_stats["fa_1sided"]
                if self.adaptor_config["replacement_strategy"] == "direct":
                    torch.save(self.student, os.path.join(self.base_dir, "model.cpl"))

                self.adaptor_config["replace_model"] = True

            else:
                self.adaptor_config["replace_model"] = False

            self.adaptor_config["current_iteration"] = (
                self.adaptor_config["current_iteration"] + 1
            )
            with open(os.path.join(self.base_dir, "config.yaml"), "w") as file:
                yaml.dump(self.adaptor_config, file)

        return self.student
