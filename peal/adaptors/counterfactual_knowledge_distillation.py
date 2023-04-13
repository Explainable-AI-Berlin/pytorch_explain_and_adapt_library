import os
import torch
import torchvision
import copy
import yaml
import shutil
import numpy as np
import matplotlib
import platform

from tensorboardX import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from typing import Union
from torch import nn

from peal.utils import load_yaml_config, embed_numberstring, set_adaptive_batch_size
from peal.data.dataloaders import (
    DataStack,
    DataloaderMixer,
    create_dataloaders_from_datasource,
)
from peal.training.trainers import ModelTrainer, calculate_test_accuracy
from peal.explainers.counterfactual_explainer import CounterfactualExplainer
from peal.data.datasets import (
    Image2MixedDataset,
)
from peal.data.dataset_interfaces import PealDataset
from peal.visualization.model_comparison import create_comparison
from peal.teachers.segmentation_mask_teacher import SegmentationMaskTeacher
from peal.teachers.teacher_factory import get_teacher
from peal.teachers.teacher_interface import TeacherInterface
from peal.generators.interfaces import InvertibleGenerator
from peal.generators.generator_factory import get_generator
from peal.adaptors.adaptor_utils import integrate_data_config_into_adaptor_config
from peal.training.training_utils import (
    retrieve_validation_statistics,
    visualize_progress,
)

matplotlib.use("Agg")


class CounterfactualKnowledgeDistillation:
    """
    This class implements the counterfactual knowledge distillation approach.
    """

    def __init__(
        self,
        student: nn.Module,
        datasource: Union(
            list[torch.utils.data.DataLoader],
            list[torch.utils.data.Dataset],
            list[PealDataset],
        ),
        output_size: int = None,
        generator: Union(
            InvertibleGenerator, Path, str
        ) = "$PEAL/configs/models/default_generator.yaml",
        base_dir: Union(str, Path) = os.path.join(
            "peal_runs", "counterfactual_knowledge_distillation"
        ),
        teacher: Union(str, TeacherInterface) = "human@8000",
        adaptor_config: Union(
            dict, str, Path
        ) = "$PEAL/configs/adaptors/counterfactual_knowledge_distillation_default.yaml",
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
            generator (Union, optional): The generator that is used for CFKD. Defaults to "$PEAL/configs/models/default_generator.yaml".
            base_dir (Union, optional): The base directory for the run. Defaults to os.path.join("peal_runs", "counterfactual_knowledge_distillation").
            teacher (Union, optional): The teacher that is used for CFKD. Defaults to "human@8000".
            adaptor_config (Union, optional): The config for the adaptor. Defaults to "$PEAL/configs/adaptors/counterfactual_knowledge_distillation_default.yaml".
            gigabyte_vram (float, optional): The amount of vram in gigabytes. Defaults to None.
            overwrite (bool, optional): The flag that indicates whether the run should be overwritten. Defaults to False.
            visualization (callable, optional): The visualization function that is used for the run. Defaults to lambda x: x.
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

        set_adaptive_batch_size(self.adaptor_config, gigabyte_vram)

        #
        self.enable_hints = bool(teacher == "SegmentationMask")
        self.adaptor_config["data"]["has_hints"] = self.enable_hints
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
        self.generator = get_generator(generator, self.adaptor_config["data"])

        #
        self.teacher = get_teacher(teacher)

        self.dataloader_mixer = DataloaderMixer(
            self.adaptor_config["training"], self.train_dataloader
        )
        self.datastack = DataStack(self.train_dataloader, self.output_size)
        self.explainer = CounterfactualExplainer(
            downstream_model=self.student,
            generator=self.generator,
            input_type=self.adaptor_config["data"]["input_type"],
            explainer_config=self.adaptor_config["explainer"],
        )
        self.logits_to_prediction = lambda logits: logits.argmax(-1)
        self.use_visualization = visualization
        self.tracked_keys = [
            "start_target_confidence",
            "result_img_collage",
            "counterfactual",
            "heatmap",
            "end_target_confidence",
            "attribution",
            "y",
            "y_pred",
            "target",
        ]

    def get_batch(
        self,
        error_distribution: torch.distributions.Categorical,
        confidence_score_stats: torch.tensor,
    ):
        x_batch = []
        source_classes_current_batch = []
        target_classes_current_batch = []
        ys_current_batch = []
        start_target_confidences = []
        current_hint_batch = []
        sample_idx = 0
        with tqdm(range(100 * self.train_dataloader.dataset.__len__())) as pbar:
            for i in pbar:
                if sample_idx >= self.adaptor_config["batch_size"]:
                    break

                cm_idx = error_distribution.sample()
                # TODO verify that this is actually balancing itself!
                source_class = int(cm_idx / self.output_size)
                target_class = int(cm_idx % self.output_size)
                X, y = self.datastack.pop(int(source_class))
                """
                TODO should be done with context manager
                if isinstance(self.teacher, SegmentationMaskTeacher):
                    y, hint = y
                """

                logits = (
                    self.student(X.to(self.device).unsqueeze(0))
                    .squeeze(0)
                    .detach()
                    .cpu()
                )
                start_target_confidence = torch.nn.Softmax()(logits)[target_class]
                prediction = self.logits_to_prediction(logits)
                if (
                    prediction == y == source_class
                    and start_target_confidence
                    > confidence_score_stats[source_class][target_class]
                ):
                    x_batch.append(X)
                    """
                    TODO should be done with context manager
                    if isinstance(self.teacher, SegmentationMaskTeacher):
                        current_hint_batch.append(hint)

                    else:
                        current_hint_batch.append(torch.zeros_like(X))
                    """
                    current_hint_batch.append(torch.zeros_like(X))
                    source_classes_current_batch.append(source_class)
                    target_classes_current_batch.append(torch.tensor(target_class))
                    ys_current_batch.append(y)
                    start_target_confidences.append(start_target_confidence)
                    sample_idx += 1

        x_batch = torch.stack(x_batch)
        target_classes_current_batch = torch.stack(target_classes_current_batch)
        return {
            "x": x_batch,
            "y": ys_current_batch,
            "target": target_classes_current_batch,
            "source": source_classes_current_batch,
            "start_target_confidences": start_target_confidences,
        }

    def generate_counterfactuals(
        self,
        error_distribution,
        confidence_score_stats,
        finetune_iteration,
        tracked_keys,
    ):
        # TODO this should be done with a context manager
        if isinstance(self.teacher, SegmentationMaskTeacher):
            for dataloader in self.datastack.datasource.dataloaders:
                dataloader.dataset.enable_hints()

            self.datastack.datasource.reset()

        self.datastack.reset()

        shutil.rmtree(
            os.path.join(self.base_dir, str(finetune_iteration), "collages"),
            ignore_errors=True,
        )
        Path(os.path.join(self.base_dir, str(finetune_iteration), "collages")).mkdir(
            parents=True, exist_ok=True
        )

        tracked_values = {key: [] for key in tracked_keys}

        continue_collecting = True
        current_acceptance_threshold = self.adaptor_config["explainer"][
            "target_confidence_goal"
        ]

        while continue_collecting:
            values = self.generate_counterfactuals_iteration(
                num_samples=self.adaptor_config["samples_per_iteration"],
                error_distribution=error_distribution,
                confidence_score_stats=confidence_score_stats,
                finetune_iteration=finetune_iteration,
                sample_idx_iteration=len(tracked_values.values()[0]),
                target_confidence_goal=current_acceptance_threshold,
                tracked_keys=tracked_keys,
            )
            for key in tracked_keys:
                tracked_values[key].extend(values[key])

            print(
                str(len(tracked_values.values()[0]))
                + "/"
                + str(self.adaptor_config["samples_per_iteration"])
            )
            if (
                len(tracked_values.values()[0])
                < self.adaptor_config["samples_per_iteration"]
            ):
                if current_acceptance_threshold == 0.51 and len(values[0]) == 0:
                    continue_collecting = False

                elif (
                    len(values.values()[0])
                    < self.adaptor_config["samples_per_iteration"] / 2
                ):
                    current_acceptance_threshold = float(
                        np.maximum(0.51, current_acceptance_threshold - 0.1)
                    )

            else:
                continue_collecting = False

        if isinstance(self.teacher, SegmentationMaskTeacher):
            for dataloader in self.datastack.datasource.dataloaders:
                dataloader.dataset.disable_hints()

            self.datastack.datasource.reset()

        return tracked_values

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

            if self.output_size == 2 and self.use_visualization:
                self.visualize_progress(
                    [os.path.join(self.base_dir, "visualization.png")]
                )

            test_accuracy = calculate_test_accuracy(
                self.student, self.test_dataloader, self.device
            )
            validation_stats = retrieve_validation_statistics(0, self.tracked_keys)

        else:
            with open(os.path.join(self.base_dir, "platform.txt"), "w") as f:
                f.write(platform.node())

            validation_stats = retrieve_validation_statistics(
                self.adaptor_config["current_iteration"], self.tracked_keys
            )

        return validation_stats, test_accuracy

    def retrieve_counterfactuals(self, validation_stats, finetune_iteration):
        if not os.path.exists(
            os.path.join(self.base_dir, str(finetune_iteration), "counterfactuals.npz")
        ):
            tracked_values = self.generate_counterfactuals(
                error_distribution=validation_stats["error_distribution"],
                confidence_score_stats=validation_stats["confidence_score_stats"],
                finetune_iteration=finetune_iteration,
                tracked_keys=self.tracked_keys,
            )

            if len(collage_paths) < self.adaptor_config["samples_per_iteration"]:
                print("No counterfactuals could be found anymore!")
                open(os.path.join(self.base_dir, "warning.txt"), "w").write(
                    "No counterfactuals could be found anymore in iteration "
                    + str(finetune_iteration)
                    + "!"
                )
                return self.student

            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), "counterfactuals.npz"
                ),
                "wb",
            ) as f:
                for key in self.tracked_keys:
                    if isinstance(tracked_values[key], torch.tensor):
                        np.savez(f, key=tracked_values[key].numpy())

        else:
            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), "counterfactuals.npz"
                ),
                "rb",
            ) as f:
                tracked_values = {}
                validation_tracked_values = np.load(f, allow_pickle=True)
                for key in validation_tracked_values.keys():
                    tracked_values[key] = list(
                        torch.tensor(validation_tracked_values[key])
                    )

                collage_paths = os.listdir(
                    os.path.join(self.base_dir, str(finetune_iteration), "collages")
                )
                tracked_values["collage_paths"] = list(
                    map(
                        lambda x: os.path.join(
                            self.base_dir, str(finetune_iteration), "collages", x
                        ),
                        collage_paths,
                    )
                )

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

            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), mode + "_feedback.txt"
                ),
                "w",
            ) as f:
                f.write(feedback)

        else:
            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), mode + "_feedback.txt"
                ),
                "r",
            ) as f:
                feedback = f.read()

        # TODO this is not correct for calculating training stats.
        num_samples = min(
            self.adaptor_config["max_" + mode + "_samples"],
            len(self.val_dataloader.dataset),
        )
        counterfactual_rate = len(tracked_values.values()[0]) / num_samples
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
                    and tracked_values["y"][x[0]] == tracked_values["y_pred"][x[0]],
                    enumerate(feedback),
                )
            )
        )
        num_false_1sided = len(
            list(
                filter(
                    lambda x: x[1] == "false"
                    and tracked_values["y"][x[0]] == tracked_values["y_pred"][x[0]],
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
        counterfactuals,
        feedback,
        source_classes,
        target_classes,
        base_dir,
        finetune_iteration,
        mode="",
        **args,
    ):
        assert (
            len(counterfactuals)
            == len(feedback)
            == len(source_classes)
            == len(target_classes)
        ), "missmatch in list lengths"

        dataset_dir = os.path.join(base_dir, str(finetune_iteration), mode + "_dataset")
        #
        current_sample_idx = 0
        x_list = []
        y_list = []
        sample_names = []
        for sample_idx in range(len(feedback)):
            if feedback[sample_idx] == "ood":
                continue

            elif feedback[sample_idx] == "true":
                sample_name = (
                    "true_"
                    + str(int(source_classes[sample_idx]))
                    + "_to_"
                    + str(int(target_classes[sample_idx]))
                    + "_"
                    + str(current_sample_idx)
                )
                x_list.append(counterfactuals[sample_idx])
                y_list.append(int(target_classes[sample_idx]))
                sample_names.append(sample_name)
                current_sample_idx += 1

            elif feedback[sample_idx] == "false":
                sample_name = (
                    "false_"
                    + str(int(source_classes[sample_idx]))
                    + "_to_"
                    + str(int(target_classes[sample_idx]))
                    + "_"
                    + str(current_sample_idx)
                )
                x_list.append(counterfactuals[sample_idx])
                y_list.append(int(source_classes[sample_idx]))
                sample_names.append(sample_name)
                current_sample_idx += 1

        self.train_dataloader.dataset.serialize_dataset(
            dataset_dir, x_list, y_list, sample_names
        )
        return dataset_dir

    def finetune_student(self, finetune_iteration, current_dataset_path):
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
            data_config = copy.deepcopy(self.adaptor_config)
            data_config["training"]["split"] = [1.0, 1.0]
            current_dataloader, _, _ = create_dataloaders_from_datasource(
                current_dataset_path, data_config
            )

            #
            val_dataset_path = os.path.join(
                self.base_dir, str(finetune_iteration), "validation_dataset"
            )
            validation_data_config = copy.deepcopy(self.adaptor_config)
            validation_data_config["training"]["split"] = [0.0, 1.0]
            _, current_dataloader_val, _ = create_dataloaders_from_datasource(
                val_dataset_path, data_config  # TODO: why current_dataset_path???
            )

            #
            priority = (
                (1 / (1 - self.adaptor_config["mixing_ratio"]))
                * self.adaptor_config["mixing_ratio"]
                * len(self.dataloader_mixer)
                / len(current_dataloader.dataset)
            )
            self.dataloader_mixer.append(current_dataloader, priority=priority)
            assert (
                abs(
                    self.dataloader_mixer.priorities[-1]
                    - self.adaptor_config["mixing_ratio"]
                )
                < 0.01
            ), "priorities do not match! " + str(self.dataloader_mixer.priorities)
            self.dataloaders_val[1] = current_dataloader_val

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
            tracked_values, stats = self.retrieve_counterfactuals(
                validation_stats=validation_stats, finetune_iteration=finetune_iteration
            )
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

            current_dataset_path = self.create_dataset(
                feedback=feedback,
                finetune_iteration=finetune_iteration,
                mode="train",
                **tracked_values,
            )
            self.finetune_student(
                finetune_iteration=finetune_iteration,
                current_dataset_path=current_dataset_path,
            )
            (
                validation_tracked_values,
                validation_stats,
            ) = retrieve_validation_statistics(finetune_iteration, self.tracked_keys)
            validation_feedback, validation_feedback_stats = self.retrieve_feedback(
                tracked_values=validation_tracked_values,
                finetune_iteration=finetune_iteration,
                mode="validation",
            )
            for key in validation_stats.keys():
                if isinstance(validation_stats[key], float):
                    writer.add_scalar(
                        "validation_" + key,
                        validation_stats[key],
                        finetune_iteration,
                    )

            self.create_dataset(
                feedback=validation_feedback,
                finetune_iteration=finetune_iteration,
                mode="validation",
                **tracked_values,
            )

            test_accuracy = calculate_test_accuracy(
                self.student, self.test_dataloader, self.device
            )
            writer.add_scalar("test_accuracy", test_accuracy, finetune_iteration)

            if self.output_size == 2 and self.use_visualization:
                self.visualize_progress(
                    [
                        os.path.join(
                            self.base_dir, str(finetune_iteration), "visualization.png"
                        ),
                        os.path.join(self.base_dir, "visualization.png"),
                    ]
                )

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
