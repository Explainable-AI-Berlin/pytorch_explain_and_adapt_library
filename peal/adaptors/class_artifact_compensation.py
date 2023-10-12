import os
import shutil
import torch
import h5py
import copy
import numpy as np
import logging
import platform

from torch import nn
from tensorboardX import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from argparse import Namespace
from collections import OrderedDict
from functools import partial
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from sklearn.svm import LinearSVC
from zennit.composites import COMPOSITES
from PIL import Image

from peal.explainers.lrp_explainer import LRPExplainer
from peal.global_utils import load_yaml_config, embed_numberstring
from peal.teachers.teacher_interface import TeacherInterface
from peal.teachers.human2model_teacher import Human2ModelTeacher
from peal.teachers.heatmap_comparison_teacher import HeatmapComparisonTeacher
from peal.teachers.segmentation_mask_teacher import SegmentationMaskTeacher
from peal.teachers.virelay_teacher import VirelayTeacher
from peal.data.dataloaders import create_dataloaders_from_datasource
from peal.data.datasets import Image2MixedDataset
from peal.visualization.model_comparison import create_comparison
from peal.training.trainers import calculate_test_accuracy


class SimpleProjection(torch.nn.Module):
    """
    Simple projection layer
    """

    def __init__(self, uncorrect_decision_strategy, correct_decision_strategy):
        """
        uncorrect_decision_strategy: List of tensors with shape [1, C, H, W]
        correct_decision_strategy: List of tensors
        """
        super().__init__()
        self.difference = torch.stack(uncorrect_decision_strategy).mean(
            0
        ) - torch.stack(correct_decision_strategy).mean(0)

    def forward(self, x):
        """
        x: Tensor with shape [B, C, H, W]
        """
        return x - self.difference


class SVMProjection(torch.nn.Module):
    def __init__(
        self, uncorrect_decision_strategy, correct_decision_strategy, adaptor_config
    ):
        super().__init__()
        self.adaptor_config = adaptor_config
        uncorrect_decision_strategy = torch.stack(uncorrect_decision_strategy)
        correct_decision_strategy = torch.stack(correct_decision_strategy)
        features = torch.cat(
            [uncorrect_decision_strategy, correct_decision_strategy], 0
        ).cpu()  # features[valid_indices]
        artifacts = torch.cat(
            [
                torch.ones([uncorrect_decision_strategy.shape[0]]),
                torch.zeros([correct_decision_strategy.shape[0]]),
            ]
        ).cpu()
        self.model = LinearSVC(
            C=1000.0, penalty="l2", max_iter=10000, class_weight="balanced", verbose=2
        )
        self.model.fit(features.reshape(features.shape[0], -1), artifacts)
        svm_cav = self.model.coef_[0]  # .reshape(features.shape[1:])
        # assuming the convention is w^T x - b = 0 (i.e. positive bias) and
        # w is not normalized
        svm_norm = (svm_cav**2).sum() ** 0.5
        # svm_cav = svm_cav / svm_norm
        self.cav = torch.tensor(
            self.adaptor_config.correction_strength * svm_cav / svm_norm
        )
        # cav_bias = svm_cav * self.model.intercept_[0] / svm_norm
        # TODO why are there two other means calculated???
        self.bias = correct_decision_strategy.mean(0).flatten()
        # bias_artifact_mean = uncorrect_decision_strategy.mean(0)

        #
        # bias = bias_artifact_mean.numpy()
        # bias = cav_bias
        # bias = bias_clean_mean.numpy()
        # TODO rewrite equation 26
        # proj = torch.outer(*[torch.from_numpy(svm_cav.ravel()).float()] * 2)
        # self.iproj = torch.eye(len(proj)) - proj
        # self.shift = torch.from_numpy(bias.ravel()).float() @ proj

    def forward(self, x):
        # return (x.flatten(start_dim=1) @ self.iproj.to(x) + self.shift.to(x)).reshape(x.shape)
        x_flat = x.flatten(1)
        return (
            x_flat
            - x_flat @ self.cav[:, None].to(x) @ self.cav[None].to(x)
            + self.bias[None].to(x) @ self.cav[:, None].to(x) @ self.cav[None].to(x)
        ).reshape(x.shape)


def extract_all_children(model):
    """
    Extracts all children of a model
    """
    children = []
    for child in model.children():
        if isinstance(child, torch.nn.Sequential):
            children.extend(extract_all_children(child))

        else:
            children.append(child)

    return children


def split_model_at_penultima(model):
    """
    Splits a model at the penultima layer
    """
    children_list = extract_all_children(model)
    feature_extractor = torch.nn.Sequential(*children_list[:-2])
    downstream_head = torch.nn.Sequential(*children_list[-2:])
    return feature_extractor, downstream_head


class ClassArtifactCompensation:
    def __init__(
        self,
        student,
        datasource,
        output_size,
        base_dir,
        teacher="virelay@8000",
        adaptor_config="<PEAL_BASE>/configs/adaptors/pclarc_projection_layer.yaml",
        gigabyte_vram=None,
        overwrite=False,
    ):
        #
        self.original_student = student
        self.original_student.eval()
        self.student = copy.deepcopy(student)
        self.student.eval()
        self.device = "cuda" if next(self.student.parameters()).is_cuda else "cpu"
        self.base_dir = base_dir
        self.output_size = output_size
        self.overwrite = overwrite
        self.adaptor_config = load_yaml_config(adaptor_config)
        if not output_size is None:
            self.output_size = output_size
            self.adaptor_config.data.output_size = output_size

        else:
            assert self.adaptor_config.data.output_size != "None"

        def integrate_task_into_adaptor_config(dataset, adaptor_config):
            if hasattr(dataset, "task_config") and not dataset.task_config is None:
                adaptor_config.task = dataset.task_config

            elif isinstance(dataset, Image2MixedDataset):
                adaptor_config.task.selection = [dataset.config.confounding_factors[0]]
                adaptor_config.task.output_size = 2
                dataset.task_config = adaptor_config.task

        if isinstance(datasource[0], torch.utils.data.Dataset):
            integrate_task_into_adaptor_config(datasource[0], self.adaptor_config)
            X, y = datasource[0].__getitem__(0)
            self.adaptor_config.data.input_size = list(X.shape)

        elif isinstance(datasource[0], torch.utils.data.DataLoader):
            integrate_task_into_adaptor_config(
                datasource[0].dataset, self.adaptor_config
            )
            X, y = datasource[0].dataset.__getitem__(0)
            self.adaptor_config.data.input_size = list(X.shape)

        else:
            assert self.adaptor_config.data.input_size != "None"

        self.input_size = self.adaptor_config.data.input_size

        if "base_batch_size" in self.adaptor_config.keys():
            multiplier = float(
                np.prod(self.adaptor_config.assumed_input_size)
                / np.prod(self.adaptor_config.data.input_size)
            )
            if (
                not gigabyte_vram is None
                and "gigabyte_vram" in self.adaptor_config.keys()
            ):
                multiplier = multiplier * (
                    gigabyte_vram / self.adaptor_config.gigabyte_vram
                )

            batch_size_adapted = max(
                1, int(self.adaptor_config.base_batch_size * multiplier)
            )
            if self.adaptor_config.batch_size == -1:
                self.adaptor_config.batch_size = batch_size_adapted
                self.adaptor_config.num_batches = (
                    int(self.adaptor_config.samples_per_iteration / batch_size_adapted)
                    + 1
                )

        #
        self.enable_hints = bool(teacher == "SegmentationMask")
        self.adaptor_config.data.has_hints = self.enable_hints
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

        #
        self.explainer = LRPExplainer(
            downstream_model=self.student,
            num_classes=self.output_size,
            explainer_config=self.adaptor_config.explainer,
        )

        #
        if isinstance(teacher, TeacherInterface):
            self.teacher = teacher

        elif isinstance(teacher, nn.Module):
            teacher.eval()
            teacher_explainer = LRPExplainer(
                teacher, self.output_size, self.adaptor_config.explainer
            )
            self.teacher = HeatmapComparisonTeacher(
                teacher_explainer, self.adaptor_config.attribution_threshold
            )

        elif teacher[:5] == "human":
            if len(teacher) == 10:
                port = int(teacher[-4:])

            else:
                port = 8000

            self.teacher = Human2ModelTeacher(port)

        elif teacher == "SegmentationMask":
            self.teacher = SegmentationMaskTeacher(
                self.adaptor_config.attribution_threshold
            )

        elif teacher[:7] == "virelay":
            self.teacher = VirelayTeacher(
                num_classes=self.output_size, port=int(teacher[-4:])
            )

    def generate_lrp_heatmaps(self, dataloader, collage_dir, max_samples=10000000):
        images = []
        heatmaps = []
        overlays = []
        gt_classes = []
        predictions = []
        hints = []
        samples = 0
        max_samples = min(len(dataloader.dataset), max_samples)
        with tqdm(enumerate(dataloader)) as pbar:
            for it, (X, y) in pbar:
                (
                    heatmap_batch,
                    overlay_batch,
                    prediction_batch,
                ) = self.explainer.explain_batch(X, y)
                predictions.append(prediction_batch)
                images.append(X)
                overlays.append(overlay_batch)
                heatmaps.append(heatmap_batch)
                gt_classes.append(y)
                # TODO implement hints
                hints.append(heatmap_batch)
                samples += X.shape[0]
                pbar.set_description(
                    "Generating LRP maps: " + str(samples) + " / " + str(max_samples)
                )
                if samples > max_samples:
                    break

        images = list(torch.cat(images))
        heatmaps = list(torch.cat(heatmaps))
        overlays = list(torch.cat(overlays))
        gt_classes = list(torch.cat(gt_classes))
        predictions = list(torch.cat(predictions))
        hints = list(torch.cat(hints))
        # TODO implement attributions in intermediate layers
        attributions = heatmaps
        collage_paths = []
        for idx in range(len(images)):
            collage = np.array(
                255 * torch.cat([images[idx], heatmaps[idx], overlays[idx]], 2).numpy(),
                dtype=np.uint8,
            ).transpose(1, 2, 0)
            collage_img = Image.fromarray(collage)
            collage_path = os.path.join(
                collage_dir, embed_numberstring(str(idx)) + ".png"
            )
            collage_img.save(collage_path)
            collage_paths.append(collage_path)

        return (
            collage_paths,
            heatmaps,
            gt_classes,
            hints,
            attributions,
            images,
            predictions,
        )

    def calculate_validation_statistics(self, finetune_iteration):
        (
            collage_paths,
            heatmaps,
            gt_classes,
            hints,
            attributions,
            images,
            predictions,
        ) = self.generate_lrp_heatmaps(
            self.val_dataloader,
            os.path.join(self.base_dir, str(finetune_iteration), "validation_collages"),
            self.adaptor_config.max_validation_samples,
        )

        feedback = self.teacher.get_feedback(
            heatmaps=heatmaps,
            collage_paths=collage_paths,
            gt_classes=gt_classes,
            hints=hints,
            attributions=attributions,
            images=images,
            source_classes=predictions,
        )

        accuracy = calculate_test_accuracy(
            self.student, self.val_dataloader, self.device
        )

        fa_absolute = len(list(filter(lambda x: x == "true", feedback))) / len(feedback)

        return accuracy, fa_absolute

    def visualize_progress(self, paths):
        task_config_buffer = copy.deepcopy(self.test_dataloader.dataset.task_config)
        criterions = {}
        if (
            isinstance(self.test_dataloader.dataset, Image2MixedDataset)
            and "Confounder" in self.test_dataloader.dataset.attributes
        ):
            self.test_dataloader.dataset.task_config = {
                "selection": [],
                "criterions": [],
            }
            criterions["class"] = lambda X, y: int(
                y[
                    self.test_dataloader.dataset.attributes.index(
                        task_config_buffer["selection"][0]
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
            criterions["pclarc"] = lambda X, y: int(
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
            criterions["pclarc"] = lambda X, y: int(
                self.student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
            )

        img = create_comparison(
            dataset=self.test_dataloader.dataset,
            criterions=criterions,
            columns={
                "LRP\nuncorrected": ["lrp", self.original_student, "uncorrected"],
                "LRP\ncorrected": ["lrp", self.student, "pclarc"],
            },
            score_reference_idx=1,
            device=self.device,
            explainer_config=self.adaptor_config.explainer,
            generator=None,
            max_samples=10,
        )
        self.test_dataloader.dataset.task_config = task_config_buffer
        for path in paths:
            img.save(path)

        return img

    def run(self):
        """
        Run the class artifact compensation algorithm
        """
        if self.overwrite:
            shutil.rmtree(self.base_dir, ignore_errors=True)

        Path(self.base_dir).mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(os.path.join(self.base_dir, "logs"))
        f = open(os.path.join(self.base_dir, "platform.txt"), "w")
        f.write(platform.node())
        f.close()

        if not os.path.exists(os.path.join(self.base_dir, "0")):
            os.makedirs(os.path.join(self.base_dir, "0"))
            os.makedirs(os.path.join(self.base_dir, "0", "collages"))
            os.makedirs(os.path.join(self.base_dir, "0", "validation_collages"))

            if self.output_size == 2:
                self.visualize_progress(
                    [os.path.join(self.base_dir, "visualization.png")]
                )

            if self.adaptor_config.max_validation_samples >= 1:
                val_accuracy, fa_absolute = self.calculate_validation_statistics(0)
                writer.add_scalar("val_accuracy", val_accuracy, 0)
                writer.add_scalar("fa_absolute", fa_absolute, 0)

            test_accuracy = calculate_test_accuracy(
                self.student, self.test_dataloader, self.device
            )
            writer.add_scalar("test_accuracy", test_accuracy, 0)

        # iterate over the finetune iterations
        for finetune_iteration in range(1, self.adaptor_config.finetune_iterations + 1):
            if not os.path.exists(os.path.join(self.base_dir, str(finetune_iteration))):
                os.makedirs(os.path.join(self.base_dir, str(finetune_iteration)))
                os.makedirs(
                    os.path.join(self.base_dir, str(finetune_iteration), "collages")
                )
                os.makedirs(
                    os.path.join(
                        self.base_dir, str(finetune_iteration), "validation_collages"
                    )
                )
                (
                    collage_paths,
                    heatmaps,
                    gt_classes,
                    hints,
                    attributions,
                    images,
                    predictions,
                ) = self.generate_lrp_heatmaps(
                    dataloader=self.train_dataloader,
                    collage_dir=os.path.join(
                        self.base_dir, str(finetune_iteration), "collages"
                    ),
                    max_samples=self.adaptor_config.samples_per_iteration,
                )

            else:
                heatmaps, gt_classes, hints, attributions, predictions = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                collage_paths = []
                files = os.listdir(
                    os.path.join(self.base_dir, str(finetune_iteration), "collages")
                )
                files.sort()
                for file in files:
                    collage_paths.append(
                        os.path.join(
                            self.base_dir, str(finetune_iteration), "collages", file
                        )
                    )

                images = []
                labels = []
                for it in range(len(collage_paths)):
                    images.append(self.train_dataloader.dataset[it][0])
                    labels.append(self.train_dataloader.dataset[it][1])

            feedback = self.teacher.get_feedback(
                images=images,
                heatmaps=heatmaps,
                collage_paths=collage_paths,
                gt_classes=gt_classes,
                attributions=attributions,
                base_dir=os.path.join(self.base_dir, str(finetune_iteration)),
                source_classes=predictions,
            )
            feature_extractor, downstream_head = split_model_at_penultima(self.student)
            feature_extractor.eval()
            uncorrect_decision_strategy = []
            correct_decision_strategy = []
            correct_X = []
            uncorrect_X = []
            if self.adaptor_config.projection_location == "features":
                with tqdm(range(len(feedback))) as pbar:
                    for it in pbar:
                        X = images[it].unsqueeze(0).to(self.device)
                        features = feature_extractor(X).detach()[0]
                        if feedback[it] == "true":
                            correct_decision_strategy.append(features)

                        else:
                            uncorrect_decision_strategy.append(features)

                        pbar.set_description(
                            f"Extracting features: {it}/{len(feedback)}"
                        )

            else:
                correct_decision_strategy = [
                    x
                    for x, fb, l in zip(images, feedback, labels)
                    if fb == "true" and l == 1
                ]
                uncorrect_decision_strategy = [
                    x
                    for x, fb, l in zip(images, feedback, labels)
                    if fb == "false" and l == 1
                ]

            if self.adaptor_config.projection_type == "simple":
                projection = SimpleProjection(
                    uncorrect_decision_strategy, correct_decision_strategy
                )

            elif self.adaptor_config.projection_type == "svm":
                projection = SVMProjection(
                    uncorrect_decision_strategy,
                    correct_decision_strategy,
                    self.adaptor_config,
                )

            if self.adaptor_config.projection_location == "features":
                self.student = torch.nn.Sequential(
                    *[feature_extractor, projection, downstream_head]
                )

            else:
                self.student = torch.nn.Sequential(*[projection, self.student])

            from IPython.core.debugger import set_trace

            set_trace()
            test_accuracy = calculate_test_accuracy(
                self.student, self.test_dataloader, self.device
            )
            writer.add_scalar("test_accuracy", test_accuracy, finetune_iteration)

            if self.adaptor_config.max_validation_samples >= 1:
                val_accuracy, fa_absolute = self.calculate_validation_statistics(
                    finetune_iteration
                )
                writer.add_scalar("val_accuracy", val_accuracy, finetune_iteration)
                writer.add_scalar("fa_absolute", fa_absolute, finetune_iteration)

            if self.output_size == 2:
                self.visualize_progress(
                    [
                        os.path.join(
                            self.base_dir, str(finetune_iteration), "visualization.png"
                        ),
                        os.path.join(self.base_dir, "visualization.png"),
                    ]
                )

            torch.save(
                self.student,
                os.path.join(self.base_dir, str(finetune_iteration), "model.cpl"),
            )
            torch.save(self.student, os.path.join(self.base_dir, "model.cpl"))

        return self.student
