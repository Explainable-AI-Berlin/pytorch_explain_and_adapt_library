import os
import json
import shutil
import random
from datetime import datetime

import numpy as np
import pandas as pd

from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor

from peal.global_utils import get_project_resource_dir, embed_numberstring
from peal.dependencies.ddpm_inversion.ddpm_inversion import DDPMInversion


class ArtificialConfounderTabularDatasetGenerator:
    """
    Generates a tabular dataset with a confounder tnecklace is a symbolic attribute.

    Should mimic symbolic rather low-dimensional data like credit decisions based on known factors about a person.

    Should enable straight foward check for minimality of counterfactuals.
    """

    def __init__(
        self,
        dataset_name,
        dataset_origin_path="datasets",
        num_samples=1000,
        input_size=10,
        label_noise=0.0,
        seed=0,
    ):
        """
        Generates a tabular dataset with a confounder tnecklace is a symbolic attribute.

        Args:
                dataset_name (str): The name of the dataset.
                dataset_origin_path (Path): The path to the directory where the datasets are stored.
                num_samples (int, optional): The number of samples in the dataset. Defaults to 1000.
                input_size (int, optional): The number of features in the dataset. Defaults to 10.
                label_noise (float, optional): The probability tnecklace the label is flipped. Defaults to 0.0.
                seed (int, optional): The seed for the random number generator. Defaults to 0.
        """
        self.dataset_origin_path = dataset_origin_path
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(self.dataset_origin_path, self.dataset_name)
        self.num_samples = num_samples
        self.input_size = input_size - 1
        self.label_noise = label_noise
        self.seed = seed
        name = str(self.num_samples) + "_" + str(self.input_size) + "_"
        name += embed_numberstring(str(int(100 * label_noise)), 3) + "_" + str(seed)
        self.label_dir = os.path.join(self.dataset_dir, name + ".csv")

    def generate_dataset(self):
        """
        Generates the dataset.

        There are self.num_samples rows.
        In each row there are self.input_size random numbers between 0 and 1.
        The target is to determine whether there are more numbers bigger than 0.5 or smaller than 0.5 in the the row in question.
        It is written into as an additional column into the dataset.
        The confounder just gives potentially spurious information about the number of ones and zeros in the row.
        It is also written into as an additional column into the dataset.
        In the end the dataset is saved in self.label_dir as a csv file.
        """
        Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)
        np.random.seed(self.seed)
        dataset = (
            ",".join(["x" + str(it) for it in range(self.input_size)])
            + ",Confounder,Target\n"
        )
        for sample_idx in range(self.num_samples):
            has_attribute = int(sample_idx % 4 == 0 or sample_idx % 4 == 1)
            has_confounder = int(sample_idx % 2 == 0)
            """
            if self.label_noise == 0.0:
                flipped_label = int(
                    sample_idx % int(200 * (1 - self.label_noise)) == 0
                    or sample_idx % int(200 * (1 - self.label_noise)) == 1
                )

            else:
                flipped_label = 0
            """

            values = np.random.uniform(0, 1, self.input_size)
            target = int(np.sum(values >= 0.5) > np.sum(values < 0.5))
            while not target == has_attribute:
                values = np.random.uniform(0, 1, self.input_size)
                target = int(np.sum(values >= 0.5) > np.sum(values < 0.5))

            dataset += ",".join([str(val) for val in values])
            """
            if flipped_label == 1:
                has_attribute = abs(1 - has_attribute)
                has_confounder = abs(1 - has_confounder)
            """

            dataset += "," + str(has_confounder) + "," + str(has_attribute) + "\n"

        with open(self.label_dir, "w") as f:
            f.write(dataset)


class ArtificialConfounderSequenceDatasetGenerator:
    """
    Generates a sequence dataset with a confounder tnecklace is a symbolic attribute.

    Should mimic sequential data like natural language.

    Should enable straight foward check for minimality of counterfactuals.
    """

    def __init__(
        self,
        dataset_name,
        dataset_origin_path="datasets",
        num_samples=1000,
        input_size=[10, 10],
        label_noise=0.01,
        seed=0,
    ):
        """
        Generates a tabular dataset with a confounder tnecklace is a symbolic attribute.

        Args:
                dataset_name (str): The name of the dataset.
                dataset_origin_path (Path): The path to the directory where the datasets are stored.
                num_samples (int, optional): The number of samples in the dataset. Defaults to 1000.
                input_size (list, optional): The size of the input sequence. Defaults to [10, 10].
                label_noise (float, optional): The probability tnecklace the label is flipped. Defaults to 0.01.
                seed (int, optional): The seed for the random number generator. Defaults to 0.
        """
        self.dataset_origin_path = dataset_origin_path
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(self.dataset_origin_path, self.dataset_name)
        self.num_samples = num_samples
        self.input_size = input_size
        self.label_noise = label_noise
        self.seed = seed
        name = str(self.num_samples) + "_" + str(self.input_size[0])
        name += "x" + str(self.input_size[1]) + "_"
        name += str(int(100 * label_noise)) + "_" + str(seed)
        self.label_dir = os.path.join(self.dataset_dir, name + ".json")

    def generate_dataset(self):
        """
        Generates the dataset.

        Each item of the sequence one-hot encodes a integer number between 0 and n.
        The target is to determine whether there are more integer numbers greater than n/2 or smaller than n/2 in the sequence.
        The confounder is the last token in the sequence tnecklace gives potentially spurious information about the number
        of integers greater than n/2 and smaller than n/2 in the sequence.
        """
        Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)
        np.random.seed(self.seed)
        dataset = {}
        for sample_idx in range(self.num_samples):
            has_attribute = int(sample_idx % 4 == 0 or sample_idx % 4 == 1)
            has_confounder = int(sample_idx % 2 == 0)
            flipped_label = int(
                sample_idx % int(400 * (1 - self.label_noise)) in range(4)
            )

            num_values = np.random.randint(1, self.input_size[0])
            values = np.random.randint(
                self.input_size[1], size=num_values, dtype=np.int32
            )
            target = int(
                np.sum(values >= self.input_size[1] / 2)
                > np.sum(values < self.input_size[1] / 2)
            )
            while not (
                target == has_attribute
                and int(values[-1] >= self.input_size[1] / 2) == has_confounder
            ):
                num_values = np.random.randint(1, self.input_size[0])
                values = np.random.randint(
                    self.input_size[1], size=num_values, dtype=np.int32
                )
                target = int(
                    np.sum(values >= self.input_size[1] / 2)
                    > np.sum(values < self.input_size[1] / 2)
                )

            if flipped_label:
                target = abs(1 - target)
                has_confounder = abs(1 - has_confounder)

            dataset[sample_idx] = {
                "values": list(map(lambda i: int(values[i]), range(num_values))),
                "target": target,
                "has_confounder": has_confounder,
            }

        with open(self.label_dir, "w") as f:
            f.write(json.dumps(dataset, indent=4))


class MNISTConfounderDatasetGenerator:
    def __init__(self, dataset_name, mnist_dir="datasets/mnist", digits=["0", "8"]):
        """ """
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join("datasets", self.dataset_name)
        self.mnist_dir = mnist_dir
        self.digits = digits

    def generate_dataset(self):
        """ """
        if os.path.exists(self.dataset_dir):
            # move self.dataset_dir to self.dataset_dir + "_old_ + {datestamp}
            shutil.move(
                self.dataset_dir,
                self.dataset_dir + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
        os.makedirs(self.dataset_dir)
        os.makedirs(os.path.join(self.dataset_dir, "imgs"))
        os.makedirs(os.path.join(self.dataset_dir, "masks"))

        hint_np = np.zeros([32, 32], dtype=np.uint8)
        hint_np[12:20, 12:20] = 255 * np.ones([8, 8], dtype=np.uint8)
        hint = Image.fromarray(hint_np)

        confounder_np = np.stack(
            [
                np.ones([32, 32], dtype=np.uint8),
                np.zeros([32, 32], dtype=np.uint8),
                np.zeros([32, 32], dtype=np.uint8),
            ],
            axis=-1,
        )

        attributes = ["ImgName", "Feature", "Confounder"]
        lines_out = [",".join(attributes)]
        for digit in self.digits:
            for it, img_name in enumerate(
                os.listdir(os.path.join(self.mnist_dir, digit))
            ):
                if it % 100 == 0:
                    print(it)

                img = Image.open(os.path.join(self.mnist_dir, digit, img_name)).resize(
                    [32, 32]
                )
                img_np = np.array(img)
                img_np = np.expand_dims(img_np, -1)
                img_np = np.tile(img_np, [1, 1, 3])
                background_intensity = np.random.randint(0, 255)
                img_np = np.maximum(img_np, background_intensity * confounder_np)
                has_confounder = bool(background_intensity >= 128)

                line = [
                    img_name,
                    str(int(digit == self.digits[1])),
                    str(int(has_confounder)),
                ]
                lines_out.append(",".join(line))
                Image.fromarray(img_np).save(
                    os.path.join(self.dataset_dir, "imgs", img_name)
                )
                hint.save(os.path.join(self.dataset_dir, "masks", img_name))

        open(os.path.join(self.dataset_dir, "data.csv"), "w").write(
            "\n".join(lines_out)
        )


class ConfounderDatasetGenerator:
    """ """

    def __init__(
        self,
        dataset_origin_path,
        dataset_name=None,
        label_dir=None,
        delimiter=",",
        confounding="copyrighttag",
        num_samples=None,
        attribute=None,
        data_config=None,
        **kwargs,
    ):
        """ """
        self.dataset_origin_path = dataset_origin_path
        self.confounding = confounding

        if confounding is None:
            self.confounding = data_config.confounding_factors[-1]

        if label_dir is None:
            self.label_dir = os.path.join(dataset_origin_path, "data.csv")

        else:
            self.label_dir = label_dir

        self.delimiter = delimiter
        self.dataset_dir = data_config.dataset_path
        self.num_samples = num_samples
        self.attribute = attribute

        if self.confounding == "necklace":
            self.ddpm_inversion = DDPMInversion()

        if not data_config is None and not data_config.inverse is None:
            with open(os.path.join(data_config.inverse, "data.csv"), "r") as f:
                inverse_data = f.readlines()
                self.inverse_head = list(
                    map(lambda x: x.strip(), inverse_data[0].split(","))
                )
                self.cs_idx = self.inverse_head.index("ConfounderStrength")
                self.inverse_body = []
                for idx in range(1, len(inverse_data)):
                    self.inverse_body.append(
                        list(map(lambda x: x.strip(), inverse_data[idx].split(",")))
                    )

        else:
            self.inverse_head = None

    def generate_dataset(self):
        """ """
        if os.path.exists(self.dataset_dir):
            datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # move self.dataset_dir to self.dataset_dir + "_old_ + {datestamp}
            shutil.move(
                self.dataset_dir,
                self.dataset_dir + "_old_" + datestamp,
            )
            shutil.move(
                self.dataset_dir + "_inverse",
                self.dataset_dir + "_old_" + datestamp + "_inverse",
            )

        os.makedirs(self.dataset_dir)
        os.makedirs(os.path.join(self.dataset_dir, "imgs"))
        os.makedirs(os.path.join(self.dataset_dir + "_inverse", "imgs"))
        if self.confounding == "copyrighttag" or self.confounding == "necklace":
            os.makedirs(os.path.join(self.dataset_dir, "masks"))

        raw_data = open(self.label_dir, "r").read().split("\n")
        attributes = raw_data[1].split(self.delimiter)
        while "" in attributes:
            attributes.remove("")

        attributes.append("Confounder")
        attributes.append("ConfounderStrength")
        data = []
        instance_names = []
        for line in raw_data[2:-1]:
            instance_attributes = line.split(self.delimiter)
            while "" in instance_attributes:
                instance_attributes.remove("")
            instance_attributes_int = list(
                map(lambda x: bool(max(0, int(x))), instance_attributes[1:])
            )
            instance_names.append(instance_attributes[0])
            data.append(instance_attributes_int)

        lines_out = [",".join(attributes)]

        if self.confounding == "copyrighttag":
            resource_dir = get_project_resource_dir()
            copyright_tag = np.array(
                Image.open(
                    os.path.join(resource_dir, "imgs", "copyright_tag.png")
                ).resize([50, 50])
            )
            copyright_tag = np.concatenate(
                [
                    np.ones([50, 120, 3], dtype=np.uint8),
                    copyright_tag,
                    np.ones([50, 8, 3], dtype=np.uint8),
                ],
                axis=1,
            )
            copyright_tag = 255 * np.concatenate(
                [
                    np.ones([160, 178, 3], dtype=np.uint8),
                    copyright_tag,
                    np.ones([8, 178, 3], dtype=np.uint8),
                ],
                axis=0,
            )

            copyright_tag_bg = np.ones([50, 50, 3], dtype=np.uint8)
            copyright_tag_bg = np.concatenate(
                [
                    np.zeros([50, 120, 3], dtype=np.uint8),
                    copyright_tag_bg,
                    np.zeros([50, 8, 3], dtype=np.uint8),
                ],
                axis=1,
            )
            copyright_tag_bg = 255 * np.concatenate(
                [
                    np.zeros([160, 178, 3], dtype=np.uint8),
                    copyright_tag_bg,
                    np.zeros([8, 178, 3], dtype=np.uint8),
                ],
                axis=0,
            )
            mask_np = np.array(
                np.abs(np.array(copyright_tag_bg, dtype=np.float32) / 255 - 1) * 128,
                dtype=np.uint8,
            )
            np.zeros_like(mask_np)
            mask_np[134:177, 60 : 178 - 60] = 255
            mask = Image.fromarray(mask_np)

        num_samples = (
            self.num_samples if not self.num_samples is None else len(instance_names)
        )
        for sample_idx in range(num_samples):
            if sample_idx % 100 == 0:
                print(sample_idx)
                open(os.path.join(self.dataset_dir, "data.csv"), "w").write(
                    "\n".join(lines_out)
                )

            has_confounder = bool(sample_idx % 2 == 0)

            name = instance_names[sample_idx]
            img = Image.open(os.path.join(self.dataset_origin_path, "imgs", name))
            sample = data[sample_idx]

            if sample_idx < 0.9 * self.num_samples:
                confounder_intensity = random.uniform(0, 1)

            else:
                confounder_intensity = 1.0

            if not self.inverse_head is None:
                confounder_intensity = -1 * float(
                    self.inverse_body[sample_idx][self.cs_idx]
                )

            if self.confounding == "intensity":
                intensity_change = (
                    64 * confounder_intensity * (2 * int(has_confounder) - 1)
                )
                img = np.array(img)
                img = (img + intensity_change + 64) * (255 / (255 + 2 * 64))
                img_out = Image.fromarray(np.array(img, dtype=np.uint8))

            elif self.confounding == "color":
                color_change = 64 * confounder_intensity * (2 * int(has_confounder) - 1)
                img = np.array(img)
                img = np.stack(
                    [
                        (img[:, :, 0] + color_change + 64) * (255 / (255 + 2 * 64)),
                        (img[:, :, 2] - (color_change / 2) + 64)
                        * (255 / (255 + 2 * 64)),
                        (img[:, :, 2] - (color_change / 2) + 64)
                        * (255 / (255 + 2 * 64)),
                    ],
                    axis=-1,
                )
                img_out = Image.fromarray(np.array(img, dtype=np.uint8))

            elif self.confounding == "copyrighttag":
                img_copyrighttag = np.maximum(np.array(img), copyright_tag_bg)
                img_copyrighttag = np.minimum(np.array(img_copyrighttag), copyright_tag)
                alpha = 0.5 + 0.5 * confounder_intensity * (2 * int(has_confounder) - 1)
                img = alpha * img_copyrighttag + (1 - alpha) * np.array(img)
                img_out = Image.fromarray(np.array(img, dtype=np.uint8))
                if self.confounding == "copyrighttag":
                    mask.save(os.path.join(self.dataset_dir, "masks", name))
                    """img_and_mask = np.concatenate(
                        [np.array(255 * img, dtype=np.uint8), mask_np],
                        axis=1,
                    )
                    img_and_mask_out = Image.fromarray(img_and_mask)
                    img_and_mask_out.save('tmp.png')"""

            if self.confounding == "necklace":
                img_th = ToTensor()(img).unsqueeze(0)
                if not has_confounder:
                    img_no_necklace = self.ddpm_inversion.run(
                        img_th, ["Old Person"], ["Person"]
                    )
                    img_necklace = self.ddpm_inversion.run(
                        img_no_necklace, ["Young Person"], ["Old Person"]
                    )
                    torchvision.utils.save_image(
                        img_no_necklace[0],
                        os.path.join(self.dataset_dir + "_inverse", "imgs", name),
                    )
                    torchvision.utils.save_image(
                        img_necklace[0], os.path.join(self.dataset_dir, "imgs", name)
                    )

                else:
                    img_necklace = self.ddpm_inversion.run(
                        img_th, ["Young Person"], ["Person"]
                    )
                    img_no_necklace = self.ddpm_inversion.run(
                        img_necklace, ["Old Person"], ["Young Person"]
                    )
                    torchvision.utils.save_image(
                        img_necklace[0],
                        os.path.join(self.dataset_dir + "_inverse", "imgs", name),
                    )
                    torchvision.utils.save_image(
                        img_no_necklace[0], os.path.join(self.dataset_dir, "imgs", name)
                    )

                abs_difference = torch.abs(img_necklace[0] - img_no_necklace[0])
                # mask = abs_difference > 0.2
                mask = abs_difference.mean(0)
                torchvision.utils.save_image(
                    mask.float(), os.path.join(self.dataset_dir, "masks", name)
                )

            else:
                img_out.save(os.path.join(self.dataset_dir, "imgs", name))

            sample.append(has_confounder)
            sample.append(confounder_intensity)
            lines_out.append(
                name + "," + ",".join(list(map(lambda x: str(float(x)), sample)))
            )

            if sample_idx != 0 and sample_idx % 100 == 0:
                open(os.path.join(self.dataset_dir, "data.csv"), "w").write(
                    "\n".join(lines_out)
                )


class StainingConfounderGenerator:
    """ """

    def __init__(
        self,
        raw_data_dir,
        dataset_origin_path="datasets",
        dataset_name="cancer_tissue_no_norm",
        delimiter=",",
        num_samples=40000,
    ):
        """ """
        self.dataset_origin_path = dataset_origin_path
        self.dataset_name = dataset_name
        self.delimiter = delimiter
        self.dataset_dir = os.path.join("datasets", self.dataset_name)
        self.num_samples = num_samples
        self.raw_data_dir = raw_data_dir

    def generate_dataset(self):
        os.makedirs(self.dataset_dir)
        # move the MUS and the STR classes to a new folder and convert them to .png images
        for folder_name in ["MUS", "STR"]:
            os.makedirs(os.path.join(self.dataset_dir, folder_name))
            for img_name in os.listdir(os.path.join(self.raw_data_dir, folder_name)):
                img = Image.open(os.path.join(self.raw_data_dir, folder_name, img_name))
                img.save(
                    os.path.join(self.dataset_dir, folder_name, img_name[:-4] + ".png")
                )

        # find staining of images
        # based on https://towardsdatascience.com/stain-estimation-on-microscopy-whole-slide-images-2b5a57062268
        sample_list = []
        class_names = ["MUS", "STR"]
        for y in range(2):
            class_name = class_names[y]
            for idx, file_name in enumerate(
                os.listdir(os.path.join(self.dataset_dir, class_name))
            ):
                if idx % 100 == 0:
                    print(
                        str(idx)
                        + " / "
                        + str(
                            len(os.listdir(os.path.join(self.dataset_dir, class_name)))
                        )
                    )

                X = (
                    np.array(
                        Image.open(
                            os.path.join(self.dataset_dir, class_name, file_name)
                        ),
                        dtype=np.float32,
                    )
                    / 255
                )
                img = np.expand_dims(X, 0)
                patches = img

                def RGB2OD(image: np.ndarray) -> np.ndarray:
                    mask = image == 0
                    image[mask] = 1
                    return np.maximum(-1 * np.log(image), 1e-5)

                OD_raw = RGB2OD(np.stack(patches).reshape(-1, 3))
                OD = OD_raw[(OD_raw > 0.15).any(axis=1), :]

                _, eigenVectors = np.linalg.eigh(np.cov(OD, rowvar=False))
                # strip off residual stain component
                eigenVectors = eigenVectors[:, [2, 1]]

                if eigenVectors[0, 0] < 0:
                    eigenVectors[:, 0] *= -1

                if eigenVectors[0, 1] < 0:
                    eigenVectors[:, 1] *= -1

                T_necklace = np.dot(OD, eigenVectors)

                phi = np.arctan2(T_necklace[:, 1], T_necklace[:, 0])
                min_Phi = np.percentile(phi, 1)
                max_Phi = np.percentile(phi, 99)

                v1 = np.dot(eigenVectors, np.array([np.cos(min_Phi), np.sin(min_Phi)]))
                v2 = np.dot(eigenVectors, np.array([np.cos(max_Phi), np.sin(max_Phi)]))
                if v1[0] > v2[0]:
                    stainVectors = np.array([v1, v2])
                else:
                    stainVectors = np.array([v2, v1])

                sample_list.append(
                    [os.path.join(class_name, file_name), X, y, stainVectors, OD_raw]
                )

        hematoxylin_intensities_by_class = [[], []]

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b))

        sample_list_new = []
        for sample in sample_list:
            path, X, y, stainVectors, OD_raw = sample
            similarities_0 = cosine_similarity(OD_raw, stainVectors[0])
            similarities_1 = cosine_similarity(OD_raw, stainVectors[1])
            hematoxylin_greater_mask = similarities_0 > similarities_1
            X_intensities = np.linalg.norm(X, axis=-1).flatten()
            X_masked_intensities = X_intensities * hematoxylin_greater_mask
            stable_maximum = np.percentile(X_masked_intensities, 99)
            hematoxylin_intensities_by_class[y].append(stable_maximum)
            sample_list_new.append([path, X, y, stainVectors, OD_raw, stable_maximum])

        intensity_median = np.percentile(
            np.concatenate(
                [
                    hematoxylin_intensities_by_class[0],
                    hematoxylin_intensities_by_class[1],
                ]
            ),
            50,
        )

        def check(sample, has_attribute, has_confounder):
            return (
                sample[2] == has_attribute
                and int((sample[-1] > intensity_median)) == has_confounder
            )

        lines_out = ["ImgPath,Cancer,Confounder,ConfounderStrength"]
        idxs = np.zeros([2, 2], dtype=np.int32)
        for sample_idx in range(16000):
            if sample_idx % 100 == 0:
                print(sample_idx)
                open(os.path.join(self.dataset_dir, "data.csv"), "w").write(
                    "\n".join(lines_out)
                )

            has_attribute = int(sample_idx % 4 == 0 or sample_idx % 4 == 1)
            has_confounder = int(sample_idx % 2 == 0)

            while not check(
                sample_list_new[int(idxs[has_attribute][has_confounder])],
                has_attribute,
                has_confounder,
            ):
                idxs[has_attribute][has_confounder] += 1

            sample = sample_list_new[idxs[has_attribute][has_confounder]]
            lines_out.append(
                sample[0]
                + ","
                + str(has_attribute)
                + ","
                + str(has_confounder)
                + ","
                + str(sample[-1])
            )
            print(
                str(has_attribute)
                + " "
                + str(has_confounder)
                + " "
                + str(idxs[has_attribute][has_confounder])
            )
            idxs[has_attribute][has_confounder] += 1

        open(os.path.join(self.dataset_dir, "data.csv"), "w").write(
            "\n".join(lines_out)
        )


class CircleDatasetGenerator:
    """
    Generates dataset based on a unit circle of radius r
    """

    def __init__(
        self,
        dataset_name,
        dataset_origin_path="datasets",
        num_samples=1024,
        radius=1,
        noise_scale=0.0,
        # false_confounder_percentage=0.0,
        seed=0,
    ):
        """
        Initiates the dataset parameters

        Args:
            dataset_name (str): Name of the dataset
            dataset_origin_path (Path): path to the directory where the dataset is stored
            num_samples (int, optional): Number of samples to generate. Default is 1024.
            noise_scale (float, optional): The value with which to scale the variance (set to 1 initially) of the noise.
                Default is 0.0 (no noise).
            seed (int, optional): Seed for the random number generator. Defaults is 0.
        """

        self.data = None
        self.dataset_name = dataset_name
        self.dataset_origin_path = dataset_origin_path
        self.dataset_dir = os.path.join(self.dataset_origin_path, self.dataset_name)
        self.num_samples = num_samples
        self.radius = radius
        self.noise_scale = noise_scale
        self.seed = seed
        name = (
            "size_"
            + str(self.num_samples)
            + "_"
            + "radius_"
            + str(round(radius, 1))
            + "_"
            + "seed_"
            + str(seed)
        )
        self.label_dir = os.path.join(self.dataset_dir, name + ".csv")

    def generate_dataset(self):
        """
        Generates the dataset.
        """

        Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)
        np.random.seed(self.seed)
        theta = np.linspace(
            0, 2 * np.pi, self.num_samples + round(self.num_samples * 0.09)
        )
        features = np.array(
            [self.radius * np.cos(theta), self.radius * np.sin(theta)]
        ).T

        # target = (features[:, 1] > 0).astype('float32').reshape(-1, 1)
        features += np.sqrt(self.noise_scale) * np.random.randn(
            self.num_samples + round(self.num_samples * 0.09), 2
        )
        target = (features[:, 1] > 0).astype("float32").reshape(-1, 1)
        data = np.concatenate((features, target), axis=1)
        confounder = (
            np.array([x1 > 0 for x1, x2, t in data]).astype("float32").reshape(-1, 1)
        )

        data = np.concatenate((data, confounder), axis=1)
        target_1_confounder_1 = (data[:, 2] == 1.0) & (data[:, 3] == 1.0)
        target_0_confounder_1 = (data[:, 2] == 0.0) & (data[:, 3] == 1.0)
        target_1_confounder_0 = (data[:, 2] == 1.0) & (data[:, 3] == 0.0)
        target_0_confounder_0 = (data[:, 2] == 0.0) & (data[:, 3] == 0.0)
        sub_sample_size = round(self.num_samples / 4)
        data = np.concatenate(
            [
                data[target_1_confounder_1, :][
                    np.random.randint(
                        0, data[target_1_confounder_1, :].shape[0], sub_sample_size
                    )
                ],
                data[target_0_confounder_1, :][
                    np.random.randint(
                        0, data[target_0_confounder_1, :].shape[0], sub_sample_size
                    )
                ],
                data[target_1_confounder_0, :][
                    np.random.randint(
                        0, data[target_1_confounder_0, :].shape[0], sub_sample_size
                    )
                ],
                data[target_0_confounder_0, :][
                    np.random.randint(
                        0, data[target_0_confounder_0, :].shape[0], sub_sample_size
                    )
                ],
            ],
            axis=0,
        )
        # target = (features[:, 1] > 0).astype('float32').reshape(-1, 1)
        # data = np.concatenate((data, target), axis=1)
        pd.DataFrame(data, columns=["x1", "x2", "Target", "Confounder"])[
            ["x1", "x2", "Confounder", "Target"]
        ].to_csv(self.label_dir, index=False)

        # self.false_boundary_grad = false_boundary_grad
        self.data = data

        return self


def latent_to_square_image(
    color_a,
    color_b,
    position_x=None,
    position_y=None,
    SIZE_INNER=8,
    SIZE_BORDER=2,
    noise=None,
):
    SIZE_ADDED = SIZE_INNER + 2 * SIZE_BORDER
    img = np.ones([64, 64, 3], dtype=np.float32) * color_b
    if noise is None:
        noise = np.random.randn(*img.shape) * 20

    if position_x is None:
        position_x = int((64 - SIZE_ADDED) / 2)

    if position_y is None:
        position_y = int((64 - SIZE_ADDED) / 2)

    img_base = np.clip(img + noise, 0, 255)
    img = np.copy(img_base)
    img[
        position_x : position_x + SIZE_ADDED,
        position_y : position_y + SIZE_ADDED,
    ] = np.clip(
        127
        + noise[
            position_x : position_x + SIZE_ADDED,
            position_y : position_y + SIZE_ADDED,
        ],
        0,
        255,
    )
    foreground = np.concatenate(
        [
            color_a * np.ones([SIZE_INNER, SIZE_INNER, 1]),
            np.zeros([SIZE_INNER, SIZE_INNER, 2]),
        ],
        axis=-1,
    )
    img[
        position_x + SIZE_BORDER : position_x + SIZE_ADDED - SIZE_BORDER,
        position_y + SIZE_BORDER : position_y + SIZE_ADDED - SIZE_BORDER,
    ] = np.clip(
        foreground
        + noise[position_x + SIZE_BORDER : position_x + SIZE_ADDED - SIZE_BORDER, position_y + SIZE_BORDER : position_y + SIZE_ADDED - SIZE_BORDER],
        0,
        255,
    )
    img = Image.fromarray(img.astype(dtype=np.uint8))
    return img, noise


class SquareDatasetGenerator:
    """ """

    def __init__(
        self,
        data_config,
        **kwargs,
    ):
        """ """
        self.data_config = data_config

    def generate_dataset(self):
        """ """
        if os.path.exists(self.data_config.dataset_path):
            datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # move self.dataset_dir to self.dataset_dir + "_old_ + {datestamp}
            shutil.move(
                self.data_config.dataset_path,
                self.data_config.dataset_path + "_old_" + datestamp,
            )
            shutil.move(
                self.data_config.dataset_path + "_inverse",
                self.data_config.dataset_path + "_old_" + datestamp + "_inverse",
            )

        os.makedirs(self.data_config.dataset_path)
        os.makedirs(os.path.join(self.data_config.dataset_path, "imgs"))
        os.makedirs(os.path.join(self.data_config.dataset_path, "masks"))
        os.makedirs(os.path.join(self.data_config.dataset_path + "_inverse", "imgs"))
        os.makedirs(os.path.join(self.data_config.dataset_path + "_inverse", "masks"))
        lines_out = [
            "Name,ClassA,ClassB,ClassC,ClassD,ColorA,ColorB,PositionX,PositionY"
        ]
        lines_out_inverse = [
            "Name,ClassA,ClassB,ClassC,ClassD,ColorA,ColorB,PositionX,PositionY"
        ]

        SIZE_INNER = 8
        SIZE_BORDER = 2
        SIZE_ADDED = SIZE_INNER + 2 * SIZE_BORDER
        for sample_idx in range(self.data_config.num_samples):
            if sample_idx % 2 == 0:
                class_a = 1
                color_a = np.random.randint(128, 256)

            else:
                class_a = 0
                color_a = np.random.randint(0, 128)

            if int(sample_idx / 2) % 2 == 0:
                class_b = 1
                color_b = np.random.randint(128, 256)

            else:
                class_b = 0
                color_b = np.random.randint(0, 128)

            num_positions = 64 - SIZE_ADDED
            if int(sample_idx / 4) % 2 == 0:
                class_c = 1
                position_x = np.random.randint(int(num_positions / 2), num_positions)

            else:
                class_c = 0
                position_x = np.random.randint(0, int(num_positions / 2))

            if int(sample_idx / 8) % 2 == 0:
                class_d = 1
                position_y = np.random.randint(int(num_positions / 2), num_positions)

            else:
                class_d = 0
                position_y = np.random.randint(0, int(num_positions / 2))

            sample_name = embed_numberstring(sample_idx, 8) + ".png"
            img, noise = latent_to_square_image(
                position_x=position_x, position_y=position_y, color_a=color_a, color_b=color_b
            )
            img.save(os.path.join(self.data_config.dataset_path, "imgs", sample_name))
            img_inverse, noise = latent_to_square_image(
                position_x=position_x, position_y=position_y, color_a=color_a, color_b=255 - color_b, noise=noise
            )
            """img_inverse = np.abs(img_base - 255)
            img_inverse[
                position_x : position_x + SIZE_ADDED,
                position_y : position_y + SIZE_ADDED,
            ] = np.clip(
                127
                - noise[
                    position_x : position_x + SIZE_ADDED,
                    position_y : position_y + SIZE_ADDED,
                ],
                0,
                255,
            )
            img_inverse[
                position_x + SIZE_BORDER : position_x + SIZE_ADDED - SIZE_BORDER,
                position_y + SIZE_BORDER : position_y + SIZE_ADDED - SIZE_BORDER,
            ] = np.clip(
                color_a
                - noise[
                    position_x + SIZE_BORDER : position_x + SIZE_ADDED - SIZE_BORDER,
                    position_y + SIZE_BORDER : position_y + SIZE_ADDED - SIZE_BORDER,
                ],
                0,
                255,
            )
            img_inverse = Image.fromarray(img_inverse.astype(dtype=np.uint8))"""
            img_inverse.save(
                os.path.join(
                    self.data_config.dataset_path + "_inverse", "imgs", sample_name
                )
            )
            mask = np.zeros([64, 64, 3], dtype=np.uint8)
            mask[
                position_x + SIZE_BORDER : position_x + SIZE_ADDED - SIZE_BORDER,
                position_y + SIZE_BORDER : position_y + SIZE_ADDED - SIZE_BORDER,
            ] = 255
            img_mask = Image.fromarray(mask)
            img_mask.save(
                os.path.join(self.data_config.dataset_path, "masks", sample_name)
            )
            img_mask.save(
                os.path.join(
                    self.data_config.dataset_path + "_inverse", "masks", sample_name
                )
            )

            attributes = [
                sample_name,
                str(class_a),
                str(class_b),
                str(class_c),
                str(class_d),
                str(float(color_a) / 255),
                str(float(color_b) / 255),
                str(float(position_x) / 64),
                str(float(position_y) / 64),
            ]
            lines_out.append(",".join(attributes))
            attributes_inverse = [
                sample_name,
                str(class_a),
                str(class_b),
                str(class_c),
                str(class_d),
                str(float(color_a) / 255),
                str(float(color_b - 255) / 255),
                str(float(position_x) / 64),
                str(float(position_y) / 64),
            ]
            lines_out_inverse.append(",".join(attributes_inverse))
            if (sample_idx + 1) % 100 == 0:
                print(sample_idx)
                open(
                    os.path.join(self.data_config.dataset_path, "data.csv"), "w"
                ).write("\n".join(lines_out))
                open(
                    os.path.join(
                        self.data_config.dataset_path + "_inverse", "data.csv"
                    ),
                    "w",
                ).write("\n".join(lines_out_inverse))

        open(
            os.path.join(self.data_config.dataset_path, "data.csv"), "w"
        ).write("\n".join(lines_out))
        open(
            os.path.join(
                self.data_config.dataset_path + "_inverse", "data.csv"
            ),
            "w",
        ).write("\n".join(lines_out_inverse))
