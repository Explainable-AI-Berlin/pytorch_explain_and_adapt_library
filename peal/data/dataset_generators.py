"""
Generators for PEAL's synthetic and confounded benchmark datasets.

Each class writes a dataset directory in the layout PEAL's dataset classes
read (``imgs/``, optional ``masks/`` with segmentation hints, and a
``data.csv`` or ``.json`` label file with one column per attribute) into
``data_config.dataset_path`` or a ``datasets/<name>`` folder. The generators
plant a known confounder next to the true feature (copyright tags, colour
shifts, staining intensity, square colour/position, specific numbers, nodule
roundness, ...) so that adaptors can be evaluated against ground truth.
"""

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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision.transforms import ToTensor

from peal.global_utils import get_project_resource_dir, embed_numberstring
from peal.log import get_logger

_log = get_logger(__name__)


# from peal.dependencies.ddpm_inversion.ddpm_inversion import DDPMInversion


class ArtificialConfounderTabularDatasetGenerator:
    """
    Generates a tabular dataset with a confounder tnecklace is a symbolic attribute.

    Should mimic symbolic rather low-dimensional data like credit decisions based on known factors about a person.

    Should enable straightforward check for minimality of counterfactuals.
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

            values = np.random.uniform(0, 1, self.input_size)
            target = int(np.sum(values >= 0.5) > np.sum(values < 0.5))
            while not target == has_attribute:
                values = np.random.uniform(0, 1, self.input_size)
                target = int(np.sum(values >= 0.5) > np.sum(values < 0.5))

            dataset += ",".join([str(val) for val in values])

            dataset += "," + str(has_confounder) + "," + str(has_attribute) + "\n"

        with open(self.label_dir, "w") as f:
            f.write(dataset)


class ArtificialConfounderSequenceDatasetGenerator:
    """
    Generates a sequence dataset with a confounder tnecklace is a symbolic attribute.

    Should mimic sequential data like natural language.

    Should enable straightforward check for minimality of counterfactuals.
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
    """
    Two-digit MNIST subset with a red background as confounder.

    Every digit image is resized to 32x32, tiled to RGB and its red channel
    is raised to a random background intensity; ``Confounder`` is 1 when
    that intensity is >= 128. Note the confounder is drawn independently of
    the digit, so the columns are not correlated by construction. A constant
    8x8 central hint mask is written for every image.

    Parameters
    ----------
    dataset_name : str
        Output folder ``datasets/<dataset_name>``.
    mnist_dir : str
        Folder with one sub-folder of image files per digit.
    digits : list of str
        Two digit folders; ``Feature`` is 1 for the second one.
    """

    def __init__(self, dataset_name, mnist_dir="datasets/mnist", digits=["0", "8"]):
        """Store the paths; nothing is written until ``generate_dataset``."""
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join("datasets", self.dataset_name)
        self.mnist_dir = mnist_dir
        self.digits = digits

    def generate_dataset(self):
        """
        Write ``imgs/``, ``masks/`` and ``data.csv`` (ImgName, Feature,
        Confounder) into the dataset folder.

        An existing folder is moved aside to ``<dataset_dir>_old_<timestamp>``
        first.
        """
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
                    _log.info("%s", it)

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
    """
    Stamp a synthetic confounder onto an existing image dataset (e.g. CelebA).

    Reads the source label file, alternates ``Confounder`` between samples
    (even index = confounded) and writes modified images plus the original
    attributes, ``Confounder`` and ``ConfounderStrength`` to
    ``data_config.dataset_path``. Supported ``confounding`` modes:
    ``"intensity"`` (brightness shift), ``"color"`` (red/blue shift),
    ``"copyrighttag"`` (a copyright tag blended into the bottom of the image,
    with a hint mask) and ``"necklace"`` (young/old edits through DDPM
    inversion, with an ``_inverse`` twin dataset and a difference mask).

    Parameters
    ----------
    dataset_origin_path : str
        Source dataset root containing ``imgs/`` and the label file.
    dataset_name : str, optional
        Unused; the output path comes from ``data_config.dataset_path``.
    label_dir : str, optional
        Label file; defaults to ``<dataset_origin_path>/data.csv``. The
        first two lines are treated as header and skipped.
    delimiter : str
        Column delimiter of the label file.
    confounding : str or None
        Mode as above; ``None`` takes ``data_config.confounding_factors[-1]``.
    num_samples : int, optional
        Number of source samples to process (all when ``None``).
    attribute : str, optional
        Stored but unused.
    data_config : DataConfig
        Supplies ``dataset_path`` and, optionally, ``inverse`` (a previously
        generated dataset whose ``ConfounderStrength`` is negated and reused).
    """

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
        """Resolve paths and the confounding mode; read the ``inverse``
        dataset's ``ConfounderStrength`` column if ``data_config.inverse``
        is set."""
        self.dataset_origin_path = dataset_origin_path
        self.confounding = confounding

        if confounding is None:
            self.confounding = data_config.confounding_factors[-1]

        if label_dir is None:
            self.label_dir = os.path.join(dataset_origin_path, "data.csv")

        else:
            self.label_dir = label_dir

        self.delimiter = delimiter
        _log.info("%s", "self.delimiter")
        _log.info("%s", self.delimiter)
        _log.info("%s", self.delimiter)
        _log.info("%s", self.delimiter)
        self.dataset_dir = data_config.dataset_path
        self.num_samples = num_samples
        self.attribute = attribute

        if self.confounding == "necklace":
            # Imported here: the vendored module pulls in diffusers at import
            # time and only this confounder needs it.
            from peal.dependencies.ddpm_inversion.ddpm_inversion import DDPMInversion

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
        """
        Write the confounded dataset to ``dataset_dir`` (and ``_inverse``).

        The first 90 % of samples get a random confounder strength in
        ``[0, 1]``, the rest strength 1. ``data.csv`` is rewritten every
        100 samples so a partial run is usable. Existing output folders are
        moved aside with a timestamp suffix.
        """
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
        attributes = raw_data[0].split(self.delimiter)
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
        _log.info("%s", lines_out)

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
                _log.info("%s", sample_idx)
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
    """
    Build a histology dataset whose confounder is the hematoxylin staining.

    Converts the ``MUS`` and ``STR`` tissue classes of a raw NCT-CRC style
    folder to PNG, estimates the stain vectors of every image (Macenko-style
    OD/PCA estimation), measures the 99th-percentile intensity of
    hematoxylin-dominated pixels and splits at the median into weak/strong
    staining. Then 16000 samples are drawn so that class (``Cancer``: STR=1)
    and ``Confounder`` (strong staining) follow a fixed 2x2 pattern.

    Parameters
    ----------
    raw_data_dir : str
        Folder with ``MUS/`` and ``STR/`` image sub-folders.
    dataset_origin_path, delimiter, num_samples
        Stored but unused by ``generate_dataset``.
    dataset_name : str
        Output folder ``datasets/<dataset_name>``.
    """

    def __init__(
        self,
        raw_data_dir,
        dataset_origin_path="datasets",
        dataset_name="cancer_tissue_no_norm",
        delimiter=",",
        num_samples=40000,
    ):
        """Store the paths; nothing is written until ``generate_dataset``."""
        self.dataset_origin_path = dataset_origin_path
        self.dataset_name = dataset_name
        self.delimiter = delimiter
        self.dataset_dir = os.path.join("datasets", self.dataset_name)
        self.num_samples = num_samples
        self.raw_data_dir = raw_data_dir

    def generate_dataset(self):
        """
        Convert the images, estimate staining and write ``data.csv``.

        The csv columns are ``ImgPath,Cancer,Confounder,ConfounderStrength``
        with ``ConfounderStrength`` the measured hematoxylin intensity. The
        output folder must not exist yet.
        """
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
                    _log.info(
                        "%s",
                        str(idx)
                        + " / "
                        + str(
                            len(os.listdir(os.path.join(self.dataset_dir, class_name)))
                        ),
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
                _log.info("%s", sample_idx)
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
            _log.info(
                "%s",
                str(has_attribute)
                + " "
                + str(has_confounder)
                + " "
                + str(idxs[has_attribute][has_confounder]),
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
        config=None,
        dataset_name=None,
        dataset_origin_path="datasets",
        num_samples=1024,
        radius=1,
        noise_scale=0.0,
        # false_confounder_percentage=0.0,
        seed=0,
        **kwargs,
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
        if config is not None:
            self.dataset_dir = config.dataset_path
            self.num_samples = config.num_samples
            self.radius = getattr(config, "radius", radius)
            self.noise_scale = getattr(config, "noise_scale", noise_scale)
            self.seed = getattr(config, "seed", seed)
        else:
            self.num_samples = num_samples
            self.radius = radius
            self.noise_scale = noise_scale
            self.seed = seed

            if dataset_name is None:
                dataset_name = (
                    "size_"
                    + str(self.num_samples)
                    + "_"
                    + "radius_"
                    + str(round(self.radius, 1))
                    + "_"
                    + "seed_"
                    + str(self.seed)
                )
            self.dataset_name = dataset_name
            self.dataset_origin_path = dataset_origin_path
            self.dataset_dir = os.path.join(self.dataset_origin_path, self.dataset_name)

        self.label_dir = os.path.join(self.dataset_dir, "data.csv")

    def generate_dataset(self):
        """
        Generates the dataset.

        Points on a circle of ``radius`` (plus Gaussian noise) get
        ``Target = x2 > 0`` and ``Confounder = x1 > 0``; the four
        target/confounder cells are resampled to ``num_samples / 4`` rows
        each and written to ``data.csv`` with columns
        ``x1,x2,Confounder,Target``.

        Returns
        -------
        CircleDatasetGenerator
            ``self``, with the generated array stored in ``self.data``.
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
    """
    Render one 64x64 "square" image from its generative factors.

    The background is a grey level ``color_b``, a grey border square of
    intensity 127 is placed at ``(position_x, position_y)`` and its inner
    ``SIZE_INNER`` square is filled with red intensity ``color_a`` (green
    and blue 0). Gaussian noise (std 20) is added to everything.

    Parameters
    ----------
    color_a : int
        Red intensity of the inner square, 0-255.
    color_b : int
        Grey level of the background, 0-255.
    position_x, position_y : int, optional
        Top-left corner of the bordered square; centred when ``None``.
    SIZE_INNER, SIZE_BORDER : int
        Inner square side and border width in pixels.
    noise : numpy.ndarray, optional
        Noise of shape ``(64, 64, 3)`` to reuse (for an ``_inverse`` twin);
        drawn fresh when ``None``.

    Returns
    -------
    img : PIL.Image.Image
        The rendered RGB image.
    noise : numpy.ndarray
        The noise that was used.
    """
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
        + noise[
            position_x + SIZE_BORDER : position_x + SIZE_ADDED - SIZE_BORDER,
            position_y + SIZE_BORDER : position_y + SIZE_ADDED - SIZE_BORDER,
        ],
        0,
        255,
    )
    img = Image.fromarray(img.astype(dtype=np.uint8))
    return img, noise


class SquareDatasetGenerator:
    """
    Synthetic 64x64 "square" dataset with four independent binary factors.

    Sample ``i`` gets ``ClassA = i % 2`` (inner square bright/dark red),
    ``ClassB = (i // 2) % 2`` (background bright/dark), ``ClassC`` and
    ``ClassD`` (square in the lower/upper half in x and y). The continuous
    factors are stored as ``ColorA, ColorB, PositionX, PositionY``. Every
    sample also gets an ``_inverse`` twin with inverted background colour and
    the same noise, and a mask of the inner square is written for both.

    Parameters
    ----------
    data_config : DataConfig
        Supplies ``dataset_path`` and ``num_samples``.
    """

    def __init__(
        self,
        data_config,
        **kwargs,
    ):
        """Store the data config; nothing is written until
        ``generate_dataset``."""
        self.data_config = data_config

    def generate_dataset(self):
        """
        Write ``imgs/``, ``masks/`` and ``data.csv`` for the dataset and its
        ``_inverse`` twin (``<dataset_path>_inverse``).

        Existing folders are moved aside with a timestamp suffix; the csv is
        rewritten every 100 samples.
        """
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
                position_x=position_x,
                position_y=position_y,
                color_a=color_a,
                color_b=color_b,
            )
            img.save(os.path.join(self.data_config.dataset_path, "imgs", sample_name))
            img_inverse, noise = latent_to_square_image(
                position_x=position_x,
                position_y=position_y,
                color_a=color_a,
                color_b=255 - color_b,
                noise=noise,
            )
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
                _log.info("%s", sample_idx)
                open(
                    os.path.join(self.data_config.dataset_path, "data.csv"), "w"
                ).write("\n".join(lines_out))
                open(
                    os.path.join(
                        self.data_config.dataset_path + "_inverse", "data.csv"
                    ),
                    "w",
                ).write("\n".join(lines_out_inverse))

        open(os.path.join(self.data_config.dataset_path, "data.csv"), "w").write(
            "\n".join(lines_out)
        )
        open(
            os.path.join(self.data_config.dataset_path + "_inverse", "data.csv"),
            "w",
        ).write("\n".join(lines_out_inverse))


class SparseNumbersDatasetGenerator:
    """
    Generates a dataset of images with a 2x4 grid of 4-digit numbers.
    Resolution: 128x128.
    Grid: 2 columns, 4 rows.
    Numbers: 0000-9999, each appears 50 times in the dataset.
    Occupancy: Each slot has a 50% chance of being present (500,000 total numbers in 125,000 samples).
    Background: Variable red intensity (white to red).
    Foreground: Black numbers.
    """

    def __init__(self, data_config, **kwargs):
        """
        Parameters
        ----------
        data_config : DataConfig
            Supplies ``dataset_path``, ``num_samples`` (default 125000) and
            ``output_split`` (number of distinct numbers, default 10000).
        **kwargs
            ``font_path`` overrides the DejaVu Sans TrueType font.
        """
        self.data_config = data_config
        self.font_path = kwargs.get(
            "font_path", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        )

    def _draw_number_in_cell(self, draw, text, box):
        """Draw ``text`` centred in ``box`` with the largest font size that
        fits 90 % of the cell (binary search per call)."""
        # box is (x0, y0, x1, y1)
        cell_w = box[2] - box[0]
        cell_h = box[3] - box[1]

        # Binary search for optimal font size
        low = 1
        high = cell_h * 2
        best_size = 1

        while low <= high:
            mid = (low + high) // 2
            try:
                font = ImageFont.truetype(self.font_path, mid)
            except OSError:
                font = ImageFont.load_default()
                best_size = mid
                break

            if hasattr(font, "getbbox"):
                bbox = font.getbbox(text)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            else:
                w, h = draw.textsize(text, font=font)

            if w <= cell_w * 0.9 and h <= cell_h * 0.9:
                best_size = mid
                low = mid + 1
            else:
                high = mid - 1

        try:
            font = ImageFont.truetype(self.font_path, best_size)
        except OSError:
            font = ImageFont.load_default()

        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        else:
            w, h = draw.textsize(text, font=font)

        x = box[0] + (cell_w - w) / 2
        y = box[1] + (cell_h - h) / 2
        draw.text((x, y), text, fill=(0, 0, 0), font=font)

    def generate_dataset(self):
        """Generates the sparse_numbers dataset.

        Writes ``imgs/<i:06d>.png`` and ``data.csv`` with columns
        ``imgs, Red1..Red8`` (background red intensity per slot) and
        ``Num1..Num8`` (number per slot, -1 if empty) to
        ``data_config.dataset_path``; an existing folder is moved aside with
        a timestamp suffix. Uses the global ``random`` module (unseeded).
        """

        dataset_path = self.data_config.dataset_path
        _log.info("%s", f"DEBUG: dataset_path={dataset_path}")
        if os.path.exists(dataset_path):
            datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(dataset_path, dataset_path + "_old_" + datestamp)

        os.makedirs(os.path.join(dataset_path, "imgs"), exist_ok=True)

        num_samples = (
            self.data_config.num_samples if self.data_config.num_samples else 125000
        )
        _log.info("%s", f"DEBUG: num_samples={num_samples}")
        n_unique_numbers = (
            self.data_config.output_split if self.data_config.output_split else 10000
        )
        num_digits = len(str(n_unique_numbers - 1))
        # Calculate repeats to aim for 50% occupancy if num_samples changed
        # Total slots = num_samples * 8. Occupancy 50% = 4 * num_samples.
        # repeats = (4 * num_samples) // n_unique_numbers
        repeats = max(1, (4 * num_samples) // n_unique_numbers)
        total_slots = num_samples * 8
        total_number_entries = min(n_unique_numbers * repeats, total_slots)

        # Prepare all numbers
        all_numbers = []
        for n in range(n_unique_numbers):
            all_numbers.extend([n] * repeats)
        random.shuffle(all_numbers)
        all_numbers = all_numbers[:total_number_entries]

        # Assign numbers to slots
        # Total slots = 125,000 * 8 = 1,000,000
        # We need to pick 500,000 slots to be occupied.
        occupied_indices = random.sample(range(total_slots), total_number_entries)

        # Map indices to samples
        sample_slots = [{} for _ in range(num_samples)]
        for i, slot_idx in enumerate(occupied_indices):
            sample_idx = slot_idx // 8
            local_slot_idx = slot_idx % 8
            sample_slots[sample_idx][local_slot_idx] = all_numbers[i]

        # Check for duplicates within samples and fix them
        # We'll use a simple swap strategy with another random sample's slot
        for s_idx in range(num_samples):
            seen = {}
            for local_idx, val in list(sample_slots[s_idx].items()):
                if val in seen:
                    # Collision! Find another sample to swap with
                    done = False
                    while not done:
                        other_s_idx = random.randint(0, num_samples - 1)
                        if other_s_idx == s_idx:
                            continue
                        other_slots = sample_slots[other_s_idx]
                        if not other_slots:
                            continue
                        # Pick a random slot in other_sample
                        other_local_idx = random.choice(list(other_slots.keys()))
                        other_val = other_slots[other_local_idx]

                        # Check if swapping causes collision in either sample
                        if other_val not in seen and val not in set(
                            other_slots.values()
                        ):
                            # Swap
                            sample_slots[s_idx][local_idx] = other_val
                            sample_slots[other_s_idx][other_local_idx] = val
                            done = True
                else:
                    seen[val] = local_idx

        # Grid layout
        # Resolution 128x128. 2 columns, 4 rows.
        # Cell size: 60x30. Gaps: 8px horiz (between cols), 2px vert.
        # Padding: 2px around.
        cell_w, cell_h = 60, 30
        col_x = [2, 66]  # x-starts for col 0 and col 1
        row_y = [2, 34, 66, 98]  # y-starts for row 0, 1, 2, 3

        slots_coord = []
        for r in range(4):
            for c in range(2):
                slots_coord.append(
                    (col_x[c], row_y[r], col_x[c] + cell_w, row_y[r] + cell_h)
                )

        lines_out = [
            "imgs,Red1,Red2,Red3,Red4,Red5,Red6,Red7,Red8,Num1,Num2,Num3,Num4,Num5,Num6,Num7,Num8"
        ]

        for s_idx in tqdm(range(num_samples), desc="Generating samples"):
            img = Image.new("RGB", (128, 128), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            row_data = [f"{s_idx:06d}.png"]
            intensities = [random.random() for _ in range(8)]
            nums_in_sample = []

            for i in range(8):
                intensity = intensities[i]
                box = slots_coord[i]

                # Draw background: White (255,255,255) to Red (255,0,0)
                # color = (255, int(255*(1-intensity)), int(255*(1-intensity)))
                # Fill the cell
                draw.rectangle(
                    box,
                    fill=(255, int(255 * (1 - intensity)), int(255 * (1 - intensity))),
                )

                if i in sample_slots[s_idx]:
                    num = sample_slots[s_idx][i]
                    text = f"{num:0{num_digits}d}"
                    self._draw_number_in_cell(draw, text, box)
                    nums_in_sample.append(num)
                else:
                    nums_in_sample.append(-1)

            row_data.extend([f"{intt:.4f}" for intt in intensities])
            row_data.extend([str(n) for n in nums_in_sample])
            lines_out.append(",".join(row_data))

            img.save(os.path.join(dataset_path, "imgs", f"{s_idx:06d}.png"))

            if (s_idx + 1) % 1000 == 0:
                with open(os.path.join(dataset_path, "data.csv"), "w") as f:
                    f.write("\n".join(lines_out))

        with open(os.path.join(dataset_path, "data.csv"), "w") as f:
            f.write("\n".join(lines_out))

        _log.info("%s", f"Dataset generated at {dataset_path}")


class OnlySparseNumbersDatasetGenerator:
    """
    Generates a dataset of 128x128 synthetic images with a 2×4 grid of numbers.

    Dataset Properties:
    - Images: 128×128 RGB, white background, black text
    - Grid: 2 columns × 4 rows = 8 slots per image
    - Numbers: Configurable range (e.g., 000–999)
    - Uniqueness: Each number appears at most once per image

    Typical Configuration:
    - num_samples: 12,500 images (default)
    - output_split: 1,000 unique numbers (000–999, default)
    - occupancy: ~50% (approximately 4 numbers per image on average)

    CSV Output Format:
    - imgs: image filename
    - Num1–Num8: per-slot labels (-1 if empty)

    Algorithm Overview:
    1. Generate a pool of numbers with repetition to achieve target occupancy
    2. Assign numbers to sample × slot pairs, ensuring uniqueness within each image
    3. Render each image with centered, sized-to-fit text in each slot
    4. Write CSV with per-sample labels
    """

    def __init__(self, data_config, **kwargs):
        """
        Args:
            data_config: Config object with attributes
            **kwargs: optional 'font_path' (default: DejaVu Sans)
        """
        self.data_config = data_config
        self.font_path = (
            data_config.font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        )
        self.seed = getattr(data_config, "seed", 0)
        self.rng = np.random.RandomState(self.seed)

    def _compute_font_size(self, draw, max_width, max_height, sample_text, margin=0.9):
        """
        Binary search for the largest font size that fits the cell.

        Args:
            draw: PIL ImageDraw object
            max_width: maximum text width (pixels)
            max_height: maximum text height (pixels)
            sample_text: text to measure (e.g., "9999")

        Returns:
            best_size: font size in points
        """
        low, high = 1, max_height * 2
        best_size = 1

        while low <= high:
            mid = (low + high) // 2
            try:
                font = ImageFont.truetype(self.font_path, mid)
            except OSError:
                font = ImageFont.load_default()
                best_size = mid
                break

            # Get text bounding box
            if hasattr(font, "getbbox"):
                bbox = font.getbbox(sample_text)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            else:
                w, h = draw.textsize(sample_text, font=font)

            # Check if text fits with margin
            if w <= max_width * margin and h <= max_height * margin:
                best_size = mid
                low = mid + 1
            else:
                high = mid - 1

        return best_size

    def _draw_number_in_cell(self, draw, text, box, font_size):
        """
        Draw a number centered in a cell.

        Args:
            draw: PIL ImageDraw object
            text: text to draw (e.g., "042")
            box: (x0, y0, x1, y1) cell bounds
            font_size: pre-computed font size (points)

        Note: Uses anchor="mm" (middle-middle) to ensure proper centering,
        accounting for font metrics and avoiding margin issues.
        """
        cell_w = box[2] - box[0]
        cell_h = box[3] - box[1]

        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except OSError:
            font = ImageFont.load_default()

        # Calculate cell center
        center_x = box[0] + cell_w / 2
        center_y = box[1] + cell_h / 2

        # Use anchor="mm" (middle-middle) to center text at the exact cell center
        draw.text((center_x, center_y), text, fill=(0, 0, 0), font=font, anchor="mm")

    def _build_number_pool(self, num_samples, n_unique_numbers):
        """
        Build a shuffled pool of numbers with repetition to achieve ~50% occupancy.

        Strategy:
        - Total slots: num_samples × 8
        - Target occupancy: 50% (4 numbers per image on average)
        - Repeats: (4 × num_samples) // n_unique_numbers
        - Pool is randomly shuffled to distribute numbers evenly

        Args:
            num_samples: number of images
            n_unique_numbers: number of unique values

        Returns:
            Tuple of (all_numbers, num_digits) where:
            - all_numbers: shuffled pool of numbers to assign
            - num_digits: number of digits needed to format numbers
        """
        num_digits = len(str(n_unique_numbers - 1))
        repeats = max(1, (4 * num_samples) // n_unique_numbers)
        total_slots = num_samples * 8
        total_number_entries = min(n_unique_numbers * repeats, total_slots)

        # Build and shuffle pool
        all_numbers = []
        for n in range(n_unique_numbers):
            all_numbers.extend([n] * repeats)
        self.rng.shuffle(all_numbers)
        all_numbers = all_numbers[:total_number_entries]

        return all_numbers, num_digits

    def _assign_numbers_to_samples(self, num_samples, all_numbers):
        """
        Assign numbers to sample/slot pairs, ensuring each number appears at most once per image.

        Strategy:
        1. Randomly select which slots (across all samples) to occupy
        2. Assign numbers from the pool to those slots
        3. Check for duplicates within each image; if found, swap with another sample to resolve

        Args:
            num_samples: number of images
            all_numbers: shuffled pool of numbers

        Returns:
            sample_slots: list of dicts, where sample_slots[i] = {slot_idx: number}
        """
        total_slots = num_samples * 8
        total_number_entries = len(all_numbers)

        # Randomly select which slots to occupy
        occupied_indices = self.rng.choice(
            total_slots, total_number_entries, replace=False
        )

        # Map slot indices to sample/slot pairs
        sample_slots = [{} for _ in range(num_samples)]
        for i, slot_idx in enumerate(occupied_indices):
            sample_idx = slot_idx // 8
            local_slot_idx = slot_idx % 8
            sample_slots[sample_idx][local_slot_idx] = all_numbers[i]

        # Resolve collisions: if a number appears twice in the same image, swap it with another sample
        for s_idx in range(num_samples):
            seen = {}
            for local_idx, val in list(sample_slots[s_idx].items()):
                if val in seen:
                    # Collision detected; find another sample to swap with
                    done = False
                    while not done:
                        other_s_idx = self.rng.randint(0, num_samples)
                        if other_s_idx == s_idx or not sample_slots[other_s_idx]:
                            continue

                        # Pick a random slot in the other sample
                        other_local_idx = self.rng.choice(
                            list(sample_slots[other_s_idx].keys())
                        )
                        other_val = sample_slots[other_s_idx][other_local_idx]

                        # Verify swap won't create a collision in either sample
                        other_sample_vals = set(sample_slots[other_s_idx].values())
                        if other_val not in seen and val not in other_sample_vals:
                            # Perform swap
                            sample_slots[s_idx][local_idx] = other_val
                            sample_slots[other_s_idx][other_local_idx] = val
                            seen[other_val] = local_idx
                            done = True
                else:
                    seen[val] = local_idx

        return sample_slots

    def generate_dataset(self):
        """
        Generate the dataset: images, masks, and CSV labels.

        Workflow:
        1. Prepare output directory (backup existing data with timestamp)
        2. Build number pool with target occupancy
        3. Assign numbers to slots, ensuring uniqueness per image
        4. Pre-compute optimal font size for all cells
        5. For each sample:

           - Create blank 128×128 image
           - For each slot: draw number if present, else leave empty
           - Save PNG and update CSV

        6. Write final CSV
        """
        dataset_path = self.data_config.dataset_path

        # Back up existing dataset
        if os.path.exists(dataset_path):
            datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(dataset_path, dataset_path + "_old_" + datestamp)

        os.makedirs(os.path.join(dataset_path, "imgs"), exist_ok=True)

        # Load config parameters
        num_samples = (
            self.data_config.num_samples if self.data_config.num_samples else 12500
        )
        n_unique_numbers = (
            self.data_config.output_split if self.data_config.output_split else 1000
        )
        num_digits = len(str(n_unique_numbers - 1))

        confounding_factors = getattr(self.data_config, "confounding_factors", None)
        if confounding_factors and len(confounding_factors) == 2:

            def _parse_num(factor_name):
                if isinstance(factor_name, str) and factor_name.startswith("Num"):
                    return int(factor_name[3:])
                return int(factor_name)

            target_num_a = _parse_num(confounding_factors[0])
            target_num_b = _parse_num(confounding_factors[1])
            other_numbers = [
                n
                for n in range(n_unique_numbers)
                if n not in (target_num_a, target_num_b)
            ]

            group_size = num_samples // 4
            sample_slots = []
            sample_confounder_labels = []

            for g_idx in range(4):
                has_a = 1 if g_idx in (2, 3) else 0
                has_b = 1 if g_idx in (1, 3) else 0

                for _ in range(group_size):
                    k = int(self.rng.randint(3, 6))
                    chosen_slots = list(self.rng.choice(8, k, replace=False))
                    slot_dict = {}
                    slots_left = list(chosen_slots)

                    if has_a:
                        idx_a = self.rng.randint(0, len(slots_left))
                        slot_a = slots_left.pop(idx_a)
                        slot_dict[slot_a] = target_num_a

                    if has_b:
                        idx_b = self.rng.randint(0, len(slots_left))
                        slot_b = slots_left.pop(idx_b)
                        slot_dict[slot_b] = target_num_b

                    rem_count = len(slots_left)
                    if rem_count > 0:
                        fill_nums = self.rng.choice(
                            other_numbers, rem_count, replace=False
                        )
                        for s_i, num in zip(slots_left, fill_nums):
                            slot_dict[s_i] = num

                    sample_slots.append(slot_dict)
                    sample_confounder_labels.append((has_a, has_b))

            perm = self.rng.permutation(len(sample_slots))
            sample_slots = [sample_slots[p] for p in perm]
            sample_confounder_labels = [sample_confounder_labels[p] for p in perm]
            header = f"imgs,Num1,Num2,Num3,Num4,Num5,Num6,Num7,Num8,{confounding_factors[0]},{confounding_factors[1]}"
        else:
            # Build number pool
            all_numbers, num_digits = self._build_number_pool(
                num_samples, n_unique_numbers
            )
            # Assign numbers to samples (ensuring uniqueness per image)
            sample_slots = self._assign_numbers_to_samples(num_samples, all_numbers)
            sample_confounder_labels = None
            header = "imgs,Num1,Num2,Num3,Num4,Num5,Num6,Num7,Num8"

        # Grid layout: 2 columns × 4 rows in 128×128 image
        # Cell size: 60×30 pixels with 2–8 pixel padding/gaps
        cell_w, cell_h = 60, 30
        col_x = [2, 66]  # x-offsets for columns
        row_y = [2, 34, 66, 98]  # y-offsets for rows

        slots_coord = []
        for r in range(4):
            for c in range(2):
                slots_coord.append(
                    (col_x[c], row_y[r], col_x[c] + cell_w, row_y[r] + cell_h)
                )

        # Pre-compute optimal font size (constant across all cells)
        sample_img = Image.new("RGB", (128, 128), (255, 255, 255))
        sample_draw = ImageDraw.Draw(sample_img)
        sample_text = (
            f"{n_unique_numbers - 1:0{num_digits}d}"  # Largest possible number
        )
        font_size = self._compute_font_size(
            sample_draw, cell_w, cell_h, sample_text, margin=0.6
        )

        # Generate images and CSV
        lines_out = [header]

        for s_idx in tqdm(range(num_samples), desc="Generating samples"):
            img = Image.new("RGB", (128, 128), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            row_data = [f"{s_idx:06d}.png"]
            nums_in_sample = []

            for i in range(8):
                box = slots_coord[i]

                if i in sample_slots[s_idx]:
                    num = sample_slots[s_idx][i]
                    text = f"{num:0{num_digits}d}"
                    self._draw_number_in_cell(draw, text, box, font_size)
                    nums_in_sample.append(num)
                else:
                    nums_in_sample.append(-1)

            row_data.extend([str(n) for n in nums_in_sample])
            if sample_confounder_labels is not None:
                has_a, has_b = sample_confounder_labels[s_idx]
                row_data.extend([str(has_a), str(has_b)])

            lines_out.append(",".join(row_data))

            # Add Gaussian noise
            img_np = np.array(img, dtype=np.float32)
            std = self.rng.uniform(0, 15)
            noise = self.rng.normal(0, std, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255)
            img = Image.fromarray(img_np.astype(np.uint8))

            img.save(os.path.join(dataset_path, "imgs", f"{s_idx:06d}.png"))

            # Checkpoint: write CSV every 1000 samples
            if (s_idx + 1) % 1000 == 0:
                with open(os.path.join(dataset_path, "data.csv"), "w") as f:
                    f.write("\n".join(lines_out))

        # Final write
        with open(os.path.join(dataset_path, "data.csv"), "w") as f:
            f.write("\n".join(lines_out))

        _log.info(
            "%s",
            f"OnlySparseNumbersDatasetGenerator: {num_samples} samples generated at {dataset_path}",
        )


class OnlySparseNumbersZipfDatasetGenerator(OnlySparseNumbersDatasetGenerator):
    """
    Variant of OnlySparseNumbersDatasetGenerator where number frequencies follow
    Zipf's law: P(n) ∝ 1/(n+1)^s with s=1 (configurable via data_config.zipf_s).
    Lower-numbered digits appear much more frequently than higher ones.
    """

    def _build_number_pool(self, num_samples, n_unique_numbers):
        """
        Sample the number pool from a Zipf distribution instead of repeating
        every number equally.

        Parameters
        ----------
        num_samples : int
            Number of images.
        n_unique_numbers : int
            Number of distinct values.

        Returns
        -------
        all_numbers : list of int
            Shuffled pool of ``min(4 * num_samples, 8 * num_samples)`` entries
            drawn with ``P(n) ~ 1 / (n + 1) ** zipf_s``.
        num_digits : int
            Digits needed to format the largest number.
        """
        num_digits = len(str(n_unique_numbers - 1))
        total_slots = num_samples * 8
        target_entries = min(4 * num_samples, total_slots)  # ~50% occupancy

        zipf_s = (
            getattr(self.data_config, "zipf_s", 1.0)
            if hasattr(self, "data_config")
            else 1.0
        )

        # Zipf weights: w(n) = 1/(n+1)^s
        weights = np.array([1.0 / (n + 1) ** zipf_s for n in range(n_unique_numbers)])
        probs = weights / weights.sum()

        # Sample numbers according to Zipf distribution
        all_numbers = self.rng.choice(
            n_unique_numbers, size=target_entries, replace=True, p=probs
        ).tolist()
        self.rng.shuffle(all_numbers)

        return all_numbers, num_digits


class SparseNumbersZipfDatasetGenerator(SparseNumbersDatasetGenerator):
    """
    Variant of SparseNumbersDatasetGenerator where sparse number frequencies follow
    Zipf's law: P(n) ∝ 1/(n+1)^s with s=1.
    The 8 dense red-intensity background variables remain unchanged.
    """

    def generate_dataset(self):
        """Generates the sparse_numbers dataset with Zipf-distributed numbers."""
        dataset_path = self.data_config.dataset_path
        _log.info("%s", f"DEBUG: dataset_path={dataset_path}")
        if os.path.exists(dataset_path):
            datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(dataset_path, dataset_path + "_old_" + datestamp)

        os.makedirs(os.path.join(dataset_path, "imgs"), exist_ok=True)

        num_samples = (
            self.data_config.num_samples if self.data_config.num_samples else 125000
        )
        n_unique_numbers = (
            self.data_config.output_split if self.data_config.output_split else 10000
        )
        num_digits = len(str(n_unique_numbers - 1))
        total_slots = num_samples * 8
        target_entries = min(4 * num_samples, total_slots)

        zipf_s = getattr(self.data_config, "zipf_s", 1.0)
        rng = np.random.RandomState(getattr(self.data_config, "seed", 0))

        # Zipf-distributed number pool
        weights = np.array([1.0 / (n + 1) ** zipf_s for n in range(n_unique_numbers)])
        probs = weights / weights.sum()

        # Confounded variant, mirroring OnlySparseNumbersDatasetGenerator: four equal
        # groups over (has_a, has_b) so that the loader's confounder_probability
        # quota (process_confounder_data_controlled) can always be filled, with the
        # two flag columns appended to the csv under the factor names. The remaining
        # slots are filled Zipf-weighted rather than uniformly so the background
        # number distribution matches the unconfounded zipf dataset.
        confounding_factors = getattr(self.data_config, "confounding_factors", None)
        if confounding_factors and len(confounding_factors) == 2:

            def _parse_num(factor_name):
                if isinstance(factor_name, str) and factor_name.startswith("Num"):
                    return int(factor_name[3:])
                return int(factor_name)

            target_num_a = _parse_num(confounding_factors[0])
            target_num_b = _parse_num(confounding_factors[1])
            other_numbers = np.array(
                [
                    n
                    for n in range(n_unique_numbers)
                    if n not in (target_num_a, target_num_b)
                ]
            )
            other_probs = probs[other_numbers] / probs[other_numbers].sum()

            group_size = num_samples // 4
            sample_slots = []
            sample_confounder_labels = []
            for g_idx in range(4):
                has_a = 1 if g_idx in (2, 3) else 0
                has_b = 1 if g_idx in (1, 3) else 0
                for _ in range(group_size):
                    k = int(rng.randint(3, 6))
                    slots_left = list(rng.choice(8, k, replace=False))
                    slot_dict = {}
                    if has_a:
                        slot_dict[slots_left.pop(rng.randint(0, len(slots_left)))] = (
                            target_num_a
                        )
                    if has_b:
                        slot_dict[slots_left.pop(rng.randint(0, len(slots_left)))] = (
                            target_num_b
                        )
                    if slots_left:
                        fill_nums = rng.choice(
                            other_numbers, len(slots_left), replace=False, p=other_probs
                        )
                        for s_i, num in zip(slots_left, fill_nums):
                            slot_dict[s_i] = int(num)
                    sample_slots.append(slot_dict)
                    sample_confounder_labels.append((has_a, has_b))

            perm = rng.permutation(len(sample_slots))
            sample_slots = [sample_slots[i] for i in perm]
            sample_confounder_labels = [sample_confounder_labels[i] for i in perm]
            num_samples = len(sample_slots)
            header = (
                "imgs,Red1,Red2,Red3,Red4,Red5,Red6,Red7,Red8,Num1,Num2,Num3,Num4,Num5,Num6,Num7,Num8,"
                f"{confounding_factors[0]},{confounding_factors[1]}"
            )
        else:
            all_numbers = rng.choice(
                n_unique_numbers, size=target_entries, replace=True, p=probs
            ).tolist()
            random.shuffle(all_numbers)

            # Assign numbers to slots
            occupied_indices = random.sample(range(total_slots), target_entries)
            sample_slots = [{} for _ in range(num_samples)]
            for i, slot_idx in enumerate(occupied_indices):
                sample_idx = slot_idx // 8
                local_slot_idx = slot_idx % 8
                sample_slots[sample_idx][local_slot_idx] = all_numbers[i]

            # Resolve within-sample duplicates
            for s_idx in range(num_samples):
                seen = {}
                for local_idx, val in list(sample_slots[s_idx].items()):
                    if val in seen:
                        done = False
                        while not done:
                            other_s_idx = random.randint(0, num_samples - 1)
                            if other_s_idx == s_idx:
                                continue
                            other_slots = sample_slots[other_s_idx]
                            if not other_slots:
                                continue
                            other_local_idx = random.choice(list(other_slots.keys()))
                            other_val = other_slots[other_local_idx]
                            if other_val not in seen and val not in set(
                                other_slots.values()
                            ):
                                sample_slots[s_idx][local_idx] = other_val
                                sample_slots[other_s_idx][other_local_idx] = val
                                done = True
                    else:
                        seen[val] = local_idx
            sample_confounder_labels = None
            header = "imgs,Red1,Red2,Red3,Red4,Red5,Red6,Red7,Red8,Num1,Num2,Num3,Num4,Num5,Num6,Num7,Num8"

        # Grid layout (same as SparseNumbersDatasetGenerator)
        cell_w, cell_h = 60, 30
        col_x = [2, 66]
        row_y = [2, 34, 66, 98]
        slots_coord = []
        for r in range(4):
            for c in range(2):
                slots_coord.append(
                    (col_x[c], row_y[r], col_x[c] + cell_w, row_y[r] + cell_h)
                )

        lines_out = [header]

        for s_idx in tqdm(range(num_samples), desc="Generating Zipf samples"):
            img = Image.new("RGB", (128, 128), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            row_data = [f"{s_idx:06d}.png"]
            intensities = [random.random() for _ in range(8)]
            nums_in_sample = []

            for i in range(8):
                intensity = intensities[i]
                box = slots_coord[i]
                draw.rectangle(
                    box,
                    fill=(255, int(255 * (1 - intensity)), int(255 * (1 - intensity))),
                )

                if i in sample_slots[s_idx]:
                    num = sample_slots[s_idx][i]
                    text = f"{num:0{num_digits}d}"
                    self._draw_number_in_cell(draw, text, box)
                    nums_in_sample.append(num)
                else:
                    nums_in_sample.append(-1)

            row_data.extend([f"{intt:.4f}" for intt in intensities])
            row_data.extend([str(n) for n in nums_in_sample])
            if sample_confounder_labels is not None:
                has_a, has_b = sample_confounder_labels[s_idx]
                row_data.extend([str(has_a), str(has_b)])
            lines_out.append(",".join(row_data))

            img.save(os.path.join(dataset_path, "imgs", f"{s_idx:06d}.png"))

            if (s_idx + 1) % 1000 == 0:
                with open(os.path.join(dataset_path, "data.csv"), "w") as f:
                    f.write("\n".join(lines_out))

        with open(os.path.join(dataset_path, "data.csv"), "w") as f:
            f.write("\n".join(lines_out))

        _log.info(
            "%s",
            f"SparseNumbersZipfDatasetGenerator: {num_samples} samples generated at {dataset_path}",
        )


class FunnyNodulesDatasetGenerator:
    """
    Generates a PEAL-compatible FunnyNodules dataset.

    The FunnyNodules dataset consists of synthetic medical nodule images with six attributes:
    - roundness (1-5): round to oval
    - spiculation (1-5): none to marked
    - edge_sharpness (1-5): sharp to soft
    - size (1-5): small to big
    - intensity (1-5): dark to bright
    - internal_structure (0-1): absent or present

    For the PEAL confounding experiment:
    - True feature (target): internal_structure (binary: 0 or 1)
    - Confounder: roundness (correlated with internal_structure)

    The confounding is controlled by confounder_probability from the data config.
    """

    def __init__(self, data_config, **kwargs):
        """Store the data config (``dataset_path``, ``num_samples``,
        ``seed``, ``confounding_factors``)."""
        self.data_config = data_config

    def generate_dataset(self):
        """
        Render the nodules and write ``imgs/``, ``masks/`` and ``data.csv``.

        Sample ``i`` gets binary labels for every name in
        ``data_config.confounding_factors`` (default
        ``["InternalStructure", "Roundness"]``) following a big-endian
        counter over the sample index, so all label combinations are equally
        frequent. Label 1 maps to a nodule parameter in {4, 5} and label 0
        to {1, 2} (``internal_structure`` uses the label directly); the
        remaining parameters are random. Images are 64x64 grayscale tiled to
        RGB; ``data.csv`` has columns ``Name`` plus the factor names.
        """
        from peal.dependencies.FunnyNodules.dataset.dataset_generator import (
            generate_nodule,
        )

        dataset_path = self.data_config.dataset_path
        if os.path.exists(dataset_path):
            datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(dataset_path, dataset_path + "_old_" + datestamp)

        os.makedirs(os.path.join(dataset_path, "imgs"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "masks"), exist_ok=True)

        num_samples = self.data_config.num_samples
        confounding_factors = getattr(
            self.data_config, "confounding_factors", ["InternalStructure", "Roundness"]
        )

        rng = np.random.RandomState(
            self.data_config.seed if self.data_config.seed else 0
        )

        header = "Name," + ",".join(confounding_factors)
        lines_out = [header]

        img_size = 64  # Default image size

        # Mapping from config names to FunnyNodules internal parameter names
        name_map = {
            "InternalStructure": "internal_structure",
            "Roundness": "roundness",
            "Spiculation": "spiculation",
            "EdgeSharpness": "edge_sharpness",
            "Size": "size_attr",
            "Intensity": "intensity",
        }

        for sample_idx in range(num_samples):
            # Deterministically assign labels to span all factors in a balanced way.
            # We want the last factor to toggle fastest (every 1 sample), following a big-endian binary pattern.
            labels = {}
            for i, factor_name in enumerate(confounding_factors):
                effective_bit = len(confounding_factors) - 1 - i
                labels[factor_name] = (sample_idx // (2**effective_bit)) % 2

            # Default values (random 1-5 or 0-1) for any attribute not in confounding_factors
            params = {
                "roundness": rng.randint(1, 6),
                "spiculation": rng.randint(1, 6),
                "edge_sharpness": rng.randint(1, 6),
                "size_attr": rng.randint(1, 6),
                "intensity": rng.randint(1, 6),
                "internal_structure": rng.randint(0, 2),
            }

            # Override parameters with deterministic high/low values for factors in the list
            for factor_name in confounding_factors:
                label = labels[factor_name]
                param_name = name_map.get(factor_name, factor_name.lower())

                if param_name == "internal_structure":
                    params[param_name] = label
                else:
                    # Pick values from {1, 2} for label 0 and {4, 5} for label 1
                    if label == 1:
                        params[param_name] = rng.randint(4, 6)
                    else:
                        params[param_name] = rng.randint(1, 3)

            # Generate the nodule image
            img_np, mask_np, _ = generate_nodule(
                size_px=img_size,
                roundness=params["roundness"],
                spiculation=params["spiculation"],
                edge_sharpness=params["edge_sharpness"],
                size_attr=params["size_attr"],
                intensity=params["intensity"],
                internal_structure=params["internal_structure"],
                seed=rng.randint(0, 2**31),
            )

            # Convert grayscale to 3-channel RGB so it works with standard architectures
            img_rgb = np.stack([img_np, img_np, img_np], axis=-1)
            img_pil = Image.fromarray(img_rgb)

            sample_name = embed_numberstring(sample_idx, 8) + ".png"
            img_pil.save(os.path.join(dataset_path, "imgs", sample_name))

            # Save mask
            mask_rgb = np.stack(
                [mask_np * 255, mask_np * 255, mask_np * 255], axis=-1
            ).astype(np.uint8)
            mask_pil = Image.fromarray(mask_rgb)
            mask_pil.save(os.path.join(dataset_path, "masks", sample_name))

            # Save the binary labels (0/1) for all confounding factors
            # This ensures consistency and fulfills the requirement of matching groups.
            attributes = [sample_name] + [str(labels[f]) for f in confounding_factors]
            lines_out.append(",".join(attributes))

            if (sample_idx + 1) % 100 == 0:
                _log.info(
                    "%s",
                    f"Generated {sample_idx + 1}/{num_samples} FunnyNodules samples",
                )
                with open(os.path.join(dataset_path, "data.csv"), "w") as f:
                    f.write("\n".join(lines_out))

        with open(os.path.join(dataset_path, "data.csv"), "w") as f:
            f.write("\n".join(lines_out))
        _log.info("%s", f"FunnyNodules dataset generated at {dataset_path}")
