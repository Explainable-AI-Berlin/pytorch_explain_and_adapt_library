import random
import os
import json
import shutil
import random
import numpy as np

from pathlib import Path
from PIL import Image
from peal.utils import get_project_resource_dir, embed_numberstring


class ArtificialConfounderTabularDatasetGenerator:
    """
    Generates a tabular dataset with a confounder that is a symbolic attribute.

    Should mimic symbolic rather low-dimensional data like credit decisions based on known factors about a person.

    Should enable straight foward check for minimality of counterfactuals.
    """

    def __init__(
        self,
        dataset_name,
        base_dataset_dir="datasets",
        num_samples=1000,
        input_size=10,
        label_noise=0.0,
        seed=0,
    ):
        """
        Generates a tabular dataset with a confounder that is a symbolic attribute.

        Args:
                dataset_name (str): The name of the dataset.
                base_dataset_dir (Path): The path to the directory where the datasets are stored.
                num_samples (int, optional): The number of samples in the dataset. Defaults to 1000.
                input_size (int, optional): The number of features in the dataset. Defaults to 10.
                label_noise (float, optional): The probability that the label is flipped. Defaults to 0.0.
                seed (int, optional): The seed for the random number generator. Defaults to 0.
        """
        self.base_dataset_dir = base_dataset_dir
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(self.base_dataset_dir, self.dataset_name)
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
    Generates a sequence dataset with a confounder that is a symbolic attribute.

    Should mimic sequential data like natural language.

    Should enable straight foward check for minimality of counterfactuals.
    """

    def __init__(
        self,
        dataset_name,
        base_dataset_dir="datasets",
        num_samples=1000,
        input_size=[10, 10],
        label_noise=0.01,
        seed=0,
    ):
        """
        Generates a tabular dataset with a confounder that is a symbolic attribute.

        Args:
                dataset_name (str): The name of the dataset.
                base_dataset_dir (Path): The path to the directory where the datasets are stored.
                num_samples (int, optional): The number of samples in the dataset. Defaults to 1000.
                input_size (list, optional): The size of the input sequence. Defaults to [10, 10].
                label_noise (float, optional): The probability that the label is flipped. Defaults to 0.01.
                seed (int, optional): The seed for the random number generator. Defaults to 0.
        """
        self.base_dataset_dir = base_dataset_dir
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(self.base_dataset_dir, self.dataset_name)
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
        The confounder is the last token in the sequence that gives potentially spurious information about the number
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
        shutil.rmtree(self.dataset_dir, ignore_errors=True)
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
        base_dataset_dir,
        dataset_name=None,
        label_dir=None,
        delimiter=",",
        confounder_type="intensity",
        num_samples=40000,
    ):
        """ """
        self.base_dataset_dir = base_dataset_dir
        self.confounder_type = confounder_type
        if dataset_name is None:
            self.dataset_name = (
                os.path.split(base_dataset_dir)[-1] + "_" + self.confounder_type
            )

        else:
            self.dataset_name = dataset_name

        if label_dir is None:
            self.label_dir = os.path.join(base_dataset_dir, "data.csv")

        else:
            self.label_dir = label_dir

        self.delimiter = delimiter
        self.dataset_dir = os.path.join("datasets", self.dataset_name)
        self.num_samples = num_samples

    def generate_dataset(self):
        """ """
        shutil.rmtree(self.dataset_dir, ignore_errors=True)
        os.makedirs(self.dataset_dir)
        os.makedirs(os.path.join(self.dataset_dir, "imgs"))
        if self.confounder_type == "copyrighttag":
            os.makedirs(os.path.join(self.dataset_dir, "masks"))

        raw_data = open(self.label_dir, "r").read().split("\n")
        attributes = raw_data[1].split(self.delimiter)
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

        num_has_confounder = 0
        lines_out = ["ImgPath," + ",".join(attributes)]

        if self.confounder_type == "copyrighttag":
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

        attribute_vs_no_attribute_idxs = np.zeros([2], dtype=np.int32)
        for sample_idx in range(self.num_samples):
            if sample_idx % 100 == 0:
                print(sample_idx)
                open(os.path.join(self.dataset_dir, "data.csv"), "w").write(
                    "\n".join(lines_out)
                )

            has_attribute = int(sample_idx % 4 == 0 or sample_idx % 4 == 1)
            has_confounder = bool(sample_idx % 2 == 0)

            while (
                not data[attribute_vs_no_attribute_idxs[has_attribute]][
                    attributes.index("Blond_Hair")
                ]
                == has_attribute
            ):
                attribute_vs_no_attribute_idxs[has_attribute] += 1

            name = instance_names[attribute_vs_no_attribute_idxs[has_attribute]]
            img = Image.open(os.path.join(self.base_dataset_dir, name))
            sample = data[attribute_vs_no_attribute_idxs[has_attribute]]

            if sample_idx < 0.9 * self.num_samples:
                confounder_intensity = random.uniform(0, 1)

            else:
                confounder_intensity = 1.0

            if self.confounder_type == "intensity":
                intensity_change = (
                    64 * confounder_intensity * (2 * int(has_confounder) - 1)
                )
                img = np.array(img)
                img = (img + intensity_change + 64) * (255 / (255 + 2 * 64))
                img_out = Image.fromarray(np.array(img, dtype=np.uint8))

            elif self.confounder_type == "color":
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

            elif self.confounder_type == "copyrighttag":
                img_copyrighttag = np.maximum(np.array(img), copyright_tag_bg)
                img_copyrighttag = np.minimum(np.array(img_copyrighttag), copyright_tag)
                alpha = 0.5 + 0.5 * confounder_intensity * (2 * int(has_confounder) - 1)
                img = alpha * img_copyrighttag + (1 - alpha) * np.array(img)
                img_out = Image.fromarray(np.array(img, dtype=np.uint8))

            img_out.save(os.path.join(self.dataset_dir, "imgs", name))
            if self.confounder_type == "copyrighttag":
                mask = Image.fromarray(
                    np.array(
                        np.abs(np.array(copyright_tag_bg, dtype=np.float32) / 255 - 1)
                        * 255,
                        dtype=np.uint8,
                    )
                )
                mask.save(os.path.join(self.dataset_dir, "masks", name))

            sample.append(has_confounder)
            sample.append(confounder_intensity)
            lines_out.append(
                name + "," + ",".join(list(map(lambda x: str(float(x)), sample)))
            )
            attribute_vs_no_attribute_idxs[has_attribute] += 1

        open(os.path.join(self.dataset_dir, "data.csv"), "w").write(
            "\n".join(lines_out)
        )


class StainingConfounderGenerator:
    """ """

    def __init__(
        self,
        raw_data_dir,
        base_dataset_dir="datasets",
        dataset_name="cancer_tissue_no_norm",
        delimiter=",",
        num_samples=40000,
    ):
        """ """
        self.base_dataset_dir = base_dataset_dir
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

                T_hat = np.dot(OD, eigenVectors)

                phi = np.arctan2(T_hat[:, 1], T_hat[:, 0])
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
