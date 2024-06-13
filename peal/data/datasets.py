import torch
import random
import os
import copy
import numpy as np
import matplotlib
import torchvision
import torchmetrics

from torchvision.transforms import ToTensor
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Union

from peal.data.dataset_interfaces import PealDataset
from peal.data.dataset_utils import parse_json, parse_csv
from peal.global_utils import embed_numberstring
from peal.generators.interfaces import Generator

matplotlib.use("Agg")


class SequenceDataset(PealDataset):
    """Sequence dataset."""

    def __init__(
        self,
        data_dir,
        mode,
        config,
        transform=ToTensor(),
        task_config=None,
        **args,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir (Path): The path to the data directory.
            mode (str): The mode of the dataset. Can be "train", "val", "test" or "all".
            config (obj): The data config object.
            transform (torchvision.transform, optional): The transform to apply to the data. Defaults to ToTensor().
            task_config (obj, optional): The task config object. Defaults to None.
        """
        self.config = config
        self.transform = transform
        self.task_config = task_config
        self.data, self.keys = parse_json(data_dir, config, mode)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        name = self.keys[idx]

        data = torch.tensor(self.data[name], dtype=torch.int64)

        x = data[:-1]
        # pad with EOS tokens
        x = torch.cat(
            [
                x,
                self.config.input_size[1]
                * torch.ones(
                    [self.config.input_size[0] - x.shape[0]], dtype=torch.int64
                ),
            ]
        )
        y = data[-1]

        return x, y

    def generate_contrastive_collage(x_in, counterfactual):
        # TODO
        return torch.zeros([x_in.shape[0], 3, 64, 64]), torch.zeros_like(x_in)

    def serialize_dataset(output_dir, x_list, y_list, sample_names=None):
        # TODO implement this!
        pass


class SymbolicDataset(PealDataset):
    """Symbolic dataset."""

    def __init__(
        self,
        data_dir,
        mode,
        config,
        transform=ToTensor(),
        task_config=None,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir (Path): The path to the data directory.
            mode (str): The mode of the dataset. Can be "train", "val", "test" or "all".
            config (obj): The data config object.
            transform (torchvision.transform, optional): The transform to apply to the data. Defaults to ToTensor().
            task_config (obj, optional): The task config object. Defaults to None.
        """
        self.config = config
        self.transform = transform
        self.task_config = task_config
        if data_dir[-4:] != ".csv":
            data_dir = data_dir + ".csv"

        self.attributes, self.data, self.keys = parse_csv(
            data_dir=data_dir,
            config=config,
            mode=mode,
            set_negative_to_zero=config.set_negative_to_zero,
        )

    def __len__(self):
        return len(self.keys)

    @property
    def output_size(self):
        if self.task_config is not None and self.task_config.y_selection is not None:
            return self.task_config.output_channels

        else:
            return self.config.output_size

    def __getitem__(self, idx):
        name = self.keys[idx]

        data = self.data[name].clone().detach().to(torch.float32)

        if (
            not self.task_config is None
            and not self.task_config.x_selection is None
            and not len(self.task_config.x_selection) == 0
        ):
            x = torch.zeros([len(self.task_config.x_selection)], dtype=torch.float32)
            for idx, selection in enumerate(self.task_config.x_selection):
                x[idx] = data[self.attributes.index(selection)]

        else:
            x = data[:-1].to(torch.float32)

        if (
            not self.task_config is None
            and not self.task_config.y_selection is None
            and not self.task_config.y_selection is None
        ):
            y = torch.zeros([len(self.task_config.y_selection)])
            for idx, selection in enumerate(self.task_config.y_selection):
                y[idx] = data[self.attributes.index(selection)]

            if y.shape[0] == 1:
                y = y[0]

        else:
            y = data[-1]

        return x, y

    def generate_contrastive_collage(
        self,
        x_list,
        x_counterfactual_list,
        y_target_list,
        y_source_list,
        target_confidence_goal,
        base_path,
        start_idx,
        classifier=None,
        **args,
    ):
        # TODO
        collage_paths = [
            os.path.join(base_path, embed_numberstring(str(start_idx + i)))
            for i in range(len(x_list))
        ]
        for path in collage_paths:
            Path(path).mkdir(parents=True, exist_ok=True)

        return x_list, collage_paths

    def serialize_dataset(self, output_dir, x_list, y_list, sample_names=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        x = torch.stack(x_list, dim=0)
        y = torch.stack([torch.tensor([y]) for y in y_list], dim=0)
        data = torch.cat([x, y], dim=1)
        if (
            not self.task_config is None
            and not self.task_config.x_selection is None
            and not len(self.task_config.x_selection) == 0
        ):
            features = copy.deepcopy(self.task_config.x_selection)

        else:
            features = copy.deepcopy(self.attributes[:-1])

        if (
            not self.task_config is None
            and not self.task_config.y_selection is None
            and not self.task_config.y_selection is None
        ):
            targets = copy.deepcopy(self.task_config.y_selection)

        else:
            targets = copy.deepcopy(self.attributes[-1:])

        np.savetxt(
            output_dir + ".csv",
            data.numpy(),
            delimiter=",",
            header=",".join(features + targets),
            comments="",
        )


class ImageDataset(PealDataset):
    def generate_contrastive_collage(
        self,
        x_list: list,
        x_counterfactual_list: list,
        y_target_list: list,
        y_source_list: list,
        y_list: list,
        y_target_start_confidence_list: list,
        y_target_end_confidence_list: list,
        base_path: str,
        start_idx: int = 0,
        y_original_teacher_list=None,
        y_counterfactual_teacher_list=None,
        feedback_list=None,
        **kwargs: dict,
    ) -> tuple:
        Path(base_path).mkdir(parents=True, exist_ok=True)
        collage_paths = []
        heatmap_list = []
        for i in range(len(x_list)):
            x = self.project_to_pytorch_default(x_list[i])
            counterfactual = self.project_to_pytorch_default(x_counterfactual_list[i])

            heatmap_red = torch.maximum(
                torch.tensor(0.0),
                torch.sum(x, dim=0) - torch.sum(counterfactual, dim=0),
            )
            heatmap_blue = torch.maximum(
                torch.tensor(0.0),
                torch.sum(counterfactual, dim=0) - torch.sum(x, dim=0),
            )
            if counterfactual.shape[0] == 3:
                heatmap_green = torch.abs(x[0] - counterfactual[0])
                heatmap_green = heatmap_green + torch.abs(x[1] - counterfactual[1])
                heatmap_green = heatmap_green + torch.abs(x[2] - counterfactual[2])
                heatmap_green = heatmap_green - heatmap_red - heatmap_blue
                x_in = torch.clone(x)
                counterfactual_rgb = torch.clone(counterfactual)

            else:
                heatmap_green = torch.zeros_like(heatmap_red)
                x_in = torch.tile(x, [3, 1, 1])
                counterfactual_rgb = torch.tile(torch.clone(counterfactual), [3, 1, 1])

            heatmap = torch.stack([heatmap_red, heatmap_green, heatmap_blue], dim=0)
            if torch.abs(heatmap.sum() - torch.abs(x - counterfactual).sum()) > 0.1:
                print(
                    "Error: Heatmap does not add up to absolute counterfactual difference."
                )

            heatmap_high_contrast = torch.clamp(heatmap / heatmap.max(), 0.0, 1.0)
            current_collage = torch.cat(
                [x_in, counterfactual_rgb, heatmap_high_contrast], -1
            )
            current_collage = torchvision.utils.make_grid(current_collage, nrow=3)
            plt.gcf()
            plt.imshow(current_collage.permute(1, 2, 0))
            title_string = (
                str(int(y_list[i]))
                + " -> "
                + str(int(y_source_list[i]))
                + " -> "
                + str(int(y_target_list[i]))
            )
            title_string += (
                ", Target: "
                + str(
                    round(
                        float(y_target_start_confidence_list[i]),
                        2,
                    )
                )
                + " -> "
            )
            title_string += str(round(float(y_target_end_confidence_list[i]), 2))
            if not feedback_list is None:
                title_string += (
                    ", Teacher: "
                    + str(int(y_original_teacher_list[i]))
                    + " -> "
                    + str(int(y_counterfactual_teacher_list[i]))
                    + " -> "
                    + str(feedback_list[i])
                )

            plt.title(title_string)
            collage_path = os.path.join(
                base_path,
                embed_numberstring(str(start_idx + i)) + "_collage.png",
            )
            plt.axis("off")
            plt.savefig(collage_path)
            print("Saved collage to " + collage_path)
            collage_paths.append(collage_path)
            heatmap_list.append(heatmap)

        return heatmap_list, collage_paths

    def serialize_dataset(self, output_dir, x_list, y_list, sample_names=None, classifier=None):
        # TODO this does not seem very clean
        for class_name in range(max(2, self.output_size)):
            Path(os.path.join(output_dir, "imgs", str(class_name))).mkdir(
                parents=True, exist_ok=True
            )

        data = []
        for idx, x in enumerate(x_list):
            class_name = int(y_list[idx])
            x = self.project_to_pytorch_default(x)
            img = Image.fromarray(
                np.array(255 * x.cpu().numpy().transpose(1, 2, 0), dtype=np.uint8)
            )
            img_name = os.path.join(str(class_name), sample_names[idx] + ".png")
            img.save(os.path.join(output_dir, "imgs", img_name))
            data.append(
                [
                    img_name,
                    class_name,
                ]
            )

        data = "ImgPath,Class\n" + "\n".join([",".join(map(str, x)) for x in data])
        with open(os.path.join(output_dir, "data.csv"), "w") as f:
            f.write(data)

    def track_generator_performance(
        self,
        generator: Union[Generator, torch.Tensor],
        batch_size=None,
        num_samples=None,
    ):
        """
        This function tracks the performance of the generator

        Args:
            generator (Generator): The generator
        """
        if batch_size is None:
            if hasattr(generator, "config"):
                batch_size = generator.config.training.val_batch_size

            else:
                batch_size = 1

        if num_samples is None:
            num_samples = batch_size

        if isinstance(generator, torch.Tensor):
            generated_images = torch.clone(generator)

        elif isinstance(generator, Generator):
            # TODO set device
            generated_images = generator.sample_x(batch_size=batch_size).detach()
            while generated_images.shape[0] < num_samples:
                generated_images = torch.cat(
                    [generated_images, generator.sample_x(batch_size=batch_size)], dim=0
                )

        else:
            raise NotImplementedError("Generator type not supported")

        generated_images = generated_images[:num_samples]
        if generated_images.shape[0] == 1:
            generated_images = torch.cat([generated_images, generated_images], dim=0)

        if not hasattr(self, "fid"):
            self.fid = torchmetrics.image.fid.FrechetInceptionDistance(
                feature=192, reset_real_features=False
            )
            self.fid.to(generated_images.device)
            real_images = []
            for i in range(min(len(self), 100)):
                real_images.append(self[i][0])

            real_images = torch.stack(real_images, dim=0).to(generated_images.device)
            if hasattr(generator, "config"):
                real_images = torchvision.transforms.Resize(
                    generator.config.data.input_size[1:]
                )(real_images)

            self.fid.update(
                torch.tensor(255 * real_images, dtype=torch.uint8), real=True
            )

        self.fid.update(
            torch.tensor(255 * generated_images, dtype=torch.uint8), real=False
        )
        fid_score = float(self.fid.compute())

        return {"fid": fid_score}

    def _initialize_performance_metrics(self):
        # self.lpips = torchmetrics.image.lpips.LPIPS(net="vgg", spatial=False).to('cuda')
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
            net_type="vgg"
        ).to("cuda")
        self.fid = torchmetrics.image.fid.FrechetInceptionDistance(
            feature=192, reset_real_features=False
        )
        real_images = [self[i][0] for i in range(100)]
        self.fid = self.fid.to("cuda")
        self.fid.update(
            torch.tensor(255 * torch.stack(real_images, dim=0), dtype=torch.uint8).to(
                "cuda"
            ),
            real=True,
        )

    def distribution_distance(self, x_list):
        fids = []
        for i in range(len(x_list)):
            self.fid.update(
                torch.tensor(
                    255 * torch.stack(x_list[i], dim=0).to("cuda"), dtype=torch.uint8
                ),
                real=False,
            )
            fid_score = float(self.fid.compute())
            fids.append(fid_score)

        return np.mean(fids)

    def pair_wise_distance(self, x1, x2):
        """
        # TODO how does this actually work?
        distances = []
        for i in range(len(x1)):
            distance = (
                self.lpips.forward(
                    torch.stack(x1[i], dim=0).to("cuda"),
                    torch.stack(x2[i], dim=0).to("cuda"),
                )
                .squeeze(1, 2, 3)
                .detach()
                .cpu()
                .numpy()
                .mean()
            )
            distances.append(distance)

        return np.mean(distances)"""
        return 0.0

    def variance(self, x_list):
        variances = []
        for i in range(len(x_list[0])):
            variance = torch.mean(
                torch.var(
                    torch.stack([x_list[j][i] for j in range(len(x_list))], dim=0),
                    dim=0,
                )
            )
            variances.append(variance)

        return np.mean(variances)

    def flip_rate(self, y_confidence_list):
        flip_rates = []
        for i in range(len(y_confidence_list)):
            flip_rate = torch.mean((torch.stack(y_confidence_list[i]) > 0.5).float())
            flip_rates.append(flip_rate)

        return np.mean(flip_rates)


class Image2MixedDataset(ImageDataset):
    """
    This class is used to load a dataset with images and other data.

    Args:
        ImageDataset (_type_): _description_
    """

    def __init__(
        self,
        config,
        mode,
        root_dir=None,
        data_dir=None,
        transform=ToTensor(),
        task_config=None,
        return_dict=False,
    ):
        """
        This class is used to load a dataset with images and other data.

        Args:
            root_dir (_type_): _description_
            mode (_type_): _description_
            config (_type_): _description_
            transform (_type_, optional): _description_. Defaults to ToTensor().
            task_config (_type_, optional): _description_. Defaults to None.
        """
        self.config = config
        if root_dir is None:
            self.root_dir = config.dataset_path

        else:
            self.root_dir = root_dir

        self.transform = transform
        self.task_config = task_config
        self.hints_enabled = False
        self.groups_enabled = False
        self.idx_enabled = False
        self.url_enabled = False
        self.string_description_enabled = False
        self.tokenizer = None
        self.return_dict = return_dict
        # TODO
        # self.config.class_ratios = None
        if data_dir is None:
            data_dir = os.path.join(self.root_dir, "data.csv")

        if not config.delimiter is None:
            delimiter = config.delimiter

        else:
            delimiter = ","

        self.data_dir = data_dir
        self.attributes, self.data, self.keys = parse_csv(
            data_dir, config, mode, key_type="name", delimiter=delimiter
        )
        self.attributes_positive = []
        self.attributes_negative = []
        for attribute in self.attributes:
            attribute_values = attribute.split('_vs_')
            if len(attribute_values) == 2:
                self.attributes_positive.append(attribute_values[0])
                self.attributes_negative.append(attribute_values[1])

            else:
                self.attributes_positive.append("Is " + attribute)
                self.attributes_negative.append("Not " + attribute)

        self.task_specific_keys = None
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_keys is None
        ):
            self.set_task_specific_keys()

    @property
    def output_size(self):
        if self.task_config is not None and self.task_config.y_selection is not None:
            return self.task_config.output_channels

        else:
            return len(self.attributes)

    def __len__(self):
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_keys is None
        ):
            self.set_task_specific_keys()

        return len(self.keys)

    def enable_hints(self):
        self.hints_enabled = True

    def disable_hints(self):
        self.hints_enabled = False

    def enable_groups(self):
        self.groups_enabled = True

    def disable_groups(self):
        self.groups_enabled = False

    def enable_idx(self):
        self.idx_enabled = True

    def disable_idx(self):
        self.idx_enabled = False

    def enable_url(self):
        self.url_enabled = True

    def disable_url(self):
        self.url_enabled = False

    def enable_string_description(self):
        self.string_description_enabled = True

    def disable_string_description(self):
        self.string_description_enabled = False

    def enable_tokens(self, tokenizer):
        self.string_description_enabled_buffer = self.string_description_enabled
        self.enable_string_description()
        self.tokenizer = tokenizer

    def disable_tokens(self):
        self.string_description_enabled = self.string_description_enabled_buffer
        self.tokenizer = None

    def enable_class_restriction(self, class_idx):
        assert not self.task_config is None, "Task config must be set"
        self.backup_keys = copy.deepcopy(self.keys)
        self.keys = []
        for key in self.backup_keys:
            if int(self.data[key][self.attributes.index(self.task_config.y_selection[0])]) == class_idx:
                self.keys.append(key)

    def disable_class_restriction(self):
        if hasattr(self, "backup_keys"):
            self.keys = copy.deepcopy(self.backup_keys)

    def set_task_specific_keys(self):
        self.task_specific_keys = []
        num_samples_per_class = np.zeros([self.output_size])
        for key in self.keys:
            num_samples_per_class[
                int(
                    self.data[key][
                        self.attributes.index(self.task_config.y_selection[0])
                    ]
                )
            ] += 1

        num_units = num_samples_per_class / np.array(self.config.class_ratios)
        min_units = int(np.min(num_units))
        num_samples_per_class_balanced = min_units * np.array(self.config.class_ratios)
        current_num_samples_per_class = np.zeros([self.output_size])
        for key in self.keys:
            class_idx = int(
                self.data[key][self.attributes.index(self.task_config.y_selection[0])]
            )
            if (
                current_num_samples_per_class[class_idx]
                < num_samples_per_class_balanced[class_idx]
            ):
                self.task_specific_keys.append(key)
                current_num_samples_per_class[class_idx] += 1

        self.keys_backup = copy.deepcopy(self.keys)
        self.keys = self.task_specific_keys

    def __getitem__(self, idx):
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_keys is None
        ):
            self.set_task_specific_keys()

        name = self.keys[idx]

        if (
            not self.task_config is None
            and hasattr(self.task_config, "x_selection")
            and not self.task_config.x_selection is None
            and not len(self.task_config.x_selection) == 0
        ):
            x_selection = self.task_config.x_selection[0]

        else:
            x_selection = "imgs"

        img = Image.open(os.path.join(self.root_dir, x_selection, name))
        state = torch.get_rng_state()
        img_tensor = self.transform(img)

        targets = self.data[name]

        if (
            not self.task_config is None
            and hasattr(self.task_config, "y_selection")
            and not self.task_config.y_selection is None
        ):
            target = torch.zeros([len(self.task_config.y_selection)])
            for i, selection in enumerate(self.task_config.y_selection):
                target[i] = targets[self.attributes.index(selection)]

        else:
            target = torch.tensor(
                targets[: self.config.output_size[0]], dtype=torch.float32
            )

        if (
            not self.task_config is None
            and hasattr(self.task_config, "criterions")
            and "ce" in self.task_config.criterions
        ):
            assert (
                target.shape[0] == 1
            ), "output shape inacceptable for singleclass classification"
            target = target[0].to(torch.int64)

        return_dict = {"x": img_tensor, "y": target}

        if self.hints_enabled:
            mask = Image.open(os.path.join(self.root_dir, "masks", name))
            torch.set_rng_state(state)
            mask_tensor = self.transform(mask)
            return_dict["hint"] = mask_tensor

        if self.groups_enabled:
            has_confounder = targets[
                self.attributes.index(self.config.confounding_factors[-1])
            ]
            return_dict["has_confounder"] = has_confounder

        if self.idx_enabled:
            return_dict["idx"] = idx

        if self.url_enabled:
            return_dict["url"] = name

        if self.string_description_enabled:
            return_dict["description"] = ""
            if hasattr(self, "task_config"):
                y_selection = self.task_config.y_selection

            else:
                y_selection = self.attributes

            for target_idx, attribute in enumerate(y_selection):
                attribute_idx = self.attributes.index(attribute)
                if len(y_selection) == 1 and target > 0.5 or len(y_selection) > 1 and target[target_idx] > 0.5:
                    return_dict["description"] += self.attributes_positive[attribute_idx]

                else:
                    return_dict["description"] += self.attributes_negative[attribute_idx]

                if not target_idx == len(y_selection) - 1:
                    return_dict["description"] += ", "

            if self.tokenizer is not None:
                return_dict["tokens"] = torch.tensor(self.tokenizer(
                    return_dict["description"],
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids)

        if self.return_dict:
            return return_dict

        else:
            return_list = list(return_dict.values())
            return (
                return_list[0],
                return_list[1:] if len(return_list) > 2 else return_list[1],
            )


class Image2ClassDataset(ImageDataset):
    """
    This dataset is used for image classification tasks.

    Args:
        ImageDataset (_type_): _description_
    """

    def __init__(
        self,
        mode,
        config,
        root_dir=None,
        data_dir=None,
        transform=ToTensor(),
        task_config=None,
        return_dict=False,
    ):
        """
        This method initializes the dataset.

        Args:
            root_dir (_type_): _description_
            mode (_type_): _description_
            config (_type_): _description_
            transform (_type_, optional): _description_. Defaults to ToTensor().
            task_config (_type_, optional): _description_. Defaults to None.
        """
        self.config = config
        if root_dir is None:
            self.root_dir = os.path.join(config.dataset_path, "imgs")

        else:
            self.root_dir = os.path.join(root_dir, "imgs")

        if self.config.has_hints:
            self.mask_dir = os.path.join(self.root_dir, "masks")
            self.all_urls = []
            self.urls_with_hints = []

        self.hints_enabled = False
        self.task_config = task_config
        self.transform = transform
        self.return_dict = return_dict
        self.urls = []
        self.idx_to_name = os.listdir(self.root_dir)

        self.idx_to_name.sort()
        for target_str in self.idx_to_name:
            files = os.listdir(os.path.join(self.root_dir, target_str))
            files.sort()
            for file in files:
                self.urls.append((target_str, file))

        random.seed(0)
        random.shuffle(self.urls)

        if mode == "train":
            self.urls = self.urls[: int(config.split[0] * len(self.urls))]

        elif mode == "val":
            self.urls = self.urls[
                int(config.split[0] * len(self.urls)) : int(
                    config.split[1] * len(self.urls)
                )
            ]

        elif mode == "test":
            self.urls = self.urls[int(config.split[1] * len(self.urls)) :]

        if self.config.has_hints:
            self.all_urls = copy.deepcopy(self.urls)
            for target_str, file in self.all_urls:
                if os.path.exists(os.path.join(self.mask_dir, file)):
                    self.urls_with_hints.append((target_str, file))

        self.task_specific_urls = None
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_urls is None
        ):
            self.set_task_specific_urls()

    def class_idx_to_name(self, class_idx):
        return self.idx_to_name[class_idx]

    def enable_hints(self):
        self.urls = copy.deepcopy(self.urls_with_hints)
        self.hints_enabled = True

    def disable_hints(self):
        self.urls = copy.deepcopy(self.all_urls)
        self.hints_enabled = False

    @property
    def output_size(self):
        return self.config.output_size

    def set_task_specific_urls(self):
        self.task_specific_urls = []
        num_samples_per_class = np.zeros([self.output_size])
        for idx in range(len(self.urls)):
            target_str, file = self.urls[idx]
            target = int(torch.tensor(self.idx_to_name.index(target_str)))
            num_samples_per_class[target] += 1

        num_units = num_samples_per_class / np.array(self.config.class_ratios)
        min_units = int(np.min(num_units))
        num_samples_per_class_balanced = min_units * np.array(self.config.class_ratios)
        current_num_samples_per_class = np.zeros([self.output_size])
        for idx in range(len(self.urls)):
            target_str, file = self.urls[idx]
            class_idx = int(torch.tensor(self.idx_to_name.index(target_str)))
            if (
                current_num_samples_per_class[class_idx]
                < num_samples_per_class_balanced[class_idx]
            ):
                self.task_specific_urls.append((target_str, file))
                current_num_samples_per_class[class_idx] += 1

        self.urls_backup = copy.deepcopy(self.urls)
        self.urls = self.task_specific_urls

    def __len__(self):
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_urls is None
        ):
            self.set_task_specific_urls()

        return len(self.urls)

    def __getitem__(self, idx):
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_urls is None
        ):
            self.set_task_specific_urls()

        target_str, file = self.urls[idx]

        img = Image.open(os.path.join(self.root_dir, target_str, file))
        state = torch.get_rng_state()
        img = self.transform(img)

        if img.shape[0] == 1 and self.config.input_size[0] != 1:
            img = torch.tile(img, [self.config.input_size[0], 1, 1])

        # target = torch.zeros([len(self.idx_to_name)], dtype=torch.float32)
        # target[self.idx_to_name.index(target_str)] = 1.0
        target = torch.tensor(self.idx_to_name.index(target_str))

        if not self.hints_enabled:
            if self.return_dict:
                return img, {}  # TODO {"target": target}

            else:
                return img, target

        else:
            # TODO how to apply same randomized transformation?
            mask = Image.open(os.path.join(self.mask_dir, file))
            torch.set_rng_state(state)
            mask = self.transform(mask)
            if self.return_dict:
                return img, {}  # TODO  {"target": target, "mask": mask}

            else:
                return img, (target, mask)
