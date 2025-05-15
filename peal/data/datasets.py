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

from peal.data.interfaces import PealDataset
from peal.data.dataset_utils import parse_csv
from peal.global_utils import embed_numberstring, high_contrast_heatmap
from peal.generators.interfaces import Generator

matplotlib.use("Agg")

from typing import Union


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
        hint_list=None,
        idx_to_info=None,
        tracking_level=1,
        history_list=None,
        **kwargs: dict,
    ) -> tuple:
        Path(base_path).mkdir(parents=True, exist_ok=True)
        collage_paths = []
        heatmap_list = []
        for i in range(len(x_list)):
            x = self.project_to_pytorch_default(x_list[i])
            counterfactual = self.project_to_pytorch_default(x_counterfactual_list[i])

            heatmap_high_contrast, x_in, counterfactual_rgb = high_contrast_heatmap(
                x, counterfactual
            )

            heatmap_list.append(heatmap_high_contrast)

            if tracking_level >= 1:
                current_collage = torch.cat(
                    [x_in, counterfactual_rgb, heatmap_high_contrast], -1
                )
                current_collage = torchvision.utils.make_grid(current_collage, nrow=3)
                plt.gcf()
                plt.imshow(current_collage.permute(1, 2, 0))
                title_string = (
                    "Original: "
                    + str(int(y_list[i]))
                    + " -> Prediction: "
                    + str(int(y_source_list[i]))
                    + " -> Target: "
                    + str(int(y_target_list[i]))
                    + "\n"
                )
                title_string += (
                    "Target Confidence: "
                    + str(
                        round(
                            float(y_target_start_confidence_list[i]),
                            2,
                        )
                    )
                    + " -> "
                )
                title_string += (
                    str(round(float(y_target_end_confidence_list[i]), 2)) + "\n"
                )
                if not hint_list is None and not idx_to_info is None:
                    title_string += (
                        idx_to_info(x_list[i], x_counterfactual_list[i], hint_list[i])
                        + "\n"
                    )

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

            else:
                collage_paths.append(None)

        return heatmap_list, collage_paths

    def serialize_dataset(
        self,
        output_dir,
        x_list,
        y_list,
        hint_list=[],
        sample_names=None,
        classifier=None,
    ):
        # TODO this does not seem very clean
        for class_name in range(max(2, self.output_size)):
            Path(os.path.join(output_dir, "imgs", str(class_name))).mkdir(
                parents=True, exist_ok=True
            )
            if not len(hint_list) == 0:
                Path(os.path.join(output_dir, "masks", str(class_name))).mkdir(
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
            if not len(hint_list) == 0:
                mask = Image.fromarray(
                    np.array(
                        255 * hint_list[idx].cpu().numpy().transpose(1, 2, 0),
                        dtype=np.uint8,
                    )
                )
                mask.save(os.path.join(output_dir, "masks", img_name))

            data.append(
                [
                    img_name,
                    class_name,
                ]
            )

        data = "ImgPath,Class\n" + "\n".join([",".join(map(str, x)) for x in data])
        with open(os.path.join(output_dir, self.config.label_rel_path), "w") as f:
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
                batch_size = generator.config.batch_size

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
            self.config.dataset_path = root_dir

        self.transform = transform
        self.task_config = task_config
        self.hints_enabled = False
        self.groups_enabled = False
        self.idx_enabled = False
        self.url_enabled = False
        self.string_description_enabled = False
        self.tokenizer = None
        self.return_dict = return_dict
        self.class_restrictions_enabled = False
        # TODO
        # self.config.class_ratios = None
        if data_dir is None:
            data_dir = os.path.join(self.root_dir, self.config.label_rel_path)

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
            attribute_values = attribute.split("_vs_")
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

        if self.config.in_memory:
            self.load_in_memory()

    def load_in_memory(self):
        print('load dataset into memory!!')
        self.in_memory_images = {}
        self.in_memory_masks = {}
        for name in self.keys:
            img = np.array(Image.open(os.path.join(self.root_dir, self.config.x_selection, name)))
            self.in_memory_images[name] = img

            if self.config.has_hints:
                option1 = os.path.join(self.root_dir, "masks", name)
                option2 = os.path.join(self.root_dir, "masks", name.split("/")[-1])
                option3 = option2[:-4] + ".png"
                option4 = option1[:-4] + ".png"
                if os.path.exists(option1):
                    mask = Image.open(os.path.join(self.root_dir, "masks", name))

                elif os.path.exists(option2):
                    mask = Image.open(option2)

                elif os.path.exists(option3):
                    mask = Image.open(option3)

                elif os.path.exists(option4):
                    mask = Image.open(option4)

                else:
                    assert (
                        not self.config.has_hints
                    ), "Hints not found despite claim that they exist!"
                    mask = Image.new("RGB", img.size, (0, 0, 0))

                self.in_memory_masks[name] = np.array(mask)

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
            if (
                int(
                    self.data[key][
                        self.attributes.index(self.task_config.y_selection[0])
                    ]
                )
                == class_idx
            ):
                self.keys.append(key)

        self.class_restrictions_enabled = True

    def disable_class_restriction(self):
        if hasattr(self, "backup_keys"):
            self.keys = copy.deepcopy(self.backup_keys)

        self.class_restrictions_enabled = True

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

        if self.config.in_memory:
            img = Image.fromarray(self.in_memory_images[name])

        else:
            img = Image.open(os.path.join(self.root_dir, self.config.x_selection, name))

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
            if self.config.in_memory:
                mask = Image.fromarray(self.in_memory_masks[name])

            else:
                option1 = os.path.join(self.root_dir, "masks", name)
                option2 = os.path.join(self.root_dir, "masks", name.split("/")[-1])
                option3 = option2[:-4] + ".png"
                option4 = option1[:-4] + ".png"
                if os.path.exists(option1):
                    mask = Image.open(os.path.join(self.root_dir, "masks", name))

                elif os.path.exists(option2):
                    mask = Image.open(option2)

                elif os.path.exists(option3):
                    mask = Image.open(option3)

                elif os.path.exists(option4):
                    mask = Image.open(option4)

                else:
                    assert (
                        not self.config.has_hints
                    ), "Hints not found despite claim that they exist!"
                    mask = Image.new("RGB", img.size, (0, 0, 0))

            torch.set_rng_state(state)
            mask_tensor = self.transform(mask)
            if not mask_tensor.shape[0] == 3:
                # TODO very very hacky
                print("Mask tensor shape " + str(mask_tensor.shape) + " is not correct")
                mask_tensor = torch.cat([mask_tensor, mask_tensor, mask_tensor])
                mask_tensor = mask_tensor[:3]

            return_dict["hint"] = mask_tensor

        if self.groups_enabled:
            if len(self.config.confounding_factors) == 2:
                has_confounder = targets[
                    self.attributes.index(self.config.confounding_factors[-1])
                ]
                return_dict["has_confounder"] = has_confounder

            else:
                has_confounder = []
                for factor in self.config.confounding_factors[1:]:
                    has_confounder = targets[self.attributes.index(factor)]

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
                if (
                    len(y_selection) == 1
                    and target > 0.5
                    or len(y_selection) > 1
                    and target[target_idx] > 0.5
                ):
                    return_dict["description"] += self.attributes_positive[
                        attribute_idx
                    ]

                else:
                    return_dict["description"] += self.attributes_negative[
                        attribute_idx
                    ]

                if not target_idx == len(y_selection) - 1:
                    return_dict["description"] += ", "

            if self.tokenizer is not None:
                return_dict["tokens"] = torch.tensor(
                    self.tokenizer(
                        return_dict["description"],
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids
                )

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
            if data_dir is None:
                root_dir = config.dataset_path

            else:
                root_dir = data_dir

        self.config.dataset_path = root_dir

        self.root_dir = os.path.join(root_dir, self.config.x_selection)

        if self.config.has_hints:
            self.mask_dir = os.path.join(root_dir, "masks")
            self.all_urls = []
            self.urls_with_hints = []

        self.hints_enabled = False
        self.idx_enabled = False
        self.url_enabled = False
        self.task_config = task_config
        self.transform = transform
        self.return_dict = return_dict
        self.urls = []
        self.idx_to_name = os.listdir(self.root_dir)
        self.string_description_enabled = False

        self.idx_to_name.sort()
        for target_str in self.idx_to_name:
            if not os.path.isdir(os.path.join(self.root_dir, target_str)):
                continue

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
                if os.path.exists(os.path.join(self.mask_dir, file)) or os.path.exists(
                    os.path.join(self.mask_dir, target_str, file)
                ):
                    self.urls_with_hints.append((target_str, file))

                else:
                    print(os.path.join(self.mask_dir, file))
                    print(os.path.join(self.mask_dir, target_str, file))
                    raise Exception("No hints available!")

        self.task_specific_urls = None
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_urls is None
        ):
            self.set_task_specific_urls()

        self.class_restriction_enabled = False
        self.backup_urls = copy.deepcopy(self.urls)

        if self.config.in_memory:
            self.load_in_memory()

    def load_in_memory(self):
        self.in_memory_images = {}
        for target_str, file in self.urls:
            img = Image.open(os.path.join(self.root_dir, target_str, file))
            self.in_memory_images[os.path.join(target_str, file)] = np.array(img)

        if self.config.has_hints:
            self.in_memory_masks = {}
            for target_str, file in self.urls:
                if os.path.exists(os.path.join(self.mask_dir, file)):
                    mask_path = os.path.join(self.mask_dir, file)

                elif os.path.exists(os.path.join(self.mask_dir, target_str, file)):
                    mask_path = os.path.join(self.mask_dir, target_str, file)

                else:
                    raise Exception(os.path.join(self.mask_dir, target_str, file) + " not found!")

                self.in_memory_masks[os.path.join(target_str, file)] = np.array(Image.open(mask_path))


    def class_idx_to_name(self, class_idx):
        return self.idx_to_name[class_idx]

    def enable_hints(self):
        self.urls = copy.deepcopy(self.urls_with_hints)
        self.hints_enabled = True

    def disable_hints(self):
        self.urls = copy.deepcopy(self.all_urls)
        self.hints_enabled = False

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

    def enable_idx(self):
        self.idx_enabled = True

    def disable_idx(self):
        self.idx_enabled = False

    def enable_url(self):
        self.url_enabled = True

    def disable_url(self):
        self.url_enabled = False

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

    def enable_class_restriction(self, class_idx):
        self.backup_urls = copy.deepcopy(self.urls)
        self.urls = []
        self.class_restriction_enabled = True
        for url in self.backup_urls:
            if url[0] == self.idx_to_name[class_idx]:
                self.urls.append(url)

    def disable_class_restriction(self):
        if hasattr(self, "backup_urls"):
            self.urls = copy.deepcopy(self.backup_urls)

        self.class_restriction_enabled = False

    def __getitem__(self, idx):
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_urls is None
        ):
            self.set_task_specific_urls()

        target_str, file = self.urls[idx]

        if self.config.in_memory:
            img = Image.fromarray(self.in_memory_images[os.path.join(target_str, file)])

        else:
            img = Image.open(os.path.join(self.root_dir, target_str, file))

        state = torch.get_rng_state()
        img = self.transform(img)

        if img.shape[0] == 1 and self.config.input_size[0] != 1:
            img = torch.tile(img, [self.config.input_size[0], 1, 1])

        # target = torch.zeros([len(self.idx_to_name)], dtype=torch.float32)
        # target[self.idx_to_name.index(target_str)] = 1.0
        return_dict = {}
        target = torch.tensor(self.idx_to_name.index(target_str))
        return_dict["target"] = target

        if self.hints_enabled:
            if self.config.in_memory:
                mask = Image.fromarray(self.in_memory_masks[os.path.join(target_str, file)])

            else:
                if os.path.exists(os.path.join(self.mask_dir, file)):
                    mask_path = os.path.join(self.mask_dir, file)

                elif os.path.exists(os.path.join(self.mask_dir, target_str, file)):
                    mask_path = os.path.join(self.mask_dir, target_str, file)

                else:
                    raise Exception(os.path.join(self.mask_dir, target_str, file) + " not found!")

                mask = Image.open(mask_path)

            torch.set_rng_state(state)
            mask = self.transform(mask)
            return_dict["mask"] = mask

        if self.idx_enabled:
            return_dict["idx"] = idx

        if self.url_enabled:
            return_dict["url"] = os.path.join(*self.urls[idx])

        if self.string_description_enabled:
            return_dict["description"] = target_str

            if self.tokenizer is not None:
                return_dict["tokens"] = torch.tensor(
                    self.tokenizer(
                        return_dict["description"],
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids
                )

        if self.return_dict:
            return img, return_dict

        elif len(return_dict.values()) == 1:
            return img, list(return_dict.values())[0]

        else:
            return img, tuple(return_dict.values())
