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
        **kargs,
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
                try:
                    y[idx] = data[self.attributes.index(selection)]

                except:
                    print(data)
                    print(self.data)
                    quit()

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
        x_list,
        x_counterfactual_list,
        y_target_list,
        y_source_list,
        y_list,
        target_confidence_goal,
        y_target_start_confidence_list,
        y_target_end_confidence_list,
        base_path,
        start_idx,
        **args,
    ):
        Path(base_path).mkdir(parents=True, exist_ok=True)
        collage_paths = []
        heatmap_list = []
        for i in range(len(x_list)):
            heatmap_red = torch.maximum(
                torch.tensor(0.0),
                torch.sum(x_list[i], dim=0) - torch.sum(x_counterfactual_list[i], dim=0),
            )
            heatmap_blue = torch.maximum(
                torch.tensor(0.0),
                torch.sum(x_counterfactual_list[i], dim=0) - torch.sum(x_list[i], dim=0),
            )
            if x_counterfactual_list[i].shape[0] == 3:
                heatmap_green = torch.abs(x_list[i][0] - x_counterfactual_list[i][0])
                heatmap_green = heatmap_green + torch.abs(
                    x_list[i][1] - x_counterfactual_list[i][1]
                )
                heatmap_green = heatmap_green + torch.abs(
                    x_list[i][2] - x_counterfactual_list[i][2]
                )
                heatmap_green = heatmap_green - heatmap_red - heatmap_blue
                x_in = torch.clone(x_list[i])
                counterfactual_rgb = torch.clone(x_counterfactual_list[i])

            else:
                heatmap_green = torch.zeros_like(heatmap_red)
                x_in = torch.tile(x_list[i], [3, 1, 1])
                counterfactual_rgb = torch.tile(torch.clone(x_counterfactual_list[i]), [3, 1, 1])

            heatmap = torch.stack([heatmap_red, heatmap_green, heatmap_blue], dim=0)
            if torch.abs(heatmap.sum() - torch.abs(x_list[i] - x_counterfactual_list[i]).sum()) > 0.1:
                print("Error: Heatmap does not add up to absolute counterfactual difference.")

            heatmap_high_contrast = torch.clamp(heatmap / heatmap.max(), 0.0, 1.0)
            current_collage = torch.cat(
                [x_in, counterfactual_rgb, heatmap_high_contrast], -1
            )
            current_collage = torchvision.utils.make_grid(
                current_collage, nrow=3
            )
            plt.gcf()
            plt.imshow(current_collage.permute(1, 2, 0))
            title_string = (
                str(int(y_list[i])) + " -> " +
                str(int(y_source_list[i]))
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
            title_string += str(
                round(float(y_target_end_confidence_list[i]), 2)
            )
            plt.title(title_string)
            collage_path = os.path.join(
                base_path,
                embed_numberstring(str(start_idx + i)) + "_collage.png",
            )
            plt.axis("off")
            plt.savefig(collage_path)
            collage_paths.append(collage_path)
            heatmap_list.append(heatmap)

        return heatmap_list, collage_paths

    def serialize_dataset(self, output_dir, x_list, y_list, sample_names=None):
        # TODO this does not seem very clean
        for class_name in range(max(2, self.output_size)):
            Path(os.path.join(output_dir, "imgs", str(class_name))).mkdir(
                parents=True, exist_ok=True
            )

        data = []
        for idx, x in enumerate(x_list):
            class_name = int(y_list[idx])
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

    def track_generator_performance(self, generator: Generator, batch_size=1):
        """
        This function tracks the performance of the generator

        Args:
            generator (Generator): The generator
        """
        if not hasattr(self, "fid"):
            self.fid = torchmetrics.image.fid.FrechetInceptionDistance(feature=64, reset_real_features=False)
            real_images = []
            for i in range(10):
                real_images.append(self[i][0])

            real_images = torch.stack(real_images, dim=0)
            self.fid.update(torch.tensor(255 * real_images, dtype=torch.uint8), real=True)

        generated_images = generator.sample_x(batch_size=batch_size)
        while generated_images.shape[0] < 10:
            generated_images = torch.cat(
                [generated_images, generator.sample_x(batch_size=batch_size)], dim=0
            )

        generated_images = generated_images[:10]
        self.fid.update(torch.tensor(255 * generated_images, dtype=torch.uint8).cpu(), real=False)
        fid_score = float(self.fid.compute())
        return {'fid': fid_score}


class Image2MixedDataset(ImageDataset):
    """
    This class is used to load a dataset with images and other data.

    Args:
        ImageDataset (_type_): _description_
    """

    def __init__(
        self,
        root_dir,
        mode,
        config,
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
        self.root_dir = root_dir
        self.config = config
        self.transform = transform
        self.task_config = task_config
        self.hints_enabled = False
        data_dir = os.path.join(root_dir, "data.csv")
        self.attributes, self.data, self.keys = parse_csv(
            data_dir, config, mode, key_type="name", delimiter=config.delimiter
        )
        self.return_dict = return_dict

    @property
    def output_size(self):
        if self.task_config is not None and self.task_config.y_selection is not None:
            return self.task_config.output_channels

        else:
            return len(self.attributes)

    def __len__(self):
        return len(self.keys)

    def enable_hints(self):
        self.hints_enabled = True

    def disable_hints(self):
        self.hints_enabled = False

    def __getitem__(self, idx):
        name = self.keys[idx]

        img = Image.open(os.path.join(self.root_dir, "imgs", name))
        # code.interact(local=dict(globals(), **locals()))
        state = torch.get_rng_state()
        img_tensor = self.transform(img)

        targets = self.data[name]

        if not self.task_config is None and not self.task_config.y_selection is None:
            target = torch.zeros([len(self.task_config.y_selection)])
            for idx, selection in enumerate(self.task_config.y_selection):
                target[idx] = targets[self.attributes.index(selection)]

        else:
            target = torch.tensor(
                targets[:self.config.output_size[0]], dtype=torch.float32
            )

        if not self.task_config is None and "ce" in self.task_config.criterions:
            assert (
                target.shape[0] == 1
            ), "output shape inacceptable for singleclass classification"
            target = target[0].to(torch.int64)

        if not self.hints_enabled:
            if self.return_dict:
                return img_tensor, {}  # TODO  {"target": target}

            else:
                return img_tensor, target

        else:
            mask = Image.open(os.path.join(self.root_dir, "masks", name))
            torch.set_rng_state(state)
            mask_tensor = self.transform(mask)
            if self.return_dict:
                return img_tensor, {}  # TODO  {"target": target, "mask": mask_tensor}

            else:
                return img_tensor, (target, mask_tensor)


class Image2ClassDataset(ImageDataset):
    """
    This dataset is used for image classification tasks.

    Args:
        ImageDataset (_type_): _description_
    """

    def __init__(
        self,
        root_dir,
        mode,
        config,
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
        self.root_dir = os.path.join(root_dir, "imgs")
        if "has_hints" in self.config.keys() and self.config.has_hints:
            self.mask_dir = os.path.join(root_dir, "masks")
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

        if "has_hints" in self.config.keys() and self.config.has_hints:
            self.all_urls = copy.deepcopy(self.urls)
            for target_str, file in self.all_urls:
                if os.path.exists(os.path.join(self.mask_dir, file)):
                    self.urls_with_hints.append((target_str, file))

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
        return len(self.config.output_size)

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
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
