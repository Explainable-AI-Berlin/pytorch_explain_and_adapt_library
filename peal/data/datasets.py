import torch
import random
import os
import copy
import numpy as np

from torchvision.transforms import ToTensor
from PIL import Image
from pathlib import Path

from peal.data.dataset_interfaces import PealDataset
from peal.data.dataset_utils import parse_json, parse_csv


class SequenceDataset(PealDataset):
    """Sequence dataset."""

    def __init__(self, data_dir, mode, config, transform=ToTensor(), task_config=None):
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
                self.config["input_size"][1]
                * torch.ones(
                    [self.config["input_size"][0] - x.shape[0]], dtype=torch.int64
                ),
            ]
        )
        y = data[-1]

        return x, y

    def generate_contrastive_collage(batch_in, counterfactual):
        # TODO
        return torch.zeros([batch_in.shape[0], 3, 64, 64]), torch.zeros_like(batch_in)

    def serialize_dataset(output_dir, x_list, y_list, sample_names=None):
        # TODO implement this!
        pass


class SymbolicDataset(PealDataset):
    """Symbolic dataset."""

    def __init__(self, data_dir, mode, config, transform=ToTensor(), task_config=None):
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
        self.attributes, self.data, self.keys = parse_csv(data_dir, config, mode)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        name = self.keys[idx]

        data = torch.tensor(self.data[name], dtype=torch.float32)

        if (
            not self.task_config is None
            and "x_selection" in self.task_config.keys()
            and not len(self.task_config["x_selection"]) == 0
        ):
            x = torch.zeros([len(self.task_config["selection"])], dtype=torch.float32)
            for idx, selection in enumerate(self.task_config["selection"]):
                x[idx] = x[self.attributes.index(selection)]

        else:
            x = data[:-1].to(torch.float32)

        if (
            not self.task_config is None
            and "y_selection" in self.task_config.keys()
            and not len(self.task_config["y_selection"]) == 0
        ):
            y = torch.zeros([len(self.task_config["y_selection"])])
            for idx, selection in enumerate(self.task_config["y_selection"]):
                y[idx] = y[self.attributes.index(selection)]

        else:
            y = data[-1]

        return x, y

    def generate_contrastive_collage(self, batch_in, counterfactual):
        # TODO
        return torch.zeros([batch_in.shape[0], 3, 64, 64]), torch.zeros_like(batch_in)

    def serialize_dataset(self, output_dir, x_list, y_list, sample_names=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        data = torch.cat([x, y], dim=1)
        np.savetxt(
            os.path.join(output_dir, "data.csv"),
            data.numpy(),
            delimiter=",",
            header=",".join(self.attributes),
        )


class ImageDataset(PealDataset):
    def generate_contrastive_collage(self, batch_in, counterfactual):
        heatmap_red = torch.maximum(
            torch.tensor(0.0),
            torch.sum(batch_in, dim=1) - torch.sum(counterfactual, dim=1),
        )
        heatmap_blue = torch.maximum(
            torch.tensor(0.0),
            torch.sum(counterfactual, dim=1) - torch.sum(batch_in, dim=1),
        )
        if counterfactual.shape[1] == 3:
            heatmap_green = torch.abs(counterfactual[:, 0] - batch_in[:, 0])
            heatmap_green = heatmap_green + torch.abs(
                counterfactual[:, 1] - batch_in[:, 1]
            )
            heatmap_green = heatmap_green + torch.abs(
                counterfactual[:, 2] - batch_in[:, 2]
            )
            heatmap_green = heatmap_green - heatmap_red - heatmap_blue
            counterfactual_rgb = counterfactual

        else:
            heatmap_green = torch.zeros_like(heatmap_red)
            batch_in = torch.tile(batch_in, [1, 3, 1, 1])
            counterfactual_rgb = torch.tile(torch.clone(counterfactual), [1, 3, 1, 1])

        heatmap = torch.stack([heatmap_red, heatmap_green, heatmap_blue], dim=1)
        if torch.abs(heatmap.sum() - torch.abs(batch_in - counterfactual).sum()) > 0.1:
            print("Error: Heatmap does not match counterfactual")

        heatmap_high_contrast = torch.clamp(heatmap / heatmap.max(), 0.0, 1.0)
        result_img_collage = torch.cat(
            [batch_in, counterfactual_rgb, heatmap_high_contrast], -1
        )
        current_collage = self.project_to_pytorch_default(
            result_img_collages[batch_idx][sample_idx]
        )
        current_collage = torchvision.utils.make_grid(
            current_collage, nrow=self.adaptor_config["batch_size"]
        )
        plt.gcf()
        plt.imshow(current_collage.permute(1, 2, 0))
        title_string = (
            str(int(ys[batch_idx][sample_idx]))
            + " -> "
            + str(targets[batch_idx][sample_idx].item())
        )
        title_string += (
            ", Target: "
            + str(
                round(
                    float(start_target_confidences[batch_idx][sample_idx]),
                    2,
                )
            )
            + " -> "
        )
        title_string += str(
            round(float(end_target_confidences[batch_idx][sample_idx]), 2)
        )
        plt.title(title_string)
        collage_path = os.path.join(
            self.base_dir,
            str(finetune_iteration),
            "validation_collages",
            embed_numberstring(str(sample_idx_iteration)) + ".png",
        )
        plt.axis("off")
        plt.savefig(collage_path)
        img_np = np.array(Image.open(collage_path))[:, 80:-80]
        img = Image.fromarray(img_np)
        img.save(collage_path)
        return result_img_collage, heatmap_high_contrast

    def serialize_dataset(self, output_dir, x_list, y_list, sample_names=None):
        for class_name in range(self.config["output_size"]):
            Path(os.path.join(output_dir, "imgs", str(class_name))).mkdir(
                parents=True, exist_ok=True
            )

        data = []
        for idx, x in enumerate(x_list):
            class_name = int(y_list[idx])
            img = Image.fromarray(x.cpu().numpy().transpose(1, 2, 0))
            img.save(
                os.path.join(output_dir, "imgs", str(class_name), sample_names[idx])
            )
            data.append(
                [
                    os.path.join("imgs", str(class_name), sample_names[idx]),
                    class_name,
                ]
            )

        np.savetxt(
            os.path.join(output_dir, "data.csv"),
            data.numpy(),
            delimiter=",",
            header=",".join(self.attributes),
        )


class Image2MixedDataset(ImageDataset):
    """
    This class is used to load a dataset with images and other data.

    Args:
        ImageDataset (_type_): _description_
    """

    def __init__(self, root_dir, mode, config, transform=ToTensor(), task_config=None):
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
        self.attributes, self.data, self.keys = parse_csv(data_dir, config, mode)

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

        if not self.task_config is None and not len(self.task_config["selection"]) == 0:
            target = torch.zeros([len(self.task_config["selection"])])
            for idx, selection in enumerate(self.task_config["selection"]):
                target[idx] = targets[self.attributes.index(selection)]

        else:
            target = torch.tensor(
                targets[: self.config["output_size"]], dtype=torch.float32
            )

        if not self.task_config is None and "ce" in self.task_config["criterions"]:
            assert (
                target.shape[0] == 1
            ), "output shape inacceptable for singleclass classification"
            target = torch.tensor(target[0], dtype=torch.int64)

        if not self.hints_enabled:
            return img_tensor, target

        else:
            mask = Image.open(os.path.join(self.root_dir, "masks", name))
            torch.set_rng_state(state)
            mask_tensor = self.transform(mask)
            return img_tensor, (target, mask_tensor)


class Image2ClassDataset(ImageDataset):
    """
    This dataset is used for image classification tasks.

    Args:
        ImageDataset (_type_): _description_
    """

    def __init__(self, root_dir, mode, config, transform=ToTensor(), task_config=None):
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
        if "has_hints" in self.config.keys() and self.config["has_hints"]:
            self.root_dir = os.path.join(root_dir, "imgs")
            self.mask_dir = os.path.join(root_dir, "masks")
            self.all_urls = []
            self.urls_with_hints = []

        else:
            self.root_dir = root_dir

        self.hints_enabled = False
        self.task_config = task_config
        self.transform = transform
        self.urls = []
        self.idx_to_name = os.listdir(self.root_dir)

        self.idx_to_name.sort()
        for label_str in self.idx_to_name:
            files = os.listdir(os.path.join(self.root_dir, label_str))
            files.sort()
            for file in files:
                self.urls.append((label_str, file))

        random.seed(0)
        random.shuffle(self.urls)

        if mode == "train":
            self.urls = self.urls[: int(config["split"][0] * len(self.urls))]

        elif mode == "val":
            self.urls = self.urls[
                int(config["split"][0] * len(self.urls)) : int(
                    config["split"][1] * len(self.urls)
                )
            ]

        elif mode == "test":
            self.urls = self.urls[int(config["split"][1] * len(self.urls)) :]

        if "has_hints" in self.config.keys() and self.config["has_hints"]:
            self.all_urls = copy.deepcopy(self.urls)
            for label_str, file in self.all_urls:
                if os.path.exists(os.path.join(self.mask_dir, file)):
                    self.urls_with_hints.append((label_str, file))

    def class_idx_to_name(self, class_idx):
        return self.idx_to_name[class_idx]

    def enable_hints(self):
        self.urls = copy.deepcopy(self.urls_with_hints)
        self.hints_enabled = True

    def disable_hints(self):
        self.urls = copy.deepcopy(self.all_urls)
        self.hints_enabled = False

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        label_str, file = self.urls[idx]

        img = Image.open(os.path.join(self.root_dir, label_str, file))
        state = torch.get_rng_state()
        img = self.transform(img)

        if img.shape[0] == 1 and self.config["input_size"][0] != 1:
            img = torch.tile(img, [self.config["input_size"][0], 1, 1])

        # label = torch.zeros([len(self.idx_to_name)], dtype=torch.float32)
        # label[self.idx_to_name.index(label_str)] = 1.0
        label = torch.tensor(self.idx_to_name.index(label_str))

        if not self.hints_enabled:
            return img, label

        else:
            # TODO how to apply same randomized transformation?
            mask = Image.open(os.path.join(self.mask_dir, file))
            torch.set_rng_state(state)
            mask = self.transform(mask)
            return img, (label, mask)
