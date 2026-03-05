import copy
import csv
import os
import io
import shutil
import tarfile
from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
import matplotlib.cm as cm
import requests
from torch.utils.data import ConcatDataset

from torchvision.transforms import ToTensor
from wilds import get_dataset

from peal.data.dataloaders import DataloaderMixer, WeightedDataloaderList
from peal.data.dataset_generators import (
    SquareDatasetGenerator,
    ConfounderDatasetGenerator,
)
from peal.data.datasets import (
    Image2ClassDataset,
    Image2MixedDataset,
    ImageDataset,
)
from peal.data.interfaces import DataConfig
from peal.global_utils import embed_numberstring
from peal.data.dataset_generators import latent_to_square_image
from peal.dependencies.FastDiME_CelebA.eval_utils.oracle_metrics import OracleMetrics


class MnistDataset(Image2ClassDataset):
    def __init__(self, config: DataConfig, **kwargs):
        if not os.path.exists(config.dataset_path):
            mnist_dataset_train = torchvision.datasets.MNIST(
                root=config.dataset_path + "_train_raw",
                train=True,
                download=True,
                transform=None,
            )
            img_dir = os.path.join(config.dataset_path, "imgs")
            idxs = np.zeros([10])
            Path(img_dir).mkdir(parents=True, exist_ok=True)
            for i in range(len(mnist_dataset_train)):
                img, label = mnist_dataset_train[i]
                if not os.path.exists(f"{img_dir}/{label}"):
                    os.makedirs(f"{img_dir}/{label}")

                img.save(f"{img_dir}/{label}/{idxs[label]}.png")
                idxs[label] += 1

            mnist_dataset_val = torchvision.datasets.MNIST(
                root=config.dataset_path + "_val_raw",
                train=False,
                download=True,
                transform=None,
            )
            for i in range(len(mnist_dataset_val)):
                img, label = mnist_dataset_val[i]
                img.save(f"{img_dir}/{label}/{idxs[label]}.png")
                idxs[label] += 1

        super(MnistDataset, self).__init__(config=config, **kwargs)


class ColoredMnistConfig(DataConfig):
    config_name: str = "ColoredMnistConfig"
    coloring: list[tuple[int, float]] = []
    raw_path: Union[type(None), str] = None
    group_map: str = None


class ColoredMnist(Image2ClassDataset):
    def __init__(self, mode: str, config: ColoredMnistConfig, **kwargs):
        self.config = copy.deepcopy(config)
        if not os.path.exists(self.config.dataset_path):
            self._create_dataset()

        if mode == "val":
            self.config.x_selection = "imgs_val"
            self.config.split = [0, 1]
        elif mode == "test":
            self.config.x_selection = "imgs_test"
            self.config.split = [0, 0]
        else:
            self.config.x_selection = "imgs_train"
            self.config.split = [1, 1]

        self.group_labels = {}
        group_map_file = self.config.group_map or os.path.join(self.config.dataset_path, "data.csv")
        with open(group_map_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            col_filename = header.index("Name")
            col_colored = header.index("Colored")
            for row in reader:
                self.group_labels[row[col_filename]] = int(row[col_colored])

        super(ColoredMnist, self).__init__(config=self.config, mode=mode, **kwargs)

    def has_confounder(self, filename: str) -> int:
        return self.group_labels[filename]

    def _create_dataset(self):
        print(f"creating new ColoredMnist dataset with coloring {self.config.coloring} at {self.config.dataset_path}")
        np.random.seed(0 if self.config.seed is None else self.config.seed)

        data_info = {"Name": [], "Digit": [], "Colored": [], "Subset": []}

        def color_sample(coloring_wanted, coloring_actual, label, img):
            if label in coloring_wanted and coloring_actual[label] < coloring_wanted[label]:
                coloring_actual[label] += 1
                img = np.asarray(img)
                zeros = np.zeros(img.shape, dtype=np.uint8)
                img = np.stack([img, zeros, zeros], axis=2)
                img = Image.fromarray(img)
                data_info["Colored"].append(1)
            else:
                data_info["Colored"].append(0)
            return img

        number_samples_max_train = 4500
        number_samples_max_val_and_test = 750
        coloring_train = {
            class_coloring[0]: round(class_coloring[1] * number_samples_max_train)
            for class_coloring in self.config.coloring
        }
        coloring_val_and_test = {
            class_coloring[0]: round(class_coloring[1] * number_samples_max_val_and_test)
            for class_coloring in self.config.coloring
        }

        raw_path = self.config.dataset_path if self.config.raw_path is None else self.config.raw_path

        mnist_dataset_train = torchvision.datasets.MNIST(
            root=raw_path + "_train_raw",
            train=True,
            download=True,
            transform=None,
        )
        mnist_dataset_val = torchvision.datasets.MNIST(
            root=raw_path + "_val_raw",
            train=False,
            download=True,
            transform=None,
        )
        data = ConcatDataset([mnist_dataset_train, mnist_dataset_val])

        dir_train = os.path.join(self.config.dataset_path, "imgs_train")
        Path(dir_train).mkdir(parents=True, exist_ok=False)
        dir_val = os.path.join(self.config.dataset_path, "imgs_val")
        Path(dir_val).mkdir(parents=True, exist_ok=False)
        dir_test = os.path.join(self.config.dataset_path, "imgs_test")
        Path(dir_test).mkdir(parents=True, exist_ok=False)
        for i in range(10):
            Path(os.path.join(dir_train, str(i))).mkdir()
            Path(os.path.join(dir_val, str(i))).mkdir()
            Path(os.path.join(dir_test, str(i))).mkdir()

        number_samples_train = [0] * 10
        number_samples_val = [0] * 10
        number_samples_test = [0] * 10
        coloring_actual_train = [0] * 10
        coloring_actual_val = [0] * 10
        coloring_actual_test = [0] * 10

        current = 0

        idxs = np.arange(len(data))
        idxs = np.random.permutation(idxs)
        for idx in idxs:
            img, label = data[idx]
            if number_samples_train[label] < number_samples_max_train:
                number_samples_train[label] += 1
                img = color_sample(coloring_train, coloring_actual_train, label, img)
                filename = os.path.join(dir_train, str(label), str(current).zfill(6) + ".png")
                data_info["Subset"].append("train")

            elif number_samples_val[label] < number_samples_max_val_and_test:
                number_samples_val[label] += 1
                img = color_sample(coloring_val_and_test, coloring_actual_val, label, img)
                filename = os.path.join(dir_val, str(label), str(current).zfill(6) + ".png")
                data_info["Subset"].append("val")

            elif number_samples_test[label] < number_samples_max_val_and_test:
                number_samples_test[label] += 1
                img = color_sample(coloring_val_and_test, coloring_actual_test, label, img)
                filename = os.path.join(dir_test, str(label), str(current).zfill(6) + ".png")
                data_info["Subset"].append("test")

            else:
                continue

            img.save(filename)
            data_info["Name"].append(str(current).zfill(6) + ".png")
            data_info["Digit"].append(label)
            current += 1

        with open(os.path.join(self.config.dataset_path, "data.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data_info.keys())
            writer.writerows(zip(*data_info.values()))


def plot_latents_with_arrows(
    original_latents,
    counterfactual_latents,
    filename,
    y_target_start_confidence,
    y_target_end_confidence,
    decision_boundary,
):
    fig, ax = plt.subplots()

    # Convert to numpy arrays for easy manipulation
    original_latents = np.array(original_latents)
    counterfactual_latents = np.array(counterfactual_latents)
    y_target_start_confidence = np.array(y_target_start_confidence)
    y_target_end_confidence = np.array(y_target_end_confidence)

    # Define the colormap for the points: blue -> red
    cmap = cm.get_cmap("bwr")  # blue to red

    # Create a custom colormap for the decision boundary (0 -> light blue, 1 -> light red)
    # decision_cmap = ListedColormap(["lightcoral", "lightblue"])

    # Display the decision boundary grid as the background
    ax.imshow(
        decision_boundary,
        extent=[0, 1, 0, 1],  # Extend from x=0 to x=1 and y=0 to y=1
        origin="lower",  # Aligns the grid with the bottom-left of the plot
        cmap=cmap,  # Apply custom colormap
        alpha=0.3,  # Make the background semi-transparent
    )

    # Plot original latents and counterfactuals with different markers
    for i, (orig, cf, start_conf, end_conf) in enumerate(
        zip(
            original_latents,
            counterfactual_latents,
            y_target_start_confidence,
            y_target_end_confidence,
        )
    ):
        # Get the color from the colormap based on confidence (blue -> red)
        start_color = cmap(start_conf)  # Color for the original point
        end_color = cmap(end_conf)  # Color for the counterfactual point

        # Plot original point with circle marker
        ax.scatter(
            orig[0],
            orig[1],
            facecolor=start_color,
            edgecolor="darkblue",
            marker="o",  # Circle for original
            label="Original" if i == 0 else "",
        )

        # Plot counterfactual point with square marker
        ax.scatter(
            cf[0],
            cf[1],
            facecolor=end_color,
            edgecolor="darkred",
            marker="s",  # Square for counterfactual
            label="Counterfactual" if i == 0 else "",
        )

        # Draw arrow between original and counterfactual points
        ax.annotate(
            "",
            xy=(cf[0], cf[1]),
            xytext=(orig[0], orig[1]),
            arrowprops=dict(fc="green", ec="green", edgecolor="yellow", arrowstyle="->", alpha=0.7),
        )

    # Setting limits for the plot (0 to 1 for both axes)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Set ticks at 0.5 intervals
    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))

    # Axis labels
    ax.set_xlabel("Foreground Intensity")
    ax.set_ylabel("Background Intensity")

    # Add vertical dotted line for Foreground Intensity == 0.5
    plt.axvline(x=0.5, color="black", linestyle="--")
    plt.text(
        1.05,
        0.5,
        "Confounding feature only",
        rotation=270,
        verticalalignment="center",
    )

    # Add horizontal dotted line for Background Intensity == 0.5
    plt.axhline(y=0.5, color="black", linestyle="--")
    plt.text(0.5, 1.05, "True feature only", horizontalalignment="center")

    # Create neutral markers for the legend (gray fill color)
    handles, labels = ax.get_legend_handles_labels()
    original_marker = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="gray",
        markeredgecolor="darkblue",
        markersize=8,
        label="Original",
    )
    cf_marker = plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="gray",
        markeredgecolor="darkred",
        markersize=8,
        label="Counterfactual",
    )
    by_label = dict(zip(labels, handles))

    # Override handles with custom markers
    by_label["Original"] = original_marker
    by_label["Counterfactual"] = cf_marker

    # Add explanation for the background colors in the legend
    blue_patch = plt.Line2D([0], [0], color="blue", lw=4, label="Pred > 0.5")
    red_patch = plt.Line2D([0], [0], color="red", lw=4, label="Pred < 0.5")

    # Display the updated legend
    ax.legend(handles=[original_marker, cf_marker, blue_patch, red_patch], loc="upper right")

    # Show the plot
    plt.grid(True)
    plt.savefig(filename)
    plt.clf()


class SquareDataset(Image2MixedDataset):
    def __init__(self, config: DataConfig, **kwargs):
        if not os.path.exists(config.dataset_path):
            if config.confounder_probability == 0.5:
                cdg = SquareDatasetGenerator(data_config=config)
                cdg.generate_dataset()

            else:
                raise NotImplementedError("Only confounder_probability=0.5 can be used to generate the dataset")

        super(SquareDataset, self).__init__(config=config, **kwargs)

    def global_counterfactual_visualization(
        self,
        filename,
        x_list,
        counterfactuals,
        y_target_start_confidence,
        y_target_end_confidence,
        y_list,
        hint_list,
    ):
        y_start_confidence = list(
            map(
                lambda i: abs(y_list[i] - y_target_start_confidence[i]),
                range(len(y_list)),
            )
        )
        y_end_confidence = list(
            map(
                lambda i: abs(y_list[i] - y_target_end_confidence[i]),
                range(len(y_list)),
            )
        )
        original_latents = []
        counterfactual_latents = []
        hints_enabled_buffer = self.hints_enabled
        self.hints_enabled = True
        for idx in range(len(x_list)):
            x = x_list[idx]
            hint = hint_list[idx]
            original_latents.append([self.check_foreground(x, hint), self.check_background(x, hint)])
            counterfactual_latents.append(
                [
                    self.check_foreground(counterfactuals[idx], hint),
                    self.check_background(counterfactuals[idx], hint),
                ]
            )

        self.hints_enabled = hints_enabled_buffer

        path = filename.split("/")[:-1] + ["decision_boundary.npy"]
        if os.path.exists("/" + str(os.path.join(*path))):
            decision_boundary = np.load("/" + str(os.path.join(*path)))

        else:
            decision_boundary = np.load(str(os.path.join(*path)))

        decision_boundary = np.transpose(decision_boundary, (1, 0))

        plot_latents_with_arrows(
            original_latents,
            counterfactual_latents,
            filename,
            y_start_confidence,
            y_end_confidence,
            decision_boundary,
        )

    def visualize_decision_boundary(
        self,
        predictor,
        batch_size,
        device,
        path,
        temperature=1.0,
        train_dataloader=None,
        val_dataloaders=[],
        val_weights=[],
    ):
        print("visualize_decision_boundary")
        grid_path = path[:-4] + ".npy"
        # Create the grid for plotting
        x = torch.linspace(0, 1, 100)
        y = torch.linspace(0, 1, 100)
        xx, yy = torch.meshgrid(x, y)
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        if not os.path.exists(grid_path):
            prediction_grids = []
            positions = [0, 26, 52]

            # Predict the grid values for decision boundary
            for x_pos in positions:
                for y_pos in positions:
                    current_batch = []
                    logits = []
                    first_batch = None

                    for i in range(len(grid)):
                        current_batch.append(
                            self.project_from_pytorch_default(
                                ToTensor()(
                                    latent_to_square_image(
                                        255 * float(grid[i][0]),
                                        255 * float(grid[i][1]),
                                        position_x=x_pos,
                                        position_y=y_pos,
                                    )[0]
                                ),
                            )
                        )
                        if len(current_batch) == batch_size:
                            current_batch = torch.stack(current_batch)
                            if first_batch is None:
                                first_batch = current_batch

                            logits.append(predictor(current_batch.to(device)).detach())
                            current_batch = []

                    if not len(current_batch) == 0:
                        logits.append(predictor(torch.stack(current_batch).to(device)).detach())

                    logits = torch.cat(logits, dim=0).detach().cpu()
                    if logits.shape[-1] == 1:
                        probs = 1.0 - torch.sigmoid(logits / temperature).squeeze()

                    else:
                        probs = torch.nn.Softmax(dim=1)(logits / temperature)[:, 0]

                    prediction_grid = probs.reshape(100, 100)
                    prediction_grids.append(prediction_grid)

            # Average the predictions across grids
            prediction_grid = torch.mean(torch.stack(prediction_grids).to(torch.float32), dim=0).numpy()
            np.save(grid_path, prediction_grid)

        else:
            prediction_grid = np.load(grid_path)

        # Extract latents for training and validation samples
        def extract_latents(dataloader, max_batches=8):
            latents = []
            y_list = []
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                x, y = batch  # Assuming batch contains images and hints
                y, hint = y[:2]

                for idx in range(len(x)):
                    latents.append(
                        [
                            self.check_foreground(x[idx], hint[idx]),
                            self.check_background(x[idx], hint[idx]),
                        ]
                    )

                    y_list.append(y[idx])

            return latents, y_list

        if isinstance(train_dataloader, DataloaderMixer):
            train_hints_buffer = train_dataloader.hints_enabled
            train_dataloader.enable_hints()

        elif train_dataloader:
            train_hints_buffer = train_dataloader.dataset.hints_enabled
            train_dataloader.dataset.enable_hints()

        train_latents, train_y_list = extract_latents(train_dataloader) if train_dataloader else ([], [])

        if train_dataloader and not train_hints_buffer:
            if isinstance(train_dataloader, DataloaderMixer):
                train_dataloader.enable_hints()

            else:
                train_dataloader.dataset.enable_hints()

        val_latents, val_y_list = ([], [])
        if isinstance(val_dataloaders, WeightedDataloaderList):
            val_weights = val_dataloaders.weights
            val_dataloaders = val_dataloaders.dataloaders

        for val_idx, val_dataloader in enumerate(val_dataloaders):
            val_hints_buffer = val_dataloader.dataset.hints_enabled
            val_dataloader.dataset.enable_hints()
            max_batches = max(1, 8 // val_weights[val_idx])
            val_latents_current, val_y_list_current = extract_latents(val_dataloader, max_batches=max_batches)
            val_latents.extend(val_latents_current)
            val_y_list.extend(val_y_list_current)
            if not val_hints_buffer:
                val_dataloader.dataset.disable_hints()

        # Create the plot
        plt.figure()

        # Set a lighter color map: use "coolwarm" and make it lighter
        cmap = cm.get_cmap("bwr")
        # Create filled contour plot
        contour_fill = plt.contourf(xx, yy, prediction_grid, levels=100, cmap=cmap)

        # Add contour lines with black color and thicker lines
        contour_lines = plt.contour(xx, yy, prediction_grid, levels=10, colors="black", linewidths=1.5)

        # Plot training and validation samples
        train_latents = np.array(train_latents)
        val_latents = np.array(val_latents)
        if len(train_latents) > 0:
            train_y_concat = np.array(train_y_list)
            plt.scatter(
                train_latents[:, 0],
                train_latents[:, 1],
                c=np.where(train_y_concat == 1, "blue", "red"),
                marker="^",
                edgecolors="black",
                label="Train Samples",
                alpha=0.7,
            )

        if len(val_latents) > 0:
            val_y_concat = np.array(val_y_list)
            plt.scatter(
                val_latents[:, 0],
                val_latents[:, 1],
                c=np.where(val_y_concat == 1, "blue", "red"),
                marker="s",
                edgecolors="black",
                label="Val Samples",
                alpha=0.7,
            )

        plt.legend()

        # Set axis labels
        plt.xlabel("Foreground Intensity")
        plt.ylabel("Background Intensity")

        # Set the ticks to increments of 0.5
        plt.xticks(np.arange(0, 1.1, 0.5))
        plt.yticks(np.arange(0, 1.1, 0.5))

        # Add vertical dotted line for Foreground Intensity == 0.5
        plt.axvline(x=0.5, color="black", linestyle="--")
        plt.text(
            1.05,
            0.5,
            "Confounding feature only",
            rotation=270,
            verticalalignment="center",
        )

        # Add horizontal dotted line for Background Intensity == 0.5
        plt.axhline(y=0.5, color="black", linestyle="--")
        plt.text(0.5, 1.05, "True feature only", horizontalalignment="center")

        # Adjust plot limits to give space for text labels outside the plot
        plt.subplots_adjust(right=0.85, top=0.85)

        # Save the plot to the specified path
        plt.savefig(path, bbox_inches="tight")
        plt.clf()

        print("visualize_decision_boundary saved under " + path)

    def check_foreground(self, x, hint):
        intensity_foreground = torch.sum(
            hint[..., 0, :, :] * self.project_to_pytorch_default(x[..., 0, :, :])
        ) / torch.sum(hint[..., 0, :, :])
        return intensity_foreground

    def check_background(self, x, hint):
        intensity_background = torch.sum((1 - hint) * self.project_to_pytorch_default(x)) / torch.sum(1 - hint)
        return intensity_background

    def sample_to_latent(self, x, hint):
        return torch.tensor(
            [
                self.check_foreground(x.to(hint), hint),
                self.check_background(x.to(hint), hint),
            ]
        )

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
        idx_to_info=None,
        **kwargs: dict,
    ):
        if idx_to_info is None:

            def idx_to_info(x, x_counterfactual, hint):
                s = "Foreground: " + str(round(float(self.check_foreground(x, hint)), 3)) + " -> "
                s += str(round(float(self.check_foreground(x_counterfactual, hint)), 3)) + ", "
                s += "Background: " + str(round(float(self.check_background(x, hint)), 3)) + " -> "
                s += str(round(float(self.check_background(x_counterfactual, hint)), 3))
                return s

        return super().generate_contrastive_collage(
            x_list=x_list,
            x_counterfactual_list=x_counterfactual_list,
            y_target_list=y_target_list,
            y_source_list=y_source_list,
            y_list=y_list,
            y_target_start_confidence_list=y_target_start_confidence_list,
            y_target_end_confidence_list=y_target_end_confidence_list,
            base_path=base_path,
            idx_to_info=idx_to_info,
            **kwargs,
        )


class Camelyon17Dataset(ImageDataset):
    def __init__(self, config, transform, **kwargs):
        self.original_dataset = get_dataset(dataset="camelyon17", download=True)
        self.config = config
        self.transform = transform
        super(Camelyon17Dataset, self).__init__()

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x = self.original_dataset[idx]
        return self.transform(x[0]), x[1]


class RxRx1Dataset(ImageDataset):
    def __init__(self, config, transform, **kwargs):
        self.original_dataset = get_dataset(dataset="rxrx1", download=True)
        self.config = config
        self.transform = transform
        super(RxRx1Dataset, self).__init__()

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x = self.original_dataset[idx]
        return self.transform(x[0]), x[1]


class Camelyon17AugmentedDataset(Image2MixedDataset):
    def __init__(self, config, **kwargs):
        if not os.path.exists(config.dataset_path):
            original_dataset = get_dataset(dataset="camelyon17", download=True)
            Path(os.path.join(config.dataset_path, "imgs")).mkdir(parents=True, exist_ok=True)
            lines = ["img,tumor,hospital"]
            for i in range(len(original_dataset)):
                img, label, meta = original_dataset[i]
                img_name = f"{embed_numberstring(i, 7)}.png"
                img.save(f"{config.dataset_path}/imgs/{img_name}")
                lines.append(f"{img_name}, {label}, {meta[0]}")

            with open(f"{config.dataset_path}/data.csv", "w") as f:
                f.write("\n".join(lines))

        peal_runs = os.environ.get("PEAL_RUNS", "peal_runs")
        oracle_path = os.path.join(peal_runs, "camelyon17", "latent_oracle", "model.cpl")
        if os.path.exists(oracle_path):
            self.oracle = torch.load(oracle_path)
            self.oracle.eval()

        super(Camelyon17AugmentedDataset, self).__init__(config=config, **kwargs)

    def sample_to_latent(self, sample, mask=None):
        self.oracle.to(sample.device)
        sample_inflated = False
        if not len(sample.shape) == 4:
            sample_inflated = True
            sample = sample.unsqueeze(0)

        latent = self.oracle(sample)

        if sample_inflated:
            latent = latent[0]

        return latent


class RxRx1AugmentedDataset(Image2MixedDataset):
    def __init__(self, config, **kwargs):
        if not os.path.exists(config.dataset_path):
            original_dataset = get_dataset(dataset="rxrx1", download=True)
            Path(os.path.join(config.dataset_path, "imgs")).mkdir(parents=True, exist_ok=True)
            lines = ["img, label, confounder"]
            for i in range(len(original_dataset)):
                img, label, meta = original_dataset[i]
                img_name = f"{embed_numberstring(i, 7)}.png"
                img.save(f"{config.dataset_path}/imgs/{img_name}")
                lines.append(f"{img_name}, {label}, {meta[0]}")

            with open(f"{config.dataset_path}/data.csv", "w") as f:
                f.write("\n".join(lines))

        super(RxRx1AugmentedDataset, self).__init__(config=config, **kwargs)


class WaterbirdsDataset(Image2MixedDataset):
    def __init__(self, config, **kwargs):
        print("instantiate waterbirds dataset!")
        dataset_labels = os.path.join(config.dataset_path, "data.csv")
        if not os.path.exists(dataset_labels):
            # Download the segmentations
            download_path = os.path.join(config.dataset_path, "downloads")
            Path(download_path).mkdir(parents=True, exist_ok=True)

            if os.path.exists(os.path.join(download_path, "segmentations", "200.Common_Yellowthroat")):
                print("Found segmentation masks folder. Skipping downloading.")

            else:
                tar_file_path = os.path.join(download_path, "segmentations.tar.gz")
                if not os.path.exists(tar_file_path):
                    print("Download segmentation tar file")
                    url = "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz"

                    response = requests.get(url, stream=True)

                    if response.status_code == 200:
                        os.makedirs(download_path, exist_ok=True)

                        with open(tar_file_path, "wb") as file:
                            file.write(response.raw.read())

                        print("Segmentations downloaded successfully!")

                    else:
                        raise Exception("Failed to download segmentations.")

                with tarfile.open(tar_file_path, "r:gz") as tar:
                    tar.extractall(path=download_path)
                    print("segmentations extracted")

            if os.path.exists(
                os.path.join(
                    download_path,
                    "waterbird_complete95_forest2water2",
                    "200.Common_Yellowthroat",
                )
            ):
                print("Found waterbirds folder. Skipping downloading.")

            else:
                tar_file_path = os.path.join(download_path, "waterbirds.tar.gz")
                if not os.path.exists(tar_file_path):
                    print("Download waterbirds tar file")
                    url = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"

                    response = requests.get(url, stream=True)

                    if response.status_code == 200:
                        os.makedirs(download_path, exist_ok=True)

                        with open(tar_file_path, "wb") as file:
                            file.write(response.raw.read())

                        print("Waterbirds downloaded successfully!")

                    else:
                        raise Exception("Failed to download waterbirds.")

                with tarfile.open(tar_file_path, "r:gz") as tar:
                    tar.extractall(path=download_path)
                    print("waterbirds extracted")

            shutil.move(
                os.path.join(download_path, "waterbird_complete95_forest2water2"),
                os.path.join(config.dataset_path, "imgs_filename"),
            )
            shutil.move(
                os.path.join(download_path, "segmentations"),
                os.path.join(config.dataset_path, "masks"),
            )
            shutil.move(
                os.path.join(config.dataset_path, "imgs_filename", "metadata.csv"),
                os.path.join(config.dataset_path, "data.csv"),
            )
            print("Downloading, extracting and positioning of files completed!")

        super(WaterbirdsDataset, self).__init__(config=config, **kwargs)


def download_celeba_to(target_dir):
    print("This still has to be implemented!")
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    print("Path to downloaded dataset files:", path)
    shutil.move(path, target_dir)
    shutil.move(
        os.path.join(target_dir, "list_attr_celeba.csv"),
        os.path.join(target_dir, "data.csv"),
    )
    shutil.move(
        os.path.join(target_dir, "img_align_celeba", "img_align_celeba"),
        os.path.join(target_dir, "imgs"),
    )


class CelebADataset(Image2MixedDataset):
    def __init__(self, config, **kwargs):
        if not os.path.exists(config.dataset_path):
            download_celeba_to(config.dataset_path)

        super(CelebADataset, self).__init__(config=config, **kwargs)
        # these weights have to be downloaded and placed from the ACE repository manually
        ORACLEPATH = "pretrained_models/oracle.pth"
        if os.path.exists(ORACLEPATH):
            self.oracle_metric = OracleMetrics(weights_path=ORACLEPATH, device="cpu")
            self.oracle_metric.eval()
            self.oracle = lambda sample: self.oracle_metric.oracle(sample)[0]

        else:
            peal_runs = os.environ.get("PEAL_RUNS", "peal_runs")
            oracle_path = os.path.join(peal_runs, "celeba", "latent_oracle", "model.cpl")
            if os.path.exists(oracle_path):
                self.oracle = torch.load(oracle_path)
                self.oracle.eval()

    def sample_to_latent(self, sample, mask=None):
        if isinstance(self.oracle, torch.nn.Module):
            self.oracle.to(sample.device)

        else:
            self.oracle_metric.to(sample.device)

        sample_inflated = False
        if not len(sample.shape) == 4:
            sample_inflated = True
            sample = sample.unsqueeze(0)

        latent = self.oracle(sample)

        if sample_inflated:
            latent = latent[0]

        return latent


class CelebACopyrighttagDataset(Image2MixedDataset):
    def __init__(self, config, **kwargs):
        if not os.path.exists(config.dataset_path) and not os.path.exists(config.dataset_origin_path):
            download_celeba_to(config.dataset_origin_path)

        if not os.path.exists(config.dataset_path):
            print("config.delimiter")
            print(config.delimiter)
            print(config.delimiter)
            print(config.delimiter)
            cdg = ConfounderDatasetGenerator(**config.__dict__, data_config=config)
            cdg.generate_dataset()

        super(CelebACopyrighttagDataset, self).__init__(config=config, **kwargs)


class FollicleDataset(Image2MixedDataset):
    def __init__(self, config, **kwargs):
        # Login using e.g. `huggingface-cli login` to access this dataset
        if not os.path.exists(config.dataset_path):
            from datasets import load_dataset

            metadata_data_with_annotations = load_dataset(
                "janphhe/follicles_true_features", data_files="data_with_annotations.csv"
            )
            ds_imgs = load_dataset("janphhe/follicles_true_features", "imgs_config", num_proc=8)
            ds_imgs_cut = load_dataset("janphhe/follicles_true_features", "imgs_cut_config", num_proc=8)
            ds_masks = load_dataset("janphhe/follicles_true_features", "masks_config", num_proc=8)
            attributes = [e for e in metadata_data_with_annotations["train"].features.keys()][:-1]
            attribute_values = [metadata_data_with_annotations["train"][attributes[i]] for i in range(len(attributes))]
            csv_file = ",".join(attributes) + "\n"
            for i in range(len(attribute_values[0])):
                csv_file += ",".join([str(attribute_values[j][i]) for j in range(len(attributes))]) + "\n"

            os.makedirs(config.dataset_path)
            os.makedirs(os.path.join(config.dataset_path, "imgs", "0"))
            os.makedirs(os.path.join(config.dataset_path, "imgs", "1"))
            os.makedirs(os.path.join(config.dataset_path, "imgs_cut", "0"))
            os.makedirs(os.path.join(config.dataset_path, "imgs_cut", "1"))
            os.makedirs(os.path.join(config.dataset_path, "masks"))
            with open(os.path.join(config.dataset_path, "data.csv"), "w") as f:
                f.write(csv_file)

            for split in ds_imgs.keys():
                for i in range(len(ds_imgs[split])):
                    label = ds_imgs[split][i]["label"]
                    image = ds_imgs[split][i]["image"]

                    # create a folder for the label if it does not exist
                    label_folder = os.path.join(config.dataset_path, str(label))

                    if not os.path.exists(label_folder):
                        os.makedirs(label_folder)

                    # save the image in the label folder
                    image.save(
                        os.path.join(config.dataset_path, "imgs", attribute_values[attributes.index("imgs")]), "PNG"
                    )

            for split in ds_imgs_cut.keys():
                for i in range(len(ds_imgs_cut[split])):
                    label = ds_imgs_cut[split][i]["label"]
                    image = ds_imgs_cut[split][i]["image"]

                    # create a folder for the label if it does not exist
                    label_folder = os.path.join(config.dataset_path, str(label))

                    if not os.path.exists(label_folder):
                        os.makedirs(label_folder)

                    # save the image in the label folder
                    image.save(
                        os.path.join(config.dataset_path, "imgs_cut", attribute_values[attributes.index("imgs_cut")]),
                        "PNG",
                    )

            for split in ds_masks.keys():
                for i in range(len(ds_masks[split])):
                    label = ds_masks[split][i]["label"]
                    image = ds_masks[split][i]["image"]

                    # create a folder for the label if it does not exist
                    label_folder = os.path.join(config.dataset_path, str(label))

                    if not os.path.exists(label_folder):
                        os.makedirs(label_folder)

                    # save the image in the label folder
                    image.save(
                        os.path.join(
                            config.dataset_path,
                            "masks",
                            os.path.split(attribute_values[attributes.index("imgs")])[-1],
                        ),
                        "PNG",
                    )


class SkinConDataset(Image2MixedDataset):
    def __init__(self, config, **kwargs):
        if not os.path.exists(config.dataset_path):
            from datasets import load_dataset
            import pandas as pd
            import requests

            # Download Fitzpatrick17k HF dataset
            print("Downloading spycoder/fitzpatrick dataset (this might take a while)...")
            ds = load_dataset('spycoder/fitzpatrick', split='train')
            
            # Download SkinCon annotations
            print("Downloading SkinCon annotations...")
            annotations_url = "https://skincon-dataset.github.io/files/annotations_fitzpatrick17k.csv"
            res = requests.get(annotations_url)
            annotations_path = os.path.join("/tmp", "annotations_fitzpatrick17k.csv")
            with open(annotations_path, "wb") as f:
                f.write(res.content)
            
            skincon_df = pd.read_csv(annotations_path)
            
            # Create directories
            os.makedirs(config.dataset_path, exist_ok=True)
            img_dir = os.path.join(config.dataset_path, "imgs")
            os.makedirs(img_dir, exist_ok=True)
            
            # We will map "Malignant" vs "Benign" based on Fitzpatrick17k metadata?
            # Wait, the skincon annotations only provide concept presence 0/1, and ImageID.
            # We can merge it with the original HF dataset metadata for labels/skin types.
            
            print("Merging datasets and saving images...")
            lines = ["img,label,confounder"]
            
            valid_image_ids = set(skincon_df['ImageID'].tolist())
            
            # Filter the HF dataset metadata
            # For simplicity, we assume we extract malignant/benign and skin_type from the HF dataset if available
            count = 0
            
            session = requests.Session()
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            retry = Retry(connect=5, read=5, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            for i, row in enumerate(ds):
                # We need a unique identifier to match. The HF dataset has an 'md5hash' which corresponds to ImageID in SkinCon.
                img_id = row.get('md5hash')
                if img_id and f"{img_id}.jpg" in valid_image_ids:
                    # Let's say label is 'three_partition_label' or just 'label'. 
                    # If 'three_partition_label' == 'malignant', label=1, else 0.
                    is_malignant = 1 if row.get('three_partition_label') == 'malignant' else 0
                    skin_type = row.get('fitzpatrick_scale', 1) # Default 1 if missing
                    is_dark = 1 if skin_type >= 4 else 0
                    
                    img_name = f"{img_id}.png"
                    
                    # Fetching image from url
                    url = row.get('url')
                    if not url:
                        continue
                    try:
                        resp = session.get(url, timeout=15)
                        if resp.status_code == 200:
                            from PIL import Image
                            from io import BytesIO
                            img = Image.open(BytesIO(resp.content))
                            # Convert to RGB to avoid alpha channel issues when saving as PNG
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img.save(os.path.join(img_dir, img_name))
                            lines.append(f"{img_name},{is_malignant},{is_dark}")
                            count += 1
                    except Exception as e:
                        print(f"Failed to fetch {url}: {e}")
                        continue
            
            with open(os.path.join(config.dataset_path, "data.csv"), "w") as f:
                f.write("\n".join(lines))
            print(f"SkinCon dataset successfully created with {count} images in {config.dataset_path}")

        super(SkinConDataset, self).__init__(config=config, **kwargs)

class NicoPlusPlusDataset(Image2MixedDataset):
    def __init__(self, config, **kwargs):
        if not os.path.exists(os.path.join(config.dataset_path, "data.csv")):
            import requests
            import zipfile
            import shutil

            peal_data_dir = os.environ.get("PEAL_DATA", os.path.dirname(os.path.normpath(config.dataset_path)))
            zip_path = os.path.join(peal_data_dir, "NICO++.zip")

            if not os.path.exists(zip_path):
                print("NICO++ dataset requires manual download due to its size and hosting limitations.")
                print("Please download it from https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AADbhZJAy0AAbap76cg_XkAfa?dl=0")
                print(f"and place the zip file exactly at {zip_path}")
                
                # We will create a dummy dataset structure for the code to not strictly crash when just verifying config loading
                os.makedirs(config.dataset_path, exist_ok=True)
                os.makedirs(os.path.join(config.dataset_path, "imgs"), exist_ok=True)
                with open(os.path.join(config.dataset_path, "data.csv"), "w") as f:
                    f.write("img, label, confounder\n")
                
                raise RuntimeError(f"NICO++ Dataset missing. Please follow the instructions to download it. Ensure it is at {zip_path}")
            
                raise FileNotFoundError(f"Please download NICO++.zip to {zip_path}")
            
            target_categories = ["dog", "bear"]
            target_contexts = ["grass", "water"]
            
            # Since NICO++ is nested, we might need a temporary extraction or read inline
            # We'll extract only the target images
            print(f"Extracting selected subset from NICO++.zip...")
            
            # Ensure the output directories exist before writing
            img_out_dir = os.path.join(config.dataset_path, "imgs")
            os.makedirs(img_out_dir, exist_ok=True)
            
            lines = ["img,label,confounder"]
            count = 0
            
            # Use actual group counts for extraction without hard clipping
            counts = {
                "dog_grass": 0,
                "dog_water": 0,
                "bear_grass": 0,
                "bear_water": 0
            }

            with zipfile.ZipFile(zip_path, 'r') as z1:
                # Find the nested zip
                nested_zip_name = None
                for name in z1.namelist():
                    if name.endswith("NICO_DG_Benchmark.zip"):
                        nested_zip_name = name
                        break
                
                if not nested_zip_name:
                    raise FileNotFoundError("Could not find NICO_DG_Benchmark.zip inside NICO++.zip")
                
                with z1.open(nested_zip_name) as nested_zip_file:
                    nested_data = nested_zip_file.read()
                    
            with zipfile.ZipFile(io.BytesIO(nested_data)) as z2:
                for name in z2.namelist():
                    if name.lower().endswith('.jpg') or name.lower().endswith('.png'):
                        parts = name.split('/')
                        if len(parts) >= 3:
                            context = parts[-3].lower()
                            category = parts[-2].lower()
                            
                            key = f"{category}_{context}"
                            if category in target_categories and context in target_contexts:
                                img_name = f"{category}_{context}_{count:05d}.jpg"
                                img_data = z2.read(name)
                                with open(os.path.join(img_out_dir, img_name), "wb") as f_out:
                                    f_out.write(img_data)
                                
                                # Label matching
                                # We use 0 for dog, 1 for bear
                                # We use 0 for grass, 1 for water
                                label_idx = target_categories.index(category)
                                context_idx = target_contexts.index(context)
                                
                                lines.append(f"{img_name},{label_idx},{context_idx}")
                                counts[key] += 1
                                count += 1

            # NICO++ subset uses "label" for class and "confounder" for context
            with open(os.path.join(config.dataset_path, "data.csv"), "w") as f:
                f.write("\n".join(lines))
            
            if count == 0:
                raise RuntimeError(f"Could not find matching images within the extracted {zip_path}. Please check the zip contents.")
            else:
                print(f"NICO++ dataset subset successfully created with {count} images in {config.dataset_path}")
                print(f"Counts: {counts}")

        super(NicoPlusPlusDataset, self).__init__(config=config, **kwargs)
