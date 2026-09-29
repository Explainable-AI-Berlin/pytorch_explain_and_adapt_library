"""
Dataset classes for the concrete benchmarks PEAL experiments run on.

Every class here wraps one dataset behind the loaders of ``peal.data.datasets``
and materialises it first when necessary: the synthetic generators of
``peal.data.dataset_generators`` (square, sparse numbers, funny nodules,
stamped-on confounders) are run on demand, and the public benchmarks (MNIST,
CelebA, Waterbirds, Camelyon17, RxRx1, NICO++, SkinCon, ISIC, ...) are
downloaded and rewritten into PEAL's ``imgs/`` plus ``data.csv`` layout.
Datasets with a known confounder additionally provide the oracle hooks
(``sample_to_2d_latent``, ``has_confounder``) that the counterfactual
explainers and the decision boundary plots call back into.
"""

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

from peal._optional import require
from peal.data.dataloaders import DataloaderMixer, WeightedDataloaderList
from peal.data.dataset_generators import (
    SquareDatasetGenerator,
    ConfounderDatasetGenerator,
    SparseNumbersDatasetGenerator,
    OnlySparseNumbersDatasetGenerator,
    OnlySparseNumbersZipfDatasetGenerator,
    SparseNumbersZipfDatasetGenerator,
    FunnyNodulesDatasetGenerator,
)
from peal.data.datasets import (
    Image2ClassDataset,
    Image2MixedDataset,
    ImageDataset,
)
from peal.data.interfaces import DataConfig
from peal.global_utils import embed_numberstring
from peal.data.dataset_generators import latent_to_square_image
from peal.data.dataset_utils import parse_csv
from peal.log import get_logger

_log = get_logger(__name__)


class MnistDataset(Image2ClassDataset):
    """
    Plain MNIST in the ImageFolder layout, without any confounder.

    On first use the torchvision MNIST train and test splits are downloaded to
    ``<dataset_path>_train_raw`` and ``<dataset_path>_val_raw`` and written out
    as PNGs under ``<dataset_path>/imgs/<digit>/<running index>.png``, so that
    ``Image2ClassDataset`` can take over and split them by ``config.split``.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``; the remaining fields are interpreted by
        ``Image2ClassDataset``.
    **kwargs
        Forwarded to ``Image2ClassDataset`` (``mode``, ``transform``, ...).
    """

    def __init__(self, config: DataConfig, **kwargs):
        """Write the MNIST PNG tree if ``config.dataset_path`` does not exist yet."""
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
    """
    ``DataConfig`` of :class:`ColoredMnist` with the colouring specification.

    Parameters
    ----------
    coloring : list of (int, float)
        Per digit class, the fraction of its samples that is tinted red. This
        is how strongly the colour confounds that class.
    raw_path : str or None
        Where the raw torchvision MNIST download is kept; defaults to
        ``dataset_path``.
    group_map : str or None
        CSV holding the group (``Colored``) flag per file; defaults to
        ``<dataset_path>/data.csv``.
    """

    config_name: str = "ColoredMnistConfig"
    coloring: list[tuple[int, float]] = []
    raw_path: Union[type(None), str] = None
    group_map: str = None


class ColoredMnist(Image2ClassDataset):
    """
    MNIST with a red tint as a controllable confounder.

    The dataset is generated once into ``config.dataset_path`` (see
    ``_create_dataset``): 4500 train and 750 val/test samples per digit are
    written to ``imgs_train``/``imgs_val``/``imgs_test``, and for every digit
    listed in ``config.coloring`` the requested fraction of its samples is
    turned into a red-channel-only RGB image. ``data.csv`` records
    ``Name, Digit, Colored, Subset`` and is read back as the group map, so that
    ``has_confounder`` can report the colouring of a single file.

    Parameters
    ----------
    mode : str
        ``"train"``, ``"val"`` or ``"test"``. Selects the ``imgs_*`` directory;
        the split fractions are fixed per mode because each subset already
        lives in its own directory.
    config : ColoredMnistConfig
        Deep-copied before the mode-specific fields are overwritten.
    **kwargs
        Forwarded to ``Image2ClassDataset``.

    Attributes
    ----------
    group_labels : dict
        Image filename to ``Colored`` flag (0 or 1).
    """

    def __init__(self, mode: str, config: ColoredMnistConfig, **kwargs):
        """Generate the dataset if missing, read the group map and load the split."""
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
        group_map_file = self.config.group_map or os.path.join(
            self.config.dataset_path, "data.csv"
        )
        with open(group_map_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            col_filename = header.index("Name")
            col_colored = header.index("Colored")
            for row in reader:
                self.group_labels[row[col_filename]] = int(row[col_colored])

        super(ColoredMnist, self).__init__(config=self.config, mode=mode, **kwargs)

    def has_confounder(self, filename: str) -> int:
        """Return 1 if the image ``filename`` was tinted red, else 0."""
        return self.group_labels[filename]

    def _create_dataset(self):
        _log.info(
            "%s",
            f"creating new ColoredMnist dataset with coloring {self.config.coloring} at {self.config.dataset_path}",
        )
        np.random.seed(0 if self.config.seed is None else self.config.seed)

        data_info = {"Name": [], "Digit": [], "Colored": [], "Subset": []}

        def color_sample(coloring_wanted, coloring_actual, label, img):
            """Tint ``img`` red while ``label`` has quota left, and record the flag."""
            if (
                label in coloring_wanted
                and coloring_actual[label] < coloring_wanted[label]
            ):
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
            class_coloring[0]: round(
                class_coloring[1] * number_samples_max_val_and_test
            )
            for class_coloring in self.config.coloring
        }

        raw_path = (
            self.config.dataset_path
            if self.config.raw_path is None
            else self.config.raw_path
        )

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
                filename = os.path.join(
                    dir_train, str(label), str(current).zfill(6) + ".png"
                )
                data_info["Subset"].append("train")

            elif number_samples_val[label] < number_samples_max_val_and_test:
                number_samples_val[label] += 1
                img = color_sample(
                    coloring_val_and_test, coloring_actual_val, label, img
                )
                filename = os.path.join(
                    dir_val, str(label), str(current).zfill(6) + ".png"
                )
                data_info["Subset"].append("val")

            elif number_samples_test[label] < number_samples_max_val_and_test:
                number_samples_test[label] += 1
                img = color_sample(
                    coloring_val_and_test, coloring_actual_test, label, img
                )
                filename = os.path.join(
                    dir_test, str(label), str(current).zfill(6) + ".png"
                )
                data_info["Subset"].append("test")

            else:
                continue

            img.save(filename)
            data_info["Name"].append(str(current).zfill(6) + ".png")
            data_info["Digit"].append(label)
            current += 1

        with open(
            os.path.join(self.config.dataset_path, "data.csv"), "w", newline=""
        ) as f:
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
    extent=[0, 1, 0, 1],
    xlabel="Foreground Intensity",
    ylabel="Background Intensity",
    attempts=1,
):
    """
    Scatter factual and counterfactual 2D latents and connect them by arrows.

    One subplot per counterfactual attempt: the decision boundary grid is drawn
    as a semi-transparent ``bwr`` background, originals are circles and
    counterfactuals squares (both filled by the target-class confidence), and a
    green arrow points from each original to its counterfactual. The figure is
    written to ``filename`` and closed. Called by
    ``ImageDataset.global_counterfactual_visualization``.

    Parameters
    ----------
    original_latents, counterfactual_latents : array_like
        ``[N, 2]`` latent coordinates. The attempts are consecutive blocks of
        ``N // attempts`` entries.
    filename : str
        Output image path.
    y_target_start_confidence, y_target_end_confidence : array_like
        Target-class confidence before and after the edit, used as the
        colormap value of the two markers.
    decision_boundary : numpy.ndarray
        Probability grid shown as the background image.
    extent : list of float, optional
        ``[xmin, xmax, ymin, ymax]`` of both the background and the axes.
    xlabel, ylabel : str, optional
        Axis labels. ``"Foreground Intensity"`` additionally draws the 0.5
        guide lines of the square dataset.
    attempts : int, optional
        Number of subplots the latents are split into.
    """
    fig, axes = plt.subplots(1, attempts, figsize=(6 * attempts, 6))
    axes = np.atleast_1d(axes)

    # Convert to numpy arrays for easy manipulation
    original_latents = np.array(original_latents)
    counterfactual_latents = np.array(counterfactual_latents)
    y_target_start_confidence = np.array(y_target_start_confidence)
    y_target_end_confidence = np.array(y_target_end_confidence)

    # Define the colormap for the points: blue -> red
    cmap = cm.get_cmap("bwr")  # blue to red

    N = len(original_latents) // attempts

    for a in range(attempts):
        ax = axes[a]
        orig_lat_a = original_latents[a * N : (a + 1) * N]
        cf_lat_a = counterfactual_latents[a * N : (a + 1) * N]
        start_conf_a = y_target_start_confidence[a * N : (a + 1) * N]
        end_conf_a = y_target_end_confidence[a * N : (a + 1) * N]

        # Display the decision boundary grid as the background
        ax.imshow(
            decision_boundary,
            extent=extent,  # Extend dynamically based on latents
            origin="lower",  # Aligns the grid with the bottom-left of the plot
            cmap=cmap,  # Apply custom colormap
            alpha=0.3,  # Make the background semi-transparent
        )

        # Plot original latents and counterfactuals with different markers
        for i, (orig, cf, start_conf, end_conf) in enumerate(
            zip(
                orig_lat_a,
                cf_lat_a,
                start_conf_a,
                end_conf_a,
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
            )

            # Plot counterfactual point with square marker
            ax.scatter(
                cf[0],
                cf[1],
                facecolor=end_color,
                edgecolor="darkred",
                marker="s",  # Square for counterfactual
            )

            # Draw arrow between original and counterfactual points
            ax.annotate(
                "",
                xy=(cf[0], cf[1]),
                xytext=(orig[0], orig[1]),
                arrowprops=dict(
                    fc="green",
                    ec="green",
                    edgecolor="yellow",
                    arrowstyle="->",
                    alpha=0.7,
                ),
            )

        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[2], extent[3]])
        ax.set_box_aspect(1)

        # Axis labels
        ax.set_xlabel(xlabel)
        if a == 0:
            ax.set_ylabel(ylabel)

        if xlabel == "Foreground Intensity":
            # Add vertical dotted line for Foreground Intensity == 0.5
            ax.axvline(x=0.5, color="black", linestyle="--")
            ax.text(
                extent[1] + (extent[1] - extent[0]) * 0.05,
                0.5,
                "Confounding feature only",
                rotation=270,
                verticalalignment="center",
            )

            # Add horizontal dotted line for Background Intensity == 0.5
            ax.axhline(y=0.5, color="black", linestyle="--")
            ax.text(
                0.5,
                extent[3] + (extent[3] - extent[2]) * 0.05,
                "True feature only",
                horizontalalignment="center",
            )

        ax.grid(True)
        if attempts > 1:
            ax.set_title(f"Attempt {a+1}")

    # Create neutral markers for the legend (gray fill color)
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

    # Add explanation for the background colors in the legend
    blue_patch = plt.Line2D([0], [0], color="blue", lw=4, label="Pred > 0.5")
    red_patch = plt.Line2D([0], [0], color="red", lw=4, label="Pred < 0.5")

    # Display the updated legend outside the subplots
    fig.legend(
        handles=[original_marker, cf_marker, blue_patch, red_patch],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )

    # Show the plot
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.clf()
    plt.close(fig)


class SquareDataset(Image2MixedDataset):
    """
    Synthetic 64x64 squares whose colour is confounded by the background.

    An image shows a bordered square with red interior intensity ``ColorA``
    (the true feature) on a grey background of intensity ``ColorB`` (the
    confounder), at one of nine positions. ``SquareDatasetGenerator`` writes
    the dataset on first use and only supports
    ``config.confounder_probability == 0.5``; confounded variants are derived
    from that pool elsewhere. Since both generative factors are known, this
    class provides an exact 2D latent readout for the decision boundary and
    counterfactual plots, either from the hint mask or from an oracle model.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``, ``confounder_probability`` and
        ``confounding_factors``.
    **kwargs
        Forwarded to ``Image2MixedDataset``.

    Attributes
    ----------
    oracle : torch.nn.Module, optional
        ``$PEAL_RUNS/square/colora_confounding_colorb/
        classifier_all_attributes_resnet18/model.cpl`` in eval mode, if
        present. Used by ``sample_to_2d_latent`` when no mask is given.

    Raises
    ------
    NotImplementedError
        If the dataset has to be generated but ``confounder_probability`` is
        not 0.5.
    """

    def __init__(self, config: DataConfig, **kwargs):
        """Generate the dataset if missing and load the latent oracle if present."""
        if not os.path.exists(config.dataset_path):
            if config.confounder_probability == 0.5:
                cdg = SquareDatasetGenerator(data_config=config)
                cdg.generate_dataset()

            else:
                raise NotImplementedError(
                    "Only confounder_probability=0.5 can be used to generate the dataset"
                )

        peal_runs = os.environ.get("PEAL_RUNS", "peal_runs")
        oracle_path = os.path.join(
            peal_runs,
            "square",
            "colora_confounding_colorb",
            "classifier_all_attributes_resnet18",
            "model.cpl",
        )
        if os.path.exists(oracle_path):
            try:
                self.oracle = torch.load(oracle_path)
            except Exception:
                self.oracle = torch.load(oracle_path, weights_only=False)
            self.oracle.eval()

        super(SquareDataset, self).__init__(config=config, **kwargs)

    def visualize_decision_boundary(self, *args, **kwargs):
        """
        Draw both the generic oracle plot and the precise hint-based plot.

        The base implementation is run with hints disabled (so the predictor sees
        plain images), the previous hint setting is restored, and
        ``visualize_decision_boundary_precise`` is called with the same arguments.
        """
        hints_enabled_buffer = self.hints_enabled
        self.disable_hints()
        super().visualize_decision_boundary(*args, **kwargs)
        if hints_enabled_buffer:
            self.enable_hints()
        self.visualize_decision_boundary_precise(*args, **kwargs)

    def visualize_decision_boundary_precise(
        self,
        predictor,
        batch_size,
        device,
        path,
        temperature=1.0,
        train_dataloader=None,
        val_dataloaders=[],
        val_weights=[],
        test_dataloader=None,
        **kwargs,
    ):
        """
        Render the decision boundary over the true generative factors.

        Unlike the base implementation, which fits a model in an estimated latent
        space, this evaluates ``predictor`` on images synthesised directly from the
        grid: for each of the nine square positions a 100x100 grid of
        ``(ColorA, ColorB)`` pairs is rendered with ``latent_to_square_image`` and
        the softmax probability of class 0 is averaged over the positions. The grid
        is cached next to the plot as a ``.npy`` file and reused on later calls.
        Train and validation samples are then overlaid at their measured foreground
        and background intensity, which needs hints and therefore temporarily
        enables them on the given loaders.

        Parameters
        ----------
        predictor : torch.nn.Module
            Classifier, evaluated on ``device`` in batches of ``batch_size``.
        batch_size : int
            Batch size of the grid evaluation.
        device : torch.device or str
            Device the predictor runs on.
        path : str
            Path of the non-precise plot; ``.png`` is replaced by
            ``_precise.png`` and the cached grid by ``_precise.npy``.
        temperature : float, optional
            Softmax temperature applied to the logits.
        train_dataloader : DataLoader or DataloaderMixer, optional
            Up to 8 batches are plotted as triangles.
        val_dataloaders : list or WeightedDataloaderList, optional
            Validation loaders, plotted as squares; each gets a budget of
            ``8 // val_weights[i]`` batches.
        val_weights : list of float, optional
            Weights of ``val_dataloaders``; taken from the list itself when it is a
            ``WeightedDataloaderList``.
        test_dataloader : DataLoader, optional
            Unused; kept for interface compatibility.
        """
        _log.info("%s", "visualize_decision_boundary_precise")
        path = path.replace(".png", "_precise.png")
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
                        logits.append(
                            predictor(torch.stack(current_batch).to(device)).detach()
                        )

                    logits = torch.cat(logits, dim=0).detach().cpu()
                    prediction_grid = torch.nn.Softmax(dim=1)(logits / temperature)[
                        :, 0
                    ].reshape(100, 100)
                    prediction_grids.append(prediction_grid)

            # Average the predictions across grids
            prediction_grid = torch.mean(
                torch.stack(prediction_grids).to(torch.float32), dim=0
            ).numpy()
            np.save(grid_path, prediction_grid)

        else:
            prediction_grid = np.load(grid_path)

        # Extract latents for training and validation samples
        def extract_latents(dataloader, max_batches=8):
            """
            Collect ``[foreground, background]`` intensities and labels of a loader.
            """
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

        train_latents, train_y_list = (
            extract_latents(train_dataloader) if train_dataloader else ([], [])
        )

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
            val_latents_current, val_y_list_current = extract_latents(
                val_dataloader, max_batches=max_batches
            )
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
        contour_lines = plt.contour(
            xx, yy, prediction_grid, levels=10, colors="black", linewidths=1.5
        )

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

        _log.info("%s", "visualize_decision_boundary saved under " + path)

    def check_foreground(self, x, hint):
        """
        Mean red intensity inside the hint mask, i.e. the square's interior.

        Parameters
        ----------
        x : torch.Tensor
            Image ``[..., C, H, W]`` in the dataset's own value range; only
            channel 0 is read.
        hint : torch.Tensor
            Binary mask of the inner square with the same shape.

        Returns
        -------
        torch.Tensor
            Scalar mean of the masked red channel.

        Notes
        -----
        In contrast to ``check_background`` the image is not projected to the
        pytorch default range first.
        """
        intensity_foreground = torch.sum(
            hint[..., 0, :, :] * x[..., 0, :, :]
        ) / torch.sum(hint[..., 0, :, :])
        return intensity_foreground

    def check_background(self, x, hint):
        """
        Mean intensity outside the hint mask, i.e. the background level.

        ``x`` is mapped to the ``[0, 1]`` pytorch default range first and averaged
        over all channels where the mask is zero.

        Parameters
        ----------
        x : torch.Tensor
            Image ``[..., C, H, W]`` in the dataset's own value range.
        hint : torch.Tensor
            Binary mask of the inner square.

        Returns
        -------
        torch.Tensor
            Scalar mean intensity of the unmasked area.
        """
        intensity_background = torch.sum(
            (1 - hint) * self.project_to_pytorch_default(x)
        ) / torch.sum(1 - hint)
        return intensity_background

    def sample_to_2d_latent(self, sample, mask=None):
        """
        Map a sample to its two generative factors.

        With a hint mask the factors are measured directly as
        ``[check_foreground, check_background]``. Without a mask the oracle
        classifier is applied instead and, if ``config.confounding_factors`` is
        set, its outputs are reduced to the attributes named there.

        Parameters
        ----------
        sample : torch.Tensor
            Image ``[C, H, W]`` or batch ``[N, C, H, W]``; a single image is
            unsqueezed for the oracle and squeezed again afterwards.
        mask : torch.Tensor, optional
            Hint mask of the inner square.

        Returns
        -------
        torch.Tensor
            The two latent values.
        """
        if mask is not None:
            sample = sample.to(mask)
            return torch.tensor(
                [
                    self.check_foreground(sample, mask),
                    self.check_background(sample, mask),
                ]
            )
        else:
            self.oracle.to(sample.device)
            sample_inflated = False
            if len(sample.shape) != 4:
                sample_inflated = True
                sample = sample.unsqueeze(0)

            latent = self.oracle(sample)

            if sample_inflated:
                latent = latent[0]

            cf = getattr(self.config, "confounding_factors", None)
            if cf and hasattr(self, "attributes"):
                indices = [self.attributes.index(f) for f in cf]
                latent = latent[indices]

            return latent

    def global_counterfactual_visualization(
        self,
        filename,
        x_list,
        counterfactuals,
        y_target_start_confidence,
        y_target_end_confidence,
        y_list,
        hint_list,
        attempts=1,
    ):
        """
        Render the counterfactual arrow plot twice: from the oracle and the hints.

        The base implementation is first called with hints disabled and without
        masks (latents from the oracle), then again with ``hint_list`` writing to
        ``*_precise.png`` - matching the two grids that
        ``visualize_decision_boundary`` saves.

        Parameters
        ----------
        filename : str
            Output path of the oracle version; the precise one replaces ``.png``
            by ``_precise.png``.
        x_list, counterfactuals : list of torch.Tensor
            Factual and counterfactual images.
        y_target_start_confidence, y_target_end_confidence : list of float
            Target-class confidence before and after the edit.
        y_list : list
            Ground truth labels.
        hint_list : list of torch.Tensor
            Masks used by the precise version.
        attempts : int, optional
            Counterfactual attempts per sample (one subplot each).
        """
        # Oracle Version (without hints)
        hints_enabled_buffer = self.hints_enabled
        self.disable_hints()
        super().global_counterfactual_visualization(
            filename,
            x_list,
            counterfactuals,
            y_target_start_confidence,
            y_target_end_confidence,
            y_list,
            None,
            attempts=attempts,
        )
        if hints_enabled_buffer:
            self.enable_hints()

        # Precise Version (with hints)
        precise_filename = filename.replace(".png", "_precise.png")
        super().global_counterfactual_visualization(
            precise_filename,
            x_list,
            counterfactuals,
            y_target_start_confidence,
            y_target_end_confidence,
            y_list,
            hint_list,
            attempts=attempts,
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
        """
        Contrastive collage annotated with the oracle and hint latents.

        When no ``idx_to_info`` callback is passed, one is built that reports per
        confounding factor the factual and counterfactual value as measured by the
        oracle and by the hint mask, e.g. ``ColorA: 0.2 (Oracle) / 0.21 (Hint) ->
        0.8 (Oracle) / 0.79 (Hint)``. Everything else is delegated to
        ``ImageDataset.generate_contrastive_collage``.

        Parameters
        ----------
        x_list, x_counterfactual_list : list of torch.Tensor
            Factual and counterfactual images.
        y_target_list, y_source_list, y_list : list
            Target class, predicted source class and ground truth per pair.
        y_target_start_confidence_list, y_target_end_confidence_list : list
            Target-class confidence before and after the edit.
        base_path : str
            Output directory of the collages.
        idx_to_info : callable, optional
            ``(x, x_counterfactual, hint) -> str`` used for the title suffix.
        **kwargs
            Passed through to the base implementation.

        Returns
        -------
        tuple
            Whatever the base implementation returns.
        """
        if idx_to_info is None:
            cf_names = getattr(self.config, "confounding_factors", None)

            def idx_to_info(x, x_counterfactual, hint):
                """
                Latent readout of one pair, measured by both the oracle and the mask.
                """
                device = x.device if hasattr(x, "device") else "cpu"
                with torch.no_grad():
                    # Hints (Precise)
                    latent_orig_hints = self.sample_to_2d_latent(x.to(device), hint)
                    latent_cf_hints = self.sample_to_2d_latent(
                        x_counterfactual.to(device), hint
                    )

                    # Oracle
                    latent_orig_oracle = self.sample_to_2d_latent(x.to(device), None)
                    latent_cf_oracle = self.sample_to_2d_latent(
                        x_counterfactual.to(device), None
                    )

                parts = []
                for i in range(len(latent_orig_hints)):
                    name = cf_names[i] if cf_names and i < len(cf_names) else f"Dim{i}"
                    parts.append(
                        f"{name}: {round(float(latent_orig_oracle[i]), 3)} (Oracle) / {round(float(latent_orig_hints[i]), 3)} (Hint) -> "
                        f"{round(float(latent_cf_oracle[i]), 3)} (Oracle) / {round(float(latent_cf_hints[i]), 3)} (Hint)"
                    )
                return ", ".join(parts)

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


class OnlySparseNumbersDataset(Image2MixedDataset):
    """
    Synthetic images showing a 2x4 grid of numbers and nothing else.

    ``OnlySparseNumbersDatasetGenerator`` writes the dataset on first use. Its
    ``data.csv`` stores one row of 8 slot columns per image, each holding the
    number drawn in that slot or ``-1``; this class expands that row into a
    binary vector over all ``config.output_split`` (default 1000) possible
    numbers, so a classifier can be trained on "does number k appear". Targets
    are therefore selected by attribute names of the form ``"Num<k>"``.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``, ``output_split``, ``output_size`` and
        ``confounding_factors``.
    **kwargs
        Forwarded to the generator and to ``Image2MixedDataset``.

    Attributes
    ----------
    attributes : list of str
        ``Num0 ... Num<output_split - 1>``, the expanded label vector;
        ``attributes_positive``/``attributes_negative`` are its phrasings.

    Notes
    -----
    Since ``attributes`` is the expanded vector while ``data[key]`` is still
    the raw slot row, the base-class helpers that index the row by attribute
    position are overridden below.
    """

    def __init__(self, config: DataConfig, **kwargs):
        """Generate the dataset if missing and build the expanded attribute names."""
        if not os.path.exists(os.path.join(config.dataset_path, "data.csv")):
            sdg = OnlySparseNumbersDatasetGenerator(data_config=config, **kwargs)
            sdg.generate_dataset()

        super().__init__(config=config, **kwargs)
        n_unique_numbers = (
            self.config.output_split if self.config.output_split else 1000
        )
        self.attributes = [f"Num{i}" for i in range(n_unique_numbers)]
        self.attributes_positive = [f"Is Num{i}" for i in range(n_unique_numbers)]
        self.attributes_negative = [f"Not Num{i}" for i in range(n_unique_numbers)]

    # ------------------------------------------------------------------
    # self.attributes is the EXPANDED 1000-slot vector (Num0..Num999) while
    # self.data[key] is the raw csv row: 8 digit slots then the Num128/Num713
    # flags. The base implementations of enable_class_restriction and
    # set_task_specific_keys index the raw row with the expanded index, so for
    # any selection past the row width (e.g. Num128 -> 128 vs a 10-wide row) the
    # bounds check fails, nothing is ever kept, and - because the body is wrapped
    # in a bare `except Exception: pass` - both classes silently come back EMPTY.
    # With class_balanced training that yields a 0-sample val loader and a bare
    # StopIteration out of the Logger. Resolve the label from the raw row the same
    # way __getitem__ does, without decoding the image.
    # ------------------------------------------------------------------
    def _selected_attr_idx(self):
        sel = self.task_config.y_selection[0]
        if isinstance(sel, str) and sel.startswith("Num"):
            return int(sel[3:])
        if isinstance(sel, int):
            return sel
        return self.attributes.index(sel)

    def _expanded_label_value(self, key, attr_idx):
        n_slots = self.config.output_size[0] if self.config.output_size else 8
        for value in self.data[key][:n_slots]:
            try:
                if int(value) == attr_idx:
                    return 1
            except (TypeError, ValueError):
                continue
        return 0

    def enable_class_restriction(self, class_idx):
        """
        Restrict ``keys`` to the samples of one class of the selected number.

        Parameters
        ----------
        class_idx : int
            0 keeps the images without the selected number, 1 those with it.
        """
        assert self.task_config is not None, "Task config must be set"
        self.backup_keys = copy.deepcopy(self.keys)
        attr_idx = self._selected_attr_idx()
        self.keys = [
            key
            for key in self.backup_keys
            if self._expanded_label_value(key, attr_idx) == class_idx
        ]
        self.class_restrictions_enabled = True

    def set_task_specific_keys(self):
        """
        Subsample ``keys`` to the proportions in ``config.class_ratios``.

        Labels are resolved with ``_expanded_label_value`` instead of by indexing
        the raw csv row (see the class notes); the unrestricted keys are kept in
        ``keys_backup``.
        """
        attr_idx = self._selected_attr_idx()
        labels = {key: self._expanded_label_value(key, attr_idx) for key in self.keys}

        num_samples_per_class = np.zeros([self.output_size])
        for key in self.keys:
            num_samples_per_class[labels[key]] += 1

        num_units = num_samples_per_class / np.array(self.config.class_ratios)
        min_units = int(np.min(num_units))
        num_samples_per_class_balanced = min_units * np.array(self.config.class_ratios)

        current_num_samples_per_class = np.zeros([self.output_size])
        self.task_specific_keys = []
        for key in self.keys:
            class_idx = labels[key]
            if (
                current_num_samples_per_class[class_idx]
                < num_samples_per_class_balanced[class_idx]
            ):
                self.task_specific_keys.append(key)
                current_num_samples_per_class[class_idx] += 1

        self.keys_backup = copy.deepcopy(self.keys)
        self.keys = self.task_specific_keys

    def __getitem__(self, idx):
        saved_task_cfg = self.task_config
        self.task_config = None
        sample = super().__getitem__(idx)
        self.task_config = saved_task_cfg

        extras = []
        if self.return_dict:
            x = sample["x"]
            y_raw = sample["y"]
        else:
            x, y_raw = sample
            # In the tuple form Image2MixedDataset collapses the label and every
            # extra key (url, hint, index) into element 1 as soon as more than one
            # is active, so y_raw is [y, *extras] rather than the label row.
            # Slicing that list as if it were the label silently produced an
            # all-zero target and dropped the url, which in turn made
            # get_predictions fall back to writing loader positions instead of
            # filenames - mispairing every distilled image with someone else's
            # prediction. Split the extras back out and pass them along.
            if isinstance(y_raw, (list, tuple)):
                extras = list(y_raw[1:])
                y_raw = y_raw[0]

        nums = y_raw[:8]
        n_unique_numbers = (
            self.config.output_split if self.config.output_split else 1000
        )
        y_binary = _numbers_to_binary_target(nums, n_unique_numbers)

        if self.task_config is not None and self.task_config.y_selection is not None:
            selected_indices = []
            for sel in self.task_config.y_selection:
                if isinstance(sel, str) and sel.startswith("Num"):
                    idx_val = int(sel[3:])
                elif isinstance(sel, int):
                    idx_val = sel
                elif sel in self.attributes:
                    idx_val = self.attributes.index(sel)
                else:
                    raise ValueError(f"Unknown attribute selection: {sel}")
                selected_indices.append(idx_val)

            target = y_binary[selected_indices]

            if (
                hasattr(self.task_config, "criterions")
                and "ce" in self.task_config.criterions
            ):
                assert (
                    target.shape[0] == 1
                ), "output shape unacceptable for singleclass classification"
                target = target[0].to(torch.int64)

            if self.return_dict:
                sample["y"] = target
                if (
                    self.groups_enabled
                    and hasattr(self.config, "confounding_factors")
                    and self.config.confounding_factors
                ):
                    conf_factor = self.config.confounding_factors[-1]
                    conf_idx = (
                        int(conf_factor[3:])
                        if isinstance(conf_factor, str)
                        and conf_factor.startswith("Num")
                        else int(conf_factor)
                    )
                    sample["has_confounder"] = y_binary[conf_idx]
                return sample
            else:
                return x, ([target] + extras if extras else target)

        if self.return_dict:
            sample["y"] = y_binary
            if (
                self.groups_enabled
                and hasattr(self.config, "confounding_factors")
                and self.config.confounding_factors
            ):
                conf_factor = self.config.confounding_factors[-1]
                conf_idx = (
                    int(conf_factor[3:])
                    if isinstance(conf_factor, str) and conf_factor.startswith("Num")
                    else int(conf_factor)
                )
                sample["has_confounder"] = y_binary[conf_idx]
            return sample
        else:
            return x, ([y_binary] + extras if extras else y_binary)


def _numbers_to_binary_target(numbers, n_unique_numbers: int):
    binary_nums = torch.zeros(n_unique_numbers)
    if numbers is None:
        return binary_nums
    if isinstance(numbers, str):
        return binary_nums
    if isinstance(numbers, (list, tuple)):
        numbers = [n for n in numbers if not isinstance(n, str)]
        if len(numbers) == 0:
            return binary_nums
    try:
        tensor_nums = torch.as_tensor(numbers).flatten()
    except (TypeError, ValueError):
        return binary_nums

    for number in tensor_nums:
        try:
            number_idx = int(number)
            if 0 <= number_idx < n_unique_numbers:
                binary_nums[number_idx] = 1.0
        except (TypeError, ValueError):
            pass
    return binary_nums


class SparseNumbersDataset(Image2MixedDataset):
    """
    Number-grid images with eight additional dense red intensities.

    ``SparseNumbersDatasetGenerator`` writes the dataset on first use. Each csv
    row holds 8 ``Red`` intensities followed by 8 number slots (the number
    drawn there, or ``-1``). ``__getitem__`` expands the slots into a binary
    vector of length ``config.output_split`` (default 1000) and concatenates
    the red intensities, so ``attributes`` is ``Num0 ... Num<n-1>`` followed by
    ``Red1 ... Red8``; the red channels act as confounders of the sparse number
    labels.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``, ``output_split``, ``class_ratios`` and
        ``confounding_factors``.
    **kwargs
        Forwarded to the generator and to ``Image2MixedDataset``.

    Attributes
    ----------
    attributes : list of str
        The expanded label vector; ``attributes_positive`` and
        ``attributes_negative`` hold its phrasings.
    """

    def __init__(self, config: DataConfig, **kwargs):
        """Generate the dataset if missing and build the expanded attribute names."""
        if not os.path.exists(os.path.join(config.dataset_path, "data.csv")):
            sdg = SparseNumbersDatasetGenerator(data_config=config, **kwargs)
            sdg.generate_dataset()

        super(SparseNumbersDataset, self).__init__(config=config, **kwargs)
        n_unique_numbers = (
            self.config.output_split if self.config.output_split else 1000
        )
        self.attributes = [f"Num{i}" for i in range(n_unique_numbers)] + [
            f"Red{i+1}" for i in range(8)
        ]
        self.attributes_positive = [f"Is Num{i}" for i in range(n_unique_numbers)] + [
            f"Red{i+1}" for i in range(8)
        ]
        self.attributes_negative = [f"Not Num{i}" for i in range(n_unique_numbers)] + [
            f"Not Red{i+1}" for i in range(8)
        ]

    def _expanded_label_value(self, key, attr_idx):
        """Value of expanded attribute `attr_idx` for `key`, from the raw csv row.

        `self.attributes` is the expanded vector (1000 binary Num slots followed by
        the 8 dense Red columns), but `self.data[key]` is still the raw 16-column csv
        row (8 Red, then 8 number slots holding the number drawn in that slot, or -1).
        Indexing the raw row with an expanded index is what the base-class helpers do,
        and for any Num above 15 it silently falls out of range. This mirrors the
        binarisation __getitem__ performs, without decoding the image.
        """
        row = self.data[key]
        n_unique_numbers = (
            self.config.output_split if self.config.output_split else 1000
        )
        if attr_idx < n_unique_numbers:
            for n in row[8:16]:
                if n >= 0 and int(n) == attr_idx:
                    return 1.0
            return 0.0

        red_idx = attr_idx - n_unique_numbers
        reds = row[:8]
        return float(reds[red_idx]) if red_idx < len(reds) else 0.0

    def _selected_attr_idx(self):
        sel = self.task_config.y_selection[0]
        return self.attributes.index(sel) if isinstance(sel, str) else int(sel)

    def enable_class_restriction(self, class_idx):
        """
        Restrict ``keys`` to the samples of one class of the selected attribute.

        Parameters
        ----------
        class_idx : int
            Value the expanded attribute must take (0 or 1).
        """
        # The base implementation tests `self.data[key][self.attributes.index(sel)]`,
        # which for y_selection ["Num128"] indexes the 16-column raw row at 128 and so
        # matches nothing — both classes come back empty. With class_balanced training
        # that empties val_dataloaders[0] and ModelTrainer's Logger dies on
        # `next(iter(val_dataloader))` with a bare StopIteration.
        assert not self.task_config is None, "Task config must be set"
        self.backup_keys = copy.deepcopy(self.keys)
        attr_idx = self._selected_attr_idx()
        self.keys = [
            key
            for key in self.backup_keys
            if int(self._expanded_label_value(key, attr_idx)) == class_idx
        ]
        self.class_restrictions_enabled = True

    def set_task_specific_keys(self):
        """
        Subsample ``keys`` to the proportions in ``config.class_ratios``.

        Labels come from ``_expanded_label_value`` so that the expanded attribute
        index is resolved against the raw csv row correctly.
        """
        # Same raw-row-vs-expanded-index mismatch as enable_class_restriction.
        self.task_specific_keys = []
        attr_idx = self._selected_attr_idx()
        num_samples_per_class = np.zeros([self.output_size])
        for key in self.keys:
            num_samples_per_class[int(self._expanded_label_value(key, attr_idx))] += 1

        num_units = num_samples_per_class / np.array(self.config.class_ratios)
        min_units = int(np.min(num_units))
        num_samples_per_class_balanced = min_units * np.array(self.config.class_ratios)
        current_num_samples_per_class = np.zeros([self.output_size])
        for key in self.keys:
            class_idx = int(self._expanded_label_value(key, attr_idx))
            if (
                current_num_samples_per_class[class_idx]
                < num_samples_per_class_balanced[class_idx]
            ):
                self.task_specific_keys.append(key)
                current_num_samples_per_class[class_idx] += 1

        self.keys_backup = copy.deepcopy(self.keys)
        self.keys = self.task_specific_keys

    def __getitem__(self, idx):
        # Mirrors OnlySparseNumbersDataset: fetch the raw csv row with task_config
        # disabled so the base class does not try to apply y_selection to the raw
        # slot columns, then binarise and apply y_selection on the expanded vector.
        saved_task_cfg = self.task_config
        self.task_config = None
        sample = super().__getitem__(idx)
        self.task_config = saved_task_cfg

        # In Image2MixedDataset, the return format depends on return_dict.
        extras = []
        if self.return_dict:
            x = sample["x"]
            y_raw = sample["y"]
        else:
            x, y_raw = sample
            # The tuple form does not always carry the label row in position 1:
            # Image2MixedDataset returns `return_list[1:]` — a LIST of every
            # remaining return_dict value (y, then url / index / hint as enabled) —
            # as soon as more than two keys are active, which is what happens once
            # distill_binary_dataset turns on url reporting. Keep the extras so the
            # tuple we hand back still matches what get_predictions unpacks.
            if isinstance(y_raw, (list, tuple)):
                extras = list(y_raw[1:])
                y_raw = y_raw[0]

        reds = y_raw[:8]
        nums = y_raw[8:16]

        n_unique_numbers = (
            self.config.output_split if self.config.output_split else 1000
        )

        binary_nums = torch.zeros(n_unique_numbers)
        for n in nums:
            if n >= 0:
                num_idx = int(n)
                if 0 <= num_idx < n_unique_numbers:
                    binary_nums[num_idx] = 1.0

        y_final = torch.cat([binary_nums, reds])

        def _attach_confounder(sample_dict):
            cf = getattr(self.config, "confounding_factors", None)
            if self.groups_enabled and cf:
                sample_dict["has_confounder"] = y_final[self.attributes.index(cf[-1])]

        if self.task_config is not None and self.task_config.y_selection is not None:
            selected_indices = [
                self.attributes.index(sel) if isinstance(sel, str) else int(sel)
                for sel in self.task_config.y_selection
            ]
            target = y_final[selected_indices]
            if (
                hasattr(self.task_config, "criterions")
                and "ce" in self.task_config.criterions
            ):
                assert (
                    target.shape[0] == 1
                ), "output shape unacceptable for singleclass classification"
                target = target[0].to(torch.int64)
            if self.return_dict:
                sample["y"] = target
                _attach_confounder(sample)
                return sample
            return x, ([target] + extras if extras else target)

        if self.return_dict:
            sample["y"] = y_final
            _attach_confounder(sample)
            return sample
        else:
            return x, ([y_final] + extras if extras else y_final)


class OnlySparseNumbersZipfDataset(OnlySparseNumbersDataset):
    """
    OnlySparseNumbers dataset where target numbers are distributed according to Zipf's law.
    """

    def __init__(self, config: DataConfig, **kwargs):
        """Generate the Zipf-distributed variant if missing and name the attributes."""
        if not os.path.exists(os.path.join(config.dataset_path, "data.csv")):
            sdg = OnlySparseNumbersZipfDatasetGenerator(data_config=config, **kwargs)
            sdg.generate_dataset()

        super(OnlySparseNumbersDataset, self).__init__(config=config, **kwargs)
        n_unique_numbers = (
            self.config.output_split if self.config.output_split else 1000
        )
        self.attributes = [f"Num{i}" for i in range(n_unique_numbers)]
        self.attributes_positive = [f"Is Num{i}" for i in range(n_unique_numbers)]
        self.attributes_negative = [f"Not Num{i}" for i in range(n_unique_numbers)]


class SparseNumbersDenseDataset(SparseNumbersDataset):
    """
    Alias/Subclass of SparseNumbersDataset with 8 dense red intensity attributes and sparse numbers.
    """


class SparseNumbersZipfDataset(SparseNumbersDataset):
    """
    SparseNumbers dataset (8 dense red intensity attributes + sparse numbers) where sparse numbers follow Zipf distribution.
    """

    def __init__(self, config: DataConfig, **kwargs):
        """Generate the Zipf-distributed variant if missing and name the attributes."""
        if not os.path.exists(os.path.join(config.dataset_path, "data.csv")):
            sdg = SparseNumbersZipfDatasetGenerator(data_config=config, **kwargs)
            sdg.generate_dataset()

        super(SparseNumbersDataset, self).__init__(config=config, **kwargs)
        n_unique_numbers = (
            self.config.output_split if self.config.output_split else 1000
        )
        self.attributes = [f"Num{i}" for i in range(n_unique_numbers)] + [
            f"Red{i+1}" for i in range(8)
        ]
        self.attributes_positive = [f"Is Num{i}" for i in range(n_unique_numbers)] + [
            f"Red{i+1}" for i in range(8)
        ]
        self.attributes_negative = [f"Not Num{i}" for i in range(n_unique_numbers)] + [
            f"Not Red{i+1}" for i in range(8)
        ]


class SparseNumbersDenseZipfDataset(SparseNumbersZipfDataset):
    """
    Alias/Subclass of SparseNumbersZipfDataset.
    """


class Camelyon17Dataset(ImageDataset):
    """
    Thin wrapper around the WILDS Camelyon17 tumour dataset.

    Delegates to ``wilds.get_dataset("camelyon17")`` (downloading it if needed)
    and returns ``(transform(image), label)``; the WILDS metadata, which holds
    the hospital of origin, is dropped. ``Camelyon17AugmentedDataset`` is the
    variant that exports the images into PEAL's csv layout and keeps the
    hospital as the confounder.

    Parameters
    ----------
    config : DataConfig
        Stored on the instance but not interpreted here.
    transform : callable
        Applied to the PIL image.
    **kwargs
        Ignored.
    """

    def __init__(self, config, transform, **kwargs):
        """Open (and if needed download) the WILDS dataset and store the transform.

        Raises
        ------
        ImportError
            If the optional ``wilds`` dependency is not installed.
        """
        wilds = require("wilds", "datasets", "loading the WILDS benchmark datasets")
        get_dataset = wilds.get_dataset

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
    """
    Thin wrapper around the WILDS RxRx1 cell-imaging dataset.

    Delegates to ``wilds.get_dataset("rxrx1")`` (downloading it if needed) and
    returns ``(transform(image), label)``, dropping the WILDS metadata that
    identifies the experimental batch.

    Parameters
    ----------
    config : DataConfig
        Stored on the instance but not interpreted here.
    transform : callable
        Applied to the PIL image.
    **kwargs
        Ignored.
    """

    def __init__(self, config, transform, **kwargs):
        """Open (and if needed download) the WILDS dataset and store the transform.

        Raises
        ------
        ImportError
            If the optional ``wilds`` dependency is not installed.
        """
        wilds = require("wilds", "datasets", "loading the WILDS benchmark datasets")
        get_dataset = wilds.get_dataset

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
    """
    Camelyon17 exported into PEAL's ``imgs/`` plus ``data.csv`` layout.

    On first use every WILDS sample is written to
    ``<dataset_path>/imgs/<7 digit index>.png`` and a ``data.csv`` with the
    columns ``img, tumor, hospital`` is created, the hospital id acting as the
    confounder of the tumour label. A latent oracle is loaded from
    ``$PEAL_RUNS/camelyon17/latent_oracle/model.cpl``, falling back to the
    ``latent_oracle_old_20250417_150756`` run, and backs
    ``sample_to_2d_latent``.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path`` and ``confounding_factors``.
    **kwargs
        Forwarded to ``Image2MixedDataset``.

    Attributes
    ----------
    oracle : torch.nn.Module
        Attribute predictor in eval mode.
    """

    def __init__(self, config, **kwargs):
        """Export the WILDS images if missing and load the latent oracle.

        Raises
        ------
        ImportError
            If the images still have to be exported and the optional ``wilds``
            dependency is not installed.
        """
        if not os.path.exists(config.dataset_path):
            wilds = require("wilds", "datasets", "loading the WILDS benchmark datasets")
            get_dataset = wilds.get_dataset

            original_dataset = get_dataset(dataset="camelyon17", download=True)
            Path(os.path.join(config.dataset_path, "imgs")).mkdir(
                parents=True, exist_ok=True
            )
            lines = ["img,tumor,hospital"]
            for i in range(len(original_dataset)):
                img, label, meta = original_dataset[i]
                img_name = f"{embed_numberstring(i, 7)}.png"
                img.save(f"{config.dataset_path}/imgs/{img_name}")
                lines.append(f"{img_name}, {label}, {meta[0]}")

            with open(f"{config.dataset_path}/data.csv", "w") as f:
                f.write("\n".join(lines))

        peal_runs = os.environ.get("PEAL_RUNS", "peal_runs")
        oracle_path = os.path.join(
            peal_runs, "camelyon17", "latent_oracle", "model.cpl"
        )
        if os.path.exists(oracle_path):
            try:
                self.oracle = torch.load(oracle_path)
            except Exception:
                self.oracle = torch.load(oracle_path, weights_only=False)
            self.oracle.eval()
        else:
            self.oracle = torch.load(
                os.path.join(
                    os.environ.get("PEAL_RUNS", "peal_runs"),
                    "camelyon17/latent_oracle_old_20250417_150756/model.cpl",
                )
            )
            self.oracle.eval()

        super(Camelyon17AugmentedDataset, self).__init__(config=config, **kwargs)

    def sample_to_2d_latent(self, sample, mask=None):
        """
        Oracle attribute readout, reduced to the confounding factors.

        Parameters
        ----------
        sample : torch.Tensor
            Image ``[C, H, W]`` or batch ``[N, C, H, W]``; a single image is
            unsqueezed for the oracle and squeezed again afterwards.
        mask : torch.Tensor, optional
            Ignored; kept for interface compatibility.

        Returns
        -------
        torch.Tensor
            Oracle outputs for the attributes named in
            ``config.confounding_factors``, or all outputs if that is unset.
        """
        self.oracle.to(sample.device)
        sample_inflated = False
        if not len(sample.shape) == 4:
            sample_inflated = True
            sample = sample.unsqueeze(0)

        latent = self.oracle(sample)

        if sample_inflated:
            latent = latent[0]

        # Index by confounding factors to get 2D output
        cf = getattr(self.config, "confounding_factors", None)
        if cf and hasattr(self, "attributes"):
            indices = [self.attributes.index(f) for f in cf]
            latent = latent[indices]

        return latent


class RxRx1AugmentedDataset(Image2MixedDataset):
    """
    RxRx1 exported into PEAL's ``imgs/`` plus ``data.csv`` layout.

    On first use every WILDS sample is written to
    ``<dataset_path>/imgs/<7 digit index>.png`` and a ``data.csv`` with the
    columns ``img, label, confounder`` is created, the confounder being the
    first WILDS metadata field (``cell_type``).

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``.
    **kwargs
        Forwarded to ``Image2MixedDataset``.
    """

    def __init__(self, config, **kwargs):
        """Export the WILDS images into the csv layout if they are missing.

        Raises
        ------
        ImportError
            If the images still have to be exported and the optional ``wilds``
            dependency is not installed.
        """
        if not os.path.exists(config.dataset_path):
            wilds = require("wilds", "datasets", "loading the WILDS benchmark datasets")
            get_dataset = wilds.get_dataset

            original_dataset = get_dataset(dataset="rxrx1", download=True)
            Path(os.path.join(config.dataset_path, "imgs")).mkdir(
                parents=True, exist_ok=True
            )
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
    """
    Waterbirds: the bird species is confounded by the background scene.

    If ``<dataset_path>/data.csv`` is missing, the CUB segmentation masks and
    the ``waterbird_complete95_forest2water2`` release are downloaded into
    ``downloads/``, extracted and moved into place as ``imgs_filename/``,
    ``masks/`` and ``data.csv`` (the release's ``metadata.csv``, which carries
    the ``y`` and ``place`` columns). Already extracted folders are detected
    and the download is skipped.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``; the rest is handled by ``Image2MixedDataset``.
    **kwargs
        Forwarded to ``Image2MixedDataset``.

    Raises
    ------
    Exception
        If one of the two downloads returns a non-200 status code.
    """

    def __init__(self, config, **kwargs):
        """Download and lay out the Waterbirds files if ``data.csv`` is missing."""
        _log.info("%s", "instantiate waterbirds dataset!")
        # The images are read from <dataset_path>/<x_selection>/<img_filename>;
        # metadata.csv names its path column "img_filename".
        image_dir = getattr(config, "x_selection", None) or "img_filename"
        dataset_labels = os.path.join(config.dataset_path, "data.csv")
        if not os.path.exists(dataset_labels):
            # Download the segmentations
            download_path = os.path.join(config.dataset_path, "downloads")
            Path(download_path).mkdir(parents=True, exist_ok=True)

            if os.path.exists(
                os.path.join(download_path, "segmentations", "200.Common_Yellowthroat")
            ):
                _log.info(
                    "%s", "Found segmentation masks folder. Skipping downloading."
                )

            else:
                tar_file_path = os.path.join(download_path, "segmentations.tar.gz")
                if not os.path.exists(tar_file_path):
                    _log.info("%s", "Download segmentation tar file")
                    url = "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz"

                    response = requests.get(url, stream=True)

                    if response.status_code == 200:
                        os.makedirs(download_path, exist_ok=True)

                        with open(tar_file_path, "wb") as file:
                            file.write(response.raw.read())

                        _log.info("%s", "Segmentations downloaded successfully!")

                    else:
                        raise Exception("Failed to download segmentations.")

                with tarfile.open(tar_file_path, "r:gz") as tar:
                    tar.extractall(path=download_path)
                    _log.info("%s", "segmentations extracted")

            if os.path.exists(
                os.path.join(
                    download_path,
                    "waterbird_complete95_forest2water2",
                    "200.Common_Yellowthroat",
                )
            ):
                _log.info("%s", "Found waterbirds folder. Skipping downloading.")

            else:
                tar_file_path = os.path.join(download_path, "waterbirds.tar.gz")
                if not os.path.exists(tar_file_path):
                    _log.info("%s", "Download waterbirds tar file")
                    url = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"

                    response = requests.get(url, stream=True)

                    if response.status_code == 200:
                        os.makedirs(download_path, exist_ok=True)

                        with open(tar_file_path, "wb") as file:
                            file.write(response.raw.read())

                        _log.info("%s", "Waterbirds downloaded successfully!")

                    else:
                        raise Exception("Failed to download waterbirds.")

                with tarfile.open(tar_file_path, "r:gz") as tar:
                    tar.extractall(path=download_path)
                    _log.info("%s", "waterbirds extracted")

            shutil.move(
                os.path.join(download_path, "waterbird_complete95_forest2water2"),
                os.path.join(config.dataset_path, image_dir),
            )
            shutil.move(
                os.path.join(download_path, "segmentations"),
                os.path.join(config.dataset_path, "masks"),
            )
            shutil.move(
                os.path.join(config.dataset_path, image_dir, "metadata.csv"),
                os.path.join(config.dataset_path, "data.csv"),
            )
            _log.info(
                "%s", "Downloading, extracting and positioning of files completed!"
            )
        elif not os.path.isdir(
            os.path.join(config.dataset_path, image_dir)
        ) and os.path.isdir(os.path.join(config.dataset_path, "imgs_filename")):
            # layout written by earlier versions, which ignored x_selection
            os.rename(
                os.path.join(config.dataset_path, "imgs_filename"),
                os.path.join(config.dataset_path, image_dir),
            )

        super(WaterbirdsDataset, self).__init__(config=config, **kwargs)


def download_celeba_to(target_dir):
    """
    Download the Kaggle CelebA release and lay it out the way PEAL expects.

    Fetches ``jessicali9530/celeba-dataset`` with ``kagglehub``, moves it to
    ``target_dir`` and renames ``list_attr_celeba.csv`` to ``data.csv`` and
    ``img_align_celeba/img_align_celeba`` to ``imgs``.

    Parameters
    ----------
    target_dir : str
        Destination dataset root.

    Notes
    -----
    The body itself prints that this path "still has to be implemented": the
    moves assume one particular kagglehub cache layout.
    """
    _log.info("%s", "This still has to be implemented!")
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    _log.info("%s %s", "Path to downloaded dataset files:", path)
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
    """
    CelebA with its 40 binary face attributes.

    The dataset is downloaded when ``config.dataset_path`` does not exist.
    After loading, the attribute names are read from the label csv - whose
    first line is a bare sample count and whose second line is the header
    ``ImgName``, the 40 attributes and ``split`` - so evaluations report real
    attribute names instead of ``label_0 ... label_39``. If
    ``$PEAL_RUNS/celeba/latent_oracle/model.cpl`` exists it is loaded as the
    attribute oracle behind ``sample_to_2d_latent``; otherwise a warning is
    printed and that method has no oracle to call.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``, ``label_rel_path``, ``delimiter`` and
        ``confounding_factors``.
    **kwargs
        Forwarded to ``Image2MixedDataset``.

    Attributes
    ----------
    attributes : list of str
        The 40 CelebA attribute names.
    oracle : torch.nn.Module
        Attribute predictor in eval mode, when the checkpoint was found.
    """

    def __init__(self, config, **kwargs):
        """Download CelebA if missing, read the attribute names and load the oracle."""
        if not os.path.exists(config.dataset_path):
            download_celeba_to(config.dataset_path)

        super(CelebADataset, self).__init__(config=config, **kwargs)

        # Expose the 40 CelebA attribute names so the SAE evaluation reports real
        # labels instead of label_0 ... label_39. The csv starts with a bare count
        # line; its header is "ImgName", the 40 attributes, then "split".
        try:
            csv_path = os.path.join(config.dataset_path, config.label_rel_path)
            with open(csv_path) as attr_file:
                attr_file.readline()
                header = attr_file.readline().rstrip("\n").split(config.delimiter)
            self.attributes = header[1:-1]
        except Exception as exc:
            _log.info("%s", f"Warning: could not read CelebA attribute names ({exc})")

        peal_runs = os.environ.get("PEAL_RUNS", "peal_runs")
        # procrustes_path = os.path.join(
        #     peal_runs,
        #     "celeba",
        #     "diffusion_autoencoder",
        #     "OrthogonalProcrustesDictionary40Comps",
        #     "weights.npz",
        # )
        # from peal.architectures.predictors import TorchvisionModel
        # self.oracle = TorchvisionModel(model="open_clip:ViT-B-32:laion2b_s34b_b79k", num_classes=40)

        # if os.path.exists(procrustes_path):
        #     try:
        #         checkpoint = torch.load(procrustes_path, map_location="cpu")
        #     except Exception:
        #         checkpoint = torch.load(procrustes_path, map_location="cpu", weights_only=False)
        #
        #     weights = checkpoint["W_ortho"]
        #     weights = weights.to(torch.float32)
        #     if weights.shape == (512, 40):
        #         weights = weights.T
        #     self.oracle.fc.weight.data = weights.to(self.oracle.fc.weight.device)
        # else:
        #     print(f"Warning: Oracle Procrustes weights not found at {procrustes_path}")

        # self.oracle.eval()

        oracle_path = os.path.join(peal_runs, "celeba", "latent_oracle", "model.cpl")
        if os.path.exists(oracle_path):
            try:
                self.oracle = torch.load(oracle_path)
            except Exception:
                self.oracle = torch.load(oracle_path, weights_only=False)
            self.oracle.eval()
        else:
            _log.info("%s", f"Warning: CelebA latent oracle not found at {oracle_path}")

    def sample_to_2d_latent(self, sample, mask=None):
        """
        Oracle attribute readout, reduced to the confounding factors.

        Works both with the loaded PEAL oracle and with the ACE ``OracleMetrics``
        wrapper kept in ``oracle_metric``.

        Parameters
        ----------
        sample : torch.Tensor
            Image ``[C, H, W]`` or batch ``[N, C, H, W]``; a single image is
            unsqueezed for the oracle and squeezed again afterwards.
        mask : torch.Tensor, optional
            Ignored; kept for interface compatibility.

        Returns
        -------
        torch.Tensor
            Oracle outputs for the attributes named in
            ``config.confounding_factors``, or all 40 outputs if that is unset.
        """
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

        # Index by confounding factors to get 2D output
        cf = getattr(self.config, "confounding_factors", None)
        if cf and hasattr(self, "attributes"):
            indices = [self.attributes.index(f) for f in cf]
            latent = latent[indices]

        return latent


def parse_datastr(data_path):
    """
    Read a CelebA-HQ style annotation file into one whitespace-separated table.

    The leading sample count (the first six characters of the file) is dropped,
    an ``idx`` column is prepended to the header for the file name, and double
    spaces are collapsed so the rest parses as a single-space-delimited table.

    Parameters
    ----------
    data_path : str
        Path of the annotation text file.

    Returns
    -------
    str
        The whole table as a single string.
    """
    with open(data_path, "r") as f:
        datastr = f.read()[6:]
        datastr = "idx " + datastr.replace("  ", " ")
    return datastr


def datastr(data_path):
    """
    Line-wise variant of :func:`parse_datastr`.

    Parameters
    ----------
    data_path : str
        Path of the annotation text file.

    Returns
    -------
    list of str
        The normalized table split into lines, as ``parse_csv`` expects it in
        its ``raw_data`` argument.
    """
    with open(data_path, "r") as f:
        datastr = f.read()[6:]
        datastr = "idx " + datastr.replace("  ", " ")
    return datastr.split("\n")


class CelebAHQDataset(Image2MixedDataset):
    """
    CelebA-HQ with the CelebAMask-HQ attribute annotations.

    The annotation file is not a csv: it starts with a sample count and is
    aligned with variable whitespace. It is therefore pre-normalized by
    ``datastr`` and handed to ``parse_csv`` as ``raw_data``, overwriting the
    ``attributes``, ``data`` and ``keys`` that ``Image2MixedDataset.__init__``
    has just built.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``, ``label_rel_path`` and ``delimiter``.
    mode : str
        Split: ``"train"``, ``"val"``, ``"test"`` or ``"all"``.
    **kwargs
        Forwarded to ``Image2MixedDataset``. ``data_dir`` overrides the
        annotation path, in which case the file is read by ``parse_csv``
        itself instead of being pre-normalized.
    """

    def __init__(self, config, mode, **kwargs):
        """Load the split, then re-parse the whitespace-aligned annotation file."""
        super(CelebAHQDataset, self).__init__(config=config, mode=mode, **kwargs)
        _log.info("%s", "CelebAHQ")
        self.root_dir = config.dataset_path
        data_dir = kwargs.get("data_dir", None)
        raw_data = None
        if not data_dir:
            data_dir = os.path.join(self.root_dir, self.config.label_rel_path)
            raw_data = datastr(data_dir)
        self.data_dir = data_dir
        delimiter = self.config.delimiter
        self.attributes, self.data, self.keys = parse_csv(
            data_dir,
            config,
            mode,
            key_type="name",
            delimiter=delimiter,
            raw_data=raw_data,
        )


class CelebACopyrighttagDataset(Image2MixedDataset):
    """
    CelebA with a copyright tag stamped into part of the images.

    When the derived dataset is missing it is built by
    ``ConfounderDatasetGenerator`` from ``config.dataset_origin_path`` (a plain
    CelebA copy, downloaded first if necessary): the generator blends the tag
    into the bottom of every second image and stores the tag area as a hint
    mask. That tag is the confounder the repair methods have to become
    invariant to.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``, ``dataset_origin_path``, ``delimiter`` and the
        generator fields consumed by ``ConfounderDatasetGenerator``.
    **kwargs
        Forwarded to ``Image2MixedDataset``.
    """

    def __init__(self, config, **kwargs):
        """Download the CelebA origin and stamp the tag onto it if needed."""
        if not os.path.exists(config.dataset_path) and not os.path.exists(
            config.dataset_origin_path
        ):
            download_celeba_to(config.dataset_origin_path)

        if not os.path.exists(config.dataset_path):
            _log.info("%s", "config.delimiter")
            _log.info("%s", config.delimiter)
            _log.info("%s", config.delimiter)
            _log.info("%s", config.delimiter)
            cdg = ConfounderDatasetGenerator(**config.__dict__, data_config=config)
            cdg.generate_dataset()

        super(CelebACopyrighttagDataset, self).__init__(config=config, **kwargs)


class FollicleDataset(Image2MixedDataset):
    """
    Ovarian follicle images with expert annotations, taken from Hugging Face.

    On first use the ``janphhe/follicles_true_features`` dataset (metadata,
    images, cut-outs and masks) is downloaded - which requires a Hugging Face
    login - and written to ``<dataset_path>`` as ``data.csv`` plus the
    directories ``imgs/``, ``imgs_cut/`` and ``masks/``.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``.
    **kwargs
        Forwarded to ``Image2MixedDataset``.

    Notes
    -----
    The export loop builds every output filename from
    ``attribute_values[attributes.index("imgs")]``, i.e. from a whole metadata
    column rather than from the current row, so in practice the dataset
    directory is expected to exist already.
    """

    def __init__(self, config, **kwargs):
        """Download and export the Hugging Face dataset if it is not present yet."""
        # Login using e.g. `huggingface-cli login` to access this dataset
        if not os.path.exists(config.dataset_path):
            from datasets import load_dataset

            metadata_data_with_annotations = load_dataset(
                "janphhe/follicles_true_features",
                data_files="data_with_annotations.csv",
            )
            ds_imgs = load_dataset(
                "janphhe/follicles_true_features", "imgs_config", num_proc=8
            )
            ds_imgs_cut = load_dataset(
                "janphhe/follicles_true_features", "imgs_cut_config", num_proc=8
            )
            ds_masks = load_dataset(
                "janphhe/follicles_true_features", "masks_config", num_proc=8
            )
            attributes = [
                e for e in metadata_data_with_annotations["train"].features.keys()
            ][:-1]
            attribute_values = [
                metadata_data_with_annotations["train"][attributes[i]]
                for i in range(len(attributes))
            ]
            csv_file = ",".join(attributes) + "\n"
            for i in range(len(attribute_values[0])):
                csv_file += (
                    ",".join(
                        [str(attribute_values[j][i]) for j in range(len(attributes))]
                    )
                    + "\n"
                )

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
                        os.path.join(
                            config.dataset_path,
                            "imgs",
                            attribute_values[attributes.index("imgs")],
                        ),
                        "PNG",
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
                        os.path.join(
                            config.dataset_path,
                            "imgs_cut",
                            attribute_values[attributes.index("imgs_cut")],
                        ),
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
                            os.path.split(attribute_values[attributes.index("imgs")])[
                                -1
                            ],
                        ),
                        "PNG",
                    )

        super(FollicleDataset, self).__init__(config=config, **kwargs)


class FunnyNodulesDataset(Image2MixedDataset):
    """
    Synthetic nodules whose internal structure is confounded by roundness.

    ``FunnyNodulesDatasetGenerator`` writes the dataset on first use and only
    supports ``config.confounder_probability == 0.5``: the unpoisoned pool has
    to exist before a poisoned variant can be derived from it. An oracle over
    the six nodule attributes is loaded from the first existing of
    ``$PEAL_RUNS/funnynodules_roundness_confounding_size/torchvision/
    foundation_poisoned098/model.cpl`` and the same path under
    ``funnynodules/``; it backs ``sample_to_2d_latent``.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``, ``confounder_probability`` and
        ``confounding_factors``.
    **kwargs
        Forwarded to ``Image2MixedDataset``.

    Attributes
    ----------
    oracle : torch.nn.Module or None
        Attribute predictor in eval mode, or ``None`` if no checkpoint exists.

    Raises
    ------
    NotImplementedError
        If the dataset has to be generated but ``confounder_probability`` is
        not 0.5.
    """

    def __init__(self, config: DataConfig, **kwargs):
        """Generate the dataset if missing and load the first available oracle."""
        if not os.path.exists(os.path.join(config.dataset_path, "data.csv")):
            if config.confounder_probability == 0.5:
                fng = FunnyNodulesDatasetGenerator(data_config=config)
                fng.generate_dataset()
            else:
                raise NotImplementedError(
                    "Only confounder_probability=0.5 can be used to generate the dataset. "
                    "Please run the unpoisoned experiment first to generate the raw pool."
                )

        super(FunnyNodulesDataset, self).__init__(config=config, **kwargs)

        # Load foundation model as oracle for sample_to_2d_latent (analogous to CelebADataset)
        peal_runs = os.environ.get("PEAL_RUNS", "peal_runs")
        oracle_candidates = [
            os.path.join(
                peal_runs,
                "funnynodules_roundness_confounding_size",
                "torchvision",
                "foundation_poisoned098",
                "model.cpl",
            ),
            os.path.join(
                peal_runs,
                "funnynodules",
                "torchvision",
                "foundation_poisoned098",
                "model.cpl",
            ),
        ]
        self.oracle = None
        for oracle_path in oracle_candidates:
            if os.path.exists(oracle_path):
                try:
                    self.oracle = torch.load(oracle_path, map_location="cpu")
                except Exception:
                    self.oracle = torch.load(
                        oracle_path, map_location="cpu", weights_only=False
                    )
                self.oracle.eval()
                break

    def sample_to_2d_latent(self, sample, mask=None):
        """
        Oracle attribute readout, reduced to the confounding factors.

        Parameters
        ----------
        sample : torch.Tensor
            Image ``[C, H, W]`` or batch ``[N, C, H, W]``; a single image is
            unsqueezed for the oracle and squeezed again afterwards.
        mask : torch.Tensor, optional
            Ignored; kept for interface compatibility.

        Returns
        -------
        torch.Tensor
            Oracle outputs for the attributes named in
            ``config.confounding_factors``, or all outputs if that is unset.

        Raises
        ------
        RuntimeError
            If no oracle checkpoint was found at construction time.
        """
        if self.oracle is None:
            raise RuntimeError(
                "FunnyNodulesDataset: No oracle model found. "
                "Train the all_attributes_resnet18 foundation model first."
            )
        self.oracle.to(sample.device)
        sample_inflated = False
        if len(sample.shape) != 4:
            sample_inflated = True
            sample = sample.unsqueeze(0)

        latent = self.oracle(sample)

        if sample_inflated:
            latent = latent[0]

        # Index by confounding factors to get 2D output
        cf = getattr(self.config, "confounding_factors", None)
        if cf and hasattr(self, "attributes"):
            indices = [self.attributes.index(f) for f in cf]
            latent = latent[indices]

        return latent


class SkinConDataset(Image2MixedDataset):
    """
    Fitzpatrick17k skin lesions with the skin tone as confounder.

    On first use the ``spycoder/fitzpatrick`` metadata is loaded, the SkinCon
    annotation csv is downloaded, and every image whose ``md5hash`` appears in
    the annotations is fetched from its source url (with retries) and stored as
    ``imgs/<md5hash>.png``. ``data.csv`` gets the columns
    ``img, label, confounder``, the label being malignant vs. not (from
    ``three_partition_label``) and the confounder a dark skin type
    (Fitzpatrick scale 4 or higher). Images whose url fails are skipped.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``.
    **kwargs
        Forwarded to ``Image2MixedDataset``.
    """

    def __init__(self, config, **kwargs):
        """Fetch metadata, annotations and images if the dataset is not present."""
        if not os.path.exists(config.dataset_path):
            from datasets import load_dataset
            import pandas as pd
            import requests

            # Download Fitzpatrick17k HF dataset
            _log.info(
                "%s",
                "Downloading spycoder/fitzpatrick dataset (this might take a while)...",
            )
            ds = load_dataset("spycoder/fitzpatrick", split="train")

            # Download SkinCon annotations
            _log.info("%s", "Downloading SkinCon annotations...")
            annotations_url = (
                "https://skincon-dataset.github.io/files/annotations_fitzpatrick17k.csv"
            )
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

            _log.info("%s", "Merging datasets and saving images...")
            lines = ["img,label,confounder"]

            valid_image_ids = set(skincon_df["ImageID"].tolist())

            # Filter the HF dataset metadata
            # For simplicity, we assume we extract malignant/benign and skin_type from the HF dataset if available
            count = 0

            session = requests.Session()
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            retry = Retry(connect=5, read=5, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            for i, row in enumerate(ds):
                # We need a unique identifier to match. The HF dataset has an 'md5hash' which corresponds to ImageID in SkinCon.
                img_id = row.get("md5hash")
                if img_id and f"{img_id}.jpg" in valid_image_ids:
                    # Let's say label is 'three_partition_label' or just 'label'.
                    # If 'three_partition_label' == 'malignant', label=1, else 0.
                    is_malignant = (
                        1 if row.get("three_partition_label") == "malignant" else 0
                    )
                    skin_type = row.get("fitzpatrick_scale", 1)  # Default 1 if missing
                    is_dark = 1 if skin_type >= 4 else 0

                    img_name = f"{img_id}.png"

                    # Fetching image from url
                    url = row.get("url")
                    if not url:
                        continue
                    try:
                        resp = session.get(url, timeout=15)
                        if resp.status_code == 200:
                            from PIL import Image
                            from io import BytesIO

                            img = Image.open(BytesIO(resp.content))
                            # Convert to RGB to avoid alpha channel issues when saving as PNG
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            img.save(os.path.join(img_dir, img_name))
                            lines.append(f"{img_name},{is_malignant},{is_dark}")
                            count += 1
                    except Exception as e:
                        _log.info("%s", f"Failed to fetch {url}: {e}")
                        continue

            with open(os.path.join(config.dataset_path, "data.csv"), "w") as f:
                f.write("\n".join(lines))
            _log.info(
                "%s",
                f"SkinCon dataset successfully created with {count} images in {config.dataset_path}",
            )

        super(SkinConDataset, self).__init__(config=config, **kwargs)


class NicoPlusPlusDataset(Image2MixedDataset):
    """
    NICO++ subset in which the object category is confounded by its context.

    NICO++ has to be downloaded manually to ``$PEAL_DATA/NICO++.zip``; if it is
    missing, an empty skeleton is created and a ``RuntimeError`` with the
    download instructions is raised. Otherwise the nested
    ``NICO_DG_Benchmark.zip`` is read into memory and only those images are
    extracted whose category is in ``config.foreground`` (default bear, dog)
    and whose context is in ``config.background`` (default grass, water). They
    are written to ``imgs/<category>_<context>_<counter>.jpg`` and listed in
    ``data.csv`` as ``img, label, confounder`` with the positions in those two
    lists as indices.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``, ``foreground`` and ``background``.
    **kwargs
        Forwarded to ``Image2MixedDataset``.

    Raises
    ------
    RuntimeError
        If the zip is missing, or if it contains no matching image.
    FileNotFoundError
        If the nested benchmark zip cannot be found inside ``NICO++.zip``.
    """

    def __init__(self, config, **kwargs):
        """Extract the configured category/context subset if ``data.csv`` is missing."""
        if not os.path.exists(os.path.join(config.dataset_path, "data.csv")):
            import zipfile

            peal_data_dir = os.environ.get(
                "PEAL_DATA", os.path.dirname(os.path.normpath(config.dataset_path))
            )
            zip_path = os.path.join(peal_data_dir, "NICO++.zip")

            if not os.path.exists(zip_path):
                _log.info(
                    "%s",
                    "NICO++ dataset requires manual download due to its size and hosting limitations.",
                )
                _log.info(
                    "%s",
                    "Please download it from https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AADbhZJAy0AAbap76cg_XkAfa?dl=0",
                )
                _log.info("%s", f"and place the zip file exactly at {zip_path}")

                # We will create a dummy dataset structure for the code to not strictly crash when just verifying config loading
                os.makedirs(config.dataset_path, exist_ok=True)
                os.makedirs(os.path.join(config.dataset_path, "imgs"), exist_ok=True)
                with open(os.path.join(config.dataset_path, "data.csv"), "w") as f:
                    f.write("img, label, confounder\n")

                raise RuntimeError(
                    f"NICO++ Dataset missing. Please follow the instructions to download it. Ensure it is at {zip_path}"
                )

            target_categories = (
                config.foreground if config.foreground else ["bear", "dog"]
            )
            target_contexts = (
                config.background if config.background else ["grass", "water"]
            )

            # Since NICO++ is nested, we might need a temporary extraction or read inline
            # We'll extract only the target images
            _log.info("%s", "Extracting selected subset from NICO++.zip...")

            # Ensure the output directories exist before writing
            img_out_dir = os.path.join(config.dataset_path, "imgs")
            os.makedirs(img_out_dir, exist_ok=True)

            lines = ["img,label,confounder"]
            count = 0

            # Use actual group counts for extraction without hard clipping
            counts = {}
            for target_category in target_categories:
                for target_context in target_contexts:
                    counts[f"{target_category}_{target_context}"] = 0

            with zipfile.ZipFile(zip_path, "r") as z1:
                # Find the nested zip
                nested_zip_name = None
                for name in z1.namelist():
                    if name.endswith("NICO_DG_Benchmark.zip"):
                        nested_zip_name = name
                        break

                if not nested_zip_name:
                    raise FileNotFoundError(
                        "Could not find NICO_DG_Benchmark.zip inside NICO++.zip"
                    )

                with z1.open(nested_zip_name) as nested_zip_file:
                    nested_data = nested_zip_file.read()

            with zipfile.ZipFile(io.BytesIO(nested_data)) as z2:
                for name in z2.namelist():
                    if name.lower().endswith(".jpg") or name.lower().endswith(".png"):
                        parts = name.split("/")
                        if len(parts) >= 3:
                            context = parts[-3].lower()
                            category = parts[-2].lower()

                            key = f"{category}_{context}"
                            if (
                                category in target_categories
                                and context in target_contexts
                            ):
                                img_name = f"{category}_{context}_{count:05d}.jpg"
                                img_data = z2.read(name)
                                with open(
                                    os.path.join(img_out_dir, img_name), "wb"
                                ) as f_out:
                                    f_out.write(img_data)

                                # Label matching and Context matching
                                # Indices are based on the order in the configuration's foreground/background lists
                                label_idx = target_categories.index(category)
                                context_idx = target_contexts.index(context)

                                lines.append(f"{img_name},{label_idx},{context_idx}")
                                counts[key] += 1
                                count += 1

            # NICO++ subset uses "label" for class and "confounder" for context
            with open(os.path.join(config.dataset_path, "data.csv"), "w") as f:
                f.write("\n".join(lines))

            if count == 0:
                raise RuntimeError(
                    f"Could not find matching images within the extracted {zip_path}. Please check the zip contents."
                )
            else:
                _log.info(
                    "%s",
                    f"NICO++ dataset subset successfully created with {count} images in {config.dataset_path}",
                )
                _log.info("%s", f"Counts: {counts}")

        super(NicoPlusPlusDataset, self).__init__(config=config, **kwargs)


class ISICAnnotatedDataset(Image2MixedDataset):
    """
    ISIC skin lesions with the annotated lesion size as confounder.

    If ``<dataset_path>/data.csv`` is missing, ``_prepare_dataset`` downloads
    the image and segmentation metadata plus ``segs.zip`` from Zenodo record
    14201693, extracts the masks to ``masks/``, measures for every lesion the
    fraction of its mask above half intensity, and writes
    ``imgs, label, confounder`` - benign vs. malignant as the label, and a mask
    area above the median area as the confounder.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``.
    **kwargs
        Forwarded to ``Image2MixedDataset``.
    """

    def __init__(self, config, **kwargs):
        """Build the dataset from the Zenodo release if ``data.csv`` is missing."""
        _log.info("%s", "instantiate ISICAnnotated dataset!")
        dataset_labels = os.path.join(config.dataset_path, "data.csv")
        if not os.path.exists(dataset_labels):
            self._prepare_dataset(config)

        super(ISICAnnotatedDataset, self).__init__(config=config, **kwargs)

    def _prepare_dataset(self, config):
        import csv
        import requests
        from PIL import Image
        import numpy as np
        from tqdm import tqdm
        from pathlib import Path

        _log.info("%s", f"Preparing ISICAnnotated dataset at {config.dataset_path}...")
        Path(config.dataset_path).mkdir(parents=True, exist_ok=True)

        img_metadata_path = os.path.join(config.dataset_path, "img_metadata.csv")
        seg_metadata_path = os.path.join(config.dataset_path, "seg_metadata.csv")
        segs_zip_path = os.path.join(config.dataset_path, "segs.zip")

        # Zenodo record URL parts
        base_zenodo_url = "https://zenodo.org/records/14201693/files/"

        files_to_download = {
            "img_metadata.csv": img_metadata_path,
            "seg_metadata.csv": seg_metadata_path,
            "segs.zip": segs_zip_path,
        }

        for filename, local_path in files_to_download.items():
            if not os.path.exists(local_path):
                _log.info("%s", f"Downloading {filename} from Zenodo...")
                url = base_zenodo_url + filename + "?download=1"
                try:
                    r = requests.get(url, stream=True)
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                except Exception as e:
                    _log.info("%s", f"Error downloading {filename}: {e}")
                    if filename != "segs.zip":  # Metadata is critical
                        raise e

        if not os.path.exists(img_metadata_path) or not os.path.exists(
            seg_metadata_path
        ):
            _log.info("%s", "Metadata files still missing after download attempt.")
            raise FileNotFoundError(f"Missing metadata in {config.dataset_path}")

        # Read img_metadata: isic_id -> benign_malignant
        img_labels = {}
        with open(img_metadata_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_labels[row["isic_id"]] = row["benign_malignant"]

        data = []
        areas = []

        mask_dir = os.path.join(config.dataset_path, "masks")
        if not os.path.exists(mask_dir) or not os.listdir(mask_dir):
            segs_zip = os.path.join(config.dataset_path, "segs.zip")
            if os.path.exists(segs_zip):
                _log.info("%s", f"Extracting {segs_zip}...")
                import zipfile

                with zipfile.ZipFile(segs_zip, "r") as zip_ref:
                    zip_ref.extractall(config.dataset_path)

                # Debugging: List top-level items after extraction
                extracted_items = os.listdir(config.dataset_path)
                _log.info(
                    "%s",
                    f"Contents of {config.dataset_path} after extraction: {extracted_items}",
                )

                # Check where it extracted. Could be 'segs' or 'masks' or 'segmentations'
                # IMA++ typically has 'segs'
                possible_sources = ["segs", "segmentations", "masks"]
                found_source = False
                for src_name in possible_sources:
                    src_path = os.path.join(config.dataset_path, src_name)
                    if os.path.exists(src_path) and os.path.isdir(src_path):
                        if src_path != mask_dir:
                            _log.info("%s", f"Renaming {src_path} to {mask_dir}...")
                            import shutil

                            if os.path.exists(mask_dir):
                                shutil.rmtree(mask_dir)
                            shutil.move(src_path, mask_dir)
                        found_source = True
                        break

                if not found_source:
                    _log.info(
                        "%s",
                        "Warning: Did not find a standard mask source directory. Zip might have extracted files directly or into a different folder.",
                    )
                    # Let's check if there are many .png files in the root now
                    png_files = [f for f in extracted_items if f.endswith(".png")]
                    if len(png_files) > 100:
                        _log.info(
                            "%s",
                            f"Found {len(png_files)} PNG files in root. Moving them to 'masks' folder...",
                        )
                        os.makedirs(mask_dir, exist_ok=True)
                        import shutil

                        for f in png_files:
                            shutil.move(
                                os.path.join(config.dataset_path, f),
                                os.path.join(mask_dir, f),
                            )

            # If still doesn't exist, create it to avoid repeating unzip/errors
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir, exist_ok=True)

        # Read seg_metadata and process
        with open(seg_metadata_path, mode="r") as f:
            reader = csv.DictReader(f)
            seg_rows = list(reader)

        _log.info("%s", f"Processing {len(seg_rows)} rows from seg_metadata.csv...")
        _log.info("%s", f"Checking for masks in: {mask_dir}")
        _log.info("%s", f"Number of img_labels available: {len(img_labels)}")

        _log.info("%s", "Calculating mask areas for confounder retrieval...")
        for row in tqdm(seg_rows):
            isic_id = row["ISIC_id"]
            seg_filename = row["seg_filename"]
            mask_path = os.path.join(mask_dir, seg_filename)

            if not os.path.exists(mask_path):
                # Retry with some variations just in case (e.g. extension)
                if not os.path.exists(mask_path):
                    continue

            if isic_id not in img_labels:
                continue

            mask = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask)
            area_ratio = np.mean(mask_arr > 127)

            label = 1 if img_labels[isic_id] == "malignant" else 0
            img_filename = row["img_filename"]

            data.append({"img": img_filename, "label": label, "area_ratio": area_ratio})
            areas.append(area_ratio)

        if not areas:
            _log.info("%s", "Error: areas list is empty.")
            # Sample check
            if seg_rows:
                sample_row = seg_rows[0]
                sample_mask = os.path.join(mask_dir, sample_row["seg_filename"])
                _log.info(
                    "%s",
                    f"Sample mask path: {sample_mask} (Exists: {os.path.exists(sample_mask)})",
                )
                _log.info(
                    "%s",
                    f"Sample ISIC_id: {sample_row['ISIC_id']} (In img_labels: {sample_row['ISIC_id'] in img_labels})",
                )
            raise Exception(
                "No images/masks found to process! Check console for debug info."
            )

        threshold = np.median(areas)
        _log.info("%s", f"Confounder threshold (median area ratio): {threshold}")

        # Ensure imgs directory exists and contains images
        imgs_dir = os.path.join(config.dataset_path, "imgs")
        if not os.path.exists(imgs_dir):
            # Try to find where images are. They might be in a folder called 'images' or just in the root
            possible_img_dirs = [
                os.path.join(config.dataset_path, "images"),
                config.dataset_path,
            ]
            found = False
            for p in possible_img_dirs:
                # Check for a few expected images
                if any(
                    os.path.exists(os.path.join(p, row["img_filename"]))
                    for row in seg_rows[:10]
                ):
                    if p == config.dataset_path:
                        # We need 'imgs' to be a subfolder. We can't easily symlink the parent to a child.
                        # We'll create 'imgs' and symlink all images into it.
                        os.makedirs(imgs_dir, exist_ok=True)
                        _log.info("%s", "Symlinking images into 'imgs' folder...")
                        for row in tqdm(seg_rows):
                            src = os.path.join(config.dataset_path, row["img_filename"])
                            dst = os.path.join(imgs_dir, row["img_filename"])
                            if os.path.exists(src) and not os.path.exists(dst):
                                try:
                                    os.symlink(os.path.relpath(src, imgs_dir), dst)
                                except Exception:
                                    pass  # Fallback to copy if symlink fails?
                    else:
                        os.symlink(os.path.relpath(p, config.dataset_path), imgs_dir)
                    found = True
                    break
            if not found:
                _log.info(
                    "%s",
                    "Warning: Could not find image files. Please ensures images are in 'imgs' folder.",
                )

        # Save to data.csv
        data_csv_path = os.path.join(config.dataset_path, "data.csv")
        with open(data_csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["imgs", "label", "confounder"])
            writer.writeheader()
            for item in data:
                confounder = 1 if item["area_ratio"] > threshold else 0
                writer.writerow(
                    {
                        "imgs": item["img"],  # This is the filename
                        "label": item["label"],
                        "confounder": confounder,
                    }
                )
        _log.info("%s", "data.csv generated successfully.")
