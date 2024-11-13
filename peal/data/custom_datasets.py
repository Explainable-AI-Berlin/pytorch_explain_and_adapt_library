import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import matplotlib.cm as cm

from torchvision.transforms import ToTensor
from matplotlib.colors import ListedColormap
from wilds import get_dataset


from peal.data.dataset_generators import SquareDatasetGenerator
from peal.data.datasets import (
    Image2ClassDataset,
    DataConfig,
    Image2MixedDataset,
    ImageDataset,
)
from peal.global_utils import embed_numberstring
from peal.data.dataset_generators import latent_to_square_image


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
    decision_cmap = ListedColormap(["lightcoral", "lightblue"])

    # Display the decision boundary grid as the background
    ax.imshow(
        decision_boundary,
        extent=[0, 1, 0, 1],  # Extend from x=0 to x=1 and y=0 to y=1
        origin="lower",  # Aligns the grid with the bottom-left of the plot
        cmap=decision_cmap,  # Apply custom colormap
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
            arrowprops=dict(
                fc="green", ec="green", edgecolor="yellow", arrowstyle="->", alpha=0.7
            ),
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
    blue_patch = plt.Line2D([0], [0], color="lightblue", lw=4, label="Pred > 0.5")
    red_patch = plt.Line2D([0], [0], color="lightcoral", lw=4, label="Pred < 0.5")

    # Display the updated legend
    ax.legend(
        handles=[original_marker, cf_marker, blue_patch, red_patch], loc="upper right"
    )

    # Show the plot
    plt.grid(True)
    plt.savefig(filename)
    plt.clf()


def plot_latents_with_arrows_old(
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
    decision_cmap = ListedColormap(["lightblue", "lightcoral"])

    # Display the decision boundary grid as the background
    ax.imshow(
        decision_boundary,
        extent=[0, 1, 0, 1],  # Extend from x=0 to x=1 and y=0 to y=1
        origin="lower",  # Aligns the grid with the bottom-left of the plot
        cmap=decision_cmap,  # Apply custom colormap
        alpha=0.3,  # Make the background semi-transparent
    )

    # Plot original latents and counterfactuals
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

        # Plot original point with darkblue border
        ax.scatter(
            orig[0],
            orig[1],
            facecolor=start_color,
            edgecolor="darkblue",
            label="Original" if i == 0 else "",
        )

        # Plot counterfactual point with darkred border
        ax.scatter(
            cf[0],
            cf[1],
            facecolor=end_color,
            edgecolor="darkred",
            label="Counterfactual" if i == 0 else "",
        )

        # Draw arrow between original and counterfactual points
        ax.annotate(
            "",
            xy=(cf[0], cf[1]),
            xytext=(orig[0], orig[1]),
            arrowprops=dict(
                fc="green", ec="green", edgecolor="yellow", arrowstyle="->", alpha=0.7
            ),
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
    neutral_marker = plt.Line2D(
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
        marker="o",
        color="w",
        markerfacecolor="gray",
        markeredgecolor="darkred",
        markersize=8,
        label="Counterfactual",
    )
    by_label = dict(zip(labels, handles))

    # Override handles with neutral markers
    by_label["Original"] = neutral_marker
    by_label["Counterfactual"] = cf_marker

    # Display the updated legend
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    # Show the plot
    plt.grid(True)
    plt.savefig(filename)
    plt.clf()


class SquareDataset(Image2MixedDataset):
    def __init__(self, config: DataConfig, **kwargs):
        if not os.path.exists(config.dataset_path):
            cdg = SquareDatasetGenerator(data_config=config)
            cdg.generate_dataset()

        super(SquareDataset, self).__init__(config=config, **kwargs)

    def global_counterfactual_visualization(
        self,
        counterfactuals,
        filename,
        num_samples,
        y_target_start_confidence,
        y_target_end_confidence,
        y_list,
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
        for idx in range(num_samples):
            x, (y, hint) = self[idx]
            original_latents.append(
                [self.check_foreground(x, hint), self.check_background(x, hint)]
            )
            counterfactual_latents.append(
                [
                    self.check_foreground(counterfactuals[idx], hint),
                    self.check_background(counterfactuals[idx], hint),
                ]
            )

        self.hints_enabled = hints_enabled_buffer

        path = filename.split("/")[:-1] + ["decision_boundary.npy"]
        decision_boundary = np.load("/" + os.path.join(*path))
        decision_boundary = np.transpose(decision_boundary, (1, 0))

        plot_latents_with_arrows(
            original_latents,
            counterfactual_latents,
            filename,
            y_start_confidence,
            y_end_confidence,
            decision_boundary,
        )

    def visualize_decision_boundary(self, predictor, batch_size, device, path, temperature=1.0):
        print("visualize_decision_boundary")

        # Create the grid for plotting
        x = torch.linspace(0, 1, 100)
        y = torch.linspace(0, 1, 100)
        xx, yy = torch.meshgrid(x, y)
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

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
                        ToTensor()(
                            latent_to_square_image(
                                255 * float(grid[i][0]),
                                255 * float(grid[i][1]),
                                position_x=x_pos,
                                position_y=y_pos,
                            )[0],
                        )
                    )
                    if len(current_batch) == batch_size:
                        current_batch = torch.stack(current_batch)
                        if first_batch is None:
                            first_batch = current_batch

                        logits.append(predictor(current_batch.to(device)).detach())
                        current_batch = []

                logits.append(predictor(torch.stack(current_batch).to(device)).detach())
                logits = torch.cat(logits, dim=0).detach().cpu()
                prediction_grid = torch.nn.Softmax(dim=1)(logits / temperature)[:,0].reshape(100, 100)
                prediction_grids.append(prediction_grid)

        """# Average the predictions across grids
        prediction_grid = torch.mean(torch.stack(prediction_grids).to(torch.float32), dim=0).numpy()
        
        # Create the plot
        plt.figure()
        
        # Set a lighter color map: use "coolwarm" and make it lighter
        cmap = cm.get_cmap("bwr")
        plt.contourf(xx, yy, prediction_grid, levels=100, cmap=cmap)
        
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
        plt.clf()"""

        # Average the predictions across grids
        prediction_grid = torch.mean(torch.stack(prediction_grids).to(torch.float32), dim=0).numpy()

        # Create the plot
        plt.figure()

        # Set a lighter color map: use "coolwarm" and make it lighter
        cmap = cm.get_cmap("bwr")
        # Create filled contour plot
        contour_fill = plt.contourf(xx, yy, prediction_grid, levels=100, cmap=cmap)

        # Add contour lines with black color and thicker lines
        contour_lines = plt.contour(xx, yy, prediction_grid, levels=10, colors='black', linewidths=1.5)

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
        np.save(path[:-4] + ".npy", prediction_grid)

        print("visualize_decision_boundary saved under " + path)

    def check_foreground(self, x, hint):
        intensity_foreground = torch.sum(
            hint[..., 0, :, :] * x[..., 0, :, :]
        ) / torch.sum(hint[..., 0, :, :])
        return intensity_foreground

    def check_background(self, x, hint):
        intensity_background = torch.sum((1 - hint) * x) / torch.sum(1 - hint)
        return intensity_background

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
                s = (
                    "Foreground: "
                    + str(round(float(self.check_foreground(x, hint)), 3))
                    + " -> "
                )
                s += (
                    str(round(float(self.check_foreground(x_counterfactual, hint)), 3))
                    + ", "
                )
                s += (
                    "Background: "
                    + str(round(float(self.check_background(x, hint)), 3))
                    + " -> "
                )
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

        super(Camelyon17AugmentedDataset, self).__init__(config=config, **kwargs)


class RxRx1AugmentedDataset(Image2MixedDataset):
    def __init__(self, config, **kwargs):
        if not os.path.exists(config.dataset_path):
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
