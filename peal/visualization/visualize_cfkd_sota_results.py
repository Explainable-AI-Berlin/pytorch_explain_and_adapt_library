import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision.transforms.functional import resize

NUM_CLUSTERS = 2


def plot_images_with_custom_padding(
    imgs, confidences, task_names, method_names, output_path
):
    num_methods, num_tasks, c, h, w = imgs.shape

    # Define padding configuration
    col_spacing = [0.5 if i in [2, 4, 6] else 0.0 for i in range(num_tasks)]
    row_spacing = [0.5 if i in [1, 3, 7] else 0.0 for i in range(num_methods)]

    # Create GridSpec for precise control over spacings
    fig = plt.figure(figsize=(2 * num_tasks, 2 * num_methods))
    grid = GridSpec(num_methods, num_tasks, figure=fig, hspace=0.2, wspace=0.1)

    for i in range(num_methods):
        for j in range(num_tasks):
            row_start = sum(row_spacing[:i])
            row_end = row_start + 1
            col_start = sum(col_spacing[:j])
            col_end = col_start + 1

            # Define individual cell positions
            ax = fig.add_subplot(grid[i, j])
            img = imgs[i, j].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")

            # Add confidence values below the image
            ax.text(
                0.5,
                -0.1,
                f"{confidences[i, j]:.2f}",
                fontsize=8,
                ha="center",
                transform=ax.transAxes,
            )

            if i == 0:  # Add task names as column headers
                ax.set_title(task_names[j], fontsize=8)

            if j == 0:  # Add method names as row labels
                ax.text(
                    -1.0,
                    0.5,
                    method_names[i],
                    fontsize=8,
                    ha="center",
                    transform=ax.transAxes,
                )

    plt.tight_layout()
    plt.subplots_adjust(
        top=1 - sum(row_spacing) / num_methods,
        bottom=0 + sum(row_spacing) / num_methods,
        left=0 + sum(col_spacing) / num_tasks,
        right=1 - sum(col_spacing) / num_tasks,
    )
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    base_path = os.environ.get("PEAL_RUNS", "peal_runs")
    base_paths = [
        base_path + "/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098",
        base_path + "/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098",
        base_path + "/celeba1k/Blond_Hair/resnet18_poisoned098",
        base_path + "/follicles_cut/classifier_natural",
        #base_path + "/camelyon17_1k/classifier_poisoned098",
    ]
    methods = [
        "sce_cfkd/0",
        "sce_cfkd/1",
    ]
    task_names = [
        "Light to Dark FG",
        "Dark to Light FG",
        "Serious to Smiling",
        "Smiling to Serious",
        "Non-Blond to Blond",
        "Blond to Non-Blond",
        "Growing to Primordial",
        "Primordial to Growing",
        #"Healthy to Cancer",
        #"Cancer to Healthy",
    ]
    method_names = [
        "Original",
        "Counterfactual Before",
        "Counterfactual After",
    ]
    # sample_idxs = [[11, 12], [5, 12, 40, 36, 103, 125], [36, 93, 140, 157], [3, 52, 150, 314, 80], [0, 1]]
    sample_idxs = [[11, 12], [103, 125], [140, 157], [80, 150]]
    imgs = torch.zeros([1 + len(methods), 2 * len(base_paths), 3, 128, 128])
    target_confidences = torch.zeros([1 + len(methods), 2 * len(base_paths)])

    for dataset_idx in range(len(base_paths)):
        for method_idx in range(len(methods)):
            tracked_values_path = os.path.join(
                base_paths[dataset_idx],
                methods[method_idx],
                "validation_tracked_values.npz",
            )
            if not os.path.exists(tracked_values_path):
                continue

            with open(tracked_values_path, "rb") as f:
                tracked_values = np.load(f, allow_pickle=True)

                for i, sample_idx in enumerate(sample_idxs[dataset_idx]):
                    if method_idx == 0:
                        imgs[0][2 * dataset_idx + i] = resize(
                            torch.from_numpy(tracked_values["x_list"][sample_idx]),
                            [128, 128],
                        )
                        target_confidences[0][2 * dataset_idx + i] = float(
                            tracked_values["y_target_start_confidence_list"][sample_idx]
                        )

                    imgs[1 + method_idx][2 * dataset_idx + i] = (
                        resize(
                            torch.from_numpy(
                                tracked_values["x_counterfactual_list"][
                                    sample_idx
                                ]
                            ),
                            [128, 128],
                        )
                    )
                    target_confidences[1 + method_idx][
                        2 * dataset_idx + i
                    ] = float(
                        tracked_values["y_target_end_confidence_list"][
                            sample_idx
                        ]
                    )
                    print(tracked_values_path)

    for method_idx, method_name in enumerate(method_names):
        plot_images_with_custom_padding(
            imgs[method_idx : method_idx + 1],
            target_confidences,
            task_names,
            [method_names[method_idx]],
            method_name + ".png",
        )

    imgs_reshaped = imgs.reshape([-1] + list(imgs.shape)[2:])
    plot_images_with_custom_padding(
        imgs,
        target_confidences,
        task_names,
        method_names,
        "collage_with_custom_padding.png",
    )
