import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision.transforms.functional import resize

from pathlib import Path

# Allow running from project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from peal.global_utils import generate_overlay, generate_ssim_overlay

NUM_CLUSTERS = 2


def plot_images_with_custom_padding(
    imgs, confidences, task_names, method_names, output_path
):
    num_methods, num_tasks, c, h, w = imgs.shape

    # Define padding configuration
    col_spacing = [0.2 if i in [2, 4, 6] else 0.0 for i in range(num_tasks)] if num_tasks > 1 else [0.0]
    row_spacing = [0.2 if i in [0, 2] else 0.0 for i in range(num_methods)] if num_methods > 1 else [0.0]

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

            # Add confidence values below the image (unless it's an overlay or ssim)
            if "Overlay" not in method_names[i] and "SSIM" not in method_names[i]:
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
    if num_methods > 1 or num_tasks > 1:
        plt.subplots_adjust(
            top=min(0.95, 1 - sum(row_spacing) / num_methods if num_methods > 0 else 1),
            bottom=max(0.05, sum(row_spacing) / num_methods if num_methods > 0 else 0),
            left=max(0.05, sum(col_spacing) / num_tasks if num_tasks > 0 else 0),
            right=min(0.95, 1 - sum(col_spacing) / num_tasks if num_tasks > 0 else 1),
        )
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    base_path = os.environ.get("PEAL_RUNS", "peal_runs")
    base_paths = [
        base_path
        + "/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098",
        base_path
        + "/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098",
        base_path + "/celeba1k/Blond_Hair/classifier_poisoned098",
        base_path + "/follicles_cut/classifier_natural",
        # base_path + "/camelyon17_1k/classifier_poisoned098",
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
        # "Healthy to Cancer",
        # "Cancer to Healthy",
    ]
    method_names = [
        "Original",
        "Counterfactual Before",
        "SSIM Before",
        "Counterfactual After",
        "SSIM After",
    ]
    # sample_idxs = [[11, 12], [5, 12, 40, 36, 103, 125], [36, 93, 140, 157], [3, 52, 150, 314, 80], [0, 1]]
    sample_idxs = [[11, 12], [103, 125], [140, 157], [80, 150]]
    num_methods = len(methods)
    rows_per_method = 2
    imgs = torch.zeros([1 + rows_per_method * num_methods, 2 * len(base_paths), 3, 128, 128])
    target_confidences = torch.zeros([1 + rows_per_method * num_methods, 2 * len(base_paths)])

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

                    cf_img = resize(
                        torch.from_numpy(
                            tracked_values["x_counterfactual_list"][sample_idx]
                        ),
                        [128, 128],
                    )
                    orig_img = imgs[0][2 * dataset_idx + i]
                    
                    method_base_idx = rows_per_method * method_idx + 1
                    imgs[method_base_idx][2 * dataset_idx + i] = cf_img
                    imgs[method_base_idx + 1][2 * dataset_idx + i] = generate_ssim_overlay(orig_img, cf_img)
                    
                    for r in range(rows_per_method):
                        target_confidences[method_base_idx + r][2 * dataset_idx + i] = float(
                            tracked_values["y_target_end_confidence_list"][sample_idx]
                        )
                    print(tracked_values_path)

    for method_idx, method_name in enumerate(method_names):
        plot_images_with_custom_padding(
            imgs[method_idx : method_idx + 1],
            target_confidences[method_idx : method_idx + 1],
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
