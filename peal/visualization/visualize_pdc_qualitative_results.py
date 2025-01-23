'''import torch
import os
import numpy as np
import torchvision.utils

NUM_CLUSTERS = 2


if __name__ == "__main__":
    base_path = os.environ.get("PEAL_RUNS", "peal_runs")
    base_paths = [
        base_path + "/celeba/Smiling/classifier_natural",
        base_path + "/celeba/Blond_Hair/classifier_natural",
        base_path + "/waterbirds/classifier_natural",
        base_path
        + "/square/colora_confounding_colorb/torchvision/classifier_poisoned100",
    ]
    methods = ["ace_cfkd", "dime_cfkd", "fastdime_cfkd", "pdc_cfkd"]
    task_names = [
        "Smiling to Serious",
        "Serious to Smiling",
        "Blond to Non-Blond",
        "Non-Blond to Blond",
        "Landbird to Waterbird",
        "Waterbird to Landbird",
        "Dark Background to Light Background",
        "Light Background to Dark Background",
    ]
    method_names = [
        "Original",
        "ACE (1)",
        "ACE (2)",
        "DiME (1)",
        "DiME (2)",
        "FastDiME (1)",
        "FastDiME (2)",
        "PDC (ours) (1)",
        "PDC (ours) (2)",
    ]
    sample_idxs = [[17, 43], [6, 23], [3, 30], [0, 11]]
    #sample_idxs = [[17, 43], [6, 23, 25], [3, 30, 43, 48, 51, 53, 63], [0, 11]]
    imgs = torch.zeros([1 + 2 * len(methods), 2 * len(base_paths), 3, 128, 128])
    target_confidences = torch.zeros([1 + 2 * len(methods), 2 * len(base_paths)])
    for dataset_idx in range(len(base_paths)):
        for method_idx in range(len(methods)):
            print([dataset_idx, method_idx])
            tracked_values_path = os.path.join(
                base_paths[dataset_idx],
                methods[method_idx],
                "0",
                "validation_tracked_cluster_values.npz",
            )
            with open(
                tracked_values_path,
                "rb",
            ) as f:
                tracked_values = np.load(f, allow_pickle=True)

                for i, sample_idx in enumerate(sample_idxs[dataset_idx]):
                    if method_idx == 0:
                        imgs[0][
                            2 * dataset_idx + i
                        ] = torchvision.transforms.Resize([128, 128])(
                            torch.from_numpy(tracked_values["x_list"][sample_idx])
                        )
                        target_confidences[0][2 * dataset_idx + i] = float(
                            tracked_values["y_target_start_confidence_list"][sample_idx]
                        )

                    for cluster_idx in range(NUM_CLUSTERS):
                        imgs[1 + 2 * method_idx + cluster_idx][
                            2 * dataset_idx + i
                        ] = torchvision.transforms.Resize([128, 128])(
                            torch.from_numpy(
                                tracked_values["clusters" + str(cluster_idx)][
                                    sample_idx
                                ]
                            )
                        )
                        target_confidences[1 + 2 * method_idx + cluster_idx][
                            2 * dataset_idx + i
                        ] = float(
                            tracked_values["cluster_confidence" + str(cluster_idx)][
                                sample_idx
                            ]
                        )

    imgs_reshaped = imgs.reshape([-1] + list(imgs.shape)[2:])
    torchvision.utils.save_image(imgs_reshaped, "collage.png", nrow=imgs.shape[1])

import torch
import os
import numpy as np
import torchvision.utils
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize

NUM_CLUSTERS = 2

def plot_images_with_labels(
    imgs, confidences, task_names, method_names, output_path, padding_config
):
    num_methods, num_tasks, c, h, w = imgs.shape
    fig, axes = plt.subplots(
        num_methods, num_tasks, figsize=(2 * num_tasks, 2 * num_methods)
    )
    fig.subplots_adjust(
        hspace=padding_config['row_padding'], wspace=padding_config['col_padding']
    )

    for i in range(num_methods):
        for j in range(num_tasks):
            img = imgs[i, j].permute(1, 2, 0).numpy()
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(
                f"{confidences[i, j]:.2f}", fontsize=8, loc="center", pad=2
            )

            if i == 0:  # Add task name labels on top
                ax.set_title(f"{task_names[j]}\n{confidences[i, j]:.2f}", fontsize=8)

            if j == 0:  # Add method name labels on the side
                ax.set_ylabel(method_names[i], rotation=0, fontsize=8, labelpad=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    base_path = os.environ.get("PEAL_RUNS", "peal_runs")
    base_paths = [
        base_path + "/celeba/Smiling/classifier_natural",
        base_path + "/celeba/Blond_Hair/classifier_natural",
        base_path + "/waterbirds/classifier_natural",
        base_path
        + "/square/colora_confounding_colorb/torchvision/classifier_poisoned100",
    ]
    methods = ["ace_cfkd", "dime_cfkd", "fastdime_cfkd", "pdc_cfkd"]
    task_names = [
        "Smiling to Serious",
        "Serious to Smiling",
        "Blond to Non-Blond",
        "Non-Blond to Blond",
        "Landbird to Waterbird",
        "Waterbird to Landbird",
        "Dark Background to Light Background",
        "Light Background to Dark Background",
    ]
    method_names = [
        "Original",
        "ACE (1)",
        "ACE (2)",
        "DiME (1)",
        "DiME (2)",
        "FastDiME (1)",
        "FastDiME (2)",
        "PDC (ours) (1)",
        "PDC (ours) (2)",
    ]
    sample_idxs = [[17, 43], [6, 23], [3, 30], [0, 11]]
    imgs = torch.zeros([1 + 2 * len(methods), 2 * len(base_paths), 3, 128, 128])
    target_confidences = torch.zeros([1 + 2 * len(methods), 2 * len(base_paths)])

    for dataset_idx in range(len(base_paths)):
        for method_idx in range(len(methods)):
            tracked_values_path = os.path.join(
                base_paths[dataset_idx],
                methods[method_idx],
                "0",
                "validation_tracked_cluster_values.npz",
            )
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

                    for cluster_idx in range(NUM_CLUSTERS):
                        imgs[1 + 2 * method_idx + cluster_idx][
                            2 * dataset_idx + i
                        ] = resize(
                            torch.from_numpy(
                                tracked_values["clusters" + str(cluster_idx)][
                                    sample_idx
                                ]
                            ),
                            [128, 128],
                        )
                        target_confidences[1 + 2 * method_idx + cluster_idx][
                            2 * dataset_idx + i
                        ] = float(
                            tracked_values["cluster_confidence" + str(cluster_idx)][
                                sample_idx
                            ]
                        )

    imgs_reshaped = imgs.reshape([-1] + list(imgs.shape)[2:])
    padding_config = {"row_padding": 0.5, "col_padding": 0.2}
    plot_images_with_labels(
        imgs,
        target_confidences,
        task_names,
        method_names,
        "collage_with_labels.png",
        padding_config,
    )


import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize

NUM_CLUSTERS = 2

def plot_images_with_custom_padding(
    imgs, confidences, task_names, method_names, output_path
):
    num_methods, num_tasks, c, h, w = imgs.shape

    # Define padding configuration
    col_spacing = [0.5 if i in [2, 4, 6] else 0.0 for i in range(num_tasks)]
    row_spacing = [0.5 if i in [1, 3, 7] else 0.0 for i in range(num_methods)]
    fig, axes = plt.subplots(
        num_methods, num_tasks, figsize=(2 * num_tasks, 2 * num_methods),
        gridspec_kw={'hspace': max(row_spacing), 'wspace': max(col_spacing)}
    )

    for i in range(num_methods):
        for j in range(num_tasks):
            img = imgs[i, j].permute(1, 2, 0).numpy()
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis("off")

            # Add confidence values below the image
            ax.text(
                0.5, -0.1, f"{confidences[i, j]:.2f}", fontsize=8, ha='center',
                transform=ax.transAxes
            )

            if i == 0:  # Add task names as column headers
                ax.set_title(task_names[j], fontsize=8)

            if j == 0:  # Add method names as row labels
                ax.text(
                    -1.0, 0.5, method_names[i], fontsize=8, ha='center',
                    transform=ax.transAxes
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    base_path = os.environ.get("PEAL_RUNS", "peal_runs")
    base_paths = [
        base_path + "/celeba/Smiling/classifier_natural",
        base_path + "/celeba/Blond_Hair/classifier_natural",
        base_path + "/waterbirds/classifier_natural",
        base_path
        + "/square/colora_confounding_colorb/torchvision/classifier_poisoned100",
    ]
    methods = ["ace_cfkd", "dime_cfkd", "fastdime_cfkd", "pdc_cfkd"]
    task_names = [
        "Smiling to Serious",
        "Serious to Smiling",
        "Blond to Non-Blond",
        "Non-Blond to Blond",
        "Landbird to Waterbird",
        "Waterbird to Landbird",
        "Dark to Light BG",
        "Light to Dark BG",
    ]
    method_names = [
        "Original",
        "ACE (1)",
        "ACE (2)",
        "DiME (1)",
        "DiME (2)",
        "FastDiME (1)",
        "FastDiME (2)",
        "PDC (ours) (1)",
        "PDC (ours) (2)",
    ]
    sample_idxs = [[17, 43], [6, 23], [3, 30], [0, 11]]
    imgs = torch.zeros([1 + 2 * len(methods), 2 * len(base_paths), 3, 128, 128])
    target_confidences = torch.zeros([1 + 2 * len(methods), 2 * len(base_paths)])

    for dataset_idx in range(len(base_paths)):
        for method_idx in range(len(methods)):
            tracked_values_path = os.path.join(
                base_paths[dataset_idx],
                methods[method_idx],
                "0",
                "validation_tracked_cluster_values.npz",
            )
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

                    for cluster_idx in range(NUM_CLUSTERS):
                        imgs[1 + 2 * method_idx + cluster_idx][
                            2 * dataset_idx + i
                        ] = resize(
                            torch.from_numpy(
                                tracked_values["clusters" + str(cluster_idx)][
                                    sample_idx
                                ]
                            ),
                            [128, 128],
                        )
                        target_confidences[1 + 2 * method_idx + cluster_idx][
                            2 * dataset_idx + i
                        ] = float(
                            tracked_values["cluster_confidence" + str(cluster_idx)][
                                sample_idx
                            ]
                        )

    imgs_reshaped = imgs.reshape([-1] + list(imgs.shape)[2:])
    plot_images_with_custom_padding(
        imgs,
        target_confidences,
        task_names,
        method_names,
        "collage_with_custom_padding.png",
    )
'''
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
    grid = GridSpec(num_methods, num_tasks, figure=fig, hspace=0, wspace=0.1)

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
                0.5, -0.1, f"{confidences[i, j]:.2f}", fontsize=8, ha='center',
                transform=ax.transAxes
            )

            if i == 0:  # Add task names as column headers
                ax.set_title(task_names[j], fontsize=8)

            if j == 0:  # Add method names as row labels
                ax.text(
                    -1.0, 0.5, method_names[i], fontsize=8, ha='center',
                    transform=ax.transAxes
                )

    plt.tight_layout()
    plt.subplots_adjust(
        top=1 - sum(row_spacing) / num_methods,
        bottom=0 + sum(row_spacing) / num_methods,
        left=0 + sum(col_spacing) / num_tasks,
        right=1 - sum(col_spacing) / num_tasks
    )
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    base_path = os.environ.get("PEAL_RUNS", "peal_runs")
    base_paths = [
        base_path + "/celeba/Smiling/classifier_natural",
        base_path + "/celeba/Blond_Hair/classifier_natural",
        base_path + "/waterbirds/classifier_natural",
        base_path
        + "/square/colora_confounding_colorb/torchvision/classifier_poisoned100",
    ]
    methods = ["ace_cfkd", "dime_cfkd", "fastdime_cfkd", "pdc_cfkd"]
    task_names = [
        "Serious to Smiling",
        "Smiling to Serious",
        "Non-Blond to Blond",
        "Blond to Non-Blond",
        "Waterbird to Landbird",
        "Landbird to Waterbird",
        "Light to Dark BG",
        "Dark to Light BG",
    ]
    method_names = [
        "Original",
        "ACE (1)",
        "ACE (2)",
        "DiME (1)",
        "DiME (2)",
        "FastDiME (1)",
        "FastDiME (2)",
        "PDC (ours) (1)",
        "PDC (ours) (2)",
    ]
    sample_idxs = [[17, 43], [6, 23], [3, 30], [0, 11]]
    imgs = torch.zeros([1 + 2 * len(methods), 2 * len(base_paths), 3, 128, 128])
    target_confidences = torch.zeros([1 + 2 * len(methods), 2 * len(base_paths)])

    for dataset_idx in range(len(base_paths)):
        for method_idx in range(len(methods)):
            tracked_values_path = os.path.join(
                base_paths[dataset_idx],
                methods[method_idx],
                "0",
                "validation_tracked_cluster_values.npz",
            )
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

                    for cluster_idx in range(NUM_CLUSTERS):
                        imgs[1 + 2 * method_idx + cluster_idx][
                            2 * dataset_idx + i
                        ] = resize(
                            torch.from_numpy(
                                tracked_values["clusters" + str(cluster_idx)][
                                    sample_idx
                                ]
                            ),
                            [128, 128],
                        )
                        target_confidences[1 + 2 * method_idx + cluster_idx][
                            2 * dataset_idx + i
                        ] = float(
                            tracked_values["cluster_confidence" + str(cluster_idx)][
                                sample_idx
                            ]
                        )

    for method_idx, method_name in enumerate(method_names):
        plot_images_with_custom_padding(
            imgs[method_idx:method_idx+1],
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

