import torch
import os
import torchvision
import numpy as np
import torchvision.utils

NUM_CLUSTERS = 2


if __name__ == "__main__":
    base_path = os.environ.get("PEAL_RUNS", "peal_runs")
    base_paths = [
        base_path + "/celeba/Smiling/classifier_natural",
        base_path + "/celeba/Blond_Hair/classifier_natural",
        base_path + "/waterbirds/classifier_natural",
        base_path + "/square/colora_confounding_colorb/torchvision/classifier_poisoned100",
    ]
    methods = ["ace_cfkd", "dime_cfkd", "fastdime_cfkd", "pdc_cfkd"]
    dataset_names = [""]
    method_names = ["ACE", "DiME", "FastDiME", "PDC (ours)"]
    sample_idxs = [[0, 1], [0, 1], [0, 1], [0, 1]]
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

                for sample_idx in sample_idxs[dataset_idx]:
                    if method_idx == 0:
                        imgs[0][2 * dataset_idx + sample_idx] = torchvision.transforms.Resize([128, 128])(
                            torch.from_numpy(tracked_values["x_list"][sample_idx])
                        )
                        target_confidences[0][2 * dataset_idx + sample_idx] = (
                            float(tracked_values["y_target_start_confidence_list"][sample_idx])
                        )

                    for cluster_idx in range(NUM_CLUSTERS):
                        imgs[1 + 2 * method_idx + cluster_idx][
                            2 * dataset_idx + sample_idx
                        ] = torchvision.transforms.Resize([128, 128])(
                            torch.from_numpy(tracked_values["x_counterfactual_list"][sample_idx])
                        )
                        target_confidences[1 + 2 * method_idx + cluster_idx][
                            2 * dataset_idx + sample_idx
                        ] = (
                            float(tracked_values["y_target_end_confidence_list"][sample_idx])
                        )
                        """imgs[2 * dataset_idx + sample_idx][
                            1 + 2 * method_idx + cluster_idx
                        ] = torchvision.transforms.Resize([128, 128])(
                            torch.from_numpy(tracked_values["clusters" + str(cluster_idx)][sample_idx])
                        )
                        target_confidences[2 * dataset_idx + sample_idx][
                            1 + 2 * method_idx + cluster_idx
                        ] = (
                            float(tracked_values["cluster_confidence" + str(cluster_idx)][sample_idx])
                        )"""

    imgs_reshaped = imgs.reshape([-1] + list(imgs.shape)[2:])
    torchvision.utils.save_image(imgs_reshaped, "collage.png", nrow=imgs.shape[1])
