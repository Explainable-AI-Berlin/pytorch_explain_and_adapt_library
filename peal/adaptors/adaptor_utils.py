import torch
import os
import copy
import numpy as np

from pathlib import Path

# TODO this is quite a bad smell... we should be able to do this without
from peal.data.datasets import Image2MixedDataset


def integrate_data_config_into_adaptor_config(
    adaptor_config, datasource, output_size=None
):
    if not output_size is None:
        output_size = output_size
        adaptor_config["data"]["output_size"] = output_size

    else:
        assert adaptor_config["data"]["output_size"] != "None"

    def integrate_task_into_adaptor_config(dataset, adaptor_config):
        if hasattr(dataset, "task_config") and not dataset.task_config is None:
            adaptor_config["task"] = dataset.task_config

        elif isinstance(dataset, Image2MixedDataset):
            adaptor_config["task"]["selection"] = [
                dataset.config["confounding_factors"][0]
            ]
            adaptor_config["task"]["output_size"] = 2
            dataset.task_config = adaptor_config["task"]

    if isinstance(datasource[0], torch.utils.data.Dataset):
        integrate_task_into_adaptor_config(datasource[0], adaptor_config)
        X, y = datasource[0].__getitem__(0)
        if hasattr(datasource[0], "config"):
            adaptor_config["data"] = datasource[0].config

        else:
            adaptor_config["data"]["input_size"] = list(X.shape)

    elif isinstance(datasource[0], torch.utils.data.DataLoader):
        integrate_task_into_adaptor_config(datasource[0].dataset, adaptor_config)
        X, y = datasource[0].dataset.__getitem__(0)
        adaptor_config["data"]["input_size"] = list(X.shape)

    else:
        assert adaptor_config["data"]["input_size"] != "None"

    return output_size


def calculate_validation_statistics(finetune_iteration, tracked_keys, base_dir):
    """
    This method calculates the validation statistics for the current finetune iteration.

    Args:
        finetune_iteration (int): The current finetune iteration.

    Returns:
        tuple: A tuple containing the validation statistics.
    """
    path = os.path.join(base_dir, str(finetune_iteration))

    #
    tracked_values = {key: [] for key in tracked_keys}

    (
        accuracy,
        confidence_score_stats,
        error_matrix,
        error_distribution,
        end_target_confidences,
    ) = calculate_validation_statistics(tracked_values)

    validation_basics = accuracy, error_distribution, confidence_score_stats
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(path, "validation_basics.npz"), "wb") as f:
        np.savez(
            f,
            accuracy=accuracy,
            error_matrix=error_matrix,
            confidence_score_stats=confidence_score_stats,
        )

    valid_tracked_values = {key: [] for key in tracked_values.keys()}
    sample_idx_iteration = 0
    Path(os.path.join(base_dir, str(finetune_iteration), "validation_collages")).mkdir(
        parents=True, exist_ok=True
    )
    for batch_idx in range(len(tracked_values.values()[0])):
        for sample_idx in range(tracked_values.values()[0][batch_idx].shape[0]):
            if end_target_confidences[batch_idx][sample_idx] > 0.5:
                for key in tracked_values.keys():
                    valid_tracked_values[key].append(tracked_values[key][sample_idx])

                sample_idx_iteration += 1

    with open(os.path.join(path, "validation_seriazed_values.npz"), "wb") as f:
        for key in valid_tracked_values.keys():
            np.savez(f, key=torch.stack(valid_tracked_values[key]).numpy())

    return validation_basics, valid_tracked_values


def retrieve_validation_statistics(finetune_iteration, tracked_keys, base_dir):
    """
    This method retrieves the validation statistics.

    Args:
        finetune_iteration (int): The finetune iteration for which the validation statistics should be retrieved.

    Returns:
        tuple: A tuple containing the validation statistics.
    """
    if not os.path.exists(
        os.path.join(
            base_dir, str(finetune_iteration), "validation_seriazed_values.npz"
        )
    ):
        (
            validation_basics,
            validation_tracked_values,
        ) = calculate_validation_statistics(finetune_iteration, tracked_keys, base_dir)
        accuracy, error_distribution, confidence_score_stats = validation_basics

    else:
        #
        with open(
            os.path.join(base_dir, str(finetune_iteration), "validation_basics.npz"),
            "rb",
        ) as f:
            validation_basics = np.load(f, allow_pickle=True)
            accuracy = float(validation_basics["accuracy"])
            error_matrix = torch.tensor(validation_basics["error_matrix"])
            error_distribution = torch.distributions.categorical.Categorical(
                error_matrix
            )
            confidence_score_stats = torch.tensor(
                validation_basics["confidence_score_stats"]
            )
        #
        with open(
            os.path.join(
                base_dir,
                str(finetune_iteration),
                "validation_seriazed_values.npz",
            ),
            "rb",
        ) as f:
            validation_seriazed_values = np.load(f, allow_pickle=True)
            validation_tracked_values = []
            for key in tracked_keys:
                validation_tracked_values.append(
                    torch.tensor(validation_seriazed_values[key])
                )

            collage_path_list = os.listdir(
                os.path.join(base_dir, str(finetune_iteration), "validation_collages")
            )
            validation_tracked_values["collage_paths"] = list(
                map(
                    lambda x: os.path.join(
                        base_dir,
                        str(finetune_iteration),
                        "validation_collages",
                        x,
                    ),
                    collage_path_list,
                )
            )

    validation_stats = {
        "accuracy": accuracy,
        "error_distribution": error_distribution,
        "confidence_score_stats": confidence_score_stats,
    }

    return validation_tracked_values, validation_stats


'''
TODO this function has to be readapted to the new project structure
def visualize_progress(paths):
    """
    This function visualizes the progress of the training process.

    Args:
        paths (list): List of paths where the visualizations are saved.

    Returns:
        torch.tensor: The visualization.
    """
    task_config_buffer = copy.deepcopy(test_dataloader.dataset.task_config)
    criterions = {}
    if (
        isinstance(test_dataloader.dataset, Image2MixedDataset)
        and "Confounder" in test_dataloader.dataset.attributes
    ):
        test_dataloader.dataset.task_config = {
            "selection": [],
            "criterions": [],
        }
        criterions["class"] = lambda X, y: int(
            y[
                test_dataloader.dataset.attributes.index(
                    task_config_buffer["selection"][0]
                )
            ]
        )
        criterions["confounder"] = lambda X, y: int(
            y[test_dataloader.dataset.attributes.index("Confounder")]
        )
        criterions["uncorrected"] = lambda X, y: int(
            original_student(X.unsqueeze(0).to(device)).squeeze(0).cpu().argmax()
        )
        criterions["cfkd"] = lambda X, y: int(
            student(X.unsqueeze(0).to(device)).squeeze(0).cpu().argmax()
        )

    else:
        criterions["class"] = lambda X, y: int(y)
        criterions["uncorrected"] = lambda X, y: int(
            original_student(X.unsqueeze(0).to(device)).squeeze(0).cpu().argmax()
        )
        criterions["cfkd"] = lambda X, y: int(
            student(X.unsqueeze(0).to(device)).squeeze(0).cpu().argmax()
        )

    img = create_comparison(
        dataset=test_dataloader.dataset,
        criterions=criterions,
        columns={
            "Counterfactual\nExplanation": [
                "cf",
                original_student,
                "uncorrected",
            ],
            "CFKD\ncorrected": ["cf", student, "cfkd"],
            "LRP\nuncorrected": ["lrp", original_student, "uncorrected"],
            "LRP\ncorrected": ["lrp", student, "cfkd"],
        },
        score_reference_idx=1,
        generator=generator,
        device=device,
        explainer_config=adaptor_config["explainer"],
    )
    test_dataloader.dataset.task_config = task_config_buffer
    for path in paths:
        img.save(path)

    return img
'''
