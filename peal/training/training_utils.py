import torch
import numpy as np
import sys

from torch import nn
from tqdm import tqdm_notebook as tqdm
from typing import Union

from peal.explainers.explainer_interface import ExplainerInterface


def calculate_validation_statistics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tracked_keys: list,
    base_path: str,
    output_size: int,
    device: Union[str, torch.device],
    logits_to_prediction: callable,
    use_confusion_matrix: bool,
    explainer: ExplainerInterface,
    max_validation_samples: int,
    min_start_target_percentile: torch.tensor,
):
    """
    This function calculates the validation statistics for a given model and dataloader.

    Args:
        model (nn.Module): _description_
        dataloader (torch.utils.data.DataLoader): _description_
        tracked_keys (list): _description_
        base_path (str): _description_
        output_size (int): _description_
        device (Union[str, torch.device]): _description_
        logits_to_prediction (callable): _description_
        use_confusion_matrix (bool): _description_
        explainer (ExplainerInterface): _description_
        max_validation_samples (int): _description_
        min_start_target_percentile (torch.tensor): _description_

    Returns:
        _type_: _description_
    """
    tracked_values = {key: [] for key in tracked_keys}
    confusion_matrix = np.zeros([output_size, output_size])
    correct = 0
    num_samples = 0
    confidence_scores = []
    for i in range(output_size):
        confidence_scores.append([])

    pbar = tqdm(
        total=int(
            min(max_validation_samples, len(dataloader.dataset)) / dataloader.batch_size
            + 0.9999
        )
        * explainer.explainer_config["gradient_steps"],
    )
    pbar.stored_values = {}
    for it, (x, y) in enumerate(dataloader):
        pred_confidences = torch.nn.Softmax(dim=-1)(model(x.to(device))).detach().cpu()
        y_pred = logits_to_prediction(pred_confidences)
        for i in range(y.shape[0]):
            if y_pred[i] == y[i]:
                correct += 1
                confidence_scores[int(y[i])].append(pred_confidences[i])

            confusion_matrix[int(y[i])][int(y_pred[i])] += 1
            num_samples += 1

        pbar.stored_values["acc"] = correct / num_samples

        batch_targets = (y_pred + 1) % output_size
        batch_target_start_confidences = []
        for sample_idx in range(pred_confidences.shape[0]):
            batch_target_start_confidences.append(
                pred_confidences[sample_idx][batch_targets[sample_idx]]
            )
        batch = {}
        batch["x_list"] = x
        batch["y_list"] = y
        batch["y_source_list"] = y_pred
        batch["y_target_list"] = batch_targets
        batch["y_target_start_confidence_list"] = torch.stack(
            batch_target_start_confidences, 0
        )
        results = explainer.explain_batch(
            batch=batch,
            base_path=base_path,
            remove_below_threshold=False,
            pbar=pbar,
            mode="Validation",
        )
        for key in set(results.keys()).intersection(set(tracked_values.keys())):
            tracked_values[key].extend(results[key])

        if num_samples >= max_validation_samples:
            break

    pbar.close()
    confidence_score_stats = []
    for i in range(output_size):
        if len(confidence_scores[i]) >= 1:
            confidence_score_stats.append(
                torch.quantile(
                    torch.stack(confidence_scores[i], dim=1),
                    min_start_target_percentile,
                    dim=1,
                )
            )

        else:
            confidence_score_stats.append(torch.zeros([output_size]))

    confidence_score_stats = torch.stack(confidence_score_stats)
    accuracy = correct / num_samples

    if use_confusion_matrix and not accuracy == 1.0:
        error_matrix = np.copy(confusion_matrix)
        for i in range(error_matrix.shape[0]):
            error_matrix[i][i] = 0.0

        error_matrix = error_matrix.flatten()
        error_matrix = error_matrix / error_matrix.sum()
        error_distribution = torch.distributions.categorical.Categorical(
            torch.tensor(error_matrix)
        )

    else:
        error_matrix = torch.ones([output_size, output_size]) - torch.eye(output_size)
        error_matrix = error_matrix.flatten() / error_matrix.sum()
        error_distribution = torch.distributions.categorical.Categorical(error_matrix)

    validation_stats = {
        "accuracy": accuracy,
        "confidence_score_stats": confidence_score_stats,
        "error_distribution": error_distribution,
    }

    return tracked_values, validation_stats
