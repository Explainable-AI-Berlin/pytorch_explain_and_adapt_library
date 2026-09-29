"""Validation helpers shared by PEAL's predictor training loops.

The one public function, :func:`calculate_validation_statistics`, walks
validation dataloaders, measures the accuracy and error structure of a
classifier and runs an explainer on every batch so that counterfactual
statistics (flip rates, hints, indices, ...) are collected alongside the plain
validation metrics. Adaptors such as CFKD call it once per fine-tuning round.
"""

import torch
import numpy as np

from torch import nn
from tqdm import tqdm
from typing import Union

from peal.explainers.interfaces import ExplainerInterface


def calculate_validation_statistics(
    model: nn.Module,
    dataloaders: torch.utils.data.DataLoader,
    tracked_keys: list,
    base_path: str,
    output_size: int,
    device: Union[str, torch.device],
    logits_to_prediction: callable,
    use_confusion_matrix: bool,
    explainer: ExplainerInterface,
    max_validation_samples: int,
):
    """Compute accuracy / error statistics and explainer outputs on validation data.

    For every dataloader the model is evaluated with temperature-scaled softmax
    (``explainer.explainer_config.temperature``), the prediction is mapped to a
    class with ``logits_to_prediction`` and the explainer is asked to explain the
    batch towards the target ``(y_pred + 1) % output_size``. Keys of the
    explainer result that appear in ``tracked_keys`` are accumulated.

    Parameters
    ----------
    model : nn.Module
        Classifier producing logits of shape ``(B, output_size)``.
    dataloaders : list of torch.utils.data.DataLoader
        Validation loaders; the sample budget is split evenly between them and
        a trailing batch smaller than ``batch_size`` stops the loop.
    tracked_keys : list of str
        Explainer result keys to collect, e.g. ``"x_counterfactual_list"``.
        ``"hint_list"`` / ``"idx_list"`` additionally unpack a list-valued ``y``
        into label, hints and indices.
    base_path : str
        Directory handed to ``explainer.explain_batch`` for its artefacts.
    output_size : int
        Number of classes.
    device : str or torch.device
        Device the inputs are moved to.
    logits_to_prediction : callable
        Maps a confidence tensor to integer predictions.
    use_confusion_matrix : bool
        If True and the accuracy is below 1.0, the normalised off-diagonal
        confusion matrix is returned as ``error_matrix``; otherwise a uniform
        off-diagonal matrix is used.
    explainer : ExplainerInterface
        Explainer whose ``explain_batch`` runs in ``mode="validation"``.
    max_validation_samples : int
        Upper bound on the number of samples over all dataloaders.

    Returns
    -------
    tracked_values : dict
        ``{key: list}`` for every key in ``tracked_keys``.
    validation_stats : dict
        ``accuracy`` (float), ``confidence_score_stats`` (zero tensor of shape
        ``(output_size, output_size)``, currently a placeholder) and
        ``error_matrix`` (flattened tensor of length ``output_size ** 2`` that
        sums to one). Only the last dataloader's values are returned.
    """
    tracked_values = {key: [] for key in tracked_keys}
    for dataloader in dataloaders:
        confusion_matrix = np.zeros([output_size, output_size])
        correct = 0
        num_samples = 0
        confidence_scores = []
        for i in range(output_size):
            confidence_scores.append([])

        pbar = tqdm(
            total=int(
                min(max_validation_samples, len(dataloader.dataset))
                / dataloader.batch_size
                + 0.9999
            )
            * (
                explainer.explainer_config.gradient_steps
                if hasattr(explainer.explainer_config, "gradient_steps")
                else 1
            ),
        )
        pbar.stored_values = {}

        for it, (x, y) in enumerate(dataloader):

            if (
                num_samples >= int(max_validation_samples / len(dataloaders))
                or x.shape[0] != dataloader.batch_size
            ):
                break

            try:
                with torch.no_grad():
                    pred_confidences = (
                        torch.nn.Softmax(dim=-1)(
                            model(x.to(device)) / explainer.explainer_config.temperature
                        )
                        .detach()
                        .cpu()
                    )
            except:
                raise
            y_pred = logits_to_prediction(pred_confidences)
            if (
                "hint_list" in tracked_keys or "idx_list" in tracked_keys
            ) and isinstance(y, list):
                y_res = y[1:]
                y = y[0]
                if "hint_list" in tracked_keys:
                    hints = y_res[0]

                if "idx_list" in tracked_keys:
                    idxs = y_res[-1]

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
            if "hint_list" in tracked_keys:
                batch["hint_list"] = hints

            if "idx_list" in tracked_keys:
                batch["idx_list"] = idxs

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
                mode="validation",
                start_idx=it * dataloader.batch_size,
                # dataloader=dataloader,
            )
            # import pdb; pdb.set_trace()
            # import torchvision; torchvision.utils.save_image(results['x_counterfactual_list'][0], "ace_066.png")
            # except TypeError:
            #    import pdb; pdb.set_trace()
            for key in set(results.keys()).intersection(set(tracked_values.keys())):
                tracked_values[key].extend(results[key])

        pbar.close()

        confidence_score_stats = []
        for i in range(output_size):
            confidence_score_stats.append(torch.zeros([output_size]))

        confidence_score_stats = torch.stack(confidence_score_stats)
        accuracy = correct / num_samples

        if use_confusion_matrix and not accuracy == 1.0:
            error_matrix = np.copy(confusion_matrix)
            for i in range(error_matrix.shape[0]):
                error_matrix[i][i] = 0.0

            error_matrix = error_matrix.flatten()
            error_matrix = error_matrix / error_matrix.sum()
            error_matrix = torch.tensor(error_matrix)

        else:
            error_matrix = torch.ones([output_size, output_size]) - torch.eye(
                output_size
            )
            error_matrix = error_matrix.flatten() / error_matrix.sum()

        validation_stats = {
            "accuracy": accuracy,
            "confidence_score_stats": confidence_score_stats,
            "error_matrix": error_matrix,
        }

    return tracked_values, validation_stats
