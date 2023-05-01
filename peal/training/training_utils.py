import torch
import numpy as np

from tqdm import tqdm


def calculate_validation_statistics(
    model,
    dataloader,
    tracked_keys,
    base_path,
    output_size,
    device,
    logits_to_prediction,
    use_confusion_matrix,
    explainer,
    max_validation_samples,
    min_start_target_percentile,
):
    tracked_values = {key: [] for key in tracked_keys}
    confusion_matrix = np.zeros([output_size, output_size])
    correct = 0
    num_samples = 0
    confidence_scores = []
    for i in range(output_size):
        confidence_scores.append([])

    with tqdm(enumerate(dataloader)) as pbar:
        for it, (X, y) in pbar:
            pred_confidences = torch.nn.Softmax()(model(X.to(device))).detach().cpu()
            y_pred = logits_to_prediction(pred_confidences)
            for i in range(y.shape[0]):
                if y_pred[i] == y[i]:
                    correct += 1
                    confidence_scores[int(y[i])].append(pred_confidences[i])

                confusion_matrix[int(y[i])][int(y_pred[i])] += 1
                num_samples += 1

            pbar.set_description(
                "Calculate Confusion Matrix: it: "
                + str(it)
                + ", current_accuracy: "
                + str(correct / num_samples)
            )

            batch_targets = (y_pred + 1) % output_size
            batch_target_start_confidences = []
            for sample_idx in range(pred_confidences.shape[0]):
                batch_target_start_confidences.append(
                    pred_confidences[sample_idx][batch_targets[sample_idx]]
                )
            tracked_values["y"].append(y)
            tracked_values["y_pred"].append(y_pred)
            tracked_values["target"].append(batch_targets)
            tracked_values["start_target_confidence"].append(
                torch.stack(batch_target_start_confidences, 0)
            )
            results = explainer.explain_batch(
                batch_in=X,
                target_classes=batch_targets,
                source_classes=y_pred,
                base_path=os.path.join(
                    base_path,
                    "collages",
                ),
            )
            for key in set(results.keys()).intersection(set(tracked_values.keys())):
                tracked_values[key].append(results[key])

            if num_samples >= max_validation_samples:
                break

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
