import torch

from tqdm import tqdm

from peal.visualization.image_grid import make_image_grid
from peal.explainers.lrp_explainer import LRPExplainer
from peal.explainers.counterfactual_explainer import CounterfactualExplainer


def change_all(x, target_index, current_index):
    if target_index == current_index:
        return torch.stack([torch.zeros_like(x[0]), torch.ones_like(x[1])])

    else:
        return torch.stack(
            [
                change_all(x[0], target_index, current_index + 1),
                change_all(x[1], target_index, current_index + 1),
            ]
        )


def create_checkbox_dict(criterion_keys):
    checkbox_dict = {}
    for idx, key in enumerate(criterion_keys):
        x = torch.zeros(len(criterion_keys) * [2], dtype=torch.long)
        checkbox_dict[key] = change_all(x, idx, 0).flatten()

    return checkbox_dict


def get_explanation(
    X,
    description,
    scores,
    checkbox_dict,
    explainer,
    batch_size=1,
):
    explanation_type, model, target_key, base_path = description
    lrp_target = checkbox_dict[target_key]
    if isinstance(explainer, LRPExplainer):
        lrp_heatmap, lrp_overlay, prediction = explainer.explain_batch(
            X, lrp_target
        )
        lrp_heatmap = lrp_heatmap.cpu()
        lrp_scores = list(
            map(
                lambda i: str(int(lrp_target[i]))
                + ":"
                + str(round(abs(abs(1 - float(scores[i])) - float(lrp_target[i])), 2)),
                range(scores.shape[0]),
            )
        )
        return [lrp_heatmap, lrp_overlay], lrp_scores

    elif isinstance(explainer, CounterfactualExplainer):
        cfkd_target = torch.abs(lrp_target - 1)
        """if explainer_config is None:
            student_cfkd_counterfactual_explainer = CounterfactualExplainer(
                downstream_model=model,
                generator=generator,
                input_type="image",
                dataset=dataset,
            )

        else:
            student_cfkd_counterfactual_explainer = CounterfactualExplainer(
                downstream_model=model,
                generator=generator,
                explainer_config=explainer_config,
                input_type="image",
                dataset=dataset,
            )"""

        current_idx = 0
        while current_idx < X.shape[0]:
            # TODO these entries are not correct
            batch = {
                "x_list": X[current_idx : current_idx + batch_size],
                "y_target_list": cfkd_target[current_idx : current_idx + batch_size],
                "y_source_list": lrp_target[current_idx : current_idx + batch_size],
                "y_list": lrp_target[current_idx : current_idx + batch_size],
                "y_target_start_confidence_list": torch.zeros([batch_size]),
            }
            student_cfkd_counterfactual_explanation = (
                explainer.explain_batch(
                    batch, remove_below_threshold=False, explainer_path=base_path
                )
            )
            try:
                cfkd_counterfactual = torch.stack(
                    student_cfkd_counterfactual_explanation["x_counterfactual_list"]
                ).cpu()

            except Exception:
                import pdb; pdb.set_trace()

            cfkd_heatmap = torch.stack(
                student_cfkd_counterfactual_explanation["x_attribution_list"]
            ).cpu()
            scores_before = scores[current_idx : current_idx + batch_size]
            scores_after = torch.abs(
                torch.abs(
                    cfkd_target[current_idx : current_idx + batch_size]
                    - torch.tensor(
                        student_cfkd_counterfactual_explanation[
                            "y_target_end_confidence_list"
                        ]
                    )
                )
                - 1
            )
            cfkd_score = list(
                map(
                    lambda i: str(round(float(scores_before[i]), 2))
                    + " -> "
                    + str(round(float(scores_after[i]), 2)),
                    range(scores_before.shape[0]),
                )
            )
            if current_idx == 0:
                cfkd_counterfactuals = cfkd_counterfactual
                cfkd_heatmaps = cfkd_heatmap
                cfkd_scores = cfkd_score

            else:
                cfkd_counterfactuals = torch.cat(
                    [cfkd_counterfactuals, cfkd_counterfactual], dim=0
                )
                cfkd_heatmaps = torch.cat([cfkd_heatmaps, cfkd_heatmap], dim=0)
                cfkd_scores = cfkd_scores + cfkd_score

            current_idx += batch_size

        return [cfkd_counterfactuals, cfkd_heatmaps], cfkd_scores


def create_comparison(
    explainer,
    dataset,
    criterions,
    columns,
    score_reference_idx,
    device,
    max_samples=100000000,
    checkbox_dict_in=None,
    batch_size=1,
):
    scores_dict = {}

    if checkbox_dict_in is None:
        checkbox_dict = create_checkbox_dict(criterions.keys())
        samples = torch.zeros([2] * len(criterions.keys()) + list(dataset[0][0].shape))
        sample_idxs = torch.zeros([2] * len(criterions.keys()))
        for key in columns.keys():
            scores_dict[key] = torch.zeros([2] * len(criterions.keys()))

    else:
        checkbox_dict = checkbox_dict_in
        samples = torch.zeros(
            [checkbox_dict["class"].shape[0]] + list(dataset[0][0].shape)
        )
        sample_idxs = torch.zeros([checkbox_dict["class"].shape[0]])
        checkbox_criterions = torch.zeros(
            [checkbox_dict["class"].shape[0], len(criterions.keys())]
        )
        for i, criterion in enumerate(criterions.keys()):
            for j in range(checkbox_criterions.shape[0]):
                checkbox_criterions[j][i] = checkbox_dict[criterion][j]

        for key in columns.keys():
            scores_dict[key] = torch.zeros(checkbox_dict["class"].shape[0])

    with tqdm(range(50, 50 + min(max_samples, len(dataset) - 50))) as pbar:
        for i in pbar:
            X, y = dataset[i]
            current_results = torch.zeros([len(criterions.keys())], dtype=torch.long)
            for idx, key in enumerate(criterions.keys()):
                current_results[idx] = criterions[key](X, y)

            if checkbox_dict_in is None:
                if sample_idxs[list(current_results)] == 0:
                    sample_idxs[list(current_results)] = 1
                    samples[list(current_results)] = X
                    for key in columns.keys():
                        scores = torch.nn.Softmax()(
                            columns[key][1](X.unsqueeze(0).to(device)).squeeze(0).cpu()
                        )
                        scores_dict[key][list(current_results)] = scores[
                            score_reference_idx
                        ]

                else:
                    sample_idxs[list(current_results)] += 1

                pbar.set_description(
                    "num_samples: " + str(torch.sum(sample_idxs != -1))
                )

            else:
                for i in range(checkbox_dict["class"].shape[0]):
                    if sample_idxs[i] == 0 and torch.sum(
                        current_results == checkbox_criterions[i]
                    ) == len(checkbox_criterions[i]):
                        sample_idxs[i] = 1
                        samples[i] = X
                        for key in columns.keys():
                            scores = torch.nn.Softmax()(
                                columns[key][1](X.unsqueeze(0).to(device))
                                .squeeze(0)
                                .cpu()
                            )
                            scores_dict[key][i] = scores[score_reference_idx]

                        break

                if torch.sum(sample_idxs) == checkbox_dict["class"].shape[0]:
                    break

    if checkbox_dict_in is None:
        X = samples.reshape(
            [2 ** len(criterions.keys())]
            + list(samples.shape[len(criterions.keys()) :])
        )
        sample_idxs = sample_idxs.flatten()
        sample_idxs_str = list(
            map(lambda i: str(int(sample_idxs[i])), range(sample_idxs.shape[0]))
        )

    else:
        X = samples
        sample_idxs_str = X.shape[0] * [""]

    for key in columns.keys():
        scores_dict[key] = scores_dict[key].flatten()

    image_dicts = {}
    image_dicts["Image"] = [X, sample_idxs_str]
    for key in columns.keys():
        explainer.predictor = columns[key][1]
        image_dicts[key] = get_explanation(
            X.cpu(),
            columns[key],
            scores_dict[key],
            checkbox_dict,
            explainer=explainer,
            batch_size=batch_size,
        )

    img = make_image_grid(checkbox_dict=checkbox_dict, image_dicts=image_dicts)

    return img
