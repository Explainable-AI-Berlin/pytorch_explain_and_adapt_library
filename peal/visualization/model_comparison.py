import torch

from tqdm import tqdm

from peal.visualization.image_grid import make_image_grid
from peal.explainers.lrp_explainer import LRPExplainer
from peal.explainers.counterfactual_explainer import CounterfactualExplainer

def change_all(x, target_index, current_index):
    if target_index == current_index:
        return torch.stack([
            torch.zeros_like(x[0]),
            torch.ones_like(x[1])
        ])
    
    else:
        return torch.stack([
            change_all(x[0], target_index, current_index + 1),
            change_all(x[1], target_index, current_index + 1)
        ])


def create_checkbox_dict(criterion_keys):
    checkbox_dict = {}
    for idx, key in enumerate(criterion_keys):
        x = torch.zeros(
            len(criterion_keys) * [2], dtype=torch.long
        )
        checkbox_dict[key] = change_all(x, idx, 0).flatten()
    
    return checkbox_dict


def get_explanation(X, description, scores, checkbox_dict, generator, score_reference_idx, explainer_config = None):
    explanation_type, model, target_key = description
    lrp_target = checkbox_dict[target_key]
    if explanation_type == 'lrp':
        lrp_explainer = LRPExplainer(
            downstream_model=model,
            num_classes=2
        )
        lrp_heatmap, lrp_overlay, prediction = lrp_explainer.explain_batch(
            X,
            lrp_target
        )
        lrp_heatmap = lrp_heatmap.cpu()
        lrp_scores = list(map(
            lambda i: str(int(lrp_target[i])) + ':' + str(round(abs(abs(1 - float(scores[i])) - float(lrp_target[i])), 2)),
            range(scores.shape[0])
        ))
        return [lrp_heatmap, lrp_overlay], lrp_scores
    
    elif explanation_type == 'cf':
        cfkd_target = torch.abs(lrp_target - 1)
        if explainer_config is None:
            student_cfkd_counterfactual_explainer = CounterfactualExplainer(
                downstream_model = model,
                generator = generator
            )

        else:
            student_cfkd_counterfactual_explainer = CounterfactualExplainer(
                downstream_model = model,
                generator = generator,
                explainer_config = explainer_config
            )
        student_cfkd_counterfactual_explanation = student_cfkd_counterfactual_explainer.explain_batch(
            X,
            cfkd_target,
            source_classes = lrp_target
        )
        cfkd_counterfactual = student_cfkd_counterfactual_explanation[1].cpu()
        cfkd_heatmap = student_cfkd_counterfactual_explanation[2].cpu()
        scores_before = scores
        scores_after = torch.abs(torch.abs(cfkd_target - torch.tensor(student_cfkd_counterfactual_explanation[3])) - 1)
        cfkd_scores = list(map(
            lambda i: str(round(float(scores_before[i]), 2)) + ' -> ' + str(round(float(scores_after[i]), 2)),
            range(scores.shape[0])
        ))
        return [cfkd_counterfactual, cfkd_heatmap], cfkd_scores


def create_comparison(dataset, criterions, columns, score_reference_idx, generator, device, max_samples = 100000000, explainer_config = None):
    checkbox_dict = create_checkbox_dict(criterions.keys())
    scores_dict = {}
    for key in columns.keys():
        scores_dict[key] = torch.zeros([2] * len(criterions.keys()))
    
    samples = torch.zeros([2] * len(criterions.keys()) + list(dataset[0][0].shape))
    sample_idxs = torch.zeros([2] * len(criterions.keys()))
    with tqdm(range(50, 50 + min(max_samples, len(dataset) - 50))) as pbar:
        for i in pbar:
            X, y = dataset[i]
            current_results = torch.zeros(
                [len(criterions.keys())],
                dtype=torch.long
            )
            for idx, key in enumerate(criterions.keys()):
                current_results[idx] = criterions[key](X, y)
            
            if sample_idxs[list(current_results)] == 0:
                sample_idxs[list(current_results)] = 1
                samples[list(current_results)] = X
                for key in columns.keys():
                    scores = torch.nn.Softmax()(columns[key][1](X.unsqueeze(0).to(device)).squeeze(0).cpu())
                    scores_dict[key][list(current_results)] = scores[score_reference_idx]

            else:
                sample_idxs[list(current_results)] += 1

            pbar.set_description('num_samples: ' + str(torch.sum(sample_idxs != -1)))
    
    X = samples.reshape([2 ** len(criterions.keys())] + list(samples.shape[len(criterions.keys()):]))
    sample_idxs = sample_idxs.flatten()
    sample_idxs = list(map(
        lambda i: str(int(sample_idxs[i])),
        range(sample_idxs.shape[0])
    ))
    for key in columns.keys():
        scores_dict[key] = scores_dict[key].flatten()
    
    image_dicts ={}
    image_dicts['Image'] = [X, sample_idxs]
    for key in columns.keys():
        image_dicts[key] = get_explanation(
            X.cpu(), columns[key], scores_dict[key], checkbox_dict, generator, score_reference_idx, explainer_config
        )
    
    img = make_image_grid(
        checkbox_dict=checkbox_dict,
        image_dicts= image_dicts
    )

    return img
        


