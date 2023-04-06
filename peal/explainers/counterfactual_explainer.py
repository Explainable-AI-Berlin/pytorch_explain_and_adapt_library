import torch

from torch import nn
from tqdm import tqdm

from peal.utils import load_yaml_config
from peal.explainers.lrp_explainer import LRPExplainer


def generate_image_collage(batch_in, counterfactual):
    heatmap_red = torch.maximum(
        torch.tensor(0.0),
        torch.sum(batch_in, dim=1) - torch.sum(counterfactual, dim=1),
    )
    heatmap_blue = torch.maximum(
        torch.tensor(0.0),
        torch.sum(counterfactual, dim=1) - torch.sum(batch_in, dim=1),
    )
    if counterfactual.shape[1] == 3:
        heatmap_green = torch.abs(counterfactual[:, 0] - batch_in[:, 0])
        heatmap_green = heatmap_green + torch.abs(counterfactual[:, 1] - batch_in[:, 1])
        heatmap_green = heatmap_green + torch.abs(counterfactual[:, 2] - batch_in[:, 2])
        heatmap_green = heatmap_green - heatmap_red - heatmap_blue
        counterfactual_rgb = counterfactual

    else:
        heatmap_green = torch.zeros_like(heatmap_red)
        batch_in = torch.tile(batch_in, [1, 3, 1, 1])
        counterfactual_rgb = torch.tile(torch.clone(counterfactual), [1, 3, 1, 1])

    heatmap = torch.stack([heatmap_red, heatmap_green, heatmap_blue], dim=1)
    if torch.abs(heatmap.sum() - torch.abs(batch_in - counterfactual).sum()) > 0.1:
        print("Error: Heatmap does not match counterfactual")

    heatmap_high_contrast = torch.clamp(heatmap / heatmap.max(), 0.0, 1.0)
    result_img_collage = torch.cat(
        [batch_in, counterfactual_rgb, heatmap_high_contrast], -1
    )
    return result_img_collage, heatmap_high_contrast


def generate_symbolic_collage(batch_in, counterfactual):
    # TODO these are still placeholders
    return torch.zeros([batch_in.shape[0], 3, 64, 64]), torch.zeros_like(batch_in)


def generate_sequence_collage(batch_in, counterfactual):
    # TODO these are still placeholders
    return torch.zeros([batch_in.shape[0], 3, 64, 64]), torch.zeros_like(batch_in)


class CounterfactualExplainer:
    def __init__(
        self,
        downstream_model,
        generator,
        input_type,
        explainer_config="$PEAL/configs/explainers/counterfactual_default.yaml",
        num_classes=2,
        from_pytorch_canonical=lambda x: x,
        to_pytorch_canonical=lambda x: x,
    ):
        self.downstream_model = downstream_model
        self.generator = generator
        self.explainer_config = load_yaml_config(explainer_config)
        self.device = (
            "cuda" if next(self.downstream_model.parameters()).is_cuda else "cpu"
        )
        self.input_type = input_type
        self.loss = torch.nn.CrossEntropyLoss()
        self.from_pytorch_canonical = from_pytorch_canonical
        self.to_pytorch_canonical = to_pytorch_canonical
    
    def gradient_based_counterfactual(batch):
        batch = self.from_pytorch_canonical(batch)
        v_original = self.generator.encode(batch.to(self.device))
        if isinstance(v_original, list):
            v = []

            for v_org in v_original:
                v.append(
                    nn.Parameter(torch.clone(v_org.detach().cpu()), requires_grad=True)
                )

        else:
            v = nn.Parameter(torch.clone(v_original.detach().cpu()), requires_grad=True)
            v = [v]

        if self.explainer_config["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(v, lr=self.explainer_config["learning_rate"])

        elif self.explainer_config["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(v, lr=self.explainer_config["learning_rate"])

        target_confidences = [0.0 for i in range(len(target_classes))]

        with tqdm(range(self.explainer_config["gradient_steps"])) as pbar:
            for i in pbar:
                if self.explainer_config["use_masking"]:
                    mask = (
                        torch.tensor(target_confidences).to(self.device)
                        < target_confidence_goal
                    )
                    if torch.sum(mask) == 0.0:
                        break

                latent_code = [v_elem.to(self.device) for v_elem in v]

                optimizer.zero_grad()

                img = self.generator.decode(latent_code)

                img = self.to_pytorch_canonical(img)

                logits = self.downstream_model(
                    img
                    + self.explainer_config["img_noise_injection"]
                    * torch.randn_like(img)
                )
                loss = self.loss(logits, target_classes.to(self.device))
                l1_losses = []
                for v_idx in range(len(v_original)):
                    l1_losses.append(
                        torch.mean(
                            torch.abs(
                                v[v_idx].to(self.device)
                                - torch.clone(v_original[v_idx]).detach()
                            )
                        )
                    )

                loss += self.explainer_config["l1_regularization"] * torch.mean(
                    torch.stack(l1_losses)
                )
                loss += self.explainer_config["log_prob_regularization"] * torch.mean(
                    self.generator.log_prob_z(latent_code)
                )

                if self.explainer_config["img_regularization"] > 0:
                    difference = img - torch.clone(batch).to(self.device).detach()
                    loss += torch.mean(torch.abs(difference) * regularization_mask)

                logit_confidences = torch.nn.Softmax()(logits).detach().cpu()
                target_confidences = [
                    float(logit_confidences[i][target_classes[i]])
                    for i in range(len(target_classes))
                ]

                pbar.set_description(
                    "Creating Counterfactuals: it: "
                    + str(i)
                    + ", loss: "
                    + str(loss.detach().item())
                    + ", target_confidences: "
                    + str(list(target_confidences)[:2])
                    + ", visual_difference: "
                    + str(torch.mean(torch.abs(batch_in - img.detach().cpu())).item())
                )

                loss.backward()

                if self.explainer_config["use_masking"]:
                    for sample_idx in range(len(target_confidences)):
                        if target_confidences[sample_idx] >= target_confidence_goal:
                            for variable_idx, v_elem in enumerate(v):
                                if self.explainer_config["optimizer"] == "Adam":
                                    optimizer = torch.optim.Adam(
                                        v, lr=self.explainer_config["learning_rate"]
                                    )

                                v_elem.grad[sample_idx].data.zero_()

                optimizer.step()

        latent_code = [v_elem.to(self.device) for v_elem in v]
        counterfactual = self.generator.decode(latent_code).detach().cpu()
        counterfactual = self.to_pytorch_canonical(counterfactual)

        attributions = []
        for v_idx in range(len(v_original)):
            attributions.append(
                torch.flatten(
                    v_original[v_idx].detach().cpu() - v[v_idx].detach().cpu(), 1
                )
            )

        attributions = torch.cat(attributions, 1)

        return counterfactual, attributions

    def explain_batch(
        self,
        batch_in,
        target_classes,
        target_confidence_goal_in=None,
        source_classes=None,
    ):
        """ """
        if target_confidence_goal_in is None:
            target_confidence_goal = self.explainer_config["target_confidence_goal"]

        else:
            target_confidence_goal = target_confidence_goal_in

        batch = batch_in.clone()

        if isinstance(self.generator, InvertibleGenerator):
            counterfactual, attributions = self.gradient_based_counterfactual(batch)        

        # TODO insert this via a decorator!
        self.generate_contrastive_collage(batch_in, counterfactual)
        '''if self.input_type == "symbolic":
            result_img_collage, heatmap = generate_symbolic_collage(
                batch_in,
                counterfactual,
            )

        elif self.input_type == "sequence":
            result_img_collage, heatmap = generate_sequence_collage(
                batch_in,
                counterfactual,
            )

        elif self.input_type == "image":
            result_img_collage, heatmap = generate_image_collage(
                batch_in,
                counterfactual,
            )'''

        return (
            result_img_collage,
            counterfactual.detach().cpu(),
            heatmap,
            target_confidences,
            attributions,
        )
