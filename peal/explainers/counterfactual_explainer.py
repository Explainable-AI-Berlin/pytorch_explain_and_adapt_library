import torch

from torch import nn
from tqdm import tqdm
from pathlib import Path

from peal.utils import load_yaml_config
from peal.generators.interfaces import InvertibleGenerator


class CounterfactualExplainer:
    """
    This class implements the counterfactual explanation method
    """

    def __init__(
        self,
        downstream_model,
        generator,
        input_type,
        generate_contrastive_collage,
        explainer_config="$PEAL/configs/explainers/counterfactual_default.yaml",
        from_pytorch_canonical=lambda x: x,
        to_pytorch_canonical=lambda x: x,
    ):
        """
        This class implements the counterfactual explanation method

        Args:
            downstream_model (_type_): _description_
            generator (_type_): _description_
            input_type (_type_): _description_
            explainer_config (str, optional): _description_. Defaults to "/configs/explainers/counterfactual_default.yaml".
            from_pytorch_canonical (_type_, optional): _description_. Defaults to lambdax:x.
            to_pytorch_canonical (_type_, optional): _description_. Defaults to lambdax:x.
        """
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
        self.generate_contrastive_collage = generate_contrastive_collage

    def gradient_based_counterfactual(
        self, batch_in, target_confidence_goal, target_classes
    ):
        """
        This function generates a counterfactual for a given batch of inputs.

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch = torch.clone(batch_in)
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

        return counterfactual, attributions, target_confidences

    def explain_batch(
        self,
        batch: dict,
        base_path="collages",
        start_idx=0,
        target_confidence_goal_in=None,
        remove_below_threshold=True,
    ):
        """
        _summary_

        Args:
            x (_type_): _description_
            target_classes (_type_): _description_
            target_confidence_goal_in (_type_, optional): _description_. Defaults to None.
            source_classes (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if target_confidence_goal_in is None:
            target_confidence_goal = self.explainer_config["target_confidence_goal"]

        else:
            target_confidence_goal = target_confidence_goal_in

        if isinstance(self.generator, InvertibleGenerator):
            (
                batch["x_counterfactual"],
                batch["z_difference"],
                batch["y_target_confidence"],
            ) = self.gradient_based_counterfactual(batch["x"], target_confidence_goal)

        batch_out = {}
        if remove_below_threshold:
            # TODO remove rows in counterfactual, attribution and x where target_confidences < target_confidence_goal
            for key in batch.keys():
                batch_out[key] = []
                for sample_idx in range(len(batch[key])):
                    if (
                        batch["y_target_confidence"][sample_idx]
                        < target_confidence_goal
                    ):
                        batch_out[key].append(batch[key][sample_idx])

        else:
            batch_out = batch

        batch_out["x_attribution"] = self.dataset.generate_contrastive_collage(
            x=batch_out["x"],
            x_counterfactual=batch_out["x_counterfactual"],
            target_confidence_goal=target_confidence_goal,
            y_target=batch_out["y_target"],
            y_source=batch_out["y_source"],
            base_path=base_path,
            start_idx=start_idx,
        )

        return batch_out
