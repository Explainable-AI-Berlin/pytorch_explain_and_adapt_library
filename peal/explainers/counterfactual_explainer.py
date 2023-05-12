import torch

from torch import nn
from tqdm import tqdm
from typing import Union

from peal.utils import load_yaml_config
from peal.generators.interfaces import InvertibleGenerator
from peal.data.dataset_interfaces import PealDataset
from peal.explainers.explainer_interface import ExplainerInterface


class CounterfactualExplainer(ExplainerInterface):
    """
    This class implements the counterfactual explanation method
    """

    def __init__(
        self,
        downstream_model: nn.Module,
        generator: InvertibleGenerator,
        input_type: str,
        dataset: PealDataset,
        explainer_config: Union[
            dict, str
        ] = "$PEAL/configs/explainers/counterfactual_default.yaml",
    ):
        """
        This class implements the counterfactual explanation method

        Args:
            downstream_model (nn.Module): _description_
            generator (InvertibleGenerator): _description_
            input_type (str): _description_
            dataset (PealDataset): _description_
            explainer_config (Union[ dict, str ], optional): _description_. Defaults to "/configs/explainers/counterfactual_default.yaml".
        """
        self.downstream_model = downstream_model
        self.generator = generator
        self.dataset = dataset
        self.explainer_config = load_yaml_config(explainer_config)
        self.device = (
            "cuda" if next(self.downstream_model.parameters()).is_cuda else "cpu"
        )
        self.input_type = input_type
        self.loss = torch.nn.CrossEntropyLoss()

    def gradient_based_counterfactual(
        self, x_in, target_confidence_goal, target_classes
    ):
        """
        This function generates a counterfactual for a given batch of inputs.

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = torch.clone(x_in)
        x = self.dataset.project_from_pytorch_default(x)
        v_original = self.generator.encode(x.to(self.device))
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

                img = self.dataset.project_to_pytorch_default(img)

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
                    + str(torch.mean(torch.abs(x_in - img.detach().cpu())).item())
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
        counterfactual = self.dataset.project_to_pytorch_default(counterfactual)

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
        base_path: str = "collages",
        start_idx: int = 0,
        y_target_goal_confidence_in: int = None,
        remove_below_threshold: bool = True,
    ) -> dict:
        """
        This function generates a counterfactual for a given batch of inputs.

        Args:
            batch (dict): The batch to explain.
            base_path (str, optional): The base path to save the counterfactuals to. Defaults to "collages".
            start_idx (int, optional): The start index for the counterfactuals. Defaults to 0.
            y_target_goal_confidence_in (int, optional): The target confidence for the counterfactuals. Defaults to None.
            remove_below_threshold (bool, optional): The flag to remove counterfactuals with a confidence below the target confidence. Defaults to True.

        Returns:
            dict: The batch with the counterfactuals added.
        """
        if y_target_goal_confidence_in is None:
            target_confidence_goal = self.explainer_config["y_target_goal_confidence"]

        else:
            target_confidence_goal = y_target_goal_confidence_in

        if isinstance(self.generator, InvertibleGenerator):
            (
                batch["x_counterfactual_list"],
                batch["z_difference_list"],
                batch["y_target_end_confidence_list"],
            ) = self.gradient_based_counterfactual(
                x_in=batch["x_list"],
                target_confidence_goal=target_confidence_goal,
                target_classes=batch["y_target_list"],
            )

        elif isinstance(self.generator, EditCapablGenerator):
            (
                batch["x_counterfactual_list"],
                batch["z_difference_list"],
                batch["y_target_end_confidence_list"],
            ) = self.generator.edit(
                x_in=batch["x_list"],
                target_confidence_goal=target_confidence_goal,
                target_classes=batch["y_target_list"],
                classifier=self.downstream_model,
            )

        batch_out = {}
        if remove_below_threshold:
            for key in batch.keys():
                batch_out[key] = []
                for sample_idx in range(len(batch[key])):
                    if (
                        batch["y_target_end_confidence_list"][sample_idx]
                        >= target_confidence_goal
                    ):
                        batch_out[key].append(batch[key][sample_idx])

        else:
            batch_out = batch

        (
            batch_out["x_attribution_list"],
            batch_out["collage_path_list"],
        ) = self.dataset.generate_contrastive_collage(
            target_confidence_goal=target_confidence_goal,
            base_path=base_path,
            start_idx=start_idx,
            **batch_out,
        )

        return batch_out
