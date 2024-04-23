import copy
import os

import torch
import torchvision

from torch import nn
from typing import Union

from peal.configs.explainers.perfect_false_counterfactual_config import (
    PerfectFalseCounterfactualConfig,
)
from peal.data.dataset_factory import get_datasets
from peal.global_utils import load_yaml_config
from peal.generators.interfaces import InvertibleGenerator, EditCapableGenerator
from peal.data.dataset_interfaces import PealDataset
from peal.explainers.explainer_interface import ExplainerInterface
from peal.configs.explainers.explainer_config import ExplainerConfig


class CounterfactualExplainer(ExplainerInterface):
    """
    This class implements the counterfactual explanation method
    """

    def __init__(
        self,
        downstream_model: nn.Module,
        generator: Union[InvertibleGenerator, EditCapableGenerator],
        input_type: str,
        dataset: PealDataset,
        tracking_level: int = 0,
        explainer_config: Union[
            dict, str, ExplainerConfig
        ] = "<PEAL_BASE>/configs/explainers/counterfactual_default.yaml",
        test_data_config: str = None,
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
        self.classifier_dataset = dataset
        self.explainer_config = load_yaml_config(explainer_config, ExplainerConfig)
        self.device = (
            "cuda" if next(self.downstream_model.parameters()).is_cuda else "cpu"
        )
        self.input_type = input_type
        self.tracking_level = tracking_level
        self.loss = torch.nn.CrossEntropyLoss()

        if isinstance(self.explainer_config, PerfectFalseCounterfactualConfig):
            inverse_config = copy.deepcopy(self.classifier_dataset.config)
            inverse_config.dataset_path += "_inverse"
            inverse_datasets = get_datasets(inverse_config)
            self.inverse_datasets = {}
            self.inverse_datasets["Training"] = inverse_datasets[0]
            self.inverse_datasets["Validation"] = inverse_datasets[1]
            if len(list(inverse_datasets)) == 3:
                self.inverse_datasets["test"] = inverse_datasets[2]

            if not test_data_config is None:
                inverse_test_config = copy.deepcopy(self.classifier_dataset.config)
                inverse_test_config.dataset_path += "_inverse"
                self.inverse_datasets["test"] = get_datasets(inverse_test_config)[-1]

    def gradient_based_counterfactual(
        self, x_in, target_confidence_goal, target_classes, pbar=None, mode=""
    ):
        """
        This function generates a counterfactual for a given batch of inputs.

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = torch.clone(x_in)
        x = self.classifier_dataset.project_to_pytorch_default(x)
        x = self.generator.dataset.project_from_pytorch_default(x)
        x = torchvision.transforms.Resize(self.generator.config.data.input_size[1:])(x)
        print("[x.min(), x.max()]")
        print([x.min(), x.max()])
        print([x.min(), x.max()])
        print([x.min(), x.max()])
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

        if self.explainer_config.optimizer == "Adam":
            optimizer = torch.optim.Adam(v, lr=self.explainer_config.learning_rate)

        elif self.explainer_config.optimizer == "SGD":
            optimizer = torch.optim.SGD(v, lr=self.explainer_config.learning_rate)

        target_confidences = [0.0 for i in range(len(target_classes))]

        for i in range(self.explainer_config.gradient_steps):
            if self.explainer_config.use_masking:
                mask = (
                    torch.tensor(target_confidences).to(self.device)
                    < target_confidence_goal
                )
                if torch.sum(mask) == 0.0:
                    break

            latent_code = [v_elem.to(self.device) for v_elem in v]

            optimizer.zero_grad()
            img = self.generator.decode(latent_code)

            img = self.generator.dataset.project_to_pytorch_default(img)
            img = torchvision.transforms.Resize(self.classifier_dataset.config.input_size[1:])(img)
            img = self.classifier_dataset.project_from_pytorch_default(img)
            print("[img.min(), img.max()]")
            print([img.min(), img.max()])
            print([img.min(), img.max()])
            print([img.min(), img.max()])

            logits = self.downstream_model(
                img + self.explainer_config.img_noise_injection * torch.randn_like(img)
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

            loss += self.explainer_config.l1_regularization * torch.mean(
                torch.stack(l1_losses)
            )
            """loss += self.explainer_config.log_prob_regularization * torch.mean(
                self.generator.log_prob_z(latent_code)
            )"""

            logit_confidences = torch.nn.Softmax(dim=-1)(logits).detach().cpu()
            target_confidences = [
                float(logit_confidences[i][target_classes[i]])
                for i in range(len(target_classes))
            ]
            if not pbar is None:
                pbar.set_description(
                    f"Creating {mode} Counterfactuals:"
                    + f"it: {i}"
                    + f"/{self.explainer_config.gradient_steps}"
                    + f", loss: {loss.detach().item():.2E}"
                    + f", target_confidence: {target_confidences[0]:.2E}"
                    + f", visual_difference: {torch.mean(torch.abs(x_in - img.detach().cpu())).item():.2E}"
                    + ", ".join(
                        [
                            key + ": " + str(pbar.stored_values[key])
                            for key in pbar.stored_values
                        ]
                    )
                )
                pbar.update(1)

            loss.backward()

            if self.explainer_config.use_masking:
                for sample_idx in range(len(target_confidences)):
                    if target_confidences[sample_idx] >= target_confidence_goal:
                        for variable_idx, v_elem in enumerate(v):
                            if self.explainer_config.optimizer == "Adam":
                                optimizer = torch.optim.Adam(
                                    v, lr=self.explainer_config.learning_rate
                                )

                            v_elem.grad[sample_idx].data.zero_()

            optimizer.step()

        latent_code = [v_elem.to(self.device) for v_elem in v]
        counterfactual = self.generator.decode(latent_code).detach().cpu()
        counterfactual = self.generator.dataset.project_to_pytorch_default(counterfactual)
        counterfactual = torchvision.transforms.Resize(
            self.classifier_dataset.config.input_size[1:]
        )(counterfactual)
        counterfactual = self.classifier_dataset.project_from_pytorch_default(counterfactual)
        print("[counterfactual.min(), counterfactual.max()]")
        print([counterfactual.min(), counterfactual.max()])
        print([counterfactual.min(), counterfactual.max()])
        print([counterfactual.min(), counterfactual.max()])

        attributions = []
        for v_idx in range(len(v_original)):
            attributions.append(
                torch.flatten(
                    v_original[v_idx].detach().cpu() - v[v_idx].detach().cpu(), 1
                )
            )

        attributions = torch.cat(attributions, 1)

        return counterfactual, attributions, target_confidences

    def perfect_false_counterfactuals(self, x_in, target_classes, idx_list, mode):
        """
        This function generates a counterfactual for a given batch of inputs.

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_counterfactual_list = []
        z_difference_list = []
        y_target_end_confidence_list = []
        for i, idx in enumerate(idx_list):
            x_counterfactual = self.inverse_datasets[mode][idx][0]
            x_counterfactual_list.append(x_counterfactual)
            preds = torch.nn.Softmax()(
                self.downstream_model(x_counterfactual.unsqueeze(0).to(self.device))
                .detach()
                .cpu()
            )
            y_target_end_confidence_list.append(preds[0][target_classes[i]])
            z_difference_list.append(x_in[i] - x_counterfactual)

        return x_counterfactual_list, z_difference_list, y_target_end_confidence_list

    def explain_batch(
        self,
        batch: dict,
        base_path: str = "collages",
        start_idx: int = 0,
        y_target_goal_confidence_in: int = None,
        remove_below_threshold: bool = True,
        pbar=None,
        mode="",
        explainer_path=None,
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
            if hasattr(self.explainer_config, "y_target_goal_confidence"):
                target_confidence_goal = self.explainer_config.y_target_goal_confidence

            else:
                target_confidence_goal = 0.51

        else:
            target_confidence_goal = y_target_goal_confidence_in

        if isinstance(self.explainer_config, PerfectFalseCounterfactualConfig):
            (
                batch["x_counterfactual_list"],
                batch["z_difference_list"],
                batch["y_target_end_confidence_list"],
            ) = self.perfect_false_counterfactuals(
                x_in=batch["x_list"],
                target_classes=batch["y_target_list"],
                idx_list=batch["idx_list"],
                mode=mode,
            )

        elif isinstance(self.generator, InvertibleGenerator):
            (
                batch["x_counterfactual_list"],
                batch["z_difference_list"],
                batch["y_target_end_confidence_list"],
            ) = self.gradient_based_counterfactual(
                x_in=batch["x_list"],
                target_confidence_goal=target_confidence_goal,
                target_classes=batch["y_target_list"],
                pbar=pbar,
                mode=mode,
            )

        elif isinstance(self.generator, EditCapableGenerator):
            if explainer_path is None:
                explainer_path = os.path.join(
                    *([os.path.abspath(os.sep)] + base_path.split(os.sep)[:-1])
                )

            (
                batch["x_counterfactual_list"],
                batch["z_difference_list"],
                batch["y_target_end_confidence_list"],
                batch["x_list"],
            ) = self.generator.edit(
                x_in=torch.tensor(batch["x_list"]),
                target_confidence_goal=target_confidence_goal,
                target_classes=torch.tensor(batch["y_target_list"]),
                source_classes=torch.tensor(batch["y_source_list"]),
                classifier=self.downstream_model,
                explainer_config=self.explainer_config,
                pbar=pbar,
                mode=mode,
                classifier_dataset=self.classifier_dataset,
                base_path=explainer_path,
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

        if self.tracking_level > 0:
            (
                batch_out["x_attribution_list"],
                batch_out["collage_path_list"],
            ) = self.classifier_dataset.generate_contrastive_collage(
                target_confidence_goal=target_confidence_goal,
                base_path=base_path,
                classifier=self.downstream_model,
                start_idx=start_idx,
                **batch_out,
            )
            print("Counterfactual created2!!!")
            print("Counterfactual created2!!!")
            print("Counterfactual created2!!!")

        return batch_out

    def run(self, dataset):
        """
        This function runs the explainer.
        """
        batches_out = []
        batch = None
        for idx in range(len(dataset)):
            x, y = dataset[idx]
            y_pred = self.downstream_model(x.unsqueeze(0).to(self.device))[0].argmax()
            y_target = (y_pred + 1) % dataset.output_size
            if (
                not batch is None
                or len(batch["x_list"]) < self.explainer_config.batch_size
            ):
                batch = {
                    "x_list": x.unsqueeze(0),
                    "y_target_list": torch.tensor([y_target]),
                    "y_source_list": torch.tensor([y_pred]),
                    "idx_list": [idx],
                }

            else:
                batch["x_list"] = torch.cat([batch["x_list"], x.unsqueeze(0)], 0)
                batch["y_target_list"] = torch.cat(
                    [batch["y_target_list"], torch.tensor([y_target])], 0
                )
                batch["y_source_list"] = torch.cat(
                    [batch["y_source_list"], torch.tensor([y_pred])], 0
                )
                batch["idx_list"].append(idx)
                batches_out.append(self.explain_batch(batch))

        batches_out_dict = {}
        for key in batches_out[0].keys():
            batches_out_dict[key] = torch.cat([batch[key] for batch in batches_out], 0)

        return batches_out_dict
