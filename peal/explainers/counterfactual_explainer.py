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
        batch_in,
        target_classes,
        output_filenames: list[Path],
        target_confidence_goal_in=None,
        remove_below_threshold=True,
        source_classes=None,
    ):
        """
        _summary_

        Args:
            batch_in (_type_): _description_
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

        x = batch_in.clone()

        if isinstance(self.generator, InvertibleGenerator):
            (
                counterfactual,
                attribution,
                target_confidences,
            ) = self.gradient_based_counterfactual(x, target_confidence_goal)

        # TODO remove rows in counterfactual, attribution and x where target_confidences < target_confidence_goal

        (
            collage,
            heatmap,
            target_confidences,
        ) = self.dataset.generate_contrastive_collage(
            x,
            counterfactual,
            target_confidence_goal,
            target_classes,
            source_classes,
            output_filenames,
        )

        return {
            "x": x,
            "collage": collage,
            "counterfactual": counterfactual.detach().cpu(),
            "heatmap": heatmap,
            "target_confidences": target_confidences,
            "attribution": attribution,
        }

    def generate_counterfactuals_iteration(
        self,
        num_samples: int,
        error_distribution: torch.distributions.Distribution,
        confidence_score_stats: torch.tensor,
        finetune_iteration: int,
        sample_idx_iteration: int,
        dataloder: torch.utils.data.DataLoader,
        tracked_keys: list[str],
        target_confidence_goal: torch.tensor = None,
    ):
        """
        _summary_

        Args:
            num_samples (int): _description_
            error_distribution (torch.distributions.Distribution): _description_
            confidence_score_stats (torch.tensor): _description_
            finetune_iteration (int): _description_
            sample_idx_iteration (int): _description_
            target_confidence_goal (torch.tensor, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        tracked_values = {key: [] for key in tracked_keys}
        collage_paths = []
        counterfactuals = []
        heatmaps = []
        source_classes = []
        target_classes = []
        hints = []
        attributions = []
        ys = []
        num_batches = int(num_samples / self.adaptor_config["batch_size"]) + 1

        for batch_idx in range(num_batches):
            print(str(batch_idx) + "/" + str(num_batches))
            get_batch(batch_idx)
            #
            (
                collage,
                counterfactual,
                heatmaps_current_batch,
                end_target_confidences,
                current_attributions,
            ) = self.explainer.explain_batch(
                batch_in=current_img_batch,
                target_classes=target_classes_current_batch,
                source_classes=torch.tensor(source_classes_current_batch),
                target_confidence_goal_in=target_confidence_goal,
            )

            #
            for sample_idx in range(collage.shape[0]):
                if (
                    end_target_confidences[sample_idx]
                    >= self.adaptor_config["explainer"]["target_confidence_goal"]
                ):
                    # TODO bad smell! high-level and low-level functionality is mixed here...creation of collage should be factored out!
                    ys.append(ys_current_batch[sample_idx])
                    hints.append(current_hint_batch[sample_idx])
                    attributions.append(current_attributions[sample_idx])
                    sample_idx_iteration += 1

        return (
            collage_paths,
            counterfactuals,
            heatmaps,
            source_classes,
            target_classes,
            hints,
            attributions,
            ys,
        )
