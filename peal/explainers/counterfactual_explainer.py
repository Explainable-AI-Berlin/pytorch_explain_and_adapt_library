import copy
import os

import torch
import torchvision
from pydantic import PositiveInt

from torch import nn
from typing import Union

from peal.data.dataset_factory import get_datasets
from peal.data.datasets import DataConfig
from peal.generators.generator_factory import get_generator
from peal.global_utils import load_yaml_config
from peal.generators.interfaces import InvertibleGenerator, EditCapableGenerator, GeneratorConfig
from peal.data.interfaces import PealDataset
from peal.explainers.interfaces import ExplainerInterface, ExplainerConfig



class ACEConfig(ExplainerConfig):
    """
    This class defines the config of a ACEConfig.
    """

    """
    The type of explanation that shall be used.
    Options: ['counterfactual', 'lrp']
    """
    explainer_type: str = "ACE"
    predictor_path: Union[str, type(None)] = None
    generator_config: Union[type(None), GeneratorConfig] = None
    data_config: Union[type(None), DataConfig] = None
    attack_iterations: Union[list, int] = [10,50]
    sampling_time_fraction: Union[list, float] = [0.1,0.3]
    dist_l1: Union[list, float] = [1.0, 0.0001]
    dist_l2: Union[list, float] = 0.0
    sampling_inpaint: Union[list, float] = [0.3, 0.1]
    sampling_dilation: Union[list, int] = 17
    timestep_respacing: Union[list, int] = 50
    attempts: int = 5
    clip_denoised: bool = True  # Clipping noise
    batch_size: int = 32  # Batch size
    gpu: str = "0"  # GPU index, should only be 1 gpu
    save_images: bool = False  # Saving all images
    num_samples: int = 500000000000  # useful to sample few examples
    # setting this to true will slow the computation time but will have identic results
    # hwhen using the checkpoint backwards
    cudnn_deterministic: bool = False
    # path args
    base_path: str = ""  # DDPM weights path
    # Experiment name (will store the results at Output/Results/exp_name)
    exp_name: str = "example_name"
    # attack args
    seed: int = 4  # Random seed
    # Attack method (currently 'PGD', 'C&W', 'GD' and 'None' supported)
    attack_method: str = "PGD"
    attack_epsilon: float = 255  # L inf epsilon bound (will be devided by 255)
    attack_step: float = 1.0  # Attack update step (will be devided by 255)
    attack_joint: bool = True  # Set to false to generate adversarial attacks
    # use checkpoint method for backward. Beware, this will substancially slow down the CE
    # generation!
    attack_joint_checkpoint: bool = False
    # number of DDPM iterations per backward process. We highly recommend have a larger
    # backward steps than batch size (e.g have 2 backward steps and batch size of 1 than 1
    # backward step and batch size 2)
    attack_checkpoint_backward_steps: int = 1
    # Use dime2 shortcut to transfer gradients. We do not recommend it.
    attack_joint_shortcut: bool = False
    # schedule for the distance loss. We did not used any for our results
    dist_schedule: str = "none"
    # filtering args
    sampling_stochastic: bool = True  # Set to False to remove the noise when sampling
    # dataset
    chunks: int = 1  # Chunking for spliting the CE generation into multiple gpus
    chunk: int = 0  # current chunk (between 0 and chunks - 1)
    merge_chunks: bool = False  # to merge all chunked results
    y_target_goal_confidence: float = 0.65


class DiffeoCFConfig(ExplainerConfig):
    """
    This class defines the config of a DiffeoCF.
    """

    """
    The type of explanation that shall be used.
    Options: ['counterfactual', 'lrp']
    """
    explanation_type: str = "DiffeoCFConfig"
    predictor_path: Union[str, type(None)] = None
    generator_config: Union[type(None), GeneratorConfig] = None
    data_config: Union[type(None), DataConfig] = None
    """
    The maximum number of gradients step done for explaining the network
    """
    gradient_steps: PositiveInt = 1000
    """
    The optimizer used for searching the counterfactual
    """
    optimizer: str = "Adam"
    """
    The learning rate used for finding the counterfactual
    """
    learning_rate: float = 0.01
    """
    The desired target confidence.
    Consider the tradeoff between minimality and clarity of counterfactual
    """
    y_target_goal_confidence: float = 0.9
    """
    Whether samples in the current search batch are masked after reaching y_target_goal_confidence
    or whether they are continued to be updated until the last surpasses the threshhold
    """
    use_masking: bool = True
    """
    How much noise to inject into the image while passing through it in the forward pass.
    Helps avoiding adversarial attacks in the case of a weak generator
    """
    img_noise_injection: float = 0.01
    """
    Regularizing factor of the L1 distance in latent space between the latent code of the
    original image and the counterfactual
    """
    l1_regularization: float = 1.0
    """
    Keeps the counterfactual in the high density area of the generative model
    """
    log_prob_regularization: float = 0.0
    """
    Regularization between counterfactual and original in image space to keep similariy high.
    """
    img_regularization: float = 0.0
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}


class TIMEConfig(ExplainerConfig):
    """
    This class defines the config of a ACEConfig.
    """

    """
    The type of explanation that shall be used.
    Options: ['counterfactual', 'lrp']
    """
    explainer_type: str = "TIME"
    predictor_path: Union[str, type(None)] = None
    generator_config: Union[type(None), GeneratorConfig] = None
    data_config: Union[type(None), DataConfig] = None
    editing_type: str = "ddpm_inversion"
    sd_model: str = "CompVis/stable-diffusion-v1-4"
    use_negative_guidance_denoise: bool = True
    use_negative_guidance_inverse: bool = True
    guidance_scale_denoising: list = [12]
    guidance_scale_invertion: list = [8]
    num_inference_steps: list = [50]
    exp_name: str = "time"
    label_target: int = -1
    label_query: int = 31
    class_custom_token: list = [
        "|<A*01>| |<A*02>| |<A*03>|",
        "|<A*11>| |<A*12>| |<A*13>|",
    ]
    base_prompt: str = "" # "A photo of a |<C*1>| |<C*2>| |<C*3>|"
    prompt_connector: str = "" # " that is "
    chunks: int = 1
    chunk: int = 0
    enable_xformers_memory_efficient_attention: bool = True
    use_fp16: bool = False
    sd_image_size: int = 128
    custom_obj_token: str = "|<C*>|"
    p: float = 0.93
    l2: float = 0.0
    inference_batch_size: int = 1
    classifier_image_size: int = 128
    recover: bool = False
    num_samples: int = 9999999999999999
    merge_chunks: bool = False
    generic_custom_tokens: list = ["|<C*1>|", "|<C*2>|", "|<C*3>|"]
    total_num_inference_steps: int = 50
    custom_tokens_context: list = ["|<C*1>|", "|<C*2>|", "|<C*3>|"]
    custom_tokens_init: list = ["<|endoftext|>", "<|endoftext|>", "<|endoftext|>"]
    mini_batch_size: int = 1
    gpu: str = "0"
    lr: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-9
    weight_decay: float = 1e-4
    iterations: int = 100 #1000
    max_epoch: int = 30
    train_batch_size: int = 64
    image_size: int = 128
    seed: int = 99999999
    y_target_goal_confidence: float = 0.9
    max_attacks: int = 1
    max_samples: Union[int, type(None)] = None
    use_lora: bool = True
    learn_dataset_embedding: bool = False


class PerfectFalseCounterfactualConfig(ExplainerConfig):
    """
    This class defines the config of a PerfectFalseCounterfactualConfig.
    """

    """
    The type of explanation that shall be used.
    """
    explainer_type: str = "PerfectFalseCounterfactual"
    data: Union[type(None), str, DataConfig] = None
    test_data: Union[type(None), str, DataConfig] = None


class CounterfactualExplainer(ExplainerInterface):
    """
    This class implements the counterfactual explanation method
    """

    def __init__(
        self,
        explainer_config: Union[
            dict, str, ExplainerConfig
        ],
        downstream_model: nn.Module = None,
        generator: Union[InvertibleGenerator, EditCapableGenerator] = None,
        input_type: str = None,
        dataset: PealDataset = None,
        tracking_level: int = 0,
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
        self.explainer_config = load_yaml_config(explainer_config, ExplainerConfig)
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not downstream_model is None:
            self.downstream_model = downstream_model

        else:
            self.downstream_model = torch.load(self.explainer_config.predictor_config.base_path).to(self.device)

        if not generator is None:
            self.generator = generator

        else:
            self.generator = get_generator(self.explainer_config.generator_config).to(self.device)

        if not dataset is None:
            self.classifier_dataset = dataset

        else:
            self.classifier_dataset = get_datasets(self.explainer_config.data_config)[1]

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
            img = torchvision.transforms.Resize(
                self.classifier_dataset.config.input_size[1:]
            )(img)
            img = self.classifier_dataset.project_from_pytorch_default(img)

            logits_perturbed = self.downstream_model(
                img + self.explainer_config.img_noise_injection * torch.randn_like(img)
            )
            loss = self.loss(logits_perturbed, target_classes.to(self.device))
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
            logit_confidences = torch.nn.Softmax(dim=-1)(logits_perturbed).detach().cpu()
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
        counterfactual = self.generator.dataset.project_to_pytorch_default(
            counterfactual
        )
        counterfactual = torchvision.transforms.Resize(
            self.classifier_dataset.config.input_size[1:]
        )(counterfactual)
        counterfactual = self.classifier_dataset.project_from_pytorch_default(
            counterfactual
        )
        logits = self.downstream_model(img)
        logit_confidences = torch.nn.Softmax(dim=-1)(logits).detach().cpu()
        target_confidences = [
            float(logit_confidences[i][target_classes[i]])
            for i in range(len(target_classes))
        ]

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

        return batch_out

    def run(self, dataset, oracle_path=None, confounder_oracle_path=None):
        """
        This function runs the explainer.
        """
        batches_out = []
        batch = None
        for idx in range(len(dataset)):
            x, y = dataset[idx]
            y_pred = self.downstream_model(x.unsqueeze(0).to(self.device))[0].argmax()
            for y_target in range(dataset.output_size):
                if y_target == y_pred:
                    continue

                if (
                    batch is None
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

                if len(batch["x_list"]) == self.explainer_config.batch_size:
                    batches_out.append(self.explain_batch(batch))
                    batch = None

        batches_out_dict = {}
        for key in batches_out[0].keys():
            batches_out_dict[key] = torch.cat([batch[key] for batch in batches_out], 0)

        dataset.generate_contrastive_collage(
            x_counterfactual_list=batches_out_dict['x_counterfactual_list'],
            y_source_list=batches_out_dict['y_source_list'],
            y_target_list=batches_out_dict['y_target_list'],
            x_list=batches_out_dict['x_list'],
            y_list=batches_out_dict['y_list'],
            y_target_end_confidence_list=batches_out_dict['y_target_end_confidence_list'],
            base_path=self.explainer_config.explanations_dir,
        )

        return batches_out_dict

    def human_annotate_counterfactuals(self):
        pass