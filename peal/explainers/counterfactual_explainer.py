import copy
import math
import os
import shutil
import threading
import time
import types
from pathlib import Path

import torch
import torchvision
from flask import render_template, Flask, request
from numpy.f2py.auxfuncs import throw_error
from pydantic import PositiveInt

from torch import nn
from typing import Union

from tqdm import tqdm

from peal.architectures.predictors import get_predictor
from peal.architectures.interfaces import TaskConfig
from peal.data.dataset_factory import get_datasets
from peal.generators.generator_factory import get_generator
from peal.global_utils import (
    load_yaml_config,
    is_port_in_use,
    dict_to_bar_chart,
    embed_numberstring,
    extract_penultima_activation,
    cprint,
)
from peal.generators.interfaces import (
    InvertibleGenerator,
    EditCapableGenerator,
    GeneratorConfig,
)
from peal.data.interfaces import PealDataset, DataConfig
from peal.explainers.interfaces import ExplainerInterface, ExplainerConfig
from peal.teachers.human2model_teacher import DataStore
from peal.training.interfaces import PredictorConfig
from peal.training.trainers import distill_predictor
from peal.visualization.visualize_counterfactual_gradients import visualize_step


class SCEConfig(ExplainerConfig):
    """
    This class defines the config of a DiffeoCF.
    """

    """
    The type of explanation that shall be used.
    """
    explainer_type: str = "SCEConfig"
    """
    The path to the predictor that shall be explained.
    """
    predictor_path: Union[str, type(None)] = None
    """
    The generator that shall be used for the counterfactual search
    """
    generator: Union[type(None), GeneratorConfig] = None
    """
    The data config used for the counterfactual search
    """
    data_config: Union[type(None), DataConfig] = None
    """
    The maximum number of gradients step done for explaining the network
    """
    gradient_steps: PositiveInt = 50
    """
    The optimizer used for searching the counterfactual
    """
    optimizer: str = "Adam"
    """
    The learning rate used for finding the counterfactual
    """
    learning_rate: float = 1.0
    """
    The desired target confidence.
    Consider the tradeoff between minimality and clarity of counterfactual
    """
    y_target_goal_confidence: Union[type(None), float] = None
    """
    Whether samples in the current search batch are masked after reaching y_target_goal_confidence.
    Otherwise they are continued to be updated until the last surpasses the threshhold
    """
    use_masking: bool = True
    """
    Regularizing factor that keeps changes between original image and counterfactual sparse.
    """
    dist_l1: float = 0.0
    """
    The batch size used for the counterfactual search
    """
    batch_size: int = 1
    """
    The config for the predictor distillation.
    """
    distilled_predictor: Union[type(None), str, dict] = None
    """
    The path to either the predictor or its config.
    """
    predictor: Union[str, type(None), dict] = None
    """
    How deep to go into the latent space for the counterfactual search
    """
    sampling_time_fraction: float = 0.3
    """
    The number of discretizations when going into the latent space
    """
    num_discretization_steps: int = 15
    """
    Whether to encode every iteration again or not.
    """
    iterationwise_encoding: bool = True
    """
    Whether to use stochastic counterfactual search (e.g. DDPM) or not (e.g. deterministic DDIM).
    """
    stochastic: Union[type(None), str] = "fully"
    """
    The level of dilation for the masking that is used for RePaint. If it is higher the mask is more coarse.
    If it is lower the mask is more finegrained.
    """
    dilation: int = 5
    """
    The threshold what is RePainted in the preexplanation. The repainting back to the original in the preexplanation
    is done for everything that has changes below 100 * inpaint percent of the maximimum change between sample and
    preexplanation.
    """
    inpaint: float = 0.0
    """
    The activation function ReLU is replaced with: leakyrelu, leakysoftplus
    This helps the distilled predictor to be more sensitive and smooth without saturating gradients.
    """
    replace_with_activation: str = "leakysoftplus"
    """
    Whether to only keep the best explanation while counterfactual search or not.
    While the greedy solution tends to be more stable it has bigger problems with strong local optima.
    """
    greedy: bool = False
    """
    Whether to visualize gradients for every step of the counterfactual search or not.
    Helpful for debugging, but decreases speed and clutters disk.
    """
    visualize_gradients: bool = False
    """
    Momentum term that prevents counterfactual search from changing the area that is changed for creating the
    counterfactual to rapidly.
    """
    mask_momentum: float = 0.0
    """
    Momentum term for the optimizer that does the updates of the preexplanations.
    """
    momentum: float = 0.9
    """
    The maximum absolut value that gradient update step can change one input variable at once.
    """
    gradient_clipping: float = 0.05
    """
    The strategy on how to merge clusters when calculating them.
    E.g. select_best uses the cluster that does the most salient changes, while merge just merges all clusters.
    """
    merge_clusters: str = "select_best"
    """
    Whether to use the diversification tool that forbids changing the same area of the input image again or not.
    """
    allow_overlap: bool = False
    """
    Whether to use a generative model to filter the gradients or not.
    """
    use_gradient_filtering: bool = True


class ACEConfig(ExplainerConfig):
    """
    This class defines the config of a ACE, DiME or FastDiME explainer.
    This config is primarily for replicating results of related work.
    """

    """
    The type of explanation that shall be used.
    Options: ['counterfactual', 'lrp']
    """
    explainer_type: str = "ACE"
    subtype: str = "ACE"
    loss_fn: Union[type(None), str] = None
    predictor: Union[str, type(None), dict] = None
    generator: Union[type(None), GeneratorConfig] = None
    data_config: Union[type(None), DataConfig] = None
    attack_iterations: Union[list, int] = 100
    sampling_time_fraction: Union[list, float] = 0.3
    dist_l1: Union[list, float] = 0.0001
    dist_l2: Union[list, float] = 0.0
    sampling_inpaint: Union[list, float] = 0.2
    sampling_dilation: Union[list, int] = 17
    timestep_respacing: Union[list, int] = 50
    distilled_predictor: Union[type(None), str, dict] = None
    attempts: int = 1
    clip_denoised: bool = True  # Clipping noise
    batch_size: int = 1  # Batch size
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
    y_target_goal_confidence: Union[type(None), float] = None
    """
    The activation function ReLU is replaced with: leaky_relu, leaky_softplus
    """
    replace_with_activation: str = ""
    # DiME params
    guided_iterations: object = 9999999
    image_size: object = 128
    l1_loss: object = 0.05
    l2_loss: object = 0.0
    l_perc: object = 30.0
    l_perc_layer: object = 18
    learn_sigma: object = True
    merge_and_eval: object = False
    model_path: object = "models/ddpm-celeba.pt"
    noise_schedule: object = "linear"
    num_batches: object = 1
    num_channels: object = 128
    num_chunks: object = 1
    num_head_channels: object = -1
    num_heads: object = 4
    num_heads_upsample: object = -1
    num_res_blocks: object = 2
    oracle_path: object = "models/oracle.pth"
    output_path: object = "/path/to/results"
    predict_xstart: object = False
    query_label: object = 31
    resblock_updown: object = True
    rescale_learned_sigmas: object = False
    rescale_timesteps: object = False
    sampling_scale: object = 1.0
    save_x_t: object = True
    save_z_t: object = True
    start_step: object = 60
    target_label: object = -1
    use_checkpoint: object = False
    use_ddim: object = False
    use_fp16: object = True
    use_kl: object = False
    use_logits: object = True
    use_new_attention_order: object = False
    use_sampling_on_x_t: object = True
    use_scale_shift_norm: object = True
    use_train: object = False
    # FastDiME params
    attention_resolutions: object = [32, 16, 8]
    channel_mult: object = ""
    class_cond: object = False
    classifier_path: object = "/scratch/ppar/models/classifier.pth"
    classifier_scales: object = [8, 10, 15]
    data_dir: object = "/scratch/ppar/data/img_align_celeba/"
    dataset: object = "CelebA"
    diffusion_steps: object = 500
    dilation: object = 5
    dropout: object = 0.0
    masking_threshold: object = 0.15
    method: object = "fastdime"
    n_samples: object = 1000
    percentage: object = 0.5
    scale_grads: object = False
    self_optimized_masking: object = True
    shortcut_label_name: object = "Smiling"
    task_label: object = 39
    task_label_name: object = "Young"
    warmup_step: object = 30


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
    generator: Union[type(None), GeneratorConfig] = None
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
    base_prompt: str = ""  # "A photo of a |<C*1>| |<C*2>| |<C*3>|"
    prompt_connector: str = ""  # " that is "
    chunks: int = 1
    chunk: int = 0
    enable_xformers_memory_efficient_attention: bool = True
    use_fp16: bool = False
    sd_image_size: int = 128
    custom_obj_token: str = "|<C*>|"
    p: float = 0.93
    l2: float = 0.0
    inference_batch_size: int = 1
    predictor_image_size: int = 128
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
    iterations: int = 100  # 1000
    max_epoch: int = 30
    train_batch_size: int = 64
    image_size: int = 128
    y_target_goal_confidence: float = 0.9
    max_attacks: int = 1
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
        explainer_config: Union[dict, str, ExplainerConfig],
        predictor: nn.Module = None,
        generator: Union[InvertibleGenerator, EditCapableGenerator] = None,
        input_type: str = None,
        datasource: list = None,
        tracking_level: int = None,
        test_data_config: str = None,
    ):
        """
        This class implements the counterfactual explanation method

        Args:
            explainer_config (Union[ dict, str ], optional): _description_. .
            predictor (nn.Module): _description_
            generator (InvertibleGenerator): _description_
            input_type (str): _description_
            datasets (list[PealDataset]): _description_
        """
        self.explainer_config = load_yaml_config(explainer_config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if predictor is None:
            predictor = explainer_config.predictor

        self.predictor, self.predictor_config = get_predictor(predictor, self.device)

        if not generator is None or isinstance(
            self.explainer_config, PerfectFalseCounterfactualConfig
        ):
            self.generator = generator

        else:
            self.generator = get_generator(self.explainer_config.generator).to(
                self.device
            )

        if self.explainer_config.validate_generator:
            if not os.path.exists(self.explainer_config.explanations_dir):
                os.makedirs(self.explainer_config.explanations_dir)

            x = self.generator.sample_x(self.explainer_config.batch_size)
            torchvision.utils.save_image(
                x,
                os.path.join(
                    self.explainer_config.explanations_dir,
                    "generator_validation.png",
                ),
            )

        if not datasource is None:
            self.predictor_datasources = datasource
            self.val_dataset = self.predictor_datasources[1].dataloaders[0].dataset

        else:
            """if not self.explainer_config.data_config is None:
                data_config = self.explainer_config.data_config

            elif not self.predictor_config is None:
                data_config = self.predictor_config.data

            else:
                print("No data config found!")
                raise ValueError

            if not self.predictor_config is None:
                task_config = TaskConfig(**self.predictor_config.task)

            else:
                task_config = None

            self.predictor_datasources = get_datasets(
                data_config, task_config=task_config
            )[:2]"""
            raise Exception("Currently not implemented correctly!")

        self.input_type = input_type
        if not tracking_level is None:
            self.tracking_level = tracking_level

        else:
            self.tracking_level = self.explainer_config.tracking_level

        self.loss = torch.nn.CrossEntropyLoss()

        if isinstance(self.explainer_config, PerfectFalseCounterfactualConfig):
            inverse_config = copy.deepcopy(self.val_dataset.config)
            inverse_config.dataset_path += "_inverse"
            inverse_datasets = get_datasets(inverse_config)
            self.inverse_datasets = {}
            self.inverse_datasets["Training"] = inverse_datasets[0]
            self.inverse_datasets["validation"] = inverse_datasets[1]
            if len(list(inverse_datasets)) == 3:
                self.inverse_datasets["test"] = inverse_datasets[2]

            if not test_data_config is None:
                inverse_test_config = copy.deepcopy(self.val_dataset.config)
                inverse_test_config.dataset_path += "_inverse"
                self.inverse_datasets["test"] = get_datasets(inverse_test_config)[-1]

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
                self.predictor(x_counterfactual.unsqueeze(0).to(self.device))
                .detach()
                .cpu()
            )
            y_target_end_confidence_list.append(preds[0][target_classes[i]])
            z_difference_list.append(x_in[i] - x_counterfactual)

        return x_counterfactual_list, z_difference_list, y_target_end_confidence_list

    def predictor_distilled_counterfactual(
        self,
        x_in,
        y_target,
        target_confidence_goal=None,
        pbar=None,
        mode="",
        base_path=None,
        batch_idx=0,
        num_attempts=1,
    ):
        """
        This function generates a counterfactual for a given batch of inputs.

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        if num_attempts > 1:
            (
                previous_counterfactual_list,
                previous_attributions_list,
                previous_target_confidences_list,
                boolmask_in,
            ) = self.predictor_distilled_counterfactual(
                x_in,
                y_target,
                target_confidence_goal,
                pbar,
                mode,
                base_path,
                batch_idx,
                num_attempts - 1,
            )

        else:
            previous_target_confidences_list = None

        def to_generator(sample):
            sample = self.generator.dataset.project_from_pytorch_default(sample)
            if not sample.shape[-2:] == self.generator.dataset.config.input_size[1:]:
                sample = torchvision.transforms.Resize(
                    self.generator.dataset.config.input_size[1:]
                )(sample)
            return sample

        def to_predictor(sample):
            sample = self.val_dataset.project_from_pytorch_default(sample)
            if not sample.shape[-2:] == self.val_dataset.config.input_size[1:]:
                sample = torchvision.transforms.Resize(
                    self.val_dataset.config.input_size[1:]
                )(sample)
            return sample

        if num_attempts == 1 or self.explainer_config.allow_overlap:
            boolmask_in = torch.ones_like(x_in)

        if not self.explainer_config.distilled_predictor is None:
            if base_path is None:
                base_path = self.explainer_config.explanations_dir

            distilled_path = os.path.join(
                base_path,
                "explainer",
                "distilled_predictor",
                "model.cpl",
            )
            if not os.path.exists(distilled_path):
                if isinstance(self.explainer_config.distilled_predictor, dict):
                    self.explainer_config.distilled_predictor[
                        "data"
                    ] = self.val_dataset.config

                elif isinstance(
                    self.explainer_config.distilled_predictor, PredictorConfig
                ):
                    self.explainer_config.distilled_predictor.data = (
                        self.val_dataset.config
                    )

                gradient_predictor = distill_predictor(
                    self.explainer_config.distilled_predictor,
                    os.path.join(base_path, "explainer"),
                    self.predictor,
                    self.predictor_datasources,
                    replace_with_activation=self.explainer_config.replace_with_activation,
                    tracking_level=self.explainer_config.tracking_level,
                )

            else:
                gradient_predictor = torch.load(
                    distilled_path, map_location=self.device
                )

            decision_boundary_path_distilled = os.path.join(
                base_path,
                "explainer",
                "distilled_predictor",
                "decision_boundary.png",
            )
            if hasattr(
                self.val_dataset, "visualize_decision_boundary"
            ) and not os.path.exists(decision_boundary_path_distilled):
                self.val_dataset.visualize_decision_boundary(
                    gradient_predictor,
                    32,
                    self.device,
                    decision_boundary_path_distilled,
                    temperature=self.explainer_config.temperature,
                )

        else:
            gradient_predictor = self.predictor

        x_predictor = torch.clone(x_in)
        print("x_predictor: " + str([x_predictor.min(), x_predictor.max()]))
        # should be always in [0,1] and in the resolution of the predictor
        x_original = self.val_dataset.project_to_pytorch_default(x_predictor)
        print("x_original: " + str([x_original.min(), x_original.max()]))

        if not self.explainer_config.iterationwise_encoding:
            """if self.explainer_config.use_gradient_filtering:
                z_original = self.generator.encode(x.to(self.device), stochastic=False)

            else:
                z_original = [x.to(self.device)]

            if isinstance(z_original, list):
                z = []

                for z_org in z_original:
                    z.append(
                        nn.Parameter(
                            torch.clone(z_org.detach().cpu()), requires_grad=True
                        )
                    )

            else:
                z = nn.Parameter(
                    torch.clone(z_original.detach().cpu()), requires_grad=True
                )
                z = [z]
                z_original = [z_original]"""
            pass

        else:
            z_original = x_original.to(self.device)
            print("z_original: " + str([z_original.min(), z_original.max()]))
            z = nn.Parameter(torch.clone(z_original.detach().cpu()), requires_grad=True)
            z_original = [z_original]
            z = [z]

        if self.explainer_config.optimizer == "Adam":
            # optimizer = torch.optim.Adam(z, lr=self.explainer_config.learning_rate)
            optimizer = torch.optim.RMSprop(z, lr=self.explainer_config.learning_rate)

        elif self.explainer_config.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                z,
                lr=self.explainer_config.learning_rate,
                momentum=self.explainer_config.momentum,
            )

        else:
            raise Exception(
                self.explainer_config.optimizer + " is not a valid optimizer!"
            )

        not_finished_mask = torch.ones(x_original.shape[0]).to(x_original)
        pred_original = (
            torch.nn.functional.softmax(
                self.predictor(x_predictor.to(self.device))
                / self.explainer_config.temperature
            )
            .detach()
            .cpu()
        )
        target_confidences = [
            pred_original[i][y_target[i]] for i in range(len(y_target))
        ]
        gradient_confidences_old = torch.zeros(x_original.shape[0]).to(x_original)
        if target_confidence_goal is None:
            target_confidence_goal_current = 1 - torch.tensor(target_confidences)

        else:
            target_confidence_goal_current = (
                torch.ones([x_predictor.shape[0]]) * target_confidence_goal
            )

        best_z = torch.clone(z[0])
        best_score = 0.5 * torch.ones_like(target_confidence_goal_current)
        best_mask = torch.ones_like(best_z)

        for i in range(self.explainer_config.gradient_steps):
            (
                best_score,
                best_z,
                best_mask,
                not_finished_mask,
                z,
                target_confidence_goal_current,
                gradient_confidences_old,
                is_break,
            ) = self.predictor_distilled_counterfactual_step(
                x_original=x_original,
                optimizer=optimizer,
                y_target=y_target,
                i=i,
                boolmask_in=boolmask_in,
                gradient_predictor=gradient_predictor,
                mode=mode,
                pbar=pbar,
                batch_idx=batch_idx,
                x_in=x_in,
                num_attempts=num_attempts,
                base_path=base_path,
                z_original=z_original,
                best_score=best_score,
                best_z=best_z,
                z=z,
                best_mask=best_mask,
                target_confidence_goal_current=target_confidence_goal_current,
                not_finished_mask=not_finished_mask,
                gradient_confidences_old=gradient_confidences_old,
                to_generator=to_generator,
                to_predictor=to_predictor,
            )

            if is_break:
                break

        if not self.explainer_config.iterationwise_encoding:
            z_encoded = [z_elem.to(self.device) for z_elem in z]
            if self.explainer_config.use_gradient_filtering:
                counterfactual = self.generator.decode(z_encoded).detach().cpu()

            else:
                counterfactual = z_encoded[0].detach().cpu()

        else:
            counterfactual = best_z.clone().detach()

        # bring the counterfactual back to the predictor normalization
        counterfactual = to_predictor(counterfactual)
        print("counterfactual: " + str([counterfactual.min(), counterfactual.max()]))
        logits = self.predictor(counterfactual.to(self.device))
        logit_confidences = (
            torch.nn.Softmax(dim=-1)(logits / self.explainer_config.temperature)
            .detach()
            .cpu()
        )
        target_confidences = [
            float(logit_confidences[i][y_target[i]]) for i in range(len(y_target))
        ]

        attributions = []
        for v_idx in range(len(z_original)):
            attributions.append(
                torch.flatten(
                    z_original[v_idx].detach().cpu() - z[v_idx].detach().cpu(), 1
                )
            )

        attributions = torch.cat(attributions, 1)

        current_counterfactuals, current_attributions, current_target_confidences = (
            list(counterfactual),
            list(attributions),
            list(target_confidences),
        )

        if not best_mask is None:
            if best_mask.shape[1] == 1:
                best_mask = torch.cat(3 * [best_mask.detach().cpu()], dim=1)

            boolmask_out = 1 - ((1 - boolmask_in) + (1 - best_mask))

        else:
            boolmask_out = None

        if not previous_target_confidences_list is None:
            current_counterfactuals = (
                current_counterfactuals + previous_counterfactual_list
            )
            current_attributions = current_attributions + previous_attributions_list
            current_target_confidences = (
                current_target_confidences + previous_target_confidences_list
            )

        return (
            current_counterfactuals,
            current_attributions,
            current_target_confidences,
            boolmask_out,
        )

    def predictor_distilled_counterfactual_step(
        self,
        x_original,
        optimizer,
        y_target,
        i,
        boolmask_in,
        gradient_predictor,
        mode,
        pbar,
        batch_idx,
        x_in,
        num_attempts,
        base_path,
        z_original,
        best_score,
        best_z,
        z,
        best_mask,
        target_confidence_goal_current,
        not_finished_mask,
        gradient_confidences_old,
        to_generator,
        to_predictor,
    ):
        if self.explainer_config.iterationwise_encoding:
            # always in [0,1] with resolution of discriminator
            print("z[0].data: " + str([z[0].data.min(), z[0].data.max()]))
            z[0].data = torch.clamp(z[0].data, 0, 1)
            z_default = z[0]

            z_predictor_original = to_predictor(z_default).to(self.device)
            print("z_predictor_original: " + str([z_predictor_original.min(), z_predictor_original.max()]))
            pred_original = torch.nn.functional.softmax(
                self.predictor(z_predictor_original.detach()) / self.explainer_config.temperature
            )
            target_confidences = torch.zeros_like(target_confidence_goal_current)
            for j in range(len(target_confidences)):
                target_confidences[j] = pred_original[j, int(y_target[j])]

            clean_img_old = torch.clone(z_default).detach().cpu()

            if self.explainer_config.use_gradient_filtering:
                z_generator = to_generator(z[0])
                print("z_generator: " + str([z_generator.min(), z_generator.max()]))
                z_encoded = self.generator.encode(
                    z_generator.to(self.device),
                    t=self.explainer_config.sampling_time_fraction,
                    stochastic=self.explainer_config.stochastic,
                )
                print("z_encoded: " + str([z_encoded.min(), z_encoded.max()]))

            else:
                z_encoded = z[0].to(self.device)

        else:
            z_encoded = [z_elem.to(self.device) for z_elem in z]

        optimizer.zero_grad()
        if self.explainer_config.use_gradient_filtering:
            img_decoded = self.generator.decode(
                z_encoded, t=self.explainer_config.sampling_time_fraction
            )
            print("img_decoded: " + str([img_decoded.min(), img_decoded.max()]))
            img_default = self.generator.dataset.project_to_pytorch_default(img_decoded)
            if not img_default.shape[-2:] == self.val_dataset.config.input_size[1:]:
                img_default = torchvision.transforms.Resize(
                    self.val_dataset.config.input_size[1:]
                )(img_default)

            print("img_default: " + str([img_default.min(), img_default.max()]))

        else:
            img_default = z_encoded

        img_default = torch.clamp(img_default, 0, 1)

        img_predictor = self.val_dataset.project_from_pytorch_default(img_default)

        if not self.explainer_config.iterationwise_encoding:
            pred_original = torch.nn.functional.softmax(
                self.predictor(img_predictor.detach())
                / self.explainer_config.temperature,
                -1,
            )
            target_confidences = [
                float(pred_original[i][y_target[i]]) for i in range(len(y_target))
            ]

        logits_gradient = (
            gradient_predictor(img_predictor) / self.explainer_config.temperature
        )
        loss = self.loss(logits_gradient, y_target.to(self.device))
        l1_losses = []
        for z_idx in range(len(z_original)):
            l1_losses.append(
                torch.mean(
                    torch.abs(
                        z[z_idx].to(self.device)
                        - torch.clone(z_original[z_idx]).detach()
                    )
                )
            )

        if num_attempts == 1 or self.explainer_config.allow_overlap:
            dist_l1 = self.explainer_config.dist_l1
            current_inpaint = self.explainer_config.inpaint

        else:
            dist_l1 = self.explainer_config.dist_l1 * (0.5 ** (num_attempts - 1))
            current_inpaint = self.explainer_config.inpaint * (
                0.5 ** (num_attempts - 1)
            )

        loss += dist_l1 * torch.mean(torch.stack(l1_losses))
        if not pbar is None:
            absolute_difference = torch.abs(x_in - img_predictor.detach().cpu())
            if self.explainer_config.tracking_level >= 1:
                description_str = f"Creating {mode} Counterfactuals:" + f"it: {i}"
                description_str += f"/{self.explainer_config.gradient_steps}"
                description_str += f", loss: {loss.detach().item():.2E}"
                description_str += (
                    f", target_confidence: [{best_score[0]:.2E}, {best_score[-1]:.2E}]"
                )
                description_str += f", visual_difference: [{torch.mean(absolute_difference[0]).item():.2E}, "
                description_str += (
                    f", gradient_confidence: [{gradient_confidences_old[0]:.2E},"
                )
                description_str += f"{gradient_confidences_old[-1]:.2E}]"
                description_str += f"{torch.mean(absolute_difference[-1]).item():.2E}]"
                description_str += ", ".join(
                    [
                        key + ": " + str(pbar.stored_values[key])
                        for key in pbar.stored_values
                    ]
                )
                if self.explainer_config.tracking_level < 4:
                    description_str = description_str[:80]

                pbar.set_description(description_str)
                pbar.update(1)

        img_predictor.retain_grad()

        loss.backward()
        for sample_idx in range(z[0].size(0)):
            norm = (
                z[0].grad[sample_idx].norm(p=float("inf"))
                * self.explainer_config.learning_rate
            )
            if norm > self.explainer_config.gradient_clipping:
                rescale_factor = (
                    self.explainer_config.gradient_clipping
                    / norm
                    / self.explainer_config.learning_rate
                )
                z[0].grad[sample_idx] = z[0].grad[sample_idx] * rescale_factor

        if self.explainer_config.use_masking:
            for sample_idx in range(len(target_confidences)):
                if not_finished_mask[sample_idx] == 0:
                    for variable_idx, v_elem in enumerate(z):
                        if self.explainer_config.optimizer == "Adam":
                            optimizer = torch.optim.Adam(
                                z, lr=self.explainer_config.learning_rate
                            )

                        v_elem.grad[sample_idx].data.zero_()

        if self.explainer_config.iterationwise_encoding:
            if current_inpaint > 0.0:
                z[0].grad = boolmask_in * z[0].grad

        # abs_grads = torch.abs(z[0].grad)
        # z[0].grad[abs_grads < (abs_grads.max() / 10)] = 0
        # z[0].grad = torch.zeros_like(z[0].grad)
        # z[0].data = z[0].data - 100.0 * z[0].grad
        optimizer.step()
        boolmask = torch.zeros_like(z[0].data)
        if self.explainer_config.iterationwise_encoding:
            z[0].data = torch.clamp(z[0].data, 0, 1)
            if self.explainer_config.use_gradient_filtering:
                """pe = self.generator.dataset.project_to_pytorch_default(
                    torch.clone(z[0]).detach().cpu()
                )
                z_default = self.generator.dataset.project_to_pytorch_default(z[0])
                """
                pe = torch.clone(z[0]).detach().cpu()
                print("pe: " + str([pe.min(), pe.max()]))
                z_default = z[0]

            else:
                pe = torch.clone(z[0]).detach().cpu()
                z_default = z[0]

            z_predictor = self.val_dataset.project_from_pytorch_default(z_default).to(
                self.device
            ).detach()
            pred_current = torch.nn.functional.softmax(
                self.predictor(z_predictor) / self.explainer_config.temperature
            )
            target_confidences_current = torch.zeros_like(
                target_confidence_goal_current
            )
            for j in range(len(target_confidences_current)):
                target_confidences_current[j] = pred_current[j, int(y_target[j])]

            no_repaint_exceptions = target_confidences_current < 0.5
            if i == self.explainer_config.gradient_steps - 1:
                no_repaint_exceptions = torch.zeros_like(no_repaint_exceptions)

            if (
                self.explainer_config.inpaint > 0.0
                and not no_repaint_exceptions.sum() == no_repaint_exceptions.shape[0]
            ):
                if not boolmask_in.shape[-2:] == self.generator.dataset.config.input_size[1:]:
                    boolmask_in = torchvision.transforms.Resize(
                        self.generator.dataset.config.input_size[1:]
                    )(boolmask_in)

                z_updated, boolmask = self.generator.repaint(
                    x=to_generator(x_original).to(self.device),
                    pe=torch.clone(to_generator(z[0].data)).to(self.device),
                    inpaint=current_inpaint,
                    dilation=self.explainer_config.dilation,
                    t=self.explainer_config.sampling_time_fraction,
                    stochastic=True,
                    boolmask_in=boolmask_in,
                    exceptions=no_repaint_exceptions,
                )
                print("z_updated: " + str([z_updated.min(), z_updated.max()]))
                if not boolmask.shape[-2:] == self.val_dataset.config.input_size[1:]:
                    boolmask = torchvision.transforms.Resize(
                        self.val_dataset.config.input_size[1:]
                    )(boolmask)

                z_generator_current = self.generator.dataset.project_to_pytorch_default(z_updated)
                print("z_generator_current: " + str([z_generator_current.min(), z_generator_current.max()]))
                if not z_generator_current.shape[-2:] == self.val_dataset.config.input_size[1:]:
                    z_generator_current = torchvision.transforms.Resize(
                        self.val_dataset.config.input_size[1:]
                    )(z_generator_current)

                for sample_idx in range(z[0].data.shape[0]):
                    if (
                        not_finished_mask[sample_idx] == 1
                        and no_repaint_exceptions[sample_idx] == 0
                    ):
                        z[0].data[sample_idx] = z_generator_current[sample_idx]

            """z_default = self.generator.dataset.project_to_pytorch_default(z[0])
            z_predictor = self.val_dataset.project_from_pytorch_default(z_default).to(
                self.device
            )
            pred_new = torch.nn.functional.softmax(
                gradient_predictor(z_predictor) / self.explainer_config.temperature
            )
            if self.explainer_config.greedy:
                for j in range(gradient_confidences_old.shape[0]):
                    if mask[j] == 1:
                        if pred_new[j, int(y_target[j])] >= gradient_confidences_old[j]:
                            gradient_confidences_old[j] = pred_new[j, int(y_target[j])]
                            print("Update " + str(j))

                        else:
                            z[0].data[j] = z_old[j]"""

        print("z_bef: " + str([z[0].data.min(), z[0].data.max()]))
        z[0].data = torch.clamp(z[0].data, 0, 1)
        z_default = z[0]

        z_predictor_original = self.val_dataset.project_from_pytorch_default(
            z_default
        ).to(self.device).detach()
        pred_original = torch.nn.functional.softmax(
            self.predictor(z_predictor_original) / self.explainer_config.temperature
        )
        target_confidences = torch.zeros_like(target_confidence_goal_current)
        for j in range(len(target_confidences)):
            target_confidences[j] = pred_original[j, int(y_target[j])]

        for j in range(img_predictor.shape[0]):
            if no_repaint_exceptions[j] or not_finished_mask[j] == 0:
                continue

            if (
                target_confidences[j] >= best_score[j]
                or i == self.explainer_config.gradient_steps - 1
                and best_score[j] <= 0.5
            ):
                best_z[j] = torch.clone(z[0][j])
                best_score[j] = target_confidences[j]
                if not boolmask is None:
                    best_mask[j] = boolmask[j]

            if target_confidences[j] >= target_confidence_goal_current[j]:
                not_finished_mask[j] = 0

        if (
            self.explainer_config.tracking_level >= 5
            and self.explainer_config.visualize_gradients
        ):
            gradients_path = str(
                os.path.join(
                    base_path,
                    mode + "_explainer_gradients",
                    embed_numberstring(batch_idx, 4) + "_" + str(num_attempts),
                )
            )
            Path(gradients_path).mkdir(parents=True, exist_ok=True)
            if self.explainer_config.use_gradient_filtering:
                z_encoded_visualization = z_encoded.detach()
                if not z_encoded_visualization.shape[-2:] == self.val_dataset.config.input_size[1:]:
                    z_encoded_visualization = torchvision.transforms.Resize(
                        self.val_dataset.config.input_size[1:]
                    )(z_encoded)

                z_encoded_visualization = (
                    self.generator.dataset.project_to_pytorch_default(
                        z_encoded_visualization.detach().cpu()
                    )
                )

            else:
                z_encoded_visualization = z_encoded.detach().cpu()

            visualize_step(
                x_original=x_original,
                z=z,
                clean_img_old=clean_img_old,
                z_encoded=z_encoded_visualization,
                img_predictor=img_predictor,
                pe=pe,
                boolmask=boolmask,
                filename=os.path.join(
                    gradients_path, embed_numberstring(i, 4) + ".png"
                ),
                boolmask_in=boolmask_in,
                best_z=best_z,
                best_mask=best_mask,
            )

        return (
            best_score,
            best_z,
            best_mask,
            not_finished_mask,
            z,
            target_confidence_goal_current,
            gradient_confidences_old,
            not_finished_mask.sum() == 0,
        )

    def explain_batch(
        self,
        batch: dict,
        base_path: str = "collages",
        start_idx: int = 0,
        y_target_goal_confidence_in: float = None,
        remove_below_threshold: bool = False,
        pbar=None,
        mode="",
        explainer_path=None,
        batchwise_clustering=False,
    ) -> dict:
        """
        This function generates a counterfactual for a given batch of inputs.

        Args:
            batch (dict): The batch to explain.
            base_path (str, optional): The base path to save the counterfactuals to. Defaults to "collages".
            start_idx (int, optional): The start index for the counterfactuals. Defaults to 0.
            y_target_goal_confidence_in (int, optional): The target confidence for the counterfactuals.
                Defaults to None.
            remove_below_threshold (bool, optional): The flag to remove counterfactuals with a confidence below the
            target confidence. Defaults to True.
            explainer_path:
            mode:
            pbar:

        Returns:
            dict: The batch with the counterfactuals added.
        """
        if explainer_path is None:
            os_sep = os.path.abspath(os.sep)
            if base_path[: len(os_sep)] == os_sep:
                path_splitted = [os_sep]

            else:
                path_splitted = []

            path_splitted += base_path.split(os.sep)[:-1]
            explainer_path = os.path.join(*path_splitted)

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

        elif isinstance(self.generator, InvertibleGenerator) and isinstance(
            self.explainer_config, SCEConfig
        ):
            (
                batch["x_counterfactual_list"],
                batch["z_difference_list"],
                batch["y_target_end_confidence_list"],
                _,
            ) = self.predictor_distilled_counterfactual(
                x_in=batch["x_list"],
                y_target=batch["y_target_list"],
                target_confidence_goal=target_confidence_goal,
                pbar=pbar,
                mode=mode,
                base_path=explainer_path,
                batch_idx=start_idx,
                num_attempts=self.explainer_config.num_attempts,
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
                batch["history_list"],
            ) = self.generator.edit(
                x_in=torch.tensor(batch["x_list"]),
                target_confidence_goal=target_confidence_goal,
                target_classes=torch.tensor(batch["y_target_list"]),
                source_classes=torch.tensor(batch["y_source_list"]),
                predictor=self.predictor,
                explainer_config=self.explainer_config,
                pbar=pbar,
                mode=mode,
                predictor_datasets=self.predictor_datasources,
                base_path=explainer_path,
            )

        if len(batch["x_list"]) < len(batch["x_counterfactual_list"]):
            n_reps = len(batch["x_counterfactual_list"]) // len(batch["x_list"])
            for key in batch.keys():
                if len(batch[key]) < len(batch["x_counterfactual_list"]):
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = torch.cat(n_reps * [batch[key]], dim=0)

                    elif isinstance(batch[key], list):
                        batch[key] = n_reps * batch[key]

                    else:
                        raise Exception

        if self.explainer_config.num_attempts >= 2 and batchwise_clustering:
            clustering_strategy_buffer = self.explainer_config.clustering_strategy
            merge_clusters_buffer = self.explainer_config.merge_clusters
            self.explainer_config.clustering_strategy = "highest_activation"
            self.explainer_config.merge_clusters = "select_best"
            batch = self.cluster_explanations(
                explanations_dict=batch,
                batch_size=int(
                    len(batch["x_list"]) / self.explainer_config.num_attempts
                ),
                n_clusters=self.explainer_config.num_attempts,
            )
            self.explainer_config.clustering_strategy = clustering_strategy_buffer
            self.explainer_config.merge_clusters = merge_clusters_buffer

        batch_out = {}
        if remove_below_threshold:
            for key in batch.keys():
                batch_out[key] = []
                for sample_idx in range(len(batch[key])):
                    if batch["y_target_end_confidence_list"][sample_idx] >= 0.5:
                        batch_out[key].append(batch[key][sample_idx])

        else:
            batch_out = batch

        if self.tracking_level >= 4:
            (
                batch_out["x_attribution_list"],
                batch_out["collage_path_list"],
            ) = self.val_dataset.generate_contrastive_collage(
                target_confidence_goal=target_confidence_goal,
                base_path=base_path,
                predictor=self.predictor,
                start_idx=start_idx * self.explainer_config.num_attempts,
                **batch_out,
            )

        else:
            x_attribution_list = []
            for i in range(len(batch_out["x_counterfactual_list"])):
                x_attribution_list.append(
                    torch.abs(
                        batch_out["x_counterfactual_list"][i] - batch_out["x_list"][i]
                    )
                )

            batch_out["x_attribution_list"] = x_attribution_list

        return batch_out

    def cluster_explanations(self, explanations_dict, batch_size=2, n_clusters=2):
        """
        This function clusters the explanations.
        """
        explanations_list = []
        for idx in range(len(explanations_dict["x_list"])):
            current_dict = {}
            for key in explanations_dict.keys():
                current_dict[key] = explanations_dict[key][idx]

            explanations_list.append(current_dict)

        assert (
            len(explanations_list) % (batch_size * n_clusters) == 0
        ), "restructing needed for clustering impossible!"
        explanations_list_by_source = [[] for i in range(n_clusters)]
        batch_counter = 0
        cluster_counter = 0
        for i, elem in enumerate(explanations_list):
            if batch_counter == batch_size:
                batch_counter = 0
                cluster_counter += 1

            if cluster_counter == n_clusters:
                cluster_counter = 0

            explanations_list_by_source[cluster_counter].append(explanations_list[i])
            batch_counter += 1

        def extract_feature_difference(explanations):
            difference_list = []
            activation_ref = extract_penultima_activation(
                explanations[0]["x_list"][None, ...].to(self.device), self.predictor
            )
            for i in range(len(explanations)):
                if (
                    torch.sum(explanations[0]["x_list"] != explanations[i]["x_list"])
                    != 0
                ):
                    if self.explainer_config.tracking_level >= 4:
                        print("x list is not matching across samples!")
                        import pdb

                        pdb.set_trace()

                    else:
                        raise Exception("x list is not matching across samples!")

                activation_current = extract_penultima_activation(
                    explanations[i]["x_counterfactual_list"][None, ...].to(self.device),
                    self.predictor,
                )
                difference_list.append(activation_current - activation_ref)

            return difference_list

        cluster_lists = [[] for i in range(n_clusters)]
        collage_path_base = None
        if self.explainer_config.clustering_strategy == "activation_clusters":
            explanations_beginning = [e[0] for e in explanations_list_by_source]
            cluster_means = extract_feature_difference(explanations_beginning)
            cluster_lists[0] = [explanations_list_by_source[0][0]]
            cluster_lists[1] = [explanations_list_by_source[1][0]]
            if "collage_path_list" in explanations_beginning[0].keys():
                collage_path_ref = explanations_beginning[0]["collage_path_list"]
                collage_path_elements = collage_path_ref.split(os.sep)[:-1]
                collage_path_base = str(
                    os.path.join(*([os.path.abspath(os.sep)] + collage_path_elements))
                )
                for cluster_idx in range(len(cluster_means)):
                    collage_path = explanations_beginning[cluster_idx][
                        "collage_path_list"
                    ]
                    Path(collage_path_base + "_" + str(cluster_idx)).mkdir(
                        parents=True, exist_ok=True
                    )
                    collage_path_new = os.path.join(
                        *[
                            collage_path_base + "_" + str(cluster_idx),
                            embed_numberstring(0, 7) + ".png",
                        ]
                    )
                    shutil.copy(collage_path, collage_path_new)

        for idx in range(len(explanations_list_by_source[0])):
            if self.explainer_config.clustering_strategy == "highest_activation":
                current_activations = []
                for source_idx in range(len(explanations_list_by_source)):
                    current_activations.append(
                        explanations_list_by_source[source_idx][idx][
                            "y_target_end_confidence_list"
                        ]
                    )

                current_activations = torch.tensor(current_activations)
                activations_order = torch.argsort(current_activations)

            elif self.explainer_config.clustering_strategy == "activation_clusters":
                if idx == 0:
                    continue

                current_differences = extract_feature_difference(
                    [e[idx] for e in explanations_list_by_source]
                )
                # build outer product between cluster means and current differences
                cosine_similarities = torch.zeros(
                    [len(cluster_means), len(current_differences)]
                )
                for i in range(len(cluster_means)):
                    for j in range(len(current_differences)):
                        cosine_similarities[i, j] = torch.nn.CosineSimilarity()(
                            cluster_means[i], current_differences[j]
                        )

                # find the cluster with the highest similarity
                cosine_similarities_abs = torch.abs(cosine_similarities)

            for i in range(n_clusters):
                if self.explainer_config.clustering_strategy == "activation_clusters":
                    idx_combined = int(torch.argmax(cosine_similarities_abs.flatten()))
                    idx_cluster = idx_combined // len(current_differences)
                    idx_current = idx_combined % len(current_differences)
                    cosine_similarities_abs[idx_cluster, :] = -1
                    cosine_similarities_abs[:, idx_current] = -1
                    # update running mean
                    cluster_means[idx_cluster] = (
                        idx * cluster_means[idx_cluster]
                        + torch.sign(cosine_similarities[idx_cluster, idx_current])
                        * current_differences[idx_current]
                    ) / (idx + 1)

                elif self.explainer_config.clustering_strategy == "highest_activation":
                    idx_cluster = activations_order[i]
                    idx_current = i

                if len(cluster_lists[idx_cluster]) > len(
                    cluster_lists[abs(1 - idx_cluster)]
                ):
                    raise Exception("cluster list lengths are not matching!")

                cluster_lists[idx_cluster].append(
                    explanations_list_by_source[idx_current][idx]
                )
                if hasattr(
                    explanations_list_by_source[idx_current][idx], "cluster_list"
                ):
                    explanations_list_by_source[idx_current][idx][
                        "cluster_list"
                    ].append(idx_cluster)

                else:
                    explanations_list_by_source[idx_current][idx]["cluster_list"] = [
                        idx_cluster
                    ]

                if (
                    self.explainer_config.clustering_strategy == "highest_activation"
                    and "collage_path_list"
                    in explanations_list_by_source[idx_current][idx].keys()
                ):
                    collage_path_ref = explanations_list_by_source[idx_current][idx][
                        "collage_path_list"
                    ]
                    collage_path_elements = collage_path_ref.split(os.sep)[:-1]
                    collage_path_base = str(
                        os.path.join(
                            *([os.path.abspath(os.sep)] + collage_path_elements)
                        )
                    )
                    Path(collage_path_base + "_" + str(int(idx_cluster))).mkdir(
                        parents=True, exist_ok=True
                    )

                if not collage_path_base is None:
                    collage_path = explanations_list_by_source[idx_current][idx][
                        "collage_path_list"
                    ]
                    collage_path_new = os.path.join(
                        *[
                            collage_path_base + "_" + str(int(idx_cluster)),
                            embed_numberstring(idx, 7) + ".png",
                        ]
                    )
                    shutil.copy(collage_path, collage_path_new)

        cluster_dicts = []
        for cluster_idx in range(len(cluster_lists)):
            cluster_dict = {}
            for sample_idx in range(len(cluster_lists[cluster_idx])):
                for key in cluster_lists[cluster_idx][sample_idx].keys():
                    if not key in cluster_dict.keys():
                        cluster_dict[key] = []

                    cluster_dict[key].append(
                        cluster_lists[cluster_idx][sample_idx][key]
                    )

            cluster_dicts.append(cluster_dict)

        cluster_scores = []
        for cluster_idx in range(len(cluster_dicts)):
            sample_scores = []
            for sample_idx in range(len(cluster_dicts[cluster_idx]["x_list"])):
                sample_scores.append(
                    cluster_dicts[cluster_idx]["y_target_end_confidence_list"][
                        sample_idx
                    ]
                )

            cluster_scores.append(torch.mean(torch.tensor(sample_scores)))

        sorted_cluster_idxs = torch.tensor(cluster_scores).argsort()
        sorted_cluster_idxs = [
            int(sorted_cluster_idxs[-1 - i])
            for i in range(self.explainer_config.num_attempts)
        ]

        explanations_dict_out = cluster_dicts[sorted_cluster_idxs[0]]

        for i in range(len(sorted_cluster_idxs)):
            explanations_dict_out["clusters" + str(int(i))] = copy.deepcopy(
                cluster_dicts[sorted_cluster_idxs[i]]["x_counterfactual_list"]
            )
            explanations_dict_out["cluster_confidence" + str(int(i))] = copy.deepcopy(
                cluster_dicts[sorted_cluster_idxs[i]]["y_target_end_confidence_list"]
            )

        if self.explainer_config.merge_clusters == "concatenate":
            for cluster_idx in range(1, len(cluster_dicts)):
                for key in cluster_dicts[sorted_cluster_idxs[cluster_idx]].keys():
                    try:
                        explanations_dict_out[key] += cluster_dicts[
                            sorted_cluster_idxs[cluster_idx]
                        ][key]

                    except Exception:
                        import pdb

                        pdb.set_trace()

        return explanations_dict_out

    def calculate_latent_difference_stats(self, explanations_dict):
        tracked_stats = {}
        latent_differences = None
        if hasattr(self.val_dataset, "sample_to_latent"):
            latents_original = []
            for i, e in enumerate(
                explanations_dict["x_list"][: len(explanations_dict["clusters0"])]
            ):
                hint = (
                    explanations_dict["hint_list"][i]
                    if "hint_list" in explanations_dict.keys()
                    else None
                )
                latents_original.append(
                    self.val_dataset.sample_to_latent(e.to(self.device), hint).cpu()
                )

            latents_counterfactual = []
            latent_differences = []
            for c in range(self.explainer_config.num_attempts):
                latents_counterfactual.append(
                    [
                        self.val_dataset.sample_to_latent(
                            e.to(self.device),
                            (
                                explanations_dict["hint_list"][i]
                                if "hint_list" in explanations_dict.keys()
                                else None
                            ),
                        ).cpu()
                        for i, e in enumerate(
                            explanations_dict["clusters" + str(c)]
                        )
                    ]
                )

                latent_differences.append(
                    [
                        latents_counterfactual[c][i] - latents_original[i]
                        for i in range(len(latents_original))
                    ]
                )

        elif "hint_list" in explanations_dict.keys():
            latent_differences = []
            for c in range(self.explainer_config.num_attempts):
                x_difference_list = [
                    explanations_dict["clusters" + str(c)][i]
                    - explanations_dict["x_list"][i]
                    for i in range(len(explanations_dict["clusters0"]))
                ]
                foreground_change = [
                    torch.sum(
                        torch.abs(
                            x_difference_list[i]
                            * explanations_dict["hint_list"][i]
                        )
                        / torch.sum(explanations_dict["hint_list"][i])
                    )
                    for i in range(len(x_difference_list))
                ]

                background_change = [
                    torch.sum(
                        torch.abs(
                            x_difference_list[i]
                            * torch.abs(1 - explanations_dict["hint_list"][i])
                        )
                        / torch.sum(
                            torch.abs(1 - explanations_dict["hint_list"][i])
                        )
                    )
                    for i in range(len(x_difference_list))
                ]
                latent_differences.append(
                    torch.transpose(
                        torch.tensor([foreground_change, background_change]), 0, 1
                    )
                )

        if not latent_differences is None:
            for latent_difference in latent_differences:
                assert len(latent_difference) == len(explanations_dict["clusters0"])

            latent_differences_valid = []
            for i in range(len(latent_differences[0])):
                if (
                    explanations_dict["y_target_end_confidence_distilled_list"][i]
                    > 0.5
                ):
                    latent_difference_current = []
                    for j in range(len(latent_differences)):
                        latent_difference_current.append(latent_differences[j][i])

                    latent_differences_valid.append(latent_difference_current)

            if len(latent_differences_valid) == 0:
                latent_sparsity = 0.0
                latent_diversity = 0.0

            else:
                latent_sparsities = []
                for i in range(len(latent_differences_valid)):
                    if latent_differences_valid[i][0].abs().max() == 0.0:
                        latent_sparsity = 0.0

                    else:
                        latent_sparsity = float(
                            (
                                latent_differences_valid[i][0].abs().sum()
                                - latent_differences_valid[i][0].abs().max()
                            )
                            / (len(latent_differences_valid[i][0]) - 1)
                            / latent_differences_valid[i][0].abs().max()
                        )

                    latent_sparsities.append(latent_sparsity)

                latent_sparsity = 1.0 - float(
                    torch.mean(torch.tensor(latent_sparsities))
                )
                if self.explainer_config.num_attempts >= 2:
                    cosine_similiarities_list = [
                        torch.abs(
                            torch.nn.CosineSimilarity(dim=0)(
                                latent_differences_valid[i][0],
                                latent_differences_valid[i][1],
                            )
                        )
                        for i in range(len(latent_differences_valid))
                    ]
                    cosine_similiarities = torch.tensor(cosine_similiarities_list)
                    latent_diversity = 1.0 - float(torch.mean(cosine_similiarities))

                else:
                    latent_diversity = 0.0

            tracked_stats["latent_sparsity"] = latent_sparsity
            cprint(
                "latent_sparsity: " + str(latent_sparsity),
                self.explainer_config.tracking_level,
                2,
            )
            tracked_stats["latent_diversity"] = latent_diversity
            cprint(
                "latent_diversity: " + str(latent_diversity),
                self.explainer_config.tracking_level,
                2,
            )

        return tracked_stats

    def run(self, oracle_path=None, confounder_oracle_path=None):
        """
        This function runs the explainer.
        """
        if not os.path.exists(self.explainer_config.explanations_dir):
            os.makedirs(self.explainer_config.explanations_dir)

        batches_out = []
        batch = None
        collage_idx = 0
        if self.val_dataset.config.has_hints:
            self.val_dataset.enable_hints()

        n = (
            self.explainer_config.max_samples
            if not self.explainer_config.max_samples is None
            else len(self.val_dataset)
        )
        pbar = tqdm(
            total=n
            * (
                self.explainer_config.gradient_steps
                if hasattr(self.explainer_config, "gradient_steps")
                else 1
            )
        )
        pbar.stored_values = {}
        pbar.stored_values["n_total"] = 0
        for idx in range(len(self.val_dataset)):
            if (
                not self.explainer_config.max_samples is None
                and collage_idx >= self.explainer_config.max_samples
            ):
                break

            x, y = self.val_dataset[idx]
            if self.val_dataset.hints_enabled:
                y, hint = y

            else:
                hint = None

            y_logits = self.predictor(x.unsqueeze(0).to(self.device))[0]
            y_pred = y_logits.argmax()
            y_confidence = torch.nn.Softmax(dim=-1)(
                y_logits / self.explainer_config.temperature
            )
            for y_target in range(self.val_dataset.task_config.output_channels):
                if y_target == y_pred:
                    continue

                if (
                    not self.explainer_config.transition_restrictions is None
                    and not [y_pred, y_target]
                    in self.explainer_config.transition_restrictions
                ):
                    continue

                if batch is None:
                    batch = {
                        "x_list": x.unsqueeze(0),
                        "y_target_list": torch.tensor([y_target]),
                        "y_source_list": torch.tensor([y_pred]),
                        "y_list": torch.tensor([y]),
                        "y_target_start_confidence_list": torch.tensor(
                            [y_confidence[y_target]]
                        ),
                        "idx_list": [idx],
                    }
                    if not hint is None:
                        batch["hint_list"] = [hint]

                else:
                    batch["x_list"] = torch.cat([batch["x_list"], x.unsqueeze(0)], 0)
                    batch["y_target_list"] = torch.cat(
                        [batch["y_target_list"], torch.tensor([y_target])], 0
                    )
                    batch["y_source_list"] = torch.cat(
                        [batch["y_source_list"], torch.tensor([y_pred])], 0
                    )
                    batch["y_list"] = torch.cat([batch["y_list"], torch.tensor([y])], 0)
                    batch["y_target_start_confidence_list"] = torch.cat(
                        [
                            batch["y_target_start_confidence_list"],
                            torch.tensor([y_confidence[y_target]]),
                        ],
                        0,
                    )
                    batch["idx_list"].append(idx)
                    if not hint is None:
                        batch["hint_list"].append(hint)

                if batch["x_list"].shape[0] == self.explainer_config.batch_size:
                    batches_out.append(
                        self.explain_batch(
                            batch,
                            base_path=os.path.join(
                                self.explainer_config.explanations_dir, "collages"
                            ),
                            start_idx=collage_idx,
                            pbar=pbar,
                        )
                    )
                    collage_idx += len(batches_out[-1]["x_list"])
                    batch = None

                pbar.stored_values["n_total"] += 1

        batches_out_dict = {}
        for key in batches_out[0].keys():
            for batch in batches_out:
                if not key in batches_out_dict.keys():
                    batches_out_dict[key] = batch[key]

                else:
                    batches_out_dict[key] += batch[key]

        return batches_out_dict

    def human_annotate_explanations(
        self,
        collage_path_list,
        y_source_list=None,
        y_target_list=None,
        **kwargs,
    ):
        """ """
        # TODO fix bug with reloading
        shutil.rmtree("static", ignore_errors=True)
        os.makedirs("static")
        self.port = self.explainer_config.port
        while is_port_in_use(self.port):
            print("port " + str(self.port) + " is occupied!")
            self.port += 1

        print("Start explainer loop!")
        #
        # host_name = "localhost"
        host_name = "0.0.0.0"
        app = Flask("feedback_loop")

        self.data = DataStore()
        self.data.i = 0
        self.data.collage_paths = []
        self.data.feedback = []

        app.config.UPLOAD_FOLDER = "static"

        @app.route("/", methods=["GET", "POST"])
        def index():
            if request.method == "POST":
                if request.form["submit_button"] == "Text":
                    self.data.feedback.append(request.form["user_input"])

                if (
                    len(self.data.collage_paths) > 0
                    and len(self.data.collage_paths) > self.data.i
                ):
                    collage_path = self.data.collage_paths[self.data.i]
                    self.data.i += 1
                    return render_template(
                        "explainer.html",
                        form=request.form,
                        counterfactual_collage=collage_path,
                    )

                else:
                    return render_template("information.html")

            elif request.method == "GET":
                if len(self.data.collage_paths) > 0:
                    collage_path = self.data.collage_paths[self.data.i]
                    self.data.i += 1
                    return render_template(
                        "explainer.html",
                        form=request.form,
                        counterfactual_collage=collage_path,
                    )

                else:
                    return render_template("information.html")

        self.thread = threading.Thread(
            target=lambda: app.run(
                host=host_name, port=self.port, debug=True, use_reloader=False
            )
        )
        self.thread.start()
        print("Feedback GUI is active on localhost:" + str(self.port))

        collage_paths_static = []
        for path in collage_path_list:
            collage_path_static = os.path.join("static", path.split("/")[-1])
            shutil.copy(path, collage_path_static)

            collage_paths_static.append(collage_path_static)

        self.data.collage_paths = collage_paths_static

        with tqdm(range(100000)) as pbar:
            for it in pbar:
                if len(self.data.feedback) >= len(self.data.collage_paths):
                    break

                else:
                    pbar.set_description(
                        "Give feedback at localhost:"
                        + str(self.port)
                        + ", Current Feedback given: "
                        + str(len(self.data.feedback))
                        + "/"
                        + str(len(self.data.collage_paths))
                    )
                    time.sleep(1.0)

        # stop_threads = True
        # thread.join()
        feedback = copy.deepcopy(self.data.feedback)
        with open(
            os.path.join(self.explainer_config.explanations_dir, "feedback.txt"), "w"
        ) as f:
            f.write("\n".join(feedback))

        return feedback

    def visualize_interpretations(self, feedback, y_source_list, y_target_list):
        if isinstance(feedback, str):
            with open(feedback, "r") as f:
                s = f.read()
                feedback = s.split("\n")

        interpretations_dir = os.path.join(
            self.explainer_config.explanations_dir, "interpretations"
        )
        Path(interpretations_dir).mkdir(parents=True, exist_ok=True)
        for source_class in range(self.val_dataset.output_size):
            for target_class in range(source_class + 1, self.val_dataset.output_size):
                interpretation = {}
                for idx, elem in enumerate(zip(feedback, y_source_list, y_target_list)):
                    feedback_elem, source_class_elem, target_class_elem = elem
                    qualifies = (
                        source_class == source_class_elem
                        and target_class == target_class_elem
                    )
                    qualifies = qualifies or (
                        source_class == target_class_elem
                        and target_class == source_class_elem
                    )
                    if qualifies:
                        if not feedback_elem in interpretation.keys():
                            interpretation[feedback_elem] = 1

                        else:
                            interpretation[feedback_elem] += 1

                dict_to_bar_chart(
                    interpretation,
                    os.path.join(
                        interpretations_dir,
                        f"{source_class}vs{target_class}",
                    ),
                )
