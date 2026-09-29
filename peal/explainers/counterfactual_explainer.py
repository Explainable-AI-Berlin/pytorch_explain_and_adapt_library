"""The counterfactual explainer of PEAL and the configs of its baselines.

``CounterfactualExplainer`` implements SCE: a counterfactual is searched by
optimising a generator latent (or the image itself) with gradients of the
predictor - optionally a distilled copy of it - until the target class is
reached, while an L1 anchor, a repaint step and an orthogonality penalty keep
the edit small and, across attempts, diverse. The same class also dispatches to
the re-implemented baselines (ACE, DiME, FastDiME, TIME, DAE distillation and a
perfect-false-counterfactual oracle) through their config classes defined here.

Besides the search, the module clusters the resulting explanations, renders
collages, computes the reported sparsity and diversity statistics, and can
collect human feedback on the collages through a small Flask GUI.
"""

import copy
import os
import shutil
import tempfile
import numpy as np
import threading
import time
import torch_kmeans
import gc
import torch
import torchvision

from pathlib import Path
from pydantic import PositiveInt
from torch import nn
from typing import Union
from tqdm import tqdm

from peal._optional import require
from peal.architectures.predictors import get_predictor
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
from peal.training.trainers import distill_predictor, load_first_loadable
from peal.visualization.visualize_counterfactual_gradients import visualize_step
from peal.log import get_logger

_log = get_logger(__name__)


class DAEdistillConfig(ExplainerConfig):
    """
    This class defines the config of the SCE explainer.
    """

    explainer_type: str = "DAEdistillConfig"
    """
    The type of explanation that shall be used.
    """
    predictor_path: Union[str, type(None)] = None
    """
    The path to the predictor that shall be explained.
    """
    generator: Union[type(None), GeneratorConfig] = None
    """
    The generator that shall be used for the counterfactual search
    """
    data_config: Union[type(None), DataConfig] = None
    """
    The data config used for the counterfactual search
    """
    distilled_predictor: Union[type(None), str, dict] = None
    """
    The config for the predictor distillation.
    """
    linesearch_factors: list = ["dynamic"]
    sampler: Union[type(None), dict] = None


class SCEConfig(ExplainerConfig):
    """
    This class defines the config of the SCE explainer.
    """

    explainer_type: str = "SCEConfig"
    """
    The type of explanation that shall be used.
    """
    predictor_path: Union[str, type(None)] = None
    """
    The path to the predictor that shall be explained.
    """
    generator: Union[type(None), GeneratorConfig] = None
    """
    The generator that shall be used for the counterfactual search
    """
    data_config: Union[type(None), DataConfig] = None
    """
    The data config used for the counterfactual search
    """
    gradient_steps: PositiveInt = 50
    """
    The maximum number of gradients step done for explaining the network
    """
    optimizer: str = "Adam"
    """
    The optimizer used for searching the counterfactual
    """
    learning_rate: float = 1.0
    """
    The learning rate used for finding the counterfactual
    """
    y_target_goal_confidence: Union[type(None), float] = None
    """
    The desired target confidence.
    Consider the tradeoff between minimality and clarity of counterfactual
    """
    use_masking: bool = True
    """
    Whether samples in the current search batch are masked after reaching y_target_goal_confidence.
    Otherwise they are continued to be updated until the last surpasses the threshold
    """
    dist_l1: float = 0.0
    """
    Regularizing factor that keeps changes between original image and counterfactual sparse.
    """
    batch_size: int = 1
    """
    The batch size used for the counterfactual search
    """
    distilled_predictor: Union[type(None), str, dict] = None
    """
    The config for the predictor distillation.
    """
    predictor: Union[str, type(None), dict] = None
    """
    The path to either the predictor or its config.
    """
    sampling_time_fraction: float = 0.3
    """
    How deep to go into the latent space for the counterfactual search
    """
    num_discretization_steps: int = 15
    """
    The number of discretizations when going into the latent space
    """
    iterationwise_encoding: bool = True
    """
    Whether to encode every iteration again or not.
    """
    stochastic: Union[type(None), str] = "fully"
    """
    Whether to use stochastic counterfactual search (e.g. DDPM) or not (e.g. deterministic DDIM).
    """
    dilation: int = 5
    """
    The level of dilation for the masking that is used for RePaint. If it is higher the mask is more coarse.
    If it is lower the mask is more finegrained.
    """
    inpaint: float = 0.0
    """
    The threshold what is RePainted in the preexplanation. The repainting back to the original in the preexplanation
    is done for everything that has changes below 100 * inpaint percent of the maximum change between sample and
    preexplanation.
    """
    replace_with_activation: str = "leakysoftplus"
    """
    The activation function ReLU is replaced with: leakyrelu, leakysoftplus
    This helps the distilled predictor to be more sensitive and smooth without saturating gradients.
    """
    greedy: bool = False
    """
    Whether to only keep the best explanation while counterfactual search or not.
    While the greedy solution tends to be more stable it has bigger problems with strong local optima.
    """
    visualize_gradients: bool = False
    """
    Whether to visualize gradients for every step of the counterfactual search or not.
    Helpful for debugging, but decreases speed and clutters disk.
    """
    mask_momentum: float = 0.0
    """
    Momentum term that prevents counterfactual search from changing the area that is changed for creating the
    counterfactual to rapidly.
    """
    momentum: float = 0.9
    """
    Momentum term for the optimizer that does the updates of the preexplanations.
    """
    gradient_clipping: float = 0.05
    """
    The maximum absolute value that gradient update step can change one input variable at once.
    """
    merge_clusters: str = "select_best"
    """
    The strategy on how to merge clusters when calculating them.
    E.g. select_best uses the cluster that does the most salient changes, while merge just merges all clusters.
    """
    allow_overlap: bool = False
    """
    Whether to use the diversification tool that forbids changing the same area of the input image again or not.
    """
    use_gradient_filtering: bool = True
    """
    Whether to use a generative model to filter the gradients or not.
    """
    orthogonalization_penatly: float = 0.0
    """
    The strength of the orthogonalization penalty for parallel counterfactuals.
    """


class ACEConfig(ExplainerConfig):
    """
    This class defines the config of a ACE, DiME or FastDiME explainer.
    This config is primarily for replicating results of related work.
    """

    explainer_type: str = "ACE"
    """
    The type of explanation that shall be used.
    Options: ['counterfactual', 'lrp']
    """
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
    cudnn_deterministic: bool = False
    base_path: str = ""  # DDPM weights path
    exp_name: str = "example_name"
    seed: int = 4  # Random seed
    attack_method: str = "PGD"
    attack_epsilon: float = 255  # L inf epsilon bound (will be divided by 255)
    attack_step: float = 1.0  # Attack update step (will be divided by 255)
    attack_joint: bool = True  # Set to false to generate adversarial attacks
    attack_joint_checkpoint: bool = False
    attack_checkpoint_backward_steps: int = 1
    attack_joint_shortcut: bool = False
    dist_schedule: str = "none"
    sampling_stochastic: bool = True  # Set to False to remove the noise when sampling
    chunks: int = 1  # Chunking for splitting the CE generation into multiple gpus
    chunk: int = 0  # current chunk (between 0 and chunks - 1)
    merge_chunks: bool = False  # to merge all chunked results
    y_target_goal_confidence: Union[type(None), float] = None
    replace_with_activation: str = ""
    """
    The activation function ReLU is replaced with: leaky_relu, leaky_softplus
    """
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

    explainer_type: str = "TIME"
    """
    The type of explanation that shall be used.
    Options: ['counterfactual', 'lrp']
    """
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

    explainer_type: str = "PerfectFalseCounterfactual"
    """
    The type of explanation that shall be used.
    """
    data: Union[type(None), str, DataConfig] = None
    test_data: Union[type(None), str, DataConfig] = None


# def load_dataset(explainer_config, training_config):

#     (
#         train_dataloader,
#         val_dataloader,
#         test_dataloader,
#     ) = create_dataloaders_from_datasource(
#         datasource=None,
#         config=explainer_config,
#         test_config=None,
#         enable_hints=False,
#     )
#     dataloaders_val = WeightedDataloaderList([val_dataloader])
#     print(training_config)
#     dataloader_mixer = DataloaderMixer(
#         train_config=training_config, initial_dataloader=train_dataloader
#     )
#     datasource = [dataloader_mixer, dataloaders_val]

#     return datasource, train_dataloader.dataset.config


def flatten_explanations(
    explanations, reference_key="x_counterfactual_list", batch_size=None
):
    """Give every entry of an explanations dict one row per counterfactual.

    With ``num_attempts > 1`` the counterfactual-derived entries hold one row per
    (factual, attempt) while the per-factual entries hold one row per factual.
    Every consumer indexes by counterfactual -- :func:`cluster_explanations`,
    ``Model2ModelTeacher.get_feedback`` and the desiderata in
    ``calculate_explainer_stats`` -- so a ragged dict makes them read the wrong
    factual or run off the end of the list.

    The expansion is per batch, not global. Counterfactuals are laid out as one
    attempt-major block per batch: for a batch of ``batch_size`` factuals and
    ``r`` attempts the order is all factuals of the batch for attempt 0, then the
    same factuals for attempt 1, and only then the next batch. So a per-factual
    entry is expanded by repeating each ``batch_size``-sized chunk ``r`` times,
    which leaves the two copies of a factual ``batch_size`` rows apart.

    That layout was read off a published run rather than assumed: in a finished
    cache of 408 counterfactuals over 204 factuals, each factual appears exactly
    twice and the two copies are 6 indices apart, 6 being the configured batch
    size. Global tiling would have put them 204 apart and pairs every
    counterfactual with the wrong factual outside the first batch.

    ``batch_size`` may be omitted when the dict covers a single batch, which is
    the case inside :meth:`explain_batch`; the whole entry is then one chunk and
    the result is the same.
    """
    n_reference = len(explanations.get(reference_key, []))
    if not n_reference:
        return explanations
    for key, value in list(explanations.items()):
        if not isinstance(value, (list, torch.Tensor)):
            continue
        n = len(value)
        if n == 0 or n == n_reference:
            continue
        if n_reference % n != 0:
            raise ValueError(
                f"cannot align '{key}' with '{reference_key}': {n} rows do not "
                f"divide {n_reference}"
            )
        repeats = n_reference // n
        chunk = batch_size or n
        if chunk <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        pieces = [value[start : start + chunk] for start in range(0, n, chunk)]
        if isinstance(value, torch.Tensor):
            explanations[key] = torch.cat(
                [p for piece in pieces for p in repeats * [piece]], dim=0
            )
        else:
            expanded = []
            for piece in pieces:
                for _ in range(repeats):
                    expanded.extend(list(piece))
            explanations[key] = expanded
    return explanations


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
        datasets: list([PealDataset]) = None,
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
        # self.predictor = self.predictor.to(self.device)
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
            # if self.explainer_config.data_config is not None:
            #     self.predictor_datasources, self.explainer_config.data_config = (
            #         load_dataset(self.explainer_config, self.predictor_config.training)
            #     )
            #     self.val_dataset = self.predictor_datasources[1].dataloaders[0].dataset
            # else:
            #     raise Exception(
            #         "currently data config need to be declared in explainer config"
            #     )
            raise Exception("Currently not implemented correctly!")
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

        self.counterfactuals_per_second = None

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
            """Map a pytorch-default image into the generator's input space.

            Applies the generator dataset's normalisation and resizes to
            ``generator.dataset.config.input_size`` if needed.
            """
            sample = self.generator.dataset.project_from_pytorch_default(sample)
            if not sample.shape[-2:] == self.generator.dataset.config.input_size[1:]:
                sample = torchvision.transforms.Resize(
                    self.generator.dataset.config.input_size[1:]
                )(sample)
            return sample

        def to_predictor(sample):
            """Map a pytorch-default image into the predictor's input space.

            Applies the validation dataset's normalisation and resizes to
            ``val_dataset.config.input_size`` if needed.
            """
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
            # Only model.cpl is a candidate here: this call site has no way to
            # rebuild an architecture from a bare state_dict, so a stub left by a
            # failed save means re-distilling rather than crashing on every run.
            gradient_predictor = load_first_loadable(
                [distilled_path], map_location=self.device
            )
            if gradient_predictor is None or isinstance(gradient_predictor, dict):
                if isinstance(self.explainer_config.distilled_predictor, dict):
                    self.explainer_config.distilled_predictor["data"] = (
                        self.val_dataset.config
                    )

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
        # should be always in [0,1] and in the resolution of the predictor
        x_original = self.val_dataset.project_to_pytorch_default(x_predictor)

        if not self.explainer_config.iterationwise_encoding:
            pass

        else:
            z_original = x_original.to(self.device)
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
        start_activations = extract_penultima_activation(
            x_predictor.to(self.device), self.predictor
        ).detach()

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
                start_activations=start_activations,
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
        start_activations,
    ):
        """Run one gradient step of the SCE counterfactual optimisation.

        The optimised variable ``z`` is decoded (through the generator when
        ``use_gradient_filtering`` is set, otherwise it already lives in image
        space), normalised for the predictor and scored. The loss is the
        distillation predictor's classification loss towards ``y_target``,
        plus an L1 penalty on the distance to ``z_original`` and, when
        ``orthogonalization_penatly`` is set, a Frobenius penalty pushing the
        penultimate-activation changes of the batch towards orthogonality
        (this is what makes a batch of attempts diverse).

        After the optimiser step the sample is optionally repainted with the
        generator so that everything outside the change mask returns to the
        original, the new target confidences are measured with the real
        predictor, and every sample that improved on its previous best is
        recorded. Samples that reached their confidence goal are switched off
        via ``not_finished_mask``.

        Parameters
        ----------
        x_original : torch.Tensor
            Original images in the pytorch default range, ``(B, C, H, W)``.
        optimizer : torch.optim.Optimizer
            Optimiser over the entries of ``z``.
        y_target : torch.Tensor
            Target class per sample, shape ``(B,)``.
        i : int
            Index of this gradient step, used for logging and for the final
            step special cases.
        boolmask_in : torch.Tensor
            Region the previous attempts are allowed to have changed; the
            gradient of ``z[0]`` is masked with it when inpainting is on.
        gradient_predictor : nn.Module
            Predictor supplying the gradients (the distilled copy when
            ``explainer_config.distilled_predictor`` is set).
        mode : str
            Dataset split name, used in progress-bar text and output paths.
        pbar : tqdm or None
            Progress bar; updated once per step when tracking is enabled.
        batch_idx : int
            Index of the current batch, used in the gradient dump path.
        x_in : torch.Tensor
            The unmodified input batch, used only for the visual difference
            shown in the progress bar.
        num_attempts : int
            1-based attempt counter. For later attempts ``dist_l1`` and
            ``inpaint`` are halved per attempt, unless ``allow_overlap``.
        base_path : str
            Run directory the gradient visualisations are written under.
        z_original : list of torch.Tensor
            The starting value of the optimised variables, the L1 anchor.
        best_score : torch.Tensor
            Best target confidence seen per sample so far; updated in place
            semantics but also returned.
        best_z, best_mask : torch.Tensor
            Variable value and repaint mask belonging to ``best_score``.
        z : list of torch.Tensor
            The optimised variables. With ``iterationwise_encoding`` the
            first entry is an image in ``[0, 1]``, otherwise a latent code.
        target_confidence_goal_current : torch.Tensor
            Per-sample confidence that counts as success.
        not_finished_mask : torch.Tensor
            1 for samples that are still being optimised, 0 for finished
            ones, whose gradients are zeroed when ``use_masking`` is set.
        gradient_confidences_old : torch.Tensor
            Confidences of the gradient predictor from the previous step,
            used for the progress bar only.
        to_generator, to_predictor : callable
            Resize/normalise helpers into the generator and predictor spaces.
        start_activations : torch.Tensor
            Penultimate activations of the originals, the reference for the
            orthogonality penalty.

        Returns
        -------
        tuple
            ``(best_score, best_z, best_mask, not_finished_mask, z,
            target_confidence_goal_current, gradient_confidences_old,
            all_finished)``, where the last entry is ``True`` once every
            sample reached its confidence goal.

        Notes
        -----
        The per-sample gradient of ``z[0]`` is rescaled so that
        ``learning_rate * max|grad|`` never exceeds
        ``explainer_config.gradient_clipping``. With ``tracking_level >= 5``
        and ``visualize_gradients`` every step is dumped as a png under
        ``<base_path>/<mode>_explainer_gradients/<batch>_<attempt>/``.
        """
        if self.explainer_config.iterationwise_encoding:
            # always in [0,1] with resolution of discriminator
            z[0].data = torch.clamp(z[0].data, 0, 1)
            z_default = z[0]

            z_predictor_original = to_predictor(z_default).to(self.device)
            pred_original = torch.nn.functional.softmax(
                self.predictor(z_predictor_original.detach())
                / self.explainer_config.temperature
            )
            target_confidences = torch.zeros_like(target_confidence_goal_current)
            for j in range(len(target_confidences)):
                target_confidences[j] = pred_original[j, int(y_target[j])]

            clean_img_old = torch.clone(z_default).detach().cpu()

            if self.explainer_config.use_gradient_filtering:
                z_generator = to_generator(z[0])
                z_encoded = self.generator.encode(
                    z_generator.to(self.device),
                    t=self.explainer_config.sampling_time_fraction,
                    stochastic=self.explainer_config.stochastic,
                )

            else:
                z_encoded = z[0].to(self.device)

        else:
            z_encoded = [z_elem.to(self.device) for z_elem in z]

        optimizer.zero_grad()
        if self.explainer_config.use_gradient_filtering:
            img_decoded = self.generator.decode(
                z_encoded, t=self.explainer_config.sampling_time_fraction
            )
            img_default = self.generator.dataset.project_to_pytorch_default(img_decoded)
            if not img_default.shape[-2:] == self.val_dataset.config.input_size[1:]:
                img_default = torchvision.transforms.Resize(
                    self.val_dataset.config.input_size[1:]
                )(img_default)

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

        if self.explainer_config.orthogonalization_penatly > 0.0:
            activations = extract_penultima_activation(img_predictor, self.predictor)
            activation_differences = activations - start_activations
            normalized_activations = (
                torch.nn.functional.normalize(activation_differences, p=2, dim=1)
                .squeeze(-1)
                .squeeze(-1)
            )
            gram_matrix = torch.matmul(normalized_activations, normalized_activations.T)
            identity_matrix = torch.eye(
                gram_matrix.shape[0], device=normalized_activations.device
            )
            orthogonality_loss = (
                self.explainer_config.orthogonalization_penatly
                * torch.norm(gram_matrix - identity_matrix, p="fro") ** 2
            )
            loss += orthogonality_loss

        else:
            gram_matrix = None

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
                if not gram_matrix is None:
                    description_str += f", orth: {gram_matrix[0][1:]}"
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
                # should be in [0,1] on predictor resolution
                pe = torch.clone(z[0]).detach().cpu()
                z_default = z[0]

            else:
                pe = torch.clone(z[0]).detach().cpu()
                z_default = z[0]

            z_predictor = (
                self.val_dataset.project_from_pytorch_default(z_default)
                .to(self.device)
                .detach()
            )
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
                if (
                    not boolmask_in.shape[-2:]
                    == self.generator.dataset.config.input_size[1:]
                ):
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
                if not boolmask.shape[-2:] == self.val_dataset.config.input_size[1:]:
                    boolmask = torchvision.transforms.Resize(
                        self.val_dataset.config.input_size[1:]
                    )(boolmask)

                z_generator_current = self.generator.dataset.project_to_pytorch_default(
                    z_updated
                )
                if (
                    not z_generator_current.shape[-2:]
                    == self.val_dataset.config.input_size[1:]
                ):
                    z_generator_current = torchvision.transforms.Resize(
                        self.val_dataset.config.input_size[1:]
                    )(z_generator_current)

                for sample_idx in range(z[0].data.shape[0]):
                    if (
                        not_finished_mask[sample_idx] == 1
                        and no_repaint_exceptions[sample_idx] == 0
                    ):
                        z[0].data[sample_idx] = z_generator_current[sample_idx]

        z[0].data = torch.clamp(z[0].data, 0, 1)
        z_default = z[0]

        z_predictor_original = (
            self.val_dataset.project_from_pytorch_default(z_default)
            .to(self.device)
            .detach()
        )
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
                if (
                    not z_encoded_visualization.shape[-2:]
                    == self.val_dataset.config.input_size[1:]
                ):
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
                img_predictor=self.val_dataset.project_to_pytorch_default(
                    img_predictor
                ),
                img_predictor_unnormalized=img_predictor,
                pe=pe,
                boolmask=boolmask,
                filename=os.path.join(
                    gradients_path, embed_numberstring(i, 4) + ".png"
                ),
                boolmask_in=boolmask_in,
                best_z=best_z,
                best_mask=best_mask,
            )

        gc.collect()
        torch.cuda.empty_cache()
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
        raw_start_idx = start_idx
        original_batch_size = len(batch["x_list"])

        if explainer_path is None:
            os_sep = os.path.abspath(os.sep)
            if base_path[: len(os_sep)] == os_sep:
                path_splitted = [os_sep]

            else:
                path_splitted = []

            path_splitted += base_path.split(os.sep)[:-1]
            explainer_path = os.path.join(*path_splitted)

        if mode == "validation":
            start_idx = (
                start_idx
                * self.explainer_config.num_attempts
                * self.explainer_config.parallel_attempts
            )

        if y_target_goal_confidence_in is None:
            if hasattr(self.explainer_config, "y_target_goal_confidence"):
                target_confidence_goal = self.explainer_config.y_target_goal_confidence

            else:
                target_confidence_goal = 0.51

        else:
            target_confidence_goal = y_target_goal_confidence_in

        if self.counterfactuals_per_second is None and start_idx != 0:
            start_time = time.perf_counter()

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
            x_in = torch.tile(
                batch["x_list"], [self.explainer_config.parallel_attempts, 1, 1, 1]
            )
            y_target = torch.tile(
                batch["y_target_list"], [self.explainer_config.parallel_attempts]
            )
            (
                batch["x_counterfactual_list"],
                batch["z_difference_list"],
                batch["y_target_end_confidence_list"],
                _,
            ) = self.predictor_distilled_counterfactual(
                x_in=x_in,
                y_target=y_target,
                target_confidence_goal=target_confidence_goal,
                pbar=pbar,
                mode=mode,
                base_path=explainer_path,
                batch_idx=start_idx,
                num_attempts=self.explainer_config.num_attempts,
            )
            if self.counterfactuals_per_second is None and start_idx != 0:
                end_time = time.perf_counter()
                total_time = end_time - start_time
                self.counterfactuals_per_second = (
                    len(batch["x_counterfactual_list"]) / total_time
                )
                _log.info(
                    "%s",
                    f"Explainer speed: {self.counterfactuals_per_second:.2f} counterfactuals per second.",
                )

        elif isinstance(self.generator, EditCapableGenerator):
            _log.info("%s", "explain batch!")
            number_attempts = self.explainer_config.num_attempts
            previous_attempts = []
            boolmask_in = None
            generator_name = self.generator.__class__.__name__
            uses_attempt_loop = generator_name in {"DiffusionGenerator", "DDPMPathLDM"}
            if uses_attempt_loop:
                for i in range(number_attempts):
                    previous_batch = copy.deepcopy(batch)

                    (
                        previous_batch["x_counterfactual_list"],
                        previous_batch["z_difference_list"],
                        previous_batch["y_target_end_confidence_list"],
                        previous_batch["x_list"],
                        previous_batch["history_list"],
                        boolmask_in,
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
                        boolmask_in=boolmask_in,
                        attempt_number=i,
                    )
                    previous_attempts.append(previous_batch)
                all_batches = {}
                for i in previous_attempts:
                    for key in i.keys():
                        if key not in all_batches.keys():
                            all_batches[key] = None
                        if isinstance(i[key], list):
                            if all_batches[key] is None:
                                all_batches[key] = i[key]

                            else:
                                all_batches[key] += i[key]

                        elif isinstance(i[key], torch.Tensor):
                            if all_batches[key] is None:
                                all_batches[key] = i[key]
                            else:
                                all_batches[key] = torch.cat(
                                    (all_batches[key], i[key]), dim=0
                                )

                        elif i[key] is None:
                            continue
                        else:
                            raise Exception(
                                f"key {key} of type {type(i[key])} not supported in batch!"
                            )
                del previous_attempts
                for key in all_batches.keys():
                    batch[key] = all_batches[key]
                del all_batches
            else:
                if explainer_path is None:
                    explainer_path = os.path.join(
                        *([os.path.abspath(os.sep)] + base_path.split(os.sep)[:-1])
                    )

                edit_result = self.generator.edit(
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
                (
                    batch["x_counterfactual_list"],
                    batch["z_difference_list"],
                    batch["y_target_end_confidence_list"],
                    batch["x_list"],
                    batch["history_list"],
                    *extra_edit_outputs,
                ) = edit_result
                if extra_edit_outputs:
                    boolmask_in = extra_edit_outputs[0]

            if self.counterfactuals_per_second is None and start_idx != 0:
                end_time = time.perf_counter()
                total_time = end_time - start_time
                self.counterfactuals_per_second = (
                    len(batch["x_counterfactual_list"]) / total_time
                )
                _log.info(
                    "%s",
                    f"Explainer speed: {self.counterfactuals_per_second:.2f} counterfactuals per second.",
                )
                _log.info(
                    "%s",
                    f"Explainer speed: {self.counterfactuals_per_second:.2f} counterfactuals per second.",
                )
                _log.info(
                    "%s",
                    f"Explainer speed: {self.counterfactuals_per_second:.2f} counterfactuals per second.",
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
            collage_start_idx = start_idx
            if mode == "validation":
                counterfactuals_per_input = (
                    len(batch_out["x_counterfactual_list"]) // original_batch_size
                )
                collage_start_idx = raw_start_idx * counterfactuals_per_input

            # Writing collages is a visualisation step and no reported metric depends
            # on it, but an exception here used to discard the whole run: a single
            # unrenderable title cost three multi-hour ACE runs. Fall back to the
            # plain attribution the tracking_level < 4 branch computes.
            try:
                (
                    batch_out["x_attribution_list"],
                    batch_out["collage_path_list"],
                ) = self.val_dataset.generate_contrastive_collage(
                    target_confidence_goal=target_confidence_goal,
                    base_path=base_path,
                    predictor=self.predictor,
                    start_idx=collage_start_idx,
                    **batch_out,
                )
            except Exception as error:  # noqa: BLE001 - cosmetic step, never fatal
                _log.info(
                    "%s",
                    f"collage writing failed, continuing without collages: {error}",
                )
                batch_out["x_attribution_list"] = [
                    torch.abs(
                        batch_out["x_counterfactual_list"][i]
                        - batch_out["x_list"][i % len(batch_out["x_list"])]
                    )
                    for i in range(len(batch_out["x_counterfactual_list"]))
                ]
                batch_out.pop("collage_path_list", None)

        else:

            if isinstance(self.generator, InvertibleGenerator) and isinstance(
                self.explainer_config, SCEConfig
            ):
                number_attempts = self.explainer_config.num_attempts
                if len(batch["x_counterfactual_list"]) > len(batch_out["x_list"]):
                    batch["x_list"] = torch.cat(
                        [batch["x_list"], batch["x_list"]], dim=0
                    )
                x_attribution_list = [
                    torch.abs(
                        batch_out["x_counterfactual_list"][i] - batch_out["x_list"][i]
                    )
                    for i in range(len(batch_out["x_counterfactual_list"]))
                ]
            else:

                x_attribution_list = []
                for i in range(len(batch_out["x_counterfactual_list"])):
                    x_attribution_list.append(
                        torch.abs(
                            batch_out["x_counterfactual_list"][i]
                            - batch_out["x_list"][i]
                        )
                    )

            batch_out["x_attribution_list"] = x_attribution_list

        # One row per counterfactual for every entry; see flatten_explanations.
        batch_out = flatten_explanations(batch_out)

        torch.cuda.empty_cache()
        return batch_out

    def cluster_explanations(self, explanations_dict, batch_size=2, n_clusters=2):
        """
        This function clusters the explanations.
        """
        clustering_strategy = self.explainer_config.clustering_strategy
        supported_strategies = {
            "activation_clusters",
            "attempt_nr",
            "highest_activation",
            "kmeans",
            "preclustered",
        }
        if clustering_strategy not in supported_strategies:
            raise ValueError(
                f"Unsupported clustering strategy {clustering_strategy!r}. "
                f"Expected one of {sorted(supported_strategies)}."
            )

        explanations_list = []
        if self.tracking_level < 4:
            explanations_dict.pop("collage_path_list", None)
        for idx in range(len(explanations_dict["x_list"])):

            current_dict = {}
            for key in explanations_dict.keys():
                val = explanations_dict[key]
                if isinstance(val, (list, tuple, torch.Tensor, np.ndarray)) and len(
                    val
                ) == len(explanations_dict["x_list"]):
                    current_dict[key] = val[idx]

            explanations_list.append(current_dict)

        assert (
            len(explanations_list) % (batch_size * n_clusters) == 0
        ), "restructuring needed for clustering impossible!"
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
            """Compare the penultimate-layer shifts of one sample's attempts.

            All entries of ``explanations`` must belong to the same original
            image (this is asserted on ``x_list``). For each of them the
            predictor's penultimate activation of the counterfactual minus
            that of the original is computed, and the resulting difference
            vectors are compared with each other.

            Parameters
            ----------
            explanations : list of dict
                One flattened explanation per attempt for the same input.

            Returns
            -------
            difference_list : list of torch.Tensor
                Activation difference per attempt.
            cosine_similarities_list : list of torch.Tensor
                For each attempt, the cosine similarities to all others.
            norm_list : list of torch.Tensor
                L2 norm of each difference.
            ratio_list : torch.Tensor
                Constant vector holding max(norm) / min(norm), i.e. how
                unevenly the attempts moved the representation.

            Raises
            ------
            Exception
                If the explanations do not all share the same input image
                (only below ``tracking_level`` 4, which just prints).
            """
            difference_list = []
            activation_ref = (
                extract_penultima_activation(
                    explanations[0]["x_list"][None, ...].to(self.device), self.predictor
                )
                .detach()
                .cpu()
            )
            for i in range(len(explanations)):
                if (
                    torch.sum(explanations[0]["x_list"] != explanations[i]["x_list"])
                    != 0
                ):
                    if self.explainer_config.tracking_level >= 4:
                        _log.info("%s", "x list is not matching across samples!")
                    else:
                        raise Exception("x list is not matching across samples!")

                activation_current = (
                    extract_penultima_activation(
                        explanations[i]["x_counterfactual_list"][None, ...].to(
                            self.device
                        ),
                        self.predictor,
                    )
                    .detach()
                    .cpu()
                )
                difference_list.append(activation_current - activation_ref)

            # import pdb; pdb.set_trace()
            cosine_similarities_list = []
            norm_list = []
            for i in range(len(difference_list)):
                norm_list.append(torch.norm(difference_list[i]))
                cosine_similarities = []
                for j in range(len(difference_list)):
                    if i != j:
                        cosine_similarities.append(
                            torch.nn.functional.cosine_similarity(
                                difference_list[i], difference_list[j]
                            )
                        )

                cosine_similarities_list.append(torch.tensor(cosine_similarities))

            ratio_list = (
                torch.ones([len(norm_list)])
                * torch.tensor(norm_list).max()
                / torch.tensor(norm_list).min()
            )
            return difference_list, cosine_similarities_list, norm_list, ratio_list

        cluster_lists = [[] for i in range(n_clusters)]
        collage_path_base = None
        if self.explainer_config.clustering_strategy == "attempt_nr":
            # Simple strategy: each attempt number is its own cluster
            for cluster_idx in range(n_clusters):
                cluster_lists[cluster_idx] = explanations_list_by_source[cluster_idx]
                # PreclusteredTeacher.get_feedback takes a per-explanation
                # cluster_list and checks membership in correct_clusters. The
                # other strategies stamp it (see the highest_activation /
                # activation_clusters path below); this branch did not, so a
                # preclustered teacher on a DAEdistill/DiDAE explainer failed
                # with a missing cluster_list. Under attempt_nr the cluster IS
                # the attempt index, so there is nothing to infer.
                for explanation in cluster_lists[cluster_idx]:
                    explanation["cluster_list"] = cluster_idx

        elif self.explainer_config.clustering_strategy == "preclustered":
            for idx, explanation in enumerate(explanations_list):
                idx_cluster = explanation["cluster_list"]
                cluster_lists[idx_cluster].append(explanation)
                collage_path = explanation["collage_path_list"]
                cluster_collage_dir = collage_path_base + "_" + str(int(idx_cluster))
                if not os.path.exists(cluster_collage_dir):
                    os.makedirs(cluster_collage_dir)

                collage_path_new = os.path.join(
                    *[
                        cluster_collage_dir,
                        embed_numberstring(idx, 7) + ".png",
                    ]
                )
                shutil.copy(collage_path, collage_path_new)

        elif clustering_strategy == "kmeans":
            for sample_idx in range(len(explanations_list_by_source[0])):
                feature_difference, cosine_similarities_list, norm_list, ratio_list = (
                    extract_feature_difference(
                        [e[sample_idx] for e in explanations_list_by_source]
                    )
                )
                for source_idx in range(len(explanations_list_by_source)):
                    explanations_list_by_source[source_idx][sample_idx][
                        "feature_difference"
                    ] = feature_difference[0]
                    explanations_list_by_source[source_idx][sample_idx][
                        "cosine_similarities"
                    ] = cosine_similarities_list[0]
                    explanations_list_by_source[source_idx][sample_idx]["norm_list"] = (
                        norm_list[0]
                    )
                    explanations_list_by_source[source_idx][sample_idx][
                        "ratio_list"
                    ] = ratio_list[0]

            for source_idx in range(len(explanations_list_by_source)):
                explanations_list_by_source[source_idx] = list(
                    filter(
                        lambda explanation: explanation["y_target_end_confidence_list"]
                        > 0.5,
                        explanations_list_by_source[source_idx],
                    )
                )

            explanations_list = []
            for source_list in explanations_list_by_source:
                explanations_list += source_list

            n_clusters_kmeans = 2 ** (n_clusters + 1) - 2
            f_diff = (
                torch.cat([e["feature_difference"] for e in explanations_list])
                .squeeze(-1)
                .squeeze(-1)
            )
            f_diff = f_diff / torch.norm(f_diff, dim=1, keepdim=True)
            kmeans = kmeans = torch_kmeans.SoftKMeans(
                n_clusters=n_clusters_kmeans, distance=torch_kmeans.CosineSimilarity
            )
            predictions = kmeans.fit_predict(
                torch.stack([f_diff, torch.randn_like(f_diff)])
            )
            cluster_lists = [[] for i in range(n_clusters_kmeans)]
            collage_path_ref = explanations_list[0]["collage_path_list"]
            collage_path_elements = collage_path_ref.split(os.sep)[:-1]
            collage_path_base = str(
                os.path.join(*([os.path.abspath(os.sep)] + collage_path_elements))
            )
            for idx, explanation in enumerate(explanations_list):
                idx_cluster = predictions[0][idx]
                cluster_lists[idx_cluster].append(explanation)
                collage_path = explanation["collage_path_list"]
                cluster_collage_dir = collage_path_base + "_" + str(int(idx_cluster))
                if not os.path.exists(cluster_collage_dir):
                    os.makedirs(cluster_collage_dir)

                collage_path_new = os.path.join(
                    *[
                        cluster_collage_dir,
                        embed_numberstring(idx, 7) + ".png",
                    ]
                )
                shutil.copy(collage_path, collage_path_new)

        else:
            if clustering_strategy == "activation_clusters":
                all_over_decision_boundary = False
                explanations_beginning = None
                lowest_similarity = 1.0
                current_idx = -1
                for search_idx in range(len(explanations_list_by_source[0])):
                    all_over_decision_boundary = True
                    current_explanations = [
                        e[search_idx] for e in explanations_list_by_source
                    ]
                    for i in range(len(current_explanations)):
                        all_over_decision_boundary &= (
                            current_explanations[i]["y_target_end_confidence_list"]
                            > 0.5
                        )

                    if not all_over_decision_boundary:
                        continue

                    _, cosine_similarity_list, _, _ = extract_feature_difference(
                        current_explanations
                    )

                    if cosine_similarity_list[0][0] < lowest_similarity:
                        explanations_beginning = current_explanations
                        lowest_similarity = cosine_similarity_list[0][0]
                        current_idx = search_idx

                # if explanations_beginning is None:
                #     # Fallback: use first available explanations even if not all crossed boundary
                #     explanations_beginning = [e[0] for e in explanations_list_by_source]
                #     print(
                #         "Warning: No valid cluster initialization found. Using first sample as fallback."
                #     )
                try:
                    cluster_means, _, _, _ = extract_feature_difference(
                        explanations_beginning
                    )
                except:
                    raise
                cluster_lists[0] = [explanations_list_by_source[0][0]]
                cluster_lists[1] = [explanations_list_by_source[1][0]]
                if "collage_path_list" in explanations_beginning[0].keys():
                    collage_path_ref = explanations_beginning[0]["collage_path_list"]
                    collage_path_elements = collage_path_ref.split(os.sep)[:-1]
                    collage_path_base = str(
                        os.path.join(
                            *([os.path.abspath(os.sep)] + collage_path_elements)
                        )
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
                if clustering_strategy == "highest_activation":
                    current_activations = []
                    for source_idx in range(len(explanations_list_by_source)):
                        current_activations.append(
                            explanations_list_by_source[source_idx][idx][
                                "y_target_end_confidence_list"
                            ]
                        )

                    current_activations = torch.tensor(current_activations)
                    activations_order = torch.argsort(current_activations)

                elif clustering_strategy == "activation_clusters":
                    if idx == current_idx:
                        continue

                    current_differences, _, _, _ = extract_feature_difference(
                        [e[idx] for e in explanations_list_by_source]
                    )
                    # build outer product between cluster means and current differences
                    cosine_similarities = torch.zeros(
                        [len(cluster_means), len(current_differences)]
                    )
                    for i in range(len(cluster_means)):
                        for j in range(len(current_differences)):
                            try:
                                cosine_similarities[i, j] = torch.nn.CosineSimilarity()(
                                    cluster_means[i], current_differences[j]
                                )

                            except Exception:
                                raise

                    # find the cluster with the highest similarity
                    cosine_similarities_abs = torch.abs(cosine_similarities)

                for i in range(n_clusters):
                    if clustering_strategy == "activation_clusters":
                        idx_combined = int(
                            torch.argmax(cosine_similarities_abs.flatten())
                        )
                        idx_cluster = idx_combined // len(current_differences)
                        idx_current = idx_combined % len(current_differences)
                        cosine_similarities_abs[idx_cluster, :] = -1
                        cosine_similarities_abs[:, idx_current] = -1
                        # update running mean

                    elif clustering_strategy == "highest_activation":
                        idx_cluster = activations_order[i]
                        idx_current = i

                    elif clustering_strategy == "attempt_nr":
                        idx_cluster = i
                        idx_current = i

                    if len(cluster_lists[idx_cluster]) > min(
                        len(cluster_list) for cluster_list in cluster_lists
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
                        explanations_list_by_source[idx_current][idx][
                            "cluster_list"
                        ] = [idx_cluster]

                    if (
                        clustering_strategy == "highest_activation"
                        and "collage_path_list"
                        in explanations_list_by_source[idx_current][idx].keys()
                    ):
                        collage_path_ref = explanations_list_by_source[idx_current][
                            idx
                        ]["collage_path_list"]
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
            try:
                for sample_idx in range(len(cluster_dicts[cluster_idx]["x_list"])):
                    sample_scores.append(
                        cluster_dicts[cluster_idx]["y_target_end_confidence_list"][
                            sample_idx
                        ]
                    )
            except:
                _log.info("%s", "error in cluster score")
                raise

            cluster_scores.append(torch.mean(torch.tensor(sample_scores)))

        sorted_cluster_idxs = torch.tensor(cluster_scores).argsort()
        sorted_cluster_idxs = [
            int(sorted_cluster_idxs[-1 - i]) for i in range(len(cluster_lists))
        ]

        explanations_dict_out = cluster_dicts[sorted_cluster_idxs[0]]

        for i in range(len(sorted_cluster_idxs)):
            explanations_dict_out["clusters" + str(int(i))] = copy.deepcopy(
                cluster_dicts[sorted_cluster_idxs[i]]["x_counterfactual_list"]
            )
            explanations_dict_out["cluster_confidence" + str(int(i))] = copy.deepcopy(
                cluster_dicts[sorted_cluster_idxs[i]]["y_target_end_confidence_list"]
            )
            # Store the original component index for this sorted cluster
            explanations_dict_out["cluster_component_idx" + str(int(i))] = int(
                sorted_cluster_idxs[i]
            )

        if self.explainer_config.merge_clusters == "concatenate":
            for cluster_idx in range(1, len(cluster_dicts)):
                for key in cluster_dicts[sorted_cluster_idxs[cluster_idx]].keys():
                    explanations_dict_out[key] += cluster_dicts[
                        sorted_cluster_idxs[cluster_idx]
                    ][key]

        return explanations_dict_out

    def calculate_latent_difference_stats(
        self,
        explanations_dict,
        explainer_stats_clusters=None,
        visualize_latent_sparsity=False,
        visualize_latent_diversity=False,
        base_dir=None,
    ):
        """Measure how sparse and how diverse the counterfactual edits are.

        Two ways of obtaining a low-dimensional edit vector are supported. If
        the validation dataset exposes ``sample_to_2d_latent`` (the synthetic
        datasets with known generative factors), the edit is the difference of
        the ground-truth latents. Otherwise, if hint masks are available, the
        edit is summarised by the mean absolute pixel change inside and
        outside the hint mask, i.e. a two-dimensional (foreground,
        background) vector. Without either, no statistics are produced.

        Only samples that actually flipped count: a sample enters the sparsity
        average when the first active cluster flipped both the original and
        the distilled predictor (confidence > 0.5), and the diversity average
        when all active clusters flipped both. Sparsity per sample is the
        normalised Hoyer measure of the edit vector, diversity is one minus
        the absolute cosine similarity between the edits of the first two
        active clusters. Both are averaged and reported as ``1 - mean``.

        Parameters
        ----------
        explanations_dict : dict
            Flattened explanations. ``clusters<i>`` lists are derived from
            ``x_counterfactual_list`` when missing, by striding with
            ``explainer_config.num_attempts``.
        explainer_stats_clusters : sequence of int, optional
            Which attempt indices to use; defaults to the first two (or one,
            if only one attempt was run).
        visualize_latent_sparsity, visualize_latent_diversity : bool, optional
            Also write the collages produced by
            ``_generate_sparsity_collages`` / ``_generate_diversity_collages``
            (requires ``base_dir``).
        base_dir : str, optional
            Directory the collages are written to.

        Returns
        -------
        dict
            ``{"latent_sparsity": ..., "latent_diversity": ...}``, or an empty
            dict when no edit vectors could be computed. Both values are 0.0
            when no sample passed the flip gate.
        """
        tracked_stats = {}
        latent_differences = None
        # TODO is this correct??
        if not "clusters0" in explanations_dict.keys():
            for cluster_idx in range(self.explainer_config.num_attempts):
                explanations_dict["clusters" + str(cluster_idx)] = []
                for i in range(
                    int(
                        len(explanations_dict["x_list"])
                        / self.explainer_config.num_attempts
                    )
                ):
                    explanations_dict["clusters" + str(cluster_idx)].append(
                        explanations_dict["x_counterfactual_list"][
                            i * self.explainer_config.num_attempts + cluster_idx
                        ]
                    )

        # Use the explicitly configured cluster indices for sparsity/diversity
        if explainer_stats_clusters is not None:
            active_cluster_indices = list(explainer_stats_clusters)
        else:
            active_cluster_indices = list(
                range(min(2, self.explainer_config.num_attempts))
            )

        if hasattr(self.val_dataset, "sample_to_2d_latent"):
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
                    self.val_dataset.sample_to_2d_latent(e.to(self.device), hint).cpu()
                )

            latents_counterfactual = []
            latent_differences = []
            for c in active_cluster_indices:
                latents_counterfactual.append(
                    [
                        self.val_dataset.sample_to_2d_latent(
                            e.to(self.device),
                            (
                                explanations_dict["hint_list"][i]
                                if "hint_list" in explanations_dict.keys()
                                else None
                            ),
                        ).cpu()
                        for i, e in enumerate(explanations_dict["clusters" + str(c)])
                    ]
                )

                latent_differences.append(
                    [
                        latents_counterfactual[-1][i] - latents_original[i]
                        for i in range(len(latents_original))
                    ]
                )

        elif "hint_list" in explanations_dict.keys():
            latent_differences = []
            for c in active_cluster_indices:
                x_difference_list = [
                    explanations_dict["clusters" + str(c)][i]
                    - explanations_dict["x_list"][i]
                    for i in range(len(explanations_dict["clusters0"]))
                ]
                foreground_change = [
                    torch.sum(
                        torch.abs(
                            x_difference_list[i] * explanations_dict["hint_list"][i]
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
                        / torch.sum(torch.abs(1 - explanations_dict["hint_list"][i]))
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

            # Paired gating: sparsity and diversity require BOTH the original predictor
            # AND the distilled predictor to flip for the relevant cluster(s)
            sparsity_valid_sample_indices = []
            latent_sparsities = []

            diversity_valid_sample_indices = []
            cosine_similiarities_list = []

            for i in range(len(latent_differences[0])):
                # Check flip for first active cluster (original + distilled)
                c0 = active_cluster_indices[0]
                conf_key0 = "cluster_confidence" + str(c0)
                dist_conf_key0 = "cluster_confidence_distilled" + str(c0)
                if conf_key0 in explanations_dict:
                    flipped_0_original = float(explanations_dict[conf_key0][i]) > 0.5
                else:
                    flipped_0_original = (
                        float(explanations_dict["y_target_end_confidence_list"][i])
                        > 0.5
                    )

                # Also check distilled predictor flip
                if dist_conf_key0 in explanations_dict:
                    flipped_0_distilled = (
                        float(explanations_dict[dist_conf_key0][i]) > 0.5
                    )
                elif "y_target_end_confidence_distilled_list" in explanations_dict:
                    flipped_0_distilled = (
                        float(
                            explanations_dict["y_target_end_confidence_distilled_list"][
                                i
                            ]
                        )
                        > 0.5
                    )
                else:
                    flipped_0_distilled = (
                        flipped_0_original  # fallback if no distilled info
                    )

                flipped_0 = flipped_0_original and flipped_0_distilled

                if flipped_0:
                    diff_0 = latent_differences[0][i]
                    if diff_0.abs().max() == 0.0:
                        sparsity = 0.0
                    else:
                        n = float(diff_0.numel())
                        l1_norm = diff_0.abs().sum()
                        l2_norm = diff_0.norm(p=2)
                        sparsity = float(
                            1.0 - (n**0.5 - l1_norm / l2_norm) / (n**0.5 - 1.0)
                        )

                    latent_sparsities.append(sparsity)
                    sparsity_valid_sample_indices.append(i)

                # Check flip for all active clusters for diversity (original + distilled)
                if len(active_cluster_indices) >= 2:
                    all_flipped = True
                    for c_idx, c in enumerate(active_cluster_indices):
                        conf_key = "cluster_confidence" + str(c)
                        dist_conf_key = "cluster_confidence_distilled" + str(c)
                        # Check original predictor flip
                        if conf_key in explanations_dict:
                            all_flipped &= float(explanations_dict[conf_key][i]) > 0.5
                        else:
                            all_flipped &= (
                                float(
                                    explanations_dict["y_target_end_confidence_list"][i]
                                )
                                > 0.5
                            )
                        # Check distilled predictor flip
                        if dist_conf_key in explanations_dict:
                            all_flipped &= (
                                float(explanations_dict[dist_conf_key][i]) > 0.5
                            )
                        elif (
                            "y_target_end_confidence_distilled_list"
                            in explanations_dict
                        ):
                            all_flipped &= (
                                float(
                                    explanations_dict[
                                        "y_target_end_confidence_distilled_list"
                                    ][i]
                                )
                                > 0.5
                            )

                    if all_flipped:
                        diff_0 = latent_differences[0][i]
                        diff_1 = latent_differences[1][i]
                        sim = torch.abs(
                            torch.nn.CosineSimilarity(dim=0)(diff_0, diff_1)
                        )
                        cosine_similiarities_list.append(sim)
                        diversity_valid_sample_indices.append(i)

            latent_sparsity = (
                1.0 - float(torch.mean(torch.tensor(latent_sparsities)))
                if len(latent_sparsities) > 0
                else 0.0
            )
            latent_diversity = (
                1.0 - float(torch.mean(torch.tensor(cosine_similiarities_list)))
                if len(cosine_similiarities_list) > 0
                else 0.0
            )

            # Generate sparsity collages if requested
            if (
                visualize_latent_sparsity
                and base_dir is not None
                and len(latent_sparsities) > 0
            ):
                self._generate_sparsity_collages(
                    explanations_dict=explanations_dict,
                    active_cluster_indices=active_cluster_indices,
                    valid_sample_indices=sparsity_valid_sample_indices,
                    latent_sparsities=latent_sparsities,
                    base_dir=base_dir,
                )

            # Generate diversity collages if requested
            if (
                visualize_latent_diversity
                and base_dir is not None
                and len(active_cluster_indices) >= 2
                and len(cosine_similiarities_list) > 0
            ):
                self._generate_diversity_collages(
                    explanations_dict=explanations_dict,
                    active_cluster_indices=active_cluster_indices,
                    valid_sample_indices=diversity_valid_sample_indices,
                    cosine_similarities=cosine_similiarities_list,
                    base_dir=base_dir,
                )

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

    def _generate_sparsity_collages(
        self,
        explanations_dict,
        active_cluster_indices,
        valid_sample_indices,
        latent_sparsities,
        base_dir,
    ):
        """
        Generate collages for latent sparsity visualization.
        Each collage contains: factual, counterfactual, target end confidence, sparsity score.
        """
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        collage_dir = os.path.join(base_dir, "latent_sparsity_collages")
        Path(collage_dir).mkdir(parents=True, exist_ok=True)

        c0 = active_cluster_indices[0]
        conf_key = "cluster_confidence" + str(c0)
        dist_conf_key = "cluster_confidence_distilled" + str(c0)

        for vi, (sample_idx, sparsity_score) in enumerate(
            zip(valid_sample_indices, latent_sparsities)
        ):
            factual = explanations_dict["x_list"][sample_idx]
            counterfactual = explanations_dict["clusters" + str(c0)][sample_idx]

            # Get target end confidence (original predictor)
            if conf_key in explanations_dict:
                target_conf = float(explanations_dict[conf_key][sample_idx])
            else:
                target_conf = float(
                    explanations_dict["y_target_end_confidence_list"][sample_idx]
                )

            # Get target end confidence (distilled predictor)
            if dist_conf_key in explanations_dict:
                target_conf_distilled = float(
                    explanations_dict[dist_conf_key][sample_idx]
                )
            elif "y_target_end_confidence_distilled_list" in explanations_dict:
                target_conf_distilled = float(
                    explanations_dict["y_target_end_confidence_distilled_list"][
                        sample_idx
                    ]
                )
            else:
                target_conf_distilled = None

            sparsity_display = (
                1.0 - sparsity_score
            )  # latent_sparsities stores raw ratio, final metric is 1 - mean

            # Build confidence subtitle
            conf_text = f"Original Conf: {target_conf:.4f}"
            if target_conf_distilled is not None:
                conf_text += f" | Distilled Conf: {target_conf_distilled:.4f}"

            is_image = len(factual.shape) >= 3 and factual.shape[0] in [1, 3]
            if is_image and hasattr(self.val_dataset, "project_to_pytorch_default"):
                factual_vis = self.val_dataset.project_to_pytorch_default(factual)
                cf_vis = self.val_dataset.project_to_pytorch_default(counterfactual)
            else:
                factual_vis = factual
                cf_vis = counterfactual

            try:
                if is_image:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(
                        factual_vis.permute(1, 2, 0).cpu().numpy().clip(0, 1)
                    )
                    axes[0].axis("off")
                    axes[0].set_title("Factual", fontweight="bold")

                    axes[1].imshow(cf_vis.permute(1, 2, 0).cpu().numpy().clip(0, 1))
                    axes[1].axis("off")
                    axes[1].set_title("Counterfactual", fontweight="bold")

                    fig.suptitle(
                        f"{conf_text}\n" f"Sparsity Score: {sparsity_display:.4f}",
                        fontsize=12,
                        fontweight="bold",
                    )
                else:
                    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
                    factual_np = factual_vis.cpu().numpy().flatten()
                    cf_np = cf_vis.cpu().numpy().flatten()
                    x_range = range(len(factual_np))

                    axes[0].bar(x_range, factual_np, color="#3498db")
                    axes[0].set_title("Factual", fontweight="bold")

                    axes[1].bar(x_range, cf_np, color="#2ecc71")
                    axes[1].set_title("Counterfactual", fontweight="bold")

                    fig.suptitle(
                        f"{conf_text}\n" f"Sparsity Score: {sparsity_display:.4f}",
                        fontsize=12,
                        fontweight="bold",
                    )

                plt.tight_layout()
                collage_path = os.path.join(collage_dir, f"{vi:07d}_sparsity.png")
                plt.savefig(collage_path, dpi=150)
            finally:
                plt.close(fig)

        plt.close("all")
        cprint(
            f"Saved {len(valid_sample_indices)} sparsity collages to {collage_dir}",
            self.explainer_config.tracking_level,
            2,
        )

    def _generate_diversity_collages(
        self,
        explanations_dict,
        active_cluster_indices,
        valid_sample_indices,
        cosine_similarities,
        base_dir,
    ):
        """
        Generate collages for latent diversity visualization.
        Each collage contains: factual, cf1, cf2, target end confidences, diversity score.
        """
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        collage_dir = os.path.join(base_dir, "latent_diversity_collages")
        Path(collage_dir).mkdir(parents=True, exist_ok=True)

        c0 = active_cluster_indices[0]
        c1 = active_cluster_indices[1]
        conf_key0 = "cluster_confidence" + str(c0)
        conf_key1 = "cluster_confidence" + str(c1)
        dist_conf_key0 = "cluster_confidence_distilled" + str(c0)
        dist_conf_key1 = "cluster_confidence_distilled" + str(c1)

        for vi, (sample_idx, cos_sim) in enumerate(
            zip(valid_sample_indices, cosine_similarities)
        ):
            factual = explanations_dict["x_list"][sample_idx]
            cf0 = explanations_dict["clusters" + str(c0)][sample_idx]
            cf1_sample = explanations_dict["clusters" + str(c1)][sample_idx]

            # Get target end confidences for both clusters (original predictor)
            if conf_key0 in explanations_dict:
                target_conf0 = float(explanations_dict[conf_key0][sample_idx])
            else:
                target_conf0 = float(
                    explanations_dict["y_target_end_confidence_list"][sample_idx]
                )

            if conf_key1 in explanations_dict:
                target_conf1 = float(explanations_dict[conf_key1][sample_idx])
            else:
                target_conf1 = float(
                    explanations_dict["y_target_end_confidence_list"][sample_idx]
                )

            # Get target end confidences for both clusters (distilled predictor)
            if dist_conf_key0 in explanations_dict:
                target_conf0_distilled = float(
                    explanations_dict[dist_conf_key0][sample_idx]
                )
            elif "y_target_end_confidence_distilled_list" in explanations_dict:
                target_conf0_distilled = float(
                    explanations_dict["y_target_end_confidence_distilled_list"][
                        sample_idx
                    ]
                )
            else:
                target_conf0_distilled = None

            if dist_conf_key1 in explanations_dict:
                target_conf1_distilled = float(
                    explanations_dict[dist_conf_key1][sample_idx]
                )
            elif "y_target_end_confidence_distilled_list" in explanations_dict:
                target_conf1_distilled = float(
                    explanations_dict["y_target_end_confidence_distilled_list"][
                        sample_idx
                    ]
                )
            else:
                target_conf1_distilled = None

            diversity_score = 1.0 - float(torch.abs(cos_sim))

            # Build subtitle strings for each CF
            cf1_subtitle = f"orig: {target_conf0:.4f}"
            if target_conf0_distilled is not None:
                cf1_subtitle += f", dist: {target_conf0_distilled:.4f}"
            cf2_subtitle = f"orig: {target_conf1:.4f}"
            if target_conf1_distilled is not None:
                cf2_subtitle += f", dist: {target_conf1_distilled:.4f}"

            is_image = len(factual.shape) >= 3 and factual.shape[0] in [1, 3]
            if is_image and hasattr(self.val_dataset, "project_to_pytorch_default"):
                factual_vis = self.val_dataset.project_to_pytorch_default(factual)
                cf0_vis = self.val_dataset.project_to_pytorch_default(cf0)
                cf1_vis = self.val_dataset.project_to_pytorch_default(cf1_sample)
            else:
                factual_vis = factual
                cf0_vis = cf0
                cf1_vis = cf1_sample

            try:
                if is_image:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(
                        factual_vis.permute(1, 2, 0).cpu().numpy().clip(0, 1)
                    )
                    axes[0].axis("off")
                    axes[0].set_title("Factual", fontweight="bold")

                    axes[1].imshow(cf0_vis.permute(1, 2, 0).cpu().numpy().clip(0, 1))
                    axes[1].axis("off")
                    axes[1].set_title(f"CF 1 ({cf1_subtitle})", fontweight="bold")

                    axes[2].imshow(cf1_vis.permute(1, 2, 0).cpu().numpy().clip(0, 1))
                    axes[2].axis("off")
                    axes[2].set_title(f"CF 2 ({cf2_subtitle})", fontweight="bold")

                    fig.suptitle(
                        f"Diversity Score: {diversity_score:.4f}",
                        fontsize=12,
                        fontweight="bold",
                    )
                else:
                    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
                    factual_np = factual_vis.cpu().numpy().flatten()
                    cf0_np = cf0_vis.cpu().numpy().flatten()
                    cf1_np = cf1_vis.cpu().numpy().flatten()
                    x_range = range(len(factual_np))

                    axes[0].bar(x_range, factual_np, color="#3498db")
                    axes[0].set_title("Factual", fontweight="bold")

                    axes[1].bar(x_range, cf0_np, color="#2ecc71")
                    axes[1].set_title(f"CF 1 ({cf1_subtitle})", fontweight="bold")

                    axes[2].bar(x_range, cf1_np, color="#e67e22")
                    axes[2].set_title(f"CF 2 ({cf2_subtitle})", fontweight="bold")

                    fig.suptitle(
                        f"Diversity Score: {diversity_score:.4f}",
                        fontsize=12,
                        fontweight="bold",
                    )

                plt.tight_layout()
                collage_path = os.path.join(collage_dir, f"{vi:07d}_diversity.png")
                plt.savefig(collage_path, dpi=150)
            finally:
                plt.close(fig)

        plt.close("all")
        cprint(
            f"Saved {len(valid_sample_indices)} diversity collages to {collage_dir}",
            self.explainer_config.tracking_level,
            2,
        )

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
        """Collect free-text human feedback on counterfactual collages.

        Serves a small Flask app in a background thread that shows the
        collages one after another and records what the annotator types. The
        collages are copied into a freshly created ``static/`` directory next
        to the working directory so that Flask can serve them; the port is
        ``explainer_config.port``, incremented until a free one is found. The
        call blocks (polling once a second with a progress bar) until as many
        feedback strings as collages have arrived.

        Parameters
        ----------
        collage_path_list : list of str
            Paths of the collage images to show, in order.
        y_source_list, y_target_list : list, optional
            Source and target classes of each collage; not used here, they
            are passed on to :meth:`visualize_interpretations`.
        **kwargs
            Ignored; keeps the signature interchangeable with the automated
            teachers.

        Returns
        -------
        list of str
            One feedback string per collage. The same lines are written to
            ``<explanations_dir>/feedback.txt``.

        Raises
        ------
        ImportError
            If the optional ``flask`` dependency is not installed.
        """
        flask = require("flask", "web", "the interactive feedback web app")
        Flask = flask.Flask
        render_template = flask.render_template
        request = flask.request

        # A per-instance temporary directory for the collages the browser is
        # served. This used to delete a *relative* ``static`` folder, i.e. one
        # in whatever the caller's working directory happened to be - a library
        # must not do that.
        self.static_dir = tempfile.mkdtemp(prefix="peal_feedback_")
        self.port = self.explainer_config.port
        while is_port_in_use(self.port):
            _log.info("%s", "port " + str(self.port) + " is occupied!")
            self.port += 1

        _log.info("%s", "Start explainer loop!")
        #
        # host_name = "localhost"
        host_name = "0.0.0.0"
        app = Flask(
            "feedback_loop", static_folder=self.static_dir, static_url_path="/static"
        )

        self.data = DataStore()
        self.data.i = 0
        self.data.collage_paths = []
        self.data.feedback = []

        app.config.UPLOAD_FOLDER = self.static_dir

        @app.route("/", methods=["GET", "POST"])
        def index():
            """Flask view showing the next collage and storing the answer.

            On POST the submitted text is appended to ``self.data.feedback``;
            both methods then advance ``self.data.i`` and render the next
            collage, or ``information.html`` once none are left.
            """
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
        _log.info("%s", "Feedback GUI is active on localhost:" + str(self.port))

        collage_paths_static = []
        for path in collage_path_list:
            # Copy into the served directory; the list handed to the template
            # holds the URL path, which Flask maps onto static_folder.
            name = path.split("/")[-1]
            shutil.copy(path, os.path.join(self.static_dir, name))
            collage_path_static = os.path.join("static", name)

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
        """Turn the collected feedback into one bar chart per class pair.

        For every unordered pair of classes the feedback strings of all
        explanations that go from one to the other (in either direction) are
        counted, and the histogram is written as
        ``<explanations_dir>/interpretations/<source>vs<target>``.

        Parameters
        ----------
        feedback : list of str or str
            The feedback strings, or the path of a file with one per line.
        y_source_list, y_target_list : list of int
            Source and target class of each explanation, aligned with
            ``feedback``.

        Returns
        -------
        None
            The result is written to disk.
        """
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
