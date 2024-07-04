import os
import random
import types
import shutil
import copy
from pathlib import Path

import torch
import io
import blobfile as bf

from mpi4py import MPI
from torch import nn
from types import SimpleNamespace

from torch.utils.tensorboard import SummaryWriter

from peal.dependencies.time.get_predictions import get_predictions
from peal.generators.interfaces import EditCapableGenerator
from peal.global_utils import load_yaml_config
from peal.dependencies.ace.run_ace import main as ace_main
from peal.dependencies.ace.guided_diffusion import logger
from peal.dependencies.ace.guided_diffusion import dist_util
from peal.dependencies.ace.guided_diffusion.resample import (
    create_named_schedule_sampler,
)
from peal.dependencies.ace.guided_diffusion.script_util import (
    create_model_and_diffusion,
)
from peal.dependencies.ace.guided_diffusion.train_util import TrainLoop
from peal.data.dataloaders import get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import PealDataset
from peal.explainers.counterfactual_explainer import ACEConfig
from peal.training.loggers import log_images_to_writer

from typing import Union

from peal.generators.interfaces import GeneratorConfig
from peal.data.datasets import DataConfig
from peal.training.trainers import ModelTrainer, PredictorConfig


class DDPMConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str = "DDPM"
    """
    The path where the generator is stored.
    """
    base_path: str = "peal_runs/ddpm"
    """
    The config of the data.
    """
    data: DataConfig = DataConfig()
    """
    The number of channels
    """
    num_channels: int = 128
    image_size: Union[int, type(None)] = None
    num_res_blocks: int = 2
    num_heads: int = 4
    num_heads_upsample: int = -1
    num_head_channels: int = -1
    attention_resolutions: str = "32,16,8"
    channel_mult: str = ""
    dropout: float = 0.0
    class_cond: bool = False
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    use_fp16: bool = False
    use_new_attention_order: bool = False
    schedule_sampler: str = "uniform"
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_anneal_steps: int = 0
    batch_size: int = 1
    microbatch: int = -1  # -1 disables microbatches
    ema_rate: str = "0.9999"  # comma-separated list of EMA values
    log_interval: int = 10
    save_interval: int = 10000
    max_steps: int = 1000000
    resume_checkpoint: str = ""
    fp16_scale_growth: float = 1e-3
    output_path: str = "peal_runs/ddpm/outputs"
    gpus: str = ""
    use_hdf5: bool = False
    learn_sigma : bool = True
    diffusion_steps : int = 1000
    noise_schedule : str = "linear"
    timestep_respacing : str = "50"
    use_kl : bool = False
    predict_xstart : bool = False
    rescale_timesteps : bool = False
    rescale_learned_sigmas : bool = False
    full_args: dict = {}
    x_selection: Union[list, type(None)] = None


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2**30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return torch.load(io.BytesIO(data), **kwargs)


class DDPM(EditCapableGenerator):
    def __init__(self, config, model_dir=None, device="cpu", classifier_dataset=None):
        super().__init__()
        self.classifier_distilled = None
        self.config = load_yaml_config(config)

        self.dataset = get_datasets(self.config.data)[0]

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data_test")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals_test")

        self.model, self.diffusion = create_model_and_diffusion(**self.config.__dict__)
        self.model.to(device)
        self.model_path = os.path.join(self.model_dir, "final.pt")
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                load_state_dict(self.model_path, map_location=device)
            )

    def sample_x(self, batch_size=1, renormalize=True):
        sample = self.diffusion.p_sample_loop(
            self.model, [batch_size] + self.config.data.input_size
        )
        if renormalize:
            sample = self.dataset.project_to_pytorch_default(sample)

        return sample

    def train_model(
        self,
    ):
        shutil.rmtree(self.model_dir, ignore_errors=True)

        # dist_util.setup_dist(self.config.gpus)
        logger.configure(dir=self.model_dir)

        schedule_sampler = create_named_schedule_sampler(
            self.config.schedule_sampler, self.diffusion
        )

        if not self.config.x_selection is None:
            self.dataset.task_config = SimpleNamespace(
                **{"x_selection": self.config.x_selection}
            )
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print("self.dataset.task_config1")
            print(self.dataset.task_config)

        logger.log("creating data loader...")
        dataloader = get_dataloader(
            self.dataset,
            mode="train",
            batch_size=self.config.batch_size,
            training_config=types.SimpleNamespace(
                **{"steps_per_epoch": self.config.max_steps}
            ),
        )

        writer = SummaryWriter(os.path.join(self.model_dir, "logs"))
        if not self.config.x_selection is None:
            self.dataset.task_config = SimpleNamespace(
                **{"x_selection": self.config.x_selection}
            )
            print("self.dataset.task_config2")
            print("self.dataset.task_config2")
            print("self.dataset.task_config2")
            print("self.dataset.task_config2")
            print("self.dataset.task_config2")
            print(self.dataset.task_config)

        log_images_to_writer(dataloader, writer, "train")
        data = iter(dataloader)

        logger.log("training...")
        train_loop = TrainLoop(
            model=self.model,
            diffusion=self.diffusion,
            data=data,
            batch_size=self.config.batch_size,
            microbatch=self.config.microbatch,
            lr=self.config.lr,
            ema_rate=self.config.ema_rate,
            log_interval=self.config.log_interval,
            save_interval=self.config.save_interval,
            resume_checkpoint=self.config.resume_checkpoint,
            use_fp16=self.config.use_fp16,
            fp16_scale_growth=self.config.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=self.config.weight_decay,
            lr_anneal_steps=self.config.lr_anneal_steps,
            model_dir=self.model_dir,
        )
        train_loop.run_loop(self.config, writer)

    def distill_classifier(self, explainer_config, base_path, classifier, classifier_dataset):
        class_predictions_path = os.path.join(base_path, "explainer", "predictions.csv")
        Path(os.path.join(base_path, "explainer")).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(class_predictions_path):
            classifier_dataset.enable_url()
            prediction_args = types.SimpleNamespace(
                batch_size=32,
                dataset=classifier_dataset,
                classifier=classifier,
                label_path=class_predictions_path,
                partition="train",
                label_query=0,
                max_samples=explainer_config.max_samples,
            )
            get_predictions(prediction_args)
            classifier_dataset.disable_url()

        distilled_dataset_config = copy.deepcopy(classifier_dataset.config)
        distilled_dataset_config.split = [0.9, 1.0]
        distilled_dataset_config.confounding_factors = None
        distilled_dataset_config.confounder_probability = None
        classifier_dataset_train, classifier_dataset_val, _ = get_datasets(
            config=distilled_dataset_config, data_dir=class_predictions_path
        )
        distilled_classifier_config = load_yaml_config(explainer_config.distilled_classifier, PredictorConfig)
        distilled_classifier_config.data = distilled_dataset_config
        explainer_config.distilled_classifier = distilled_classifier_config
        classifier_dataset_train.task_config = explainer_config.distilled_classifier.task
        self.classifier_distilled = copy.deepcopy(classifier)
        distillation_trainer = ModelTrainer(
            config=explainer_config.distilled_classifier,
            model=self.classifier_distilled,
            datasource=(classifier_dataset_train, classifier_dataset_val),
            model_path=os.path.join(
                base_path, "explainer", "distilled_classifier"
            )
        )
        distillation_trainer.fit()

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        classifier: nn.Module,
        explainer_config: ACEConfig,
        classifier_dataset: PealDataset,
        pbar=None,
        mode="",
        base_path="",
    ):
        if not explainer_config.distilled_classifier is None:
            if not os.path.exists(
                os.path.join(base_path, "explainer", "distilled_classifier", "model.cpl")
            ):
                self.distill_classifier(explainer_config, base_path, classifier, classifier_dataset)

            gradient_classifier = self.classifier_distilled

        else:
            gradient_classifier = classifier

        dataset = [
            (
                torch.zeros([len(x_in)], dtype=torch.long),
                x_in,
                [source_classes, target_classes],
            )
        ]
        args = copy.deepcopy(self.config).dict()
        # args = copy.deepcopy(self.config.full_args)
        args = SimpleNamespace(**args)
        args.output_path = self.counterfactual_path
        args.batch_size = x_in.shape[0]
        #
        x_counterfactuals = None
        for idx in range(explainer_config.attempts):
            # a0_0 = a_min + (a_max - a_min) / 2
            # a1_0 = a_min + 0 * (a_max - a_min) / (explainer_config.attempts - 1)
            # a1_1 = a_min + 1 * (a_max - a_min) / (explainer_config.attempts - 1)
            # a2_0 = a_min + 0 * (a_max - a_min) / (explainer_config.attempts - 1)
            # a2_0 = a_min + 1 * (a_max - a_min) / (explainer_config.attempts - 1)
            # a2_0 = a_min + 2 * (a_max - a_min) / (explainer_config.attempts - 1)
            if explainer_config.attempts > 1:
                multiplier = idx / (explainer_config.attempts - 1)

            else:
                multiplier = 0.5

            args.attack_iterations = int(
                explainer_config.attack_iterations
                if not isinstance(explainer_config.attack_iterations, list)
                else int(
                    explainer_config.attack_iterations[0]
                    + (
                        explainer_config.attack_iterations[1]
                        - explainer_config.attack_iterations[0]
                    )
                    * multiplier
                )
            )
            print("args.attack_iterations")
            print(args.attack_iterations)
            args.sampling_time_fraction = float(
                explainer_config.sampling_time_fraction
                if not isinstance(explainer_config.sampling_time_fraction, list)
                else float(
                    explainer_config.sampling_time_fraction[0]
                    + (
                        explainer_config.sampling_time_fraction[1]
                        - explainer_config.sampling_time_fraction[0]
                    )
                    * multiplier
                )
            )
            print("args.sampling_time_fraction")
            print(args.sampling_time_fraction)
            args.dist_l1 = float(
                explainer_config.dist_l1
                if not isinstance(explainer_config.dist_l1, list)
                else explainer_config.dist_l1[0]
                * (
                    (explainer_config.dist_l1[1] / explainer_config.dist_l1[0])
                    ** (1 / (explainer_config.attempts - 1))
                )
                ** idx
            )
            print("args.dist_l1")
            print(args.dist_l1)
            args.dist_l2 = float(
                explainer_config.dist_l2
                if not isinstance(explainer_config.dist_l2, list)
                else explainer_config.dist_l1[0]
                * (
                    (explainer_config.dist_l2[1] / explainer_config.dist_l2[0])
                    ** (1 / (explainer_config.attempts - 1))
                )
                ** idx
            )
            args.sampling_inpaint = float(
                explainer_config.sampling_inpaint
                if not isinstance(explainer_config.sampling_inpaint, list)
                else float(
                    explainer_config.sampling_inpaint[0]
                    - (
                        explainer_config.sampling_inpaint[0]
                        - explainer_config.sampling_inpaint[1]
                    )
                    * multiplier
                )
            )
            print("args.sampling_inpaint")
            print(args.sampling_inpaint)
            args.__dict__.update(
                {
                    k: v
                    for k, v in explainer_config.__dict__.items()
                    if k not in args.__dict__
                }
            )
            args.timestep_respacing = explainer_config.timestep_respacing
            args.dataset = dataset
            args.classifier_dataset = classifier_dataset
            args.generator_dataset = self.dataset
            args.model_path = os.path.join(self.model_dir, "final.pt")
            args.classifier = gradient_classifier
            args.diffusion = self.diffusion
            args.model = self.model
            #
            x_counterfactuals_current = ace_main(args=args)
            x_counterfactuals_current = torch.cat(x_counterfactuals_current, dim=0)

            device = [p for p in classifier.parameters()][0].device
            preds = torch.nn.Softmax(dim=-1)(
                classifier(x_counterfactuals_current.to(device)).detach().cpu()
            )
            print("preds_final: " + str(list(preds)))
            y_target_end_confidence_current = torch.zeros([x_in.shape[0]])
            for i in range(x_in.shape[0]):
                y_target_end_confidence_current[i] = preds[i, target_classes[i]]

            if x_counterfactuals is None:
                x_counterfactuals = x_counterfactuals_current
                y_target_end_confidence = y_target_end_confidence_current

            else:
                for i in range(x_in.shape[0]):
                    if y_target_end_confidence[i] < 0.51:
                        x_counterfactuals[i] = x_counterfactuals_current[i]
                        y_target_end_confidence[i] = y_target_end_confidence_current[i]

            num_successful = torch.sum(y_target_end_confidence >= 0.51).item()
            print("num_successful")
            print(num_successful)
            if num_successful == x_in.shape[0]:
                break

        return (
            list(x_counterfactuals),
            list(x_in - x_counterfactuals),
            list(y_target_end_confidence),
            list(x_in),
        )
