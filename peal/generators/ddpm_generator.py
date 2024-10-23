import os
import random
import types
import shutil
import copy
from datetime import datetime
from pathlib import Path

import torch
import io
import blobfile as bf

from mpi4py import MPI
from torch import nn
from types import SimpleNamespace

from torch.utils.tensorboard import SummaryWriter

from peal.dependencies.ace.core.utils import generate_mask
from peal.dependencies.time.get_predictions import get_predictions
from peal.generators.interfaces import EditCapableGenerator, InvertibleGenerator
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
from peal.training.trainers import ModelTrainer, PredictorConfig, distill_predictor


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
    learn_sigma: bool = True
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    timestep_respacing: str = "50"
    use_kl: bool = False
    predict_xstart: bool = False
    rescale_timesteps: bool = False
    rescale_learned_sigmas: bool = False
    stochastic: bool = True
    x_selection: Union[list, type(None)] = None
    is_trained: bool = False
    best_fid: float = 1e9
    is_loaded: bool = False


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


class DDPM(EditCapableGenerator, InvertibleGenerator):
    def __init__(self, config, model_dir=None, device="cpu", predictor_dataset=None):
        super().__init__()
        self.predictor_distilled = None
        self.config = load_yaml_config(config)

        self.dataset = get_datasets(self.config.data)[0]

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data_test")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals_test")

        self.model, self.diffusion = create_model_and_diffusion(**self.config.__dict__)
        self.device = device
        self.model.to(device)
        self.model_path = os.path.join(self.model_dir, "final.pt")
        if os.path.exists(self.model_path) and self.config.is_trained:
            print('load model!!!')
            self.model.load_state_dict(
                load_state_dict(self.model_path, map_location=device)
            )

        else:
            print('No model weights yet!!!')
            if os.path.exists(self.model_dir):
                import pdb; pdb.set_trace()
                shutil.move(self.model_dir, self.model_dir + "_old" + datetime.now().strftime("%Y%m%d_%H%M%S"))

            Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.config.is_trained = True
        self.noise_fn = torch.randn_like if self.config.stochastic else torch.zeros_like

    def sample_x(self, batch_size=None, renormalize=True):
        if batch_size is None:
            batch_size = self.config.batch_size

        sample = self.diffusion.p_sample_loop(
            self.model, [batch_size] + self.config.data.input_size
        )
        if renormalize:
            sample = self.dataset.project_to_pytorch_default(sample)

        return sample

    def encode(self, x, t=1.0, stochastic=None, num_steps=None):
        if stochastic is None:
            stochastic = self.config.stochastic

        respaced_steps = int(t * int(self.config.timestep_respacing))
        timesteps = list(range(respaced_steps))[::-1]
        def local_forward(x, t, idx, noise, steps, diffusion, model):
            out = diffusion.p_mean_variance(model, x, t, clip_denoised=True)

            x = out["mean"]

            if idx != (steps - 1):
                x += torch.exp(0.5 * out["log_variance"]) * noise

            return x

        for idx, t in enumerate(timesteps):
            t = torch.tensor([t] * x.size(0), device=x.device)

            if idx == 0:
                x = self.diffusion.q_sample(x, t, noise=self.noise_fn(x))

            if hasattr(self, "fix_noise") and self.fix_noise:
                noise = self.noise[idx + 1, ...].unsqueeze(dim=0)

            elif stochastic:
                noise = torch.randn_like(x)

            else:
                noise = torch.zeros_like(x)

            x = local_forward(x, t, idx, noise, respaced_steps, self.diffusion, self.model)

        # TODO why are gradients in ACE scaled???
        #t = torch.tensor([self.steps - 1] * x.size(0), device=x.device)
        return x

    def decode(self, z, t=1.0, stochastic=None, num_steps=None):
        if isinstance(z, list) and len(z) == 1:
            z = z[0]

        if stochastic is None:
            stochastic = self.config.stochastic

        # TODO test decode function via sampling function
        respaced_steps = int(t * int(self.config.timestep_respacing))
        timesteps = list(range(respaced_steps))[::-1]
        for idx, t in enumerate(timesteps):
            t = torch.tensor([t] * z.size(0), device=z.device)

            if idx == 0:
                z = self.diffusion.q_sample(z, t, noise=self.noise_fn(z))

            out = self.diffusion.p_mean_variance(self.model, z, t, clip_denoised=True)

            z = out["mean"]

            if idx != (respaced_steps - 1):
                if self.config.stochastic:
                    z += torch.exp(0.5 * out["log_variance"]) * self.noise_fn(z)

        return z

    def repaint(self, x, pe, inpaint, dilation, t, stochastic):
        respaced_steps = int(t * int(self.config.timestep_respacing))
        indices = list(range(respaced_steps))[::-1]
        x_normalized = self.dataset.project_to_pytorch_default(x)
        pe_normalized = self.dataset.project_to_pytorch_default(pe)
        mask, dil_mask = generate_mask(x_normalized, pe_normalized, dilation)
        boolmask = (dil_mask < inpaint).float()

        noise_fn = torch.randn_like if stochastic else torch.zeros_like

        ce = torch.clone(pe)
        for idx, t in enumerate(indices):
            # filter the with the diffusion model
            t = torch.tensor([t] * ce.size(0), device=ce.device)

            if idx == 0:
                ce = self.diffusion.q_sample(ce, t, noise=noise_fn(ce))

            if inpaint != 0:
                ce = ce * (1 - boolmask) + boolmask * self.diffusion.q_sample(
                    x, t, noise=noise_fn(ce)
                )

            out = self.diffusion.p_mean_variance(self.model, ce, t, clip_denoised=True)

            ce = out["mean"]

            if stochastic and (idx != (respaced_steps - 1)):
                noise = torch.randn_like(ce)
                ce += torch.exp(0.5 * out["log_variance"]) * noise

        ce = ce * (1 - boolmask) + boolmask * x
        return ce, boolmask

    def train_model(
        self,
    ):
        if not self.config.is_trained and os.path.exists(self.model_dir):
            shutil.move(
                self.model_dir,
                self.model_dir + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            )

        self.config.is_trained = True
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

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        predictor: nn.Module,
        explainer_config: ACEConfig,
        predictor_datasets: list,
        pbar=None,
        base_path: str = "",
        mode: str = "",
    ):
        if not self.config.is_trained:
            print('Model not trained yet. Model will be trained now!')
            import pdb; pdb.set_trace()
            self.train_model()
            
        if not explainer_config.distilled_predictor is None:
            distilled_path = os.path.join(
                base_path, "explainer", "distilled_predictor", "model.cpl"
            )
            if not os.path.exists(distilled_path):
                gradient_predictor = distill_predictor(
                    explainer_config, base_path, predictor, predictor_datasets
                )

            else:
                gradient_predictor = torch.load(
                    distilled_path, map_location=self.device
                )

        else:
            gradient_predictor = predictor
            import pdb; pdb.set_trace()

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
            args.predictor_dataset = predictor_datasets[1]
            args.generator_dataset = self.dataset
            args.model_path = os.path.join(self.model_dir, "final.pt")
            args.classifier = gradient_predictor
            args.original_classifier = predictor
            args.diffusion = self.diffusion
            args.model = self.model
            #
            x_counterfactuals_current, histories = ace_main(args=args)
            x_counterfactuals_current = torch.cat(x_counterfactuals_current, dim=0)

            device = [p for p in predictor.parameters()][0].device
            preds = torch.nn.Softmax(dim=-1)(
                predictor(x_counterfactuals_current.to(device)).detach().cpu()
            )
            print(
                "preds_final: "
                + str(
                    [
                        float(preds[i][target_classes[i]])
                        for i in range(len(target_classes))
                    ]
                )
            )
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
            list(histories),
        )
