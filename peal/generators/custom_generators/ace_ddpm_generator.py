import os
import random
import types
import shutil
import copy
import torch
import io
import blobfile as bf

from mpi4py import MPI
from torch import nn
from types import SimpleNamespace

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
from peal.data.dataset_interfaces import PealDataset
from peal.configs.explainers.ace_config import ACEConfig


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
    def __init__(self, config, model_dir=None, device="cpu"):
        super().__init__()
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
            self.model.load_state_dict(load_state_dict(self.model_path, map_location=device))

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

        dist_util.setup_dist(self.config.gpus)
        logger.configure(dir=self.model_dir)

        schedule_sampler = create_named_schedule_sampler(
            self.config.schedule_sampler, self.diffusion
        )

        logger.log("creating data loader...")
        data = iter(
            get_dataloader(
                self.dataset,
                mode="train",
                batch_size=self.config.batch_size,
                training_config={"steps_per_epoch": self.config.max_steps},
            )
        )

        logger.log("training...")
        TrainLoop(
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
        ).run_loop()

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
    ):
        dataset = [
            (
                torch.zeros([len(x_in)], dtype=torch.long),
                x_in,
                [source_classes, target_classes],
            )
        ]
        args = copy.deepcopy(self.config).dict()
        args = SimpleNamespace(**args)
        args.output_path = self.counterfactual_path
        args.batch_size = x_in.shape[0]
        #
        args.attack_iterations = int(
            explainer_config.attack_iterations
            if not isinstance(explainer_config.attack_iterations, list)
            else random.randint(
                explainer_config.attack_iterations[0],
                explainer_config.attack_iterations[1],
            )
        )
        args.sampling_time_fraction = float(
            explainer_config.sampling_time_fraction
            if not isinstance(explainer_config.sampling_time_fraction, list)
            else random.uniform(
                explainer_config.sampling_time_fraction[0],
                explainer_config.sampling_time_fraction[1],
            )
        )
        args.dist_l1 = float(
            explainer_config.dist_l1
            if not isinstance(explainer_config.dist_l1, list)
            else random.uniform(
                explainer_config.dist_l1[0],
                explainer_config.dist_l1[1],
            )
        )
        args.dist_l2 = float(
            explainer_config.dist_l2
            if not isinstance(explainer_config.dist_l2, list)
            else random.uniform(
                explainer_config.dist_l2[0],
                explainer_config.dist_l2[1],
            )
        )
        args.sampling_inpaint = float(
            explainer_config.sampling_inpaint
            if not isinstance(explainer_config.sampling_inpaint, list)
            else random.uniform(
                explainer_config.sampling_inpaint[0],
                explainer_config.sampling_inpaint[1],
            )
        )
        args.sampling_dilation = int(
            explainer_config.sampling_dilation
            if not isinstance(explainer_config.sampling_dilation, list)
            else random.randint(
                explainer_config.sampling_dilation[0],
                explainer_config.sampling_dilation[1],
            )
        )
        args.timestep_respacing = str(
            explainer_config.timestep_respacing
            if not isinstance(explainer_config.timestep_respacing, list)
            else random.uniform(
                explainer_config.timestep_respacing[0],
                explainer_config.timestep_respacing[1],
            )
        )
        args.__dict__.update({k: v for k, v in explainer_config.__dict__.items() if k not in args.__dict__})
        print("args.sampling_inpaint")
        print(args.sampling_inpaint)
        args.dataset = dataset
        args.classifier_dataset = classifier_dataset
        args.generator_dataset = self.dataset
        args.model_path = os.path.join(self.model_dir, "final.pt")
        args.classifier = classifier
        args.diffusion = self.diffusion
        args.model = self.model
        #
        x_counterfactuals = ace_main(args=args)
        x_counterfactuals = torch.cat(x_counterfactuals, dim=0)

        device = [p for p in classifier.parameters()][0].device
        preds = torch.nn.Softmax(dim=-1)(
            classifier(x_counterfactuals.to(device)).detach().cpu()
        )

        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]

        return (
            list(x_counterfactuals),
            list(x_in - x_counterfactuals),
            list(y_target_end_confidence),
            list(x_in),
        )
