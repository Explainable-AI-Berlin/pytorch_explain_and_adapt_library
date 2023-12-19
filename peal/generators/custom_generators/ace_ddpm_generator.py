import os
import types
import shutil
import copy
import torch
import io
import blobfile as bf

from mpi4py import MPI
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor

from peal.generators.interfaces import EditCapableGenerator
from peal.data.datasets import Image2ClassDataset
from peal.global_utils import load_yaml_config, embed_numberstring
from run_ace import main as ace_main
from ace.guided_diffusion import dist_util, logger
from ace.guided_diffusion.resample import create_named_schedule_sampler
from ace.guided_diffusion.script_util import (
    create_model_and_diffusion,
)
from ace.guided_diffusion.train_util import TrainLoop
from peal.data.dataset_factory import get_datasets
from peal.data.dataloaders import get_dataloader

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


class AceDDPMAdaptor(EditCapableGenerator):
    def __init__(self, config, dataset=None, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.classifier_dataset = dataset

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

    def sample_x(self, batch_size=1):
        return self.diffusion.p_sample_loop(
            self.model, [batch_size] + self.classifier_dataset.config.input_size
        )

    def train_model(
            self,
            dataset_train,
            training_config="<PEAL_BASE>/configs/training/train_ddpm.yaml",
    ):
        training_config = load_yaml_config(training_config)
        args = types.SimpleNamespace(**training_config)
        shutil.rmtree(self.model_dir, ignore_errors=True)

        dist_util.setup_dist(args.gpus)
        logger.configure(dir=self.model_dir)

        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, self.diffusion
        )

        logger.log("creating data loader...")
        data = iter(
            get_dataloader(
                dataset_train,
                mode="train",
                batch_size=args.batch_size,
                training_config={"steps_per_epoch": training_config.max_steps},
            )
        )

        logger.log("training...")
        TrainLoop(
            model=self.model,
            diffusion=self.diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            model_dir=self.model_dir,
        ).run_loop()

    def edit(
            self,
            x_in: torch.Tensor,
            target_confidence_goal: float,
            source_classes: torch.Tensor,
            target_classes: torch.Tensor,
            classifier: nn.Module,
            pbar=None,
            mode="",
    ):
        shutil.rmtree(self.data_dir, ignore_errors=True)
        shutil.rmtree(self.counterfactual_path, ignore_errors=True)

        #x_in = self.dataset.project_to_pytorch_default(x_in)
        self.classifier_dataset.serialize_dataset(
            output_dir=self.data_dir,
            x_list=x_in,
            y_list=target_classes,
            sample_names=list(
                map(lambda x: embed_numberstring(str(x)) + ".jpg", range(x_in.shape[0]))
            ),
        )

        args = copy.deepcopy(self.config)
        dataset_config = copy.deepcopy(self.config.data)
        dataset_config['split'] = [0.0, 1.0]
        dataset_config['num_samples'] = len(x_in)
        dataset_config['output_type'] = "singleclass"
        dataset_config['confounding_factors'] = None
        dataset_config['dataset_path'] = self.data_dir
        dataset_config['dataset_class'] = None
        dataset_config['invariances'] = []
        dataset_config['crop_size'] = None
        dataset_config['normalization'] = None
        dataset_config['has_hints'] = False
        dataset = get_datasets(dataset_config)[1]
        # args.dataset = Image2ClassDataset(
        #    root_dir=self.data_dir,
        #    mode=None,
        #    config=copy.deepcopy(dataset_config),
        #    transform=self.dataset.transform,
        # )
        args.dataset = dataset
        args.model_path = os.path.join(self.model_dir, "final.pt")
        args.classifier = classifier
        args.diffusion = self.diffusion
        args.model = self.model
        args.output_path = self.counterfactual_path
        args.batch_size = x_in.shape[0]
        # args.target_classes = target_classes
        #import pdb;
        #pdb.set_trace()
        if self.config.method == "ace":
            # TODO this does not use the target_classes yet!!!
            ace_main(args=args)
            ending = ".png"

        x_counterfactuals = []
        x_list = []
        base_path = os.path.join(
            self.counterfactual_path,
            "Results",
            self.config.exp_name,
        )
        if self.config.method == "ace":
            base_path = os.path.join(base_path, "explanation")

        for i in range(x_in.shape[0]):
            path_correct = os.path.join(
                base_path, "CC", "CCF", "CF", f"{embed_numberstring(str(i))}" + ending
            )
            path_correct2 = os.path.join(
                base_path, "CC", "ICF", "CF", f"{embed_numberstring(str(i))}" + ending
            )
            path_incorrect = os.path.join(
                base_path, "IC", "CCF", "CF", f"{embed_numberstring(str(i))}" + ending
            )
            path_incorrect2 = os.path.join(
                base_path, "IC", "ICF", "CF", f"{embed_numberstring(str(i))}" + ending
            )
            if os.path.exists(path_correct):
                path_counterfactual = path_correct

            elif os.path.exists(path_correct2):
                path_counterfactual = path_correct2

            elif os.path.exists(path_incorrect):
                path_counterfactual = path_incorrect

            elif os.path.exists(path_incorrect2):
                path_counterfactual = path_incorrect2

            else:
                print("No counterfactual found for image " + str(i))
                import pdb

                pdb.set_trace()

            # x_counterfactuals.append(torchvision.io.read_image(path))
            x_counterfactuals.append(ToTensor()(Image.open(path_counterfactual)))
            path_correct = os.path.join(
                self.counterfactual_path,
                "Original",
                "Correct",
                f"{embed_numberstring(str(i))}" + ending,
            )
            path_incorrect = os.path.join(
                self.counterfactual_path,
                "Original",
                "Incorrect",
                f"{embed_numberstring(str(i))}" + ending,
            )
            if os.path.exists(path_correct):
                path_original = path_correct

            elif os.path.exists(path_incorrect):
                path_original = path_incorrect

            else:
                print("No original image found " + str(i))
                import pdb

                pdb.set_trace()

            x_list.append(ToTensor()(Image.open(path_original)))

        x_counterfactuals = torch.stack(x_counterfactuals)

        #x_counterfactuals = self.dataset.project_from_pytorch_default(x_counterfactuals)
        device = [p for p in classifier.parameters()][0].device
        preds = torch.nn.Softmax(dim=-1)(
            classifier(x_counterfactuals.to(device)).detach().cpu()
        )
        # fix this later
        #original_class = torch.nn.Softmax(dim=-1)(classifier(self.dataset.project_from_pytorch_default(x_in).to(device)).detach().cpu()).argmax(dim=-1)
        #target_class_classifier = torch.nn.Softmax(dim=-1)(classifier(self.dataset.project_from_pytorch_default(x_in).to(device)).detach().cpu()).argmin(dim=-1)

        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]
        #x_counterfactuals = self.dataset.project_to_pytorch_default(x_counterfactuals)
        #x_list = list(map(lambda x: self.classifier_dataset.project_from_pytorch_default(x), x_list))
        #import pdb; pdb.set_trace()
        return (
            list(x_counterfactuals),
            list(x_in - x_counterfactuals),
            list(y_target_end_confidence),
            list(x_list),
        )
