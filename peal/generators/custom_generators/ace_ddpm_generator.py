import torch
import torchvision
import os
import types
import shutil
import copy

from pathlib import Path
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor

from peal.generators.interfaces import EditCapableGenerator
from peal.data.datasets import Image2ClassDataset
from peal.global_utils import load_yaml_config, embed_numberstring
from run_dime import main as dime_main
from run_ace import main as ace_main
from ace.guided_diffusion import dist_util, logger
from ace.guided_diffusion.resample import create_named_schedule_sampler
from ace.guided_diffusion.script_util import (
    create_model_and_diffusion,
)
from ace.guided_diffusion.train_util import TrainLoop
from peal.data.dataset_factory import get_datasets
from peal.data.dataloaders import get_dataloader
from dime2.core.dist_util import (
    load_state_dict,
)


class AceDDPMAdaptor(EditCapableGenerator):
    def __init__(self, config, dataset=None, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.dataset = (
            dataset if not dataset is None else get_datasets(self.config.data)[0]
        )
        if not self.config.image_size is None:
            self.config.image_size = self.dataset.config.input_size[-1]

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals")

        self.model, self.diffusion = create_model_and_diffusion(**self.config.__dict__)
        self.model.to(device)
        self.model_path = os.path.join(self.model_dir, "final.pt")
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                load_state_dict(self.model_path, map_location=device)
            )

    def sample_x(self, batch_size=1):
        return self.diffusion.p_sample_loop(
            self.model, [batch_size] + self.dataset.config.input_size
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
        self.dataset.serialize_dataset(
            output_dir=self.data_dir,
            x_list=x_in,
            y_list=target_classes,
            sample_names=list(
                map(lambda x: embed_numberstring(str(x)) + ".jpg", range(x_in.shape[0]))
            ),
        )

        args = copy.deepcopy(self.config)
        args.dataset = Image2ClassDataset(
            root_dir=self.data_dir,
            mode=None,
            config=copy.deepcopy(self.dataset.config),
            transform=self.dataset.transform,
        )
        args.model_path = os.path.join(self.model_dir, "final.pt")
        args.classifier = classifier
        args.diffusion = self.diffusion
        args.model = self.model
        args.output_path = self.counterfactual_path
        args.batch_size = x_in.shape[0]
        if self.config.method == "ace":
            ace_main(args=args)
            ending = ".png"

        elif self.config.method == "dime":
            dime_main(args=args)
            ending = ".jpg"

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
        x_counterfactuals = self.dataset.project_from_pytorch_default(x_counterfactuals)
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
            list(x_list),
        )
