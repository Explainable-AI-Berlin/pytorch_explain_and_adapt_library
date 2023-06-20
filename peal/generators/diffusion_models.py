import torch
import torchvision
import os
import types
import shutil
import copy

from pathlib import Path
from torch import nn

from peal.generators.interfaces import EditCapableGenerator
from peal.data.datasets import Image2ClassDataset
from peal.utils import load_yaml_config
from dime2.main import main


class DDPM(EditCapableGenerator):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = load_yaml_config(config)
        self.dataset = dataset

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
        shutil.rmtree(self.config["base_path"], ignore_errors=True)
        self.dataset.serialize_dataset(
            output_dir=self.config["data_dir"],
            x_list=x_in,
            y_list=target_classes,
            sample_names=list(map(lambda x: str(x) + ".png", range(x_in.shape[0]))),
        )

        args = types.SimpleNamespace(**self.config)
        args.dataset = Image2ClassDataset(
            root_dir=self.config["data_dir"],
            mode=None,
            config=copy.deepcopy(self.dataset.config),
            transform=self.dataset.transform,
        )
        args.classifier = classifier
        main(args=args)
        x_counterfactuals = []
        base_path = os.path.join(
            self.config["output_path"],
            "Results",
            self.config["exp_name"],
        )
        for i in range(x_in.shape[0]):
            path_correct = os.path.join(base_path, "CC", "CCF", "CF", f"{i}.png")
            path_incorrect = os.path.join(base_path, "IC", "CCF", "CF", f"{i}.png")
            if os.path.exists(path_correct):
                path = path_correct

            elif os.path.exists(path_incorrect):
                path = path_incorrect

            else:
                print("No counterfactual found for image " + str(i))
                import pdb

                pdb.set_trace()

            x_counterfactuals.append(torchvision.io.read_image(path))

        x_counterfactuals = torch.stack(x_counterfactuals)
        x_counterfactuals = self.dataset.project_from_pytorch_default(x_counterfactuals)
        return {
            "x_counterfactual_list": list(x_counterfactuals),
            "z_difference_list": list(x_in - x_counterfactuals),
            "y_target_end_confidence_list": list(
                classifier(x_counterfactuals)[~target_classes]
            ),
        }
