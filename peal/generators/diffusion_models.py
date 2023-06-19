import torch
import torchvision
import os

from pathlib import Path
from torch import nn

from peal.generators.interfaces import EditCapableGenerator
from peal.utils import load_yaml_config
from DiME.main import main


class DDPM(EditCapableGenerator):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = load_yaml_config(config)
        self.dataset = dataset

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        target_classes: torch.Tensor,
        classifier: nn.Module,
        pbar=None,
        mode="",
    ):
        Path.mkdir(self.config["output_dir"], parents=True, exist_ok=True)
        torch.save(classifier, self.config["classifier_path"])
        x_canonized = self.dataset.project_to_pytorch_default(x_in)
        data_str = str(x_in.shape[0]) + "\n" + "ImgPath Label"
        for i in range(x_canonized.shape[0]):
            torchvision.utils.save_image(
                x_canonized[i],
                os.path.join(self.config["data_dir"], "img_align_celeba", f"{i}.png"),
            )
            data_str += "\n" + f"{i}.png" + " " + str(target_classes[i].item())

        with open(
            os.path.join(self.config["data_dir"], "list_attr_celeba.txt"), "w"
        ) as f:
            f.write(data_str)

        main(args=self.config)
        x_counterfactuals = []
        base_path = os.path.join(
            self.config["output_dir"],
            "Results",
            self.config["exp_name"],
        )
        for i in range(x_canonized.shape[0]):
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
