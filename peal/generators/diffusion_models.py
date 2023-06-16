import torch

from pathlib import Path
from torch import nn

from peal.generators.interfaces import EditCapableGenerator
from peal.utils import load_yaml_config
from DiME.main import main

class DDPM(EditCapableGenerator):
    def __init__(self, config):
        super().__init__()
        self.config = load_yaml_config(config)

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        target_classes: torch.Tensor,
        classifier: nn.Module,
    ):
        # TODO: Implement this
        Path.mkdir(self.config["output_dir"], parents=True, exist_ok=True)
        torch.save(classifier, self.config["classifier_path"])
        main(args=self.config)
