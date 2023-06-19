import torch

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
    ):
        # TODO: Implement this
        Path.mkdir(self.config["output_dir"], parents=True, exist_ok=True)
        torch.save(classifier, self.config["classifier_path"])
        x_canonized = self.dataset.project_to_pytorch_default(x_in)
        for i in range(x_canonized.shape[0]):
            torchvision.utils.save_image(x_canonized[i], )

        main(args=self.config)

