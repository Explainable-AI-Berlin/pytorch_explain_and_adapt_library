import torch

from torch import nn

from peal.generators.interfaces import EditCapableGenerator
from DiME.main import main

class DDPM(EditCapableGenerator):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        target_classes: torch.Tensor,
        classifier: nn.Module,
    ):
        # TODO: Implement this
        main(args=self.config)
