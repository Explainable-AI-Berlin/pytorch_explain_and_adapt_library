import torch
import os
import shutil
import copy
import types

from torch import nn

from peal.generators.interfaces import EditCapableGenerator
from peal.data.datasets import Image2ClassDataset
from peal.global_utils import load_yaml_config, embed_numberstring
from run_asyrp import asyrp_main
from peal.data.dataset_factory import get_datasets

def recursive_dict_to_namespace(d):
    for key in d.keys():
        if isinstance(d[key], dict):
            d[key] = recursive_dict_to_namespace(d[key])

    return types.SimpleNamespace(**d)


class AsyrpDDIMAdaptor(EditCapableGenerator):
    def __init__(self, config, dataset=None, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.config.config = recursive_dict_to_namespace(self.config.config)
        self.dataset = (
            dataset if not dataset is None else get_datasets(self.config.data)[0]
        )

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals")

    def sample_x(self, batch_size=1):
        return None

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
        """
        This function edits the input image to match the target confidence goal.
        Args:
            x_in:
            target_confidence_goal:
            source_classes:
            target_classes:
            classifier:
            pbar:
            mode:

        Returns:

        """
        shutil.rmtree(self.data_dir, ignore_errors=True)
        shutil.rmtree(self.counterfactual_path, ignore_errors=True)
        os.makedirs(self.counterfactual_path, exist_ok=True)
        self.dataset.serialize_dataset(
            output_dir=self.data_dir,
            x_list=x_in,
            y_list=target_classes,
            sample_names=list(
                map(lambda x: embed_numberstring(str(x)) + ".jpg", range(x_in.shape[0]))
            ),
        )

        args = copy.deepcopy(self.config)
        dataset = Image2ClassDataset(
            root_dir=self.data_dir,
            mode=None,
            config=copy.deepcopy(self.dataset.config),
            transform=self.dataset.transform,
        )
        args.datasets = dataset, dataset
        args.model_path = os.path.join(self.model_dir, "final.pt")
        args.classifier = classifier
        args.exp = self.counterfactual_path
        args.batch_size = x_in.shape[0]
        x_counterfactuals = asyrp_main(args=args)
        x_list = x_in

        x_counterfactuals = torch.stack(x_counterfactuals)
        x_counterfactuals = self.dataset.project_from_pytorch_default(x_counterfactuals)
        device = [p for p in classifier.parameters()][0].device
        try:
            preds = torch.nn.Softmax(dim=-1)(
                classifier(x_counterfactuals.to(device)).detach().cpu()
            )

        except Exception as e:
            import pdb; pdb.set_trace()

        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]

        return (
            list(x_counterfactuals),
            list(x_in - x_counterfactuals),
            list(y_target_end_confidence),
            list(x_list),
        )
