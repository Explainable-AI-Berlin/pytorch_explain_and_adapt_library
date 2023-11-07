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
from peal.configs.data.data_template import DataConfig


def recursive_dict_to_namespace(d):
    for key in d.keys():
        if isinstance(d[key], dict):
            d[key] = recursive_dict_to_namespace(d[key])

    return types.SimpleNamespace(**d)


class AsyrpDDIMAdaptor(EditCapableGenerator):
    def __init__(self, config, dataset, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.config.config = recursive_dict_to_namespace(self.config.config)
        self.classifier_dataset = dataset
        if not hasattr(self.config, "data"):
            self.config.data = self.classifier_dataset.config

        else:
            self.config.data = DataConfig(**self.config.data)

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
        dataset_config.split = [0.0, 1.0]
        dataset_config.num_samples = len(x_in)
        dataset_config.output_type = "singleclass"
        dataset_config.confounding_factors = None
        dataset_config.dataset_path = self.data_dir
        dataset_config.dataset_class = None
        dataset = get_datasets(dataset_config)[1]
        renormalize_classifier_to_generator = (
            lambda x: dataset.project_from_pytorch_default(
                self.classifier_dataset.project_to_pytorch_default(x)
            )
        )
        renormalize_generator_to_classifier = (
            lambda x: self.classifier_dataset.project_from_pytorch_default(
                dataset.project_to_pytorch_default(x)
            )
        )
        print(f"x_in: [{x_in.min()}, {x_in.max()}], {x_in.shape}")
        print(f"x_in: [{x_in.min()}, {x_in.max()}]")
        print(f"x_in: [{x_in.min()}, {x_in.max()}]")
        test_dataset = list(map(renormalize_classifier_to_generator, list(x_in)))
        print(f"test_dataset: [{torch.stack(test_dataset).min()}, {torch.stack(test_dataset).max()}], {torch.stack(test_dataset).shape}")
        print(f"test_dataset: [{torch.stack(test_dataset).min()}, {torch.stack(test_dataset).max()}]")
        print(f"test_dataset: [{torch.stack(test_dataset).min()}, {torch.stack(test_dataset).max()}]")
        test_dataset = list(zip(test_dataset, target_classes))
        args.datasets = dataset, test_dataset
        args.model_path = os.path.join(self.model_dir, "final.pt")
        args.classifier = lambda x : classifier(renormalize_generator_to_classifier(x))
        args.exp = self.counterfactual_path
        args.batch_size = x_in.shape[0]
        x_counterfactuals = asyrp_main(args=args)
        x_list = x_in

        x_counterfactuals = torch.stack(x_counterfactuals)
        print(f"x_counterfactuals1: [{x_counterfactuals.min()}, {x_counterfactuals.max()}], {x_counterfactuals.shape}")
        print(f"x_counterfactuals1: [{x_counterfactuals.min()}, {x_counterfactuals.max()}]")
        print(f"x_counterfactuals1: [{x_counterfactuals.min()}, {x_counterfactuals.max()}]")
        x_counterfactuals = dataset.project_to_pytorch_default(x_counterfactuals)
        x_counterfactuals = self.classifier_dataset.project_from_pytorch_default(
            x_counterfactuals
        )
        print(f"x_counterfactuals2: [{x_counterfactuals.min()}, {x_counterfactuals.max()}], {x_counterfactuals.shape}")
        print(f"x_counterfactuals2: [{x_counterfactuals.min()}, {x_counterfactuals.max()}]")
        print(f"x_counterfactuals2: [{x_counterfactuals.min()}, {x_counterfactuals.max()}]")
        device = [p for p in classifier.parameters()][0].device
        try:
            preds = torch.nn.Softmax(dim=-1)(
                classifier(x_counterfactuals.to(device)).detach().cpu()
            )

        except Exception as e:
            import pdb

            pdb.set_trace()

        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]

        return (
            list(x_counterfactuals),
            list(x_in - x_counterfactuals),
            list(y_target_end_confidence),
            list(x_list),
        )
