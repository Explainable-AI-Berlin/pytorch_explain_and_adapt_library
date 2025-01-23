import copy
import os
import pathlib
from collections import defaultdict
from typing import Union

import torch
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.models import ResNet
from tqdm import tqdm

from peal.adaptors.interfaces import AdaptorConfig, Adaptor
from peal.architectures.predictors import TaskConfig
from peal.data.dataloaders import create_dataloaders_from_datasource, get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.data.datasets import DataConfig
from peal.training.trainers import TrainingConfig


class PClArCConfig(AdaptorConfig):

    category: str = "adaptor"

    adaptor_type: str = "PClArC"

    model_path: str

    data: DataConfig

    unpoisoned_data: DataConfig = None

    poisoned_data: DataConfig = None

    training: TrainingConfig

    task: TaskConfig = None

    base_dir: str

    projection_type: str

    projection_location: list[int]

    correction_strength: list[float]

    __name__: str = "peal.AdaptorConfig"

    def __init__(self,
                 model_path: str = None,
                 data: Union[dict, DataConfig] = None,
                 unpoisoned_data: Union[dict, DataConfig] = None,
                 poisoned_data: Union[dict, DataConfig] = None,
                 training: Union[dict, TrainingConfig] = None,
                 task: Union[dict, TaskConfig] = None,
                 base_dir: str = None,
                 projection_location: list[int] = None,
                 projection_type: str = "svm",
                 correction_strength: list[float] = None,
                 **kwargs):

        self.model_path = model_path
        self.base_dir = base_dir
        self.projection_location = projection_location if projection_location is not None else [-1]
        self.projection_type = projection_type
        self.correction_strength = correction_strength if correction_strength is not None else [1.0]

        if isinstance(data, DataConfig):
            self.data = data
        else:
            self.data = DataConfig(**data)

        if isinstance(unpoisoned_data, DataConfig):
            self.unpoisoned_data = unpoisoned_data
        elif not unpoisoned_data is None:
            self.unpoisoned_data = DataConfig(**unpoisoned_data)

        if isinstance(poisoned_data, DataConfig):
            self.poisoned_data = poisoned_data
        elif not poisoned_data is None:
            self.poisoned_data = DataConfig(**poisoned_data)

        if isinstance(training, TrainingConfig):
            self.training = training
        else:
            self.training = TrainingConfig(**training)

        if isinstance(task, TaskConfig):
            self.task = task
        elif not task is None:
            self.task = TaskConfig(**task)

        self.kwargs = kwargs


class PClArC(Adaptor):

    def __init__(self, adaptor_config: PClArCConfig):
        self.adaptor_config = adaptor_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.original_model = torch.load(self.adaptor_config.model_path, map_location=self.device)
        if hasattr(self.original_model, 'model'):
            self.original_model = self.original_model.model
        self.model = copy.deepcopy(self.original_model)

        pathlib.Path(self.adaptor_config.base_dir).mkdir(exist_ok=True)

    def run(self, *args, **kwargs):

        train_dataloader, val_dataloader, test_dataloader = create_dataloaders_from_datasource(self.adaptor_config)
        train_dataloader.dataset.return_dict = True
        train_dataloader.dataset.url_enabled = True
        train_dataloader.dataset.disable_class_restriction()
        train_dataloader.dataset.enable_class_restriction(0)

        # models = []
        models = [(0, 0, self.adaptor_config.model_path)]
        location, correction_strength = np.meshgrid(self.adaptor_config.projection_location, self.adaptor_config.correction_strength)
        for loc, cs in zip(location.flatten(), correction_strength.flatten()):
            print(f"performing p-clarc in layer {loc} with correction strength {cs} and projection type {self.adaptor_config.projection_type}")
            model_path = self._run(train_dataloader, projection_location=loc, correction_strength=cs)
            models.append((loc, cs, model_path))
            self.model = copy.deepcopy(self.original_model)

        test_data = [("original", test_dataloader)]
        if self.adaptor_config.unpoisoned_data is not None:
            test_data_unpoisoned = get_datasets(self.adaptor_config.unpoisoned_data)[2]
            test_data_unpoisoned = get_dataloader(test_data_unpoisoned, mode="test", batch_size=self.adaptor_config.training.test_batch_size, task_config=self.adaptor_config.task)
            test_data.append(("unpoisoned", test_data_unpoisoned))
        if self.adaptor_config.poisoned_data is not None:
            test_data_poisoned = get_datasets(self.adaptor_config.poisoned_data)[2]
            test_data_poisoned = get_dataloader(test_data_poisoned, mode="test", batch_size=self.adaptor_config.training.test_batch_size, task_config=self.adaptor_config.task)
            test_data.append(("fully_poisoned", test_data_poisoned))

        for description, dataloader in test_data:
            results = defaultdict(list)
            for loc, cs, model_path in models:
                print(f"loading model from file {model_path}")
                model = torch.load(model_path, map_location=self.device)
                model.eval()
                results["projection_location"].append(loc)
                results["correction_strength"].append(cs)
                for k, v in self.get_accuracies(dataloader, model).items():
                    results[k].append(v)
            filename = f"{self.adaptor_config.projection_type}_projection_{description}_dataset.csv"
            results = pd.DataFrame(results)
            results.fillna(0, inplace=True)
            results.to_csv(os.path.join(self.adaptor_config.base_dir, filename), index=False)
            print(f"\n\n### results on {description} dataset ###\n{results.to_string()}")


    def _run(self, dataloader, projection_location: int = -1, correction_strength: float = 1.0) -> str:
        self.model.eval()

        # train_dataloader.dataset.visualize_decision_boundary(self.model, 32, self.device, os.path.join(self.adaptor_config.base_dir, "decision_boundary_corrected_pcav.png"))
        # exit()

        feature_extractor, downstream_head = None, None
        if projection_location != 0:
            feature_extractor, downstream_head = split_model(self.model, projection_location, self.device)

        activations, annotations, targets = self.get_annotations_and_activations_simplified(dataloader, feature_extractor=feature_extractor)
        projection = get_projection_type(self.adaptor_config.projection_type)
        projection = projection(activations, annotations, device=self.device, dtype=activations.dtype, correction_strength=correction_strength)

        if projection_location != 0:
            self.model = torch.nn.Sequential(*[feature_extractor, projection, downstream_head])
        else:
            self.model = torch.nn.Sequential(*[projection, self.model])

        filename = f"corrected_model_{projection_location}_{self.adaptor_config.projection_type}_cs{correction_strength}.cpl"
        corrected_model_path = os.path.join(self.adaptor_config.base_dir, filename)
        print("saving corrected model to: " + corrected_model_path)
        torch.save(self.model.to("cpu"), corrected_model_path)

        return corrected_model_path


    def get_annotations_and_activations_simplified(self, dataloader: DataLoader, feature_extractor: Module = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        targets = []
        annotations = []
        activations = []
        filenames = []

        for it, batch in enumerate(dataloader):
            x = batch["x"].to(self.device)
            if feature_extractor is None:
                activations.append(x)
            else:
                activations.append(feature_extractor(x).detach())

            if self.adaptor_config.data.dataset_class == "SquareDataset":
                targets.append(batch["y"][:, 0].detach())
                annotations.append(batch["y"][:, 1].detach())
            elif self.adaptor_config.data.dataset_class == "MnistDataset":
                targets.append(batch["y"].detach())
                annotations.append(get_perfect_annotations_mnist(x))

            filenames.extend(batch["url"])

        activations = torch.cat(activations, 0)
        targets = torch.cat(targets, 0).to(self.device)
        assert torch.all(targets == 0), "Only class 0"
        annotations = torch.cat(annotations, dim=0).to(self.device)

        num_artifact_samples = torch.sum(annotations == 1).item()
        print(f"Number of artifact samples: {num_artifact_samples}; Number of non-artifact samples: {len(annotations) - num_artifact_samples}")
        return activations, annotations, targets

    def get_accuracies(self, dataloader: DataLoader, model: Module) -> dict:
        model.eval()
        dataloader.dataset.return_dict = True

        annotations = []
        targets = []
        acc = []
        total_its = len(dataloader)
        with tqdm(enumerate(dataloader)) as pbar:
            for it, batch in pbar:
                x = batch["x"].to(self.device)
                prediction = model(x)

                if self.adaptor_config.data.dataset_class == "SquareDataset":
                    y = batch["y"][:, 0].to(self.device)
                    annotations.append(batch["y"][:, 1].detach())
                elif self.adaptor_config.data.dataset_class == "MnistDataset":
                    y = batch["y"].to(self.device)
                    annotations.append(get_perfect_annotations_mnist(x))
                else:
                    raise ValueError("Unknown dataset class")

                acc.append((prediction.argmax(dim=1) == y).to(torch.int))
                targets.append(y)
                pbar.set_description(f'it: {it}/{total_its}')

        annotations = torch.cat(annotations, dim=0).to(self.device).int()
        targets = torch.cat(targets, dim=0).to(self.device).int()
        acc = torch.cat(acc, dim=0).to(self.device).float()

        results = {"n": len(annotations),
                    "artifact_freq": annotations.sum().item()/len(annotations),
                    "accuracy": acc.mean().item(),
                    "artifact_accuracy": acc[annotations == 1].mean().item(),
                    "non-artifact_accuracy": acc[annotations == 0].mean().item()}

        for y in torch.unique(targets):
            idx = targets == y
            results[f"c{y.item()}_n"] = len(annotations[idx])
            results[f"c{y.item()}_artifact_freq"] = annotations[idx].sum().item()/len(annotations[idx])
            results[f"c{y.item()}_accuracy"] = acc[idx].mean().item()
            results[f"c{y.item()}_artifact_accuracy"] = acc[(annotations == 1) * idx].mean().item()
            results[f"c{y.item()}_non-artifact_accuracy"] = acc[(annotations == 0) * idx].mean().item()

        return results



def split_model(model: Module, split_at: int, device):
    '''
    Splits a model at the penultima layer
    '''
    children_list = extract_all_children(model)
    feature_extractor = torch.nn.Sequential(*children_list[:split_at])
    if isinstance(model, ResNet):
        downstream_head = torch.nn.Sequential(*children_list[split_at:-1], torch.nn.Flatten(start_dim=1), children_list[-1])
    else:
        downstream_head = torch.nn.Sequential(*children_list[split_at:])
    # print("original model:\n", model)
    # print("\n\n\nfeature extractor:\n", feature_extractor)
    # print("\n\n\ndownstream head:\n", downstream_head)
    # exit()
    return feature_extractor.to(device), downstream_head.to(device)

def extract_all_children(model):
    '''
    Extracts all children of a model
    '''
    children = []
    for child in model.children():
        if isinstance(child, torch.nn.Sequential):
            children.extend(extract_all_children(child))

        else:
            children.append(child)

    return children

class PCAVProjection(torch.nn.Module):
    def __init__(self,
                 activations: torch.Tensor,
                 annotations: torch.Tensor,
                 **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.dtype = kwargs.get("dtype", torch.float32)
        self.correction_strength = kwargs.get("correction_strength", 1.0)

        annotations[annotations == 0] = -1
        annotations = annotations.to(self.dtype)
        t_centered = annotations - annotations.mean()
        activations = activations.reshape(activations.shape[0], -1)
        actvs_centered = activations - activations.mean(dim=0)[None]
        covar = (actvs_centered * t_centered[:, None]).sum(dim=0) / (annotations.shape[0] - 1)
        vary = torch.sum(t_centered ** 2, dim=0) / (annotations.shape[0] - 1)
        w = (covar / vary)[:, None]

        cav = self.correction_strength * w / torch.sqrt((w ** 2).sum())
        self.projection = cav @ cav.T

        self.z = torch.mean(activations[annotations == -1], dim=0, keepdim=True).to(device=self.device, dtype=self.dtype)

    def forward(self, x):
        x_flat = x.flatten(1)
        out = (x_flat - (x_flat @ self.projection) + (self.z @ self.projection))
        out = torch.nn.functional.relu(out)
        return out.reshape(x.shape)

class SVMProjection(torch.nn.Module):
    def __init__(self,
                 activations: torch.Tensor,
                 annotations: torch.Tensor,
                 **kwargs):

        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.dtype = kwargs.get("dtype", torch.float32)
        self.correction_strength = kwargs.get("correction_strength", 1.0)

        activations = activations.reshape(activations.shape[0], -1).cpu()
        annotations = annotations.cpu()
        self.model = LinearSVC(C=1.0, penalty='l2', max_iter=10000, class_weight='balanced', verbose = 2)
        self.model.fit(activations, annotations)

        cav = torch.tensor(self.model.coef_[0], device=self.device, dtype=self.dtype)
        cav = self.correction_strength * cav / ((cav ** 2).sum() ** .5).item() # normalize
        self.projection = cav[:, None] @ cav[None]

        self.z = torch.mean(activations[annotations == 0], dim=0, keepdim=True).to(device=self.device, dtype=self.dtype)

    def forward(self, x):
        x_flat = x.flatten(1)
        out = (x_flat - (x_flat @ self.projection) + (self.z @ self.projection))
        out = torch.nn.functional.relu(out)
        return out.reshape(x.shape)

class SimpleProjection(torch.nn.Module):
    def __init__(self,
                 activations: torch.Tensor,
                 annotations: torch.Tensor,
                 **kwargs):
        '''
        uncorrect_decision_strategy: List of tensors with shape [1, C, H, W]
        correct_decision_strategy: List of tensors
        '''
        super().__init__()
        activations = activations.flatten(start_dim=1)
        self.difference = activations[annotations == 1].mean(0) - activations[annotations == 0].mean(0)

    def forward(self, x):
        '''
        x: Tensor with shape [B, C, H, W]
        '''
        x_flat = x.flatten(start_dim=1)
        out = x_flat - self.difference
        out = torch.nn.functional.relu(out)
        return out.reshape(x.shape)

projections = {
    "pcav": PCAVProjection,
    "svm": SVMProjection,
    "simple": SimpleProjection
}
def get_projection_type(projection_type: str):
    return projections.get(projection_type, SimpleProjection)

def get_perfect_annotations_mnist(x: torch.Tensor):
    assert len(x.shape) == 4
    assert x.shape[1] == 3
    annotations = ~((x[:,0] == x[:,1]) * (x[:,0] == x[:,2])).all(dim=1).all(dim=1)
    return annotations.int()