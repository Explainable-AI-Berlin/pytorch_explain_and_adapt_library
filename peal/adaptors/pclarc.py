import copy
import os
import pathlib
from typing import Iterable, Union

import torch
from sklearn.svm import LinearSVC
from torch.nn import Module
from torch.utils.data import DataLoader
from zennit.image import imsave

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

    unpoisoned_data: DataConfig

    training: TrainingConfig

    task: TaskConfig

    base_dir: str

    projection_type: str

    projection_location: str

    __name__: str = "peal.AdaptorConfig"

    def __init__(self,
                 model_path: str = None,
                 data: Union[dict, DataConfig] = None,
                 unpoisoned_data: Union[dict, DataConfig] = None,
                 training: Union[dict, TrainingConfig] = None,
                 task: Union[dict, TaskConfig] = None,
                 base_dir: str = None,
                 projection_location: str = "features",
                 projection_type: str = "svm",
                 **kwargs):

        self.model_path = model_path
        self.base_dir = base_dir
        self.projection_location = projection_location
        self.projection_type = projection_type

        if isinstance(data, DataConfig):
            self.data = data
        else:
            self.data = DataConfig(**data)

        if isinstance(unpoisoned_data, DataConfig):
            self.unpoisoned_data = unpoisoned_data
        else:
            self.unpoisoned_data = DataConfig(**unpoisoned_data)

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
            self.model = copy.deepcopy(self.original_model.model)
        else:
            self.model = copy.deepcopy(self.original_model)

    def run(self, *args, **kwargs):
        pathlib.Path(self.adaptor_config.base_dir).mkdir(exist_ok=True)
        self.model.eval()

        train_dataloader, val_dataloader, test_dataloader = create_dataloaders_from_datasource(self.adaptor_config)
        train_dataloader.dataset.enable_idx()
        train_dataloader.dataset.return_dict = True
        train_dataloader.dataset.url_enabled = True
        train_dataloader.dataset.enable_class_restriction(0)

        projection = SVMProjection if self.adaptor_config.projection_type == "svm" else SimpleProjection

        if self.adaptor_config.projection_location == "input":
            activations, annotations, targets = self.get_annotations_and_activations_simplified(train_dataloader)
            projection = projection(activations, annotations, device=self.device, dtype=activations.dtype, flatten_output=False)
            self.model = torch.nn.Sequential(*[projection, self.model])

        else:
            feature_extractor, downstream_head = split_model_at_penultima(self.model)
            # print("original model:\n", self.model)
            # print("\n\n\nfeature extractor:\n", feature_extractor)
            # print("\n\n\ndownstream head:\n", downstream_head)
            # exit()
            activations, annotations, targets = self.get_annotations_and_activations_simplified(train_dataloader, feature_extractor=feature_extractor)
            projection = SVMProjection(activations, annotations, device=self.device, dtype=activations.dtype, flatten_output=True)
            self.model = torch.nn.Sequential(*[feature_extractor, projection, downstream_head])


        classifications = []
        for x, y in test_dataloader:
            x = x.to(self.device)
            y = y[:,0].to(self.device)
            prediction = self.model(x)
            classifications.append(prediction.argmax(1) == y)
        print("test accuracy after correction on poisoned data: ", torch.cat(classifications).float().mean().item())

        test_data_unpoisoned = get_datasets(self.adaptor_config.unpoisoned_data)[2]
        test_data_unpoisoned = get_dataloader(test_data_unpoisoned, mode="test", batch_size=self.adaptor_config.training.test_batch_size)

        classifications = []
        for x, y in test_data_unpoisoned:
            x = x.to(self.device)
            y = y[:,0].to(self.device)
            prediction = self.model(x)
            classifications.append(prediction.argmax(1) == y)
        print("test accuracy after correction on unpoisoned data: ", torch.cat(classifications).float().mean().item())

        filename = f"corrected_model_{self.adaptor_config.projection_location}_{self.adaptor_config.projection_type}.cpl"
        corrected_model_path = os.path.join(self.adaptor_config.base_dir, filename)
        print("saving corrected model to: " + corrected_model_path)
        torch.save(self.model.to("cpu"), corrected_model_path)


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
            targets.append(batch["y"][:, 0].detach())
            annotations.append(batch["y"][:, 1].detach())
            filenames.extend(batch["url"])

        activations = torch.cat(activations, 0)
        targets = torch.cat(targets, 0).to(self.device)
        assert torch.all(targets == 0), "Only class 0"
        annotations = torch.cat(annotations, dim=0).to(self.device)
        # annotations[annotations == 0] = -1

        light_background_dir = os.path.join(self.adaptor_config.base_dir, "light_background")
        dark_background_dir = os.path.join(self.adaptor_config.base_dir, "dark_background")
        pathlib.Path(light_background_dir).mkdir(exist_ok=True)
        pathlib.Path(dark_background_dir).mkdir(exist_ok=True)
        for x, t, file in zip(activations, annotations, filenames):
            if t.item() == 0:
                imsave(os.path.join(dark_background_dir, file), x)
            elif t.item() == 1:
                imsave(os.path.join(light_background_dir, file), x)
            else:
                raise ValueError("Invalid annotation")

        exit()
        return activations, annotations, targets

def split_model_at_penultima(model):
    '''
    Splits a model at the penultima layer
    '''
    children_list = extract_all_children(model)
    feature_extractor = torch.nn.Sequential(*children_list[:-1])
    downstream_head = torch.nn.Sequential(*children_list[-1:])
    return feature_extractor, downstream_head

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


class SVMProjection(torch.nn.Module):
    def __init__(self,
                 activations: torch.Tensor,
                 annotations: torch.Tensor,
                 **kwargs):

        super().__init__()
        self.flatten_output = kwargs.get("flatten_output", False)
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
        if self.flatten_output:
            return out
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
        self.flatten_output = kwargs.get("flatten_output", False)
        activations = activations.flatten(start_dim=1)
        self.difference = activations[annotations == 1].mean(0) - activations[annotations == 0].mean(0)

    def forward(self, x):
        '''
        x: Tensor with shape [B, C, H, W]
        '''
        x_flat = x.flatten(start_dim=1)
        out = x_flat - self.difference
        if self.flatten_output:
            return out
        return out.reshape(x.shape)