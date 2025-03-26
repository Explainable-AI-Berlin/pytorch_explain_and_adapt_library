import copy
import json
import os
import pathlib
from collections import defaultdict

from sklearn.svm import LinearSVC
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models import ResNet
from tqdm import tqdm

from peal.adaptors.interfaces import AdaptorConfig, Adaptor
from peal.architectures.interfaces import TaskConfig
from peal.data.dataloaders import create_dataloaders_from_datasource, get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig
from peal.training.interfaces import TrainingConfig

import torch
import numpy as np
import pandas as pd


class ClArCConfig(AdaptorConfig):

    __name__: str = "peal.AdaptorConfig"
    category: str = "adaptor"
    seed: int = 0
    model_path: str
    base_dir: str
    data: DataConfig
    unpoisoned_data: DataConfig = None
    group_labels: str = None
    training: TrainingConfig
    task: TaskConfig
    projection_type: str = "pcav"
    layer_index: list[int] = [-1]
    correction_strength: list[float] = [1]
    attacked_class: int = 0
    cav_mode: str = None
    save_model: bool = True
    use_perfect_annotations: bool = False


class PClArCConfig(ClArCConfig):
    adaptor_type: str = "PClArC"


class RRClArCConfig(ClArCConfig):
    adaptor_type: str = "RRClArC"
    gradient_target: str = "all"
    mean_grad: bool = False
    rrc_loss: str = "l2"


class ClArC(Adaptor):

    def __init__(self, adaptor_config: ClArCConfig):
        pathlib.Path(adaptor_config.base_dir).mkdir(exist_ok=True)

        self.adaptor_config = adaptor_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("running on device: ", self.device)
        self.original_model = torch.load(self.adaptor_config.model_path, map_location=self.device)
        if hasattr(self.original_model, 'model'):
            self.original_model = self.original_model.model
        self.model = copy.deepcopy(self.original_model)
        self.attacked_class = adaptor_config.attacked_class


    def run(self):
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders_from_datasource(self.adaptor_config)
        train_dataloader.dataset.return_dict = True
        train_dataloader.dataset.url_enabled = True
        train_dataloader.dataset.disable_class_restriction()
        if self.adaptor_config.attacked_class is not None:
            train_dataloader.dataset.enable_class_restriction(self.adaptor_config.attacked_class)

        test_data = [("original", test_dataloader)]
        evaluation = {"original": defaultdict(list)}
        val_data_unpoisoned = None
        if self.adaptor_config.unpoisoned_data is not None:
            _, val_data_unpoisoned, test_data_unpoisoned = get_datasets(self.adaptor_config.unpoisoned_data)
            test_data_unpoisoned = get_dataloader(test_data_unpoisoned, mode="test", batch_size=self.adaptor_config.training.test_batch_size, task_config=self.adaptor_config.task)
            val_data_unpoisoned = get_dataloader(val_data_unpoisoned, mode="val", batch_size=self.adaptor_config.training.val_batch_size, task_config=self.adaptor_config.task)
            # val_data_unpoisoned = test_data_unpoisoned
            test_data.append(("unpoisoned", test_data_unpoisoned))
            evaluation["unpoisoned"] = defaultdict(list)

        self.model.eval()
        for description, dataloader in test_data:
            evaluation[description]["projection_location"].append("uncorrected")
            evaluation[description]["correction_strength"].append("uncorrected")
            for k, v in get_accuracies(dataloader, self.model, self.adaptor_config.data.dataset_class, self.device).items():
                evaluation[description][k].append(v)

        layers, correction_strength = np.meshgrid(self.adaptor_config.layer_index, self.adaptor_config.correction_strength)
        for layer, cs in zip(layers.flatten(), correction_strength.flatten()):
            self._run(train_dataloader, layer_index=layer, correction_strength=cs, data_val_unpoisoned=val_data_unpoisoned)
            self.model.eval()
            for description, dataloader in test_data:
                evaluation[description]["projection_location"].append(layer)
                evaluation[description]["correction_strength"].append(cs)
                for k, v in get_accuracies(dataloader, self.model, self.adaptor_config.data.dataset_class, self.device).items():
                    evaluation[description][k].append(v)

            self.model = copy.deepcopy(self.original_model)

        for dataset_name, results in evaluation.items():
            filename = self.get_evaluation_filename(dataset_name)
            results = pd.DataFrame(results)
            results.fillna(0, inplace=True)
            results.to_csv(os.path.join(self.adaptor_config.base_dir, filename), index=False)
            print(f"\n\n### results on {dataset_name} dataset ###\n{results.to_string()}")

    def _run(self, *args, **kwargs):
        pass

    def get_evaluation_filename(self, dataset_name: str) -> str:
        pass

    def get_annotations_and_activations(self, dataloader: DataLoader, feature_extractor: Module = None) -> tuple[torch.Tensor, torch.Tensor]:

        if self.adaptor_config.use_perfect_annotations:
            activations, annotations = self._get_perfect_annotations_and_activations(dataloader, feature_extractor)
        else:
            group_label_map = json.load(open(self.adaptor_config.group_labels, 'r'))
            confounder_filenames = np.asarray(group_label_map["confounders"])
            confounder_filenames = np.random.choice(confounder_filenames, size=min(len(confounder_filenames), 10000), replace=False)
            non_confounder_filenames = np.asarray(group_label_map["non_confounders"])
            non_confounder_filenames = np.random.choice(non_confounder_filenames, size=min(len(non_confounder_filenames), 10000), replace=False)

            confounders = []
            non_confounders = []
            for it, batch in enumerate(dataloader):
                x = batch["x"].to(self.device)
                activation = x if feature_extractor is None else feature_extractor(x).detach()

                filenames = np.asarray(batch["url"])
                confounders.append(activation[np.isin(filenames, confounder_filenames)])
                non_confounders.append(activation[np.isin(filenames, non_confounder_filenames)])

            confounders = torch.cat(confounders, dim=0)
            non_confounders = torch.cat(non_confounders, dim=0)
            activations = torch.cat([confounders, non_confounders], dim=0)
            annotations = torch.cat([torch.ones(len(confounders)), torch.zeros(len(non_confounders))], dim=0).to(self.device)

        if self.adaptor_config.cav_mode == "cavs_max":
            activations = activations.flatten(start_dim=2).max(2).values
        elif self.adaptor_config.cav_mode == "cavs_mean":
            activations = activations.mean((2, 3))
        else:
            activations = activations.flatten(start_dim=1)

        num_artifact_samples = torch.sum(annotations == 1).item()
        print(f"Number of artifact samples: {num_artifact_samples}; Number of non-artifact samples: {len(annotations) - num_artifact_samples}")
        return activations, annotations

    def _get_perfect_annotations_and_activations(self, dataloader: DataLoader, feature_extractor: Module = None):
        annotations = []
        activations = []
        for batch in dataloader:
            x = batch["x"].to(self.device)
            activations.append(x if feature_extractor is None else feature_extractor(x).detach())

            if self.adaptor_config.data.dataset_class == "SquareDataset":
                annotations.append(batch["y"][:, 1].detach())
            elif self.adaptor_config.data.dataset_class == "MnistDataset":
                annotations.append(get_perfect_annotations_mnist(x))

        annotations = torch.cat(annotations, dim=0).to(self.device)
        activations = torch.cat(activations, 0)
        return activations, annotations


class PClArC(ClArC):

    def __init__(self, adaptor_config: PClArCConfig):
        super().__init__(adaptor_config)
        self.adaptor_config = adaptor_config

    def get_evaluation_filename(self, dataset_name: str) -> str:
        attacked_class = f"_attacked-c{self.adaptor_config.attacked_class}" if self.adaptor_config.attacked_class is not None else ""
        return f"correction_{dataset_name}-dataset_{self.adaptor_config.projection_type}-projection_mode-{self.adaptor_config.cav_mode}{attacked_class}.csv"

    def run(self, *args, **kwargs):
        super().run()


    def _run(self, dataloader, layer_index: int = -1, correction_strength: float = 1.0, **kwargs):
        print(f"\n\nperforming p-clarc in layer {layer_index} with correction strength {correction_strength} and projection type {self.adaptor_config.projection_type}")
        self.model.eval()

        # train_dataloader.dataset.visualize_decision_boundary(self.model, 32, self.device, os.path.join(self.adaptor_config.base_dir, "decision_boundary_corrected_pcav.png"))
        # exit()

        feature_extractor, downstream_head = None, None
        if layer_index != 0:
            feature_extractor, downstream_head = split_model(self.model, layer_index, self.device)

        activations, annotations = self.get_annotations_and_activations(dataloader, feature_extractor=feature_extractor)

        projection = SimpleProjection if self.adaptor_config.projection_type == "simple" else CavProjection
        projection = projection(activations, annotations, projection_type=self.adaptor_config.projection_type, cav_mode=self.adaptor_config.cav_mode, device=self.device, correction_strength=correction_strength)

        if layer_index != 0:
            self.model = torch.nn.Sequential(*[feature_extractor, projection, downstream_head])
        else:
            self.model = torch.nn.Sequential(*[projection, self.model])

        if self.adaptor_config.save_model:
            attacked_class = f"_attacked-c{self.adaptor_config.attacked_class}" if self.adaptor_config.attacked_class is not None else ""
            filename = f"corrected_model{attacked_class}_{self.adaptor_config.projection_type}{layer_index}_mode-{self.adaptor_config.cav_mode}_cs{correction_strength}.cpl"
            corrected_model_path = os.path.join(self.adaptor_config.base_dir, filename)
            print("saving corrected model to: " + corrected_model_path)
            torch.save(self.model.to("cpu"), corrected_model_path)
            self.model.to(self.device)


class RRClArC(ClArC):

    def __init__(self, adaptor_config: RRClArCConfig):
        super().__init__(adaptor_config)
        self.adaptor_config = adaptor_config
        self.log_writer = SummaryWriter(log_dir=self.adaptor_config.base_dir + "/finetuning-logs")

    def get_evaluation_filename(self, dataset_name: str) -> str:
        attacked_class = f"_attacked-c{self.adaptor_config.attacked_class}" if self.adaptor_config.attacked_class is not None else ""
        mean_grad = f"_mean-grad" if self.adaptor_config.mean_grad else ""
        return f"correction_{dataset_name}-dataset_{self.adaptor_config.projection_type}-projection_mode-{self.adaptor_config.cav_mode}{attacked_class}_{self.adaptor_config.rrc_loss}-loss_target-{self.adaptor_config.gradient_target}{mean_grad}.csv"

    def run(self):
        super().run()
        self.log_writer.close()

    def _run(self, dataloader: DataLoader,
             layer_index: int = -2,
             correction_strength: float = 1.0,
             data_val_unpoisoned: DataLoader = None):

        print(f"\n\nperforming rr-clarc in layer {layer_index} with cav_mode={self.adaptor_config.cav_mode} and correction_strength={correction_strength}")

        attacked_class = f"_attacked-c{self.adaptor_config.attacked_class}" if self.adaptor_config.attacked_class is not None else ""
        mean_grad = f"_mean-grad" if self.adaptor_config.mean_grad else ""
        model_name = f"corrected_model_layer-{layer_index}_mode-{self.adaptor_config.cav_mode}_lamb{correction_strength}_{self.adaptor_config.rrc_loss}-loss{mean_grad}{attacked_class}"

        assert layer_index != 0, "rr-clarc not implemented for input layer"
        feature_extractor, downstream_head = split_model(self.model, layer_index, self.device)
        feature_extractor.eval()

        activations, annotations = self.get_annotations_and_activations(dataloader, feature_extractor=feature_extractor)
        cav = calculate_cav(activations, annotations, self.adaptor_config.projection_type)
        cav = cav.to(self.device, activations.dtype)
        self.finetune(dataloader, cav, feature_extractor, downstream_head, correction_strength, model_name=model_name, data_val_unpoisoned=data_val_unpoisoned)
        self.model = torch.nn.Sequential(*[feature_extractor, downstream_head])
        self.model.eval()

        if self.adaptor_config.save_model:
            corrected_model_path = os.path.join(self.adaptor_config.base_dir, model_name + ".cpl")
            print("saving corrected model to: " + corrected_model_path)
            torch.save(self.model.to("cpu"), corrected_model_path)
            self.model.to(self.device)

    def finetune(self,
                 dataloader,
                 cav,
                 feature_extractor,
                 downstream_head,
                 lamb,
                 model_name = "corrected_model",
                 data_val_unpoisoned: DataLoader = None):

        downstream_head.train()
        dataloader.dataset.disable_class_restriction()

        optimizer = torch.optim.SGD(downstream_head.parameters(), lr=self.adaptor_config.training.learning_rate, momentum=0.95)
        for epoch in range(self.adaptor_config.training.max_epochs):
            ce_losses = []
            rrc_losses = []
            accuracies = []

            val_accuracy = None
            if data_val_unpoisoned is not None:
                val_accuracy = self.get_val_accuracy(data_val_unpoisoned, feature_extractor, downstream_head)
                self.log_writer.add_scalar(model_name + "/val_unpoisoned/accuracy", val_accuracy, epoch)

            with tqdm(dataloader) as pbar:
                for batch in pbar:
                    optimizer.zero_grad()

                    optimizer.zero_grad()
                    x = batch["x"].to(self.device)
                    y = batch["y"].to(self.device)
                    if self.adaptor_config.data.dataset_class == "SquareDataset":
                        y = y[:, 0]

                    representation = feature_extractor(x).requires_grad_()
                    prediction = downstream_head(representation)
                    prediction_filtered = self.get_gradient_target(prediction)

                    grad = torch.autograd.grad(outputs=prediction_filtered,
                                               inputs=representation,
                                               create_graph=True,
                                               retain_graph=True,
                                               grad_outputs=torch.ones_like(prediction_filtered))[0]
                    if self.adaptor_config.mean_grad:
                        grad = grad.mean((2, 3), keepdim=True).expand_as(grad)

                    if self.adaptor_config.cav_mode is not None:
                        grad = grad.permute(1, 0, 2, 3).flatten(start_dim=1).permute(1, 0)
                    else:
                        grad = grad.flatten(start_dim=1)

                    if self.adaptor_config.rrc_loss == "l2":
                        rrc_loss = ((grad * cav).sum(1) ** 2).mean(0)
                    elif self.adaptor_config.rrc_loss == "cosine":
                        rrc_loss = torch.nn.functional.cosine_similarity(grad, cav).abs().mean(0)
                    ce_loss = torch.nn.functional.cross_entropy(prediction, y.to(torch.long))
                    accuracy = (prediction.argmax(1) == y).float()
                    rrc_losses.append(rrc_loss.item())
                    ce_losses.append(ce_loss.item())
                    accuracies.append(accuracy.detach())

                    loss = ce_loss + lamb * rrc_loss.to(torch.float64)
                    if torch.isnan(loss) or torch.isinf(loss):
                        if self.adaptor_config.attacked_class is not None:
                            dataloader.dataset.enable_class_restriction(self.adaptor_config.attacked_class)
                        return

                    loss.backward()
                    optimizer.step()

                    pbar.set_description(f'fine tuning epoch {epoch+1}/{self.adaptor_config.training.max_epochs}: accuracy={accuracy.mean()}, rrc_loss={rrc_loss}, ce_loss={ce_loss}')

            epoch_accuracy = torch.cat(accuracies).mean().item()
            epoch_ce_loss = torch.tensor(ce_losses).mean().item()
            epoch_rrc_loss = torch.tensor(rrc_losses).mean().item()

            self.log_writer.add_scalar(model_name + "/train/accuracy", epoch_accuracy, epoch)
            self.log_writer.add_scalar(model_name + "/train/ce_loss", epoch_ce_loss, epoch)
            self.log_writer.add_scalar(model_name + "/train/rrc_loss", epoch_rrc_loss, epoch)

            print(f"Epoch {epoch+1}: train_accuracy={epoch_accuracy}, accuracy_unpoisoned={val_accuracy}, ce_loss={epoch_ce_loss}, rrc_loss={epoch_rrc_loss}")


        self.log_writer.flush()
        if self.adaptor_config.attacked_class is not None:
            dataloader.dataset.enable_class_restriction(self.adaptor_config.attacked_class)

    def get_gradient_target(self, prediction):
        if self.adaptor_config.gradient_target == 'max':
            return prediction.max(1)[0]
        elif self.adaptor_config.gradient_target == 'attacked_class':
            return prediction[:, self.adaptor_config.attacked_class]
        elif self.adaptor_config.gradient_target == 'all':
            return prediction.sum(1)
        elif self.adaptor_config.gradient_target == 'all_random':
            return (prediction * torch.sign(0.5 - torch.rand_like(prediction))).sum(1)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def get_val_accuracy(self, dataloader, feature_extractor, downstream_head):
        dataloader.dataset.return_dict = True
        acc = []
        for batch in dataloader:
            x = batch["x"].to(self.device)
            prediction = downstream_head(feature_extractor(x))

            if self.adaptor_config.data.dataset_class == "SquareDataset":
                y = batch["y"][:, 0].to(self.device)
            else:
                y = batch["y"].to(self.device)

            acc.append((prediction.argmax(dim=1) == y))

        return torch.cat(acc, dim=0).float().mean().item()


def split_model(model: Module, split_at: int, device) -> (Module, Module):
    children_list = extract_all_children(model)[0]
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

def extract_all_children(model: Module, prefix: str = "") -> (list[Module], list[str]):
    children = []
    children_names = []
    for name, child in model.named_children():
        if prefix:
            name = prefix + "." + name
        if isinstance(child, torch.nn.Sequential):
            grandchildren, grandchildren_names = extract_all_children(child, prefix=name)
            children.extend(grandchildren)
            children_names.extend(grandchildren_names)

        else:
            children.append(child)
            children_names.append(name)

    return children, children_names

def get_layer_name(model: Module, layer_index: int) -> str:
    return extract_all_children(model)[1][layer_index-1]

def calculate_cav(activations: torch.Tensor, annotations: torch.Tensor, projection_type: str) -> torch.Tensor:
    if projection_type == "pcav":
        return calculate_pcav(activations, annotations)
    elif projection_type == "svm":
        return calculate_svm_cav(activations, annotations)
    else:
        raise NotImplementedError

def calculate_pcav(activations: torch.Tensor, annotations: torch.Tensor) -> torch.Tensor:
    actvs_centered = activations - activations.mean(dim=0)[None]

    annotations = annotations.clone()
    annotations[annotations == 0] = -1
    annotations = annotations.to(activations.dtype)
    annotations_centered = annotations - annotations.mean()

    covar = (actvs_centered * annotations_centered[:, None]).sum(dim=0) / (annotations.shape[0] - 1)
    vary = torch.sum(annotations_centered ** 2, dim=0) / (annotations.shape[0] - 1)
    w = (covar / vary)[None]

    cav = w / torch.sqrt((w ** 2).sum())
    print("cav shape:", cav.shape)
    return cav

def calculate_svm_cav(activations, annotations) -> torch.Tensor:
    activations = activations.cpu()
    annotations = annotations.cpu()
    model = LinearSVC(C=1.0, penalty='l2', max_iter=10000, class_weight='balanced', verbose = 2)
    model.fit(activations, annotations)

    cav = torch.tensor(model.coef_[0])
    print("cav shape:", cav.shape)
    return cav / ((cav ** 2).sum() ** .5).item() # normalize


@torch.no_grad()
def get_accuracies(dataloader: DataLoader, model: Module, dataset_name: str, device: str) -> dict:
    model.eval()
    dataloader.dataset.return_dict = True

    annotations = []
    targets = []
    acc = []
    with tqdm(dataloader) as pbar:
        pbar.set_description(f'evaluation')
        for batch in pbar:
            x = batch["x"].to(device)
            prediction = model(x)

            if dataset_name == "SquareDataset":
                y = batch["y"][:, 0].to(device)
                annotations.append(batch["y"][:, 1].detach())
            elif dataset_name == "MnistDataset":
                y = batch["y"].to(device)
                annotations.append(get_perfect_annotations_mnist(x))
            else:
                raise ValueError("Unknown dataset class")

            acc.append((prediction.argmax(dim=1) == y).to(torch.int))
            targets.append(y)

    annotations = torch.cat(annotations, dim=0).to(device).int()
    targets = torch.cat(targets, dim=0).to(device).int()
    acc = torch.cat(acc, dim=0).to(device).float()

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
        # print(f"class {y}: {results[f'c{y.item()}_n']} items, artfiact freq: {results[f'c{y.item()}_artifact_freq']}")

    return results


class CavProjection(torch.nn.Module):
    def __init__(self,
                 activations: torch.Tensor,
                 annotations: torch.Tensor,
                 projection_type: str = "svm",
                 cav_mode: str = None,
                 device: str = "cpu",
                 correction_strength: float = 1.0):
        super().__init__()
        self.correction_strength = correction_strength
        self.projection_type = projection_type
        self.cav_mode = cav_mode
        self.device = device

        cav = calculate_cav(activations, annotations.clone(), projection_type).to(device, dtype=activations.dtype)
        self.cav = self.correction_strength * cav.reshape(-1, 1)
        z = torch.mean(activations[annotations == 0], dim=0, keepdim=True).to(device=device, dtype=activations.dtype)
        self.z = z @ self.cav @ self.cav.T

    def forward(self, x):

        out = x + 0
        if self.cav_mode is None:
            out = out.flatten(1)
            out = out - (out @ self.cav @ self.cav.T) + self.z
            out = out.reshape(x.shape)
        else:
            if self.cav_mode == "cavs_max":
                out = out.flatten(start_dim=2).max(2).values
            elif self.cav_mode == "cavs_mean":
                out = out.mean((2, 3))
            else:
                raise NotImplementedError
            out = x - (out @ self.cav @ self.cav.T)[:,:,None,None] + self.z[:,:,None,None]

        return torch.nn.functional.relu(out)


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

def get_perfect_annotations_mnist(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 4
    assert x.shape[1] == 3
    annotations = ~((x[:,0] == x[:,1]) * (x[:,0] == x[:,2])).all(dim=1).all(dim=1)
    return annotations.int()