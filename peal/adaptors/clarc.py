import copy
import json
import math
import os
import pathlib
import sys
import traceback
from collections import defaultdict, namedtuple

from sklearn.svm import LinearSVC
from torch.nn import Module, CrossEntropyLoss, Sequential
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
    model_path: str
    base_dir: str
    data: DataConfig
    unpoisoned_data: DataConfig = None
    training: TrainingConfig
    task: TaskConfig
    projection_type: str = "pcav"
    layer_index: list[int] = [-1]
    correction_strength: list[float] = [1]
    attacked_class: int = 0
    cav_mode: str = None
    save_model: bool = True
    max_samples: int = 999999
    reverse_cav_direction: bool = False


class PClArCConfig(ClArCConfig):
    adaptor_type: str = "PClArC"
    finetune: bool = False


class RRClArCConfig(ClArCConfig):
    adaptor_type: str = "RRClArC"
    gradient_target: str = "all"
    mean_grad: bool = False
    rrc_loss: str = "l2"


class ClArC(Adaptor):

    def __init__(self, adaptor_config: ClArCConfig):
        pathlib.Path(adaptor_config.base_dir).mkdir(exist_ok=True)

        self.config = adaptor_config
        torch.manual_seed(self.config.seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("running on device: ", self.device)
        self.original_model = torch.load(self.config.model_path, map_location=self.device)
        if hasattr(self.original_model, 'model'):
            self.original_model = self.original_model.model
        self.model = copy.deepcopy(self.original_model)
        self.attacked_class = adaptor_config.attacked_class
        self.use_perfect_annotations = True if self.config.data.spray_label_file is not None else False

        self.cav_cache = CavCache("", "", "", "")


    def run(self):
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders_from_datasource(self.config)
        train_dataloader.dataset.return_dict = True
        train_dataloader.dataset.url_enabled = True
        train_dataloader.dataset.enable_groups()
        val_dataloader.dataset.return_dict = True
        val_dataloader.dataset.url_enabled = True
        val_dataloader.dataset.enable_groups()
        # test_dataloader.dataset.return_dict = True
        # test_dataloader.dataset.url_enabled = True
        # test_dataloader.dataset.enable_groups()
        train_dataloader.dataset.disable_class_restriction()
        val_dataloader.dataset.disable_class_restriction()
        # test_dataloader.dataset.disable_class_restriction()
        if self.config.attacked_class is not None:
            train_dataloader.dataset.enable_class_restriction(self.config.attacked_class)

        eval_dataloaders = [("original-val", val_dataloader, self.use_perfect_annotations)]
        # eval_dataloaders.append(("original-test", test_dataloader, self.use_perfect_annotations))
        evaluation = {"original-val": defaultdict(list), "original-test": defaultdict(list)}

        if self.config.unpoisoned_data is not None:
            test_data_unpoisoned = get_datasets(self.config.unpoisoned_data, return_dict=True)[-1]
            test_data_unpoisoned.enable_groups()
            test_data_unpoisoned = get_dataloader(test_data_unpoisoned, mode="test", batch_size=self.config.training.test_batch_size, task_config=self.config.task)
            eval_dataloaders.append(("unpoisoned-test", test_data_unpoisoned, True))
            evaluation["unpoisoned-test"] = defaultdict(list)

        self.model.eval()
        for description, dataloader, perfect_annotations in eval_dataloaders:
            evaluation[description]["projection_location"].append("uncorrected")
            evaluation[description]["correction_strength"].append("uncorrected")
            evaluation[description]["epochs_finetuned"].append("uncorrected")
            for k, v in self.get_accuracies(dataloader, self.model).items():
                evaluation[description][k].append(v)

        for layer in self.config.layer_index:
            for cs in self.config.correction_strength:
                model, number_epochs_finetuned = self._run(train_dataloader, val_dataloader, layer_index=layer, correction_strength=cs)
                model.eval()
                for description, dataloader, perfect_annotations in eval_dataloaders:
                    evaluation[description]["projection_location"].append(layer)
                    evaluation[description]["correction_strength"].append(cs)
                    evaluation[description]["epochs_finetuned"].append(number_epochs_finetuned)
                    for k, v in self.get_accuracies(dataloader, model).items():
                        if k == "accuracy":
                            print("accuracy: ", v)
                        evaluation[description][k].append(v)

                self.model = copy.deepcopy(self.original_model)

        for dataset_name, results in evaluation.items():
            filename = self.get_evaluation_filename(dataset_name)
            results = pd.DataFrame(results)
            results.fillna("empty", inplace=True)
            results.to_csv(os.path.join(self.config.base_dir, filename), index=False)
            print(f"\n\n### results on {dataset_name} dataset ###\n{results.to_string()}")

    def _run(self, *args, **kwargs) -> (Module, int):
        pass

    def get_evaluation_filename(self, dataset_name: str) -> str:
        pass

    def get_annotations_and_activations(self, dataloader: DataLoader, feature_extractor: Module = None) -> tuple[torch.Tensor, torch.Tensor]:

        confounders = []
        non_confounders = []
        for it, batch in enumerate(dataloader):
            x = batch["x"].to(self.device)
            activation = x if feature_extractor is None else feature_extractor(x).detach()

            if self.config.cav_mode == "cavs_max":
                activation = activation.flatten(start_dim=2).max(2).values
            elif self.config.cav_mode == "cavs_mean":
                activation = activation.mean((2, 3))
            else:
                activation = activation.flatten(start_dim=1)

            group_labels = batch["has_confounder"].squeeze()

            if self.config.reverse_cav_direction:
                group_labels = group_labels * -1 + 1

            if len(non_confounders) < self.config.max_samples:
                non_confounders.extend(activation[group_labels == 0])
            if len(confounders) < self.config.max_samples:
                confounders.extend(activation[group_labels == 1])

        confounders = torch.stack(confounders)
        non_confounders = torch.stack(non_confounders)
        activations = torch.cat((non_confounders, confounders))
        annotations = torch.cat((torch.zeros(len(non_confounders)), torch.ones(len(confounders)))).to(self.device)

        num_artifact_samples = torch.sum(annotations == 1).item()
        print(f"Number of artifact samples: {num_artifact_samples}; Number of non-artifact samples: {len(annotations) - num_artifact_samples}")
        return activations, annotations

    @torch.no_grad()
    def get_accuracies(self, dataloader: DataLoader, model: Module) -> dict:
        model.eval()
        annotations = []
        targets = []
        acc = []
        with tqdm(dataloader) as pbar:
            pbar.set_description(f'evaluation')
            for batch in pbar:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device).squeeze()
                targets.append(y)
                annotations.append(batch["has_confounder"].squeeze())

                prediction = model(x)
                acc.append((prediction.argmax(dim=1) == y).to(torch.int))

        annotations = torch.cat(annotations, dim=0).to(self.device).int()
        targets = torch.cat(targets, dim=0).to(self.device).int()
        acc = torch.cat(acc, dim=0).to(self.device).float()

        results = {"n": len(targets),
                   "artifact_freq": annotations.sum().item()/len(annotations) if len(targets) == len(annotations) else "---",
                   "accuracy": acc.mean().item(),
                   "avg_group_acc": "empty",
                   "worst_group_acc": "empty",
                   "artifact_accuracy": acc[annotations == 1].mean().item(),
                   "non-artifact_accuracy": acc[annotations == 0].mean().item()
                   }

        group_accuracies = []
        for y in torch.unique(targets):
            idx = targets == y
            results[f"c{y.item()}_n"] = len(targets[idx])
            results[f"c{y.item()}_artifact_freq"] = annotations[idx].sum().item()/len(annotations[idx]) if len(targets) == len(annotations) else "---"
            results[f"c{y.item()}_accuracy"] = acc[idx].mean().item()
            results[f"c{y.item()}_artifact_accuracy"] = acc[(annotations == 1) * idx].mean().item()
            group_accuracies.append(results[f"c{y.item()}_artifact_accuracy"])
            results[f"c{y.item()}_non-artifact_accuracy"] = acc[(annotations == 0) * idx].mean().item()
            group_accuracies.append(results[f"c{y.item()}_non-artifact_accuracy"])
            # print(f"class {y}: {results[f'c{y.item()}_n']} items, artfiact freq: {results[f'c{y.item()}_artifact_freq']}")

        print(f"group accuracies:", group_accuracies)
        group_accuracies = [num for num in group_accuracies if not math.isnan(num)]
        results["avg_group_acc"] = np.mean(group_accuracies).item()
        results["worst_group_acc"] = np.min(group_accuracies).item()
        return results


class PClArC(ClArC):

    def __init__(self, adaptor_config: PClArCConfig):
        super().__init__(adaptor_config)
        self.adaptor_config = adaptor_config

    def get_evaluation_filename(self, dataset_name: str) -> str:
        attacked_class = f"_attacked-c{self.adaptor_config.attacked_class}" if self.adaptor_config.attacked_class is not None else ""
        label_type = ("true" if self.config.data.spray_label_file is None else "spray") + "-group-labels"
        finetune = f"_{self.config.training.max_epochs}epochs-finetune" if self.adaptor_config.finetune else ""
        return f"correction_{dataset_name}-dataset_{label_type}_{self.adaptor_config.projection_type}-projection_mode-{self.adaptor_config.cav_mode}{attacked_class}{finetune}.csv"

    def run(self, *args, **kwargs):
        super().run()

    def _run(self, data_train: DataLoader, data_val: DataLoader, layer_index: int = -1, correction_strength: float = 1.0, **kwargs) -> (Module, int):
        torch.manual_seed(self.config.seed)

        print(f"\n\nperforming p-clarc in layer {layer_index} with correction strength {correction_strength} and projection type {self.adaptor_config.projection_type}")
        self.model.eval()

        feature_extractor, downstream_head = None, None
        if layer_index != 0:
            feature_extractor, downstream_head = split_model(self.model, layer_index, self.device)

        if layer_index != self.cav_cache.layer:
            activations, annotations = self.get_annotations_and_activations(data_train, feature_extractor=feature_extractor)
            cav = calculate_cav(activations, annotations.clone(), self.adaptor_config.projection_type).to(self.device, dtype=activations.dtype)
            self.cav_cache = CavCache(layer=layer_index, cav=cav, annotations=annotations, activations=activations)

        projection = SimpleProjection if self.adaptor_config.projection_type == "simple" else CavProjection
        projection = projection(self.cav_cache.activations, self.cav_cache.annotations, cav=self.cav_cache.cav, cav_mode=self.adaptor_config.cav_mode, correction_strength=correction_strength)

        number_epochs_finetuned = 0
        if layer_index != 0:
            if self.adaptor_config.finetune:
                downstream_head, number_epochs_finetuned = self.finetune(projection, downstream_head, data_train, data_val, feature_extractor=feature_extractor)
            self.model = torch.nn.Sequential(feature_extractor, projection, downstream_head)
        else:
            if self.adaptor_config.finetune:
                self.model, number_epochs_finetuned = self.finetune(projection, self.model, data_train, data_val)
            self.model = torch.nn.Sequential(projection, self.model)

        if self.adaptor_config.save_model:
            attacked_class = f"_attacked-c{self.adaptor_config.attacked_class}" if self.adaptor_config.attacked_class is not None else ""
            filename = f"corrected_model{attacked_class}_{self.adaptor_config.projection_type}{layer_index}_mode-{self.adaptor_config.cav_mode}_cs{correction_strength}_epoch{number_epochs_finetuned}.cpl"
            corrected_model_path = os.path.join(self.adaptor_config.base_dir, filename)
            print("saving corrected model to: " + corrected_model_path)
            torch.save(self.model.to("cpu"), corrected_model_path)
            self.model.to(self.device)

        return self.model, number_epochs_finetuned

    def finetune(self, projection_layer: Module, downstream_head: Module, data_train: DataLoader, data_val: DataLoader, feature_extractor: Module = None) -> (Module, int):

        if feature_extractor is not None:
            feature_extractor.eval()
        projection_layer.eval()
        downstream_head.train()
        data_train.dataset.disable_class_restriction()

        optimizer = torch.optim.SGD(downstream_head.parameters(), lr=self.adaptor_config.training.learning_rate, momentum=0.9, weight_decay=0.0001)
        loss = CrossEntropyLoss()

        composite = Sequential(feature_extractor, projection_layer, downstream_head) if feature_extractor is not None else Sequential(projection_layer, downstream_head)
        val_accuracies = self.get_accuracies(data_val, composite)
        print(f"Epoch 0: avg_group_acc={val_accuracies['avg_group_acc']}, worst_group_acc={val_accuracies['worst_group_acc']}")
        best_model = copy.deepcopy(downstream_head)
        best_val_group_acc = val_accuracies['avg_group_acc']
        number_epochs_trained = 0

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.adaptor_config.training.max_epochs):
            losses = []
            accuracies = []

            with tqdm(data_train) as pbar:
                for batch in pbar:
                    optimizer.zero_grad()

                    x = batch["x"].to(self.device)
                    y = batch["y"].to(self.device).squeeze().long()
                    if feature_extractor is None:
                        prediction = downstream_head(projection_layer(x))
                    else:
                        prediction = downstream_head(projection_layer(feature_extractor(x)))

                    ce_loss = loss(prediction, y)
                    try:
                        ce_loss.backward()
                    except:
                        print(traceback.format_exc())
                        if self.config.attacked_class is not None:
                            data_train.dataset.enable_class_restriction(self.attacked_class)
                        return best_model, number_epochs_trained

                    optimizer.step()

                    accuracy = (prediction.argmax(1) == y).float().detach()
                    losses.append(ce_loss.item())
                    accuracies.append(accuracy)

                    pbar.set_description(f'fine tuning epoch {epoch+1}/{self.adaptor_config.training.max_epochs}: accuracy={accuracy.mean()}, loss={ce_loss}')

            epoch_accuracy = torch.cat(accuracies).mean().item()
            epoch_ce_loss = torch.tensor(losses).mean().item()
            composite = Sequential(feature_extractor, projection_layer, downstream_head) if feature_extractor is not None else Sequential(projection_layer, downstream_head)
            val_accuracies = self.get_accuracies(data_val, composite)

            if val_accuracies['avg_group_acc'] > best_val_group_acc:
                best_val_group_acc = val_accuracies['avg_group_acc']
                best_model = copy.deepcopy(downstream_head)
                number_epochs_trained = epoch + 1

            print(f"Epoch {epoch+1}: train_acc={epoch_accuracy}, avg_group_acc={val_accuracies['avg_group_acc']}, worst_group_acc={val_accuracies['worst_group_acc']}, ce_loss={epoch_ce_loss}")

        if self.config.attacked_class is not None:
            data_train.dataset.enable_class_restriction(self.attacked_class)

        return best_model, number_epochs_trained



class RRClArC(ClArC):

    def __init__(self, adaptor_config: RRClArCConfig):
        super().__init__(adaptor_config)
        self.adaptor_config = adaptor_config
        self.log_writer = SummaryWriter(log_dir=self.adaptor_config.base_dir + "/finetuning-logs")

    def get_evaluation_filename(self, dataset_name: str) -> str:
        attacked_class = f"_attacked-c{self.adaptor_config.attacked_class}" if self.adaptor_config.attacked_class is not None else ""
        mean_grad = f"_mean-grad" if self.adaptor_config.mean_grad else ""
        label_type = ("true" if self.config.data.spray_label_file is None else "spray") + "-group-labels"
        return f"correction_{dataset_name}-dataset_{label_type}_{self.adaptor_config.projection_type}-projection_mode-{self.adaptor_config.cav_mode}{attacked_class}_{self.adaptor_config.rrc_loss}-loss_target-{self.adaptor_config.gradient_target}{mean_grad}_{self.adaptor_config.training.max_epochs}-epochs.csv"

    def run(self):
        super().run()
        self.log_writer.close()

    def _run(self,
             data_train: DataLoader,
             data_val: DataLoader,
             layer_index: int = -2,
             correction_strength: float = 1.0) -> (Module, int):

        torch.manual_seed(self.config.seed)
        print(f"\n\nperforming rr-clarc in layer {layer_index} with cav_mode={self.adaptor_config.cav_mode} and correction_strength={correction_strength}")

        attacked_class = f"_attacked-c{self.adaptor_config.attacked_class}" if self.adaptor_config.attacked_class is not None else ""
        mean_grad = f"_mean-grad" if self.adaptor_config.mean_grad else ""
        model_name = f"corrected_model_layer-{layer_index}_mode-{self.adaptor_config.cav_mode}_lamb{correction_strength}_{self.adaptor_config.rrc_loss}-loss{mean_grad}{attacked_class}_{self.adaptor_config.training.max_epochs}-epochs"

        self.model.eval()
        feature_extractor, downstream_head = None, None
        if layer_index == 0:
            downstream_head = self.model
        else:
            feature_extractor, downstream_head = split_model(self.model, layer_index, self.device)

        if layer_index != self.cav_cache.layer:
            activations, annotations = self.get_annotations_and_activations(data_train, feature_extractor=feature_extractor)
            cav = calculate_cav(activations, annotations, self.adaptor_config.projection_type)
            cav = cav.to(self.device, activations.dtype)
            self.cav_cache = CavCache(layer=layer_index, cav=cav, annotations=annotations, activations=activations)

        downstream_head, number_epochs_finetuned = self.finetune(data_train, data_val, self.cav_cache.cav, downstream_head, correction_strength, feature_extractor=feature_extractor, model_name=model_name)
        if layer_index == 0:
            self.model = downstream_head
        else:
            self.model = torch.nn.Sequential(*[feature_extractor, downstream_head])
        self.model.eval()

        if self.adaptor_config.save_model:
            corrected_model_path = os.path.join(self.adaptor_config.base_dir, model_name + ".cpl")
            print("saving corrected model to: " + corrected_model_path)
            torch.save(self.model.to("cpu"), corrected_model_path)
            self.model.to(self.device)

        return self.model, number_epochs_finetuned

    def finetune(self,
                 data_train: DataLoader,
                 data_val: DataLoader,
                 cav: torch.Tensor,
                 downstream_head: Module,
                 lamb: float,
                 feature_extractor: Module = None,
                 model_name: str = "corrected_model"):

        best_model = copy.deepcopy(downstream_head)
        best_rrc_loss = sys.maxsize
        if feature_extractor is None:
            val_accuracies = self.get_accuracies(data_val, downstream_head)
        else:
            feature_extractor.eval()
            val_accuracies = self.get_accuracies(data_val, Sequential(feature_extractor, downstream_head))
        best_val_group_acc = val_accuracies["avg_group_acc"]
        number_epochs_finetuned = 0

        downstream_head.train()
        data_train.dataset.disable_class_restriction()

        optimizer = torch.optim.SGD(downstream_head.parameters(), lr=self.adaptor_config.training.learning_rate, momentum=0.95)
        for epoch in range(self.adaptor_config.training.max_epochs):
            ce_losses = []
            rrc_losses = []
            accuracies = []

            with tqdm(data_train) as pbar:
                for batch in pbar:
                    optimizer.zero_grad()

                    optimizer.zero_grad()
                    x = batch["x"].to(self.device)
                    y = batch["y"].to(self.device).squeeze()

                    representation = (x if feature_extractor is None else feature_extractor(x)).requires_grad_()
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
                        break

                    loss.backward()
                    optimizer.step()

                    pbar.set_description(f'fine tuning epoch {epoch+1}/{self.adaptor_config.training.max_epochs}: accuracy={accuracy.mean()}, rrc_loss={rrc_loss}, ce_loss={ce_loss}')

            epoch_accuracy = torch.cat(accuracies).mean().item()
            epoch_ce_loss = torch.tensor(ce_losses).mean().item()
            epoch_rrc_loss = torch.tensor(rrc_losses).mean().item()
            if feature_extractor is None:
                val_accuracies = self.get_accuracies(data_val, downstream_head)
            else:
                val_accuracies = self.get_accuracies(data_val, Sequential(feature_extractor, downstream_head))

            if val_accuracies["avg_group_acc"] > best_val_group_acc or (epoch_rrc_loss < best_rrc_loss and val_accuracies["avg_group_acc"] == best_val_group_acc):
                best_val_group_acc = val_accuracies["avg_group_acc"]
                best_rrc_loss = epoch_rrc_loss
                best_model = copy.deepcopy(downstream_head)
                number_epochs_finetuned = epoch + 1

            self.log_writer.add_scalar(model_name + "/val/emp_acc", val_accuracies["accuracy"], epoch)
            self.log_writer.add_scalar(model_name + "/val/avg_group_acc", val_accuracies["avg_group_acc"], epoch)
            self.log_writer.add_scalar(model_name + "/val/worst_group_acc", val_accuracies["worst_group_acc"], epoch)
            self.log_writer.add_scalar(model_name + "/train/accuracy", epoch_accuracy, epoch)
            self.log_writer.add_scalar(model_name + "/train/ce_loss", epoch_ce_loss, epoch)
            self.log_writer.add_scalar(model_name + "/train/rrc_loss", epoch_rrc_loss, epoch)

            print(f"Epoch {epoch+1}: train_accuracy={epoch_accuracy}, val_accuracy={val_accuracies['accuracy']}, avg_group_acc={val_accuracies['avg_group_acc']}, worst_group_acc={val_accuracies['worst_group_acc']}, ce_loss={epoch_ce_loss}, rrc_loss={epoch_rrc_loss}")


        self.log_writer.flush()
        if self.adaptor_config.attacked_class is not None:
            data_train.dataset.enable_class_restriction(self.adaptor_config.attacked_class)

        return best_model, number_epochs_finetuned

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


def split_model(model: Module, split_at: int, device) -> (Module, Module):
    children_list = extract_all_children(model)[0]
    print(f"splitting model into {len(children_list)} children")

    # for i, node in enumerate(children_list):
    #     print(f"layer {i+1}: {node}")
    # exit()

    feature_extractor = torch.nn.Sequential(*children_list[:split_at])
    if isinstance(model, ResNet):
        downstream_head = torch.nn.Sequential(*children_list[split_at:-1], torch.nn.Flatten(start_dim=1), children_list[-1])
    else:
        downstream_head = torch.nn.Sequential(*children_list[split_at:])

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
    return cav / ((cav ** 2).sum() ** .5).item()


class CavProjection(torch.nn.Module):
    def __init__(self,
                 activations: torch.Tensor,
                 annotations: torch.Tensor,
                 cav: torch.Tensor,
                 cav_mode: str = None,
                 correction_strength: float = 1.0):
        super().__init__()
        self.correction_strength = correction_strength
        self.cav_mode = cav_mode

        self.cav = self.correction_strength * cav.reshape(-1, 1)
        z = torch.mean(activations[annotations == 0], dim=0, keepdim=True).to(device=cav.device, dtype=cav.dtype)
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
        super().__init__()
        activations = activations.flatten(start_dim=1)
        self.difference = activations[annotations == 1].mean(0) - activations[annotations == 0].mean(0)

    def forward(self, x):
        x_flat = x.flatten(start_dim=1)
        out = x_flat - self.difference
        out = torch.nn.functional.relu(out)
        return out.reshape(x.shape)

CavCache = namedtuple('CavCache', ["layer", "cav", "annotations", "activations"])