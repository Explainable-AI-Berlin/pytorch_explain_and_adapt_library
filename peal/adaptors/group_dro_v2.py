import copy
import os
from datetime import datetime
from typing import Union

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
import numpy as np

from peal.adaptors.interfaces import Adaptor, AdaptorConfig
from peal.architectures.interfaces import TaskConfig
from peal.data.dataloaders import create_dataloaders_from_datasource, get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig
from peal.global_utils import save_yaml_config
from peal.training.interfaces import TrainingConfig


class GroupDroLossComputer:

    def __init__(self, group_counts, alpha, gamma=0.1, adj=None, min_var_weight=0, step_size=0.01, normalize_loss=False, btl=False, device="cpu"):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.device = device

        self.n_groups = 4
        self.group_counts = group_counts
        self.group_frac = self.group_counts/self.group_counts.sum()
        self.group_str = ["0-0", "0-1", "1-0", "1-1"]

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().to(self.device)
        else:
            self.adj = torch.zeros(self.n_groups, device=device).float()

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups, device=device)/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups, device=device)
        self.exp_avg_initialized = torch.zeros(self.n_groups, device=device).byte()

        self.reset_stats()

    def loss(self, yhat, y, group_idx):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss)
        else:
            actual_loss, weights = self.compute_robust_loss(group_loss)

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss):
        adjusted_loss = self.exp_avg_loss + self.adj/torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0)<=self.alpha
        weights = mask.float() * sorted_frac /self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac*self.min_var_weight + weights*(1-self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups, device=self.device)
        self.update_data_counts = torch.zeros(self.n_groups, device=self.device)
        self.update_batch_counts = torch.zeros(self.n_groups, device=self.device)
        self.avg_group_loss = torch.zeros(self.n_groups, device=self.device)
        self.avg_group_acc = torch.zeros(self.n_groups, device=self.device)
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        self.update_data_counts += group_count*((weights>0).float())
        self.update_batch_counts += ((group_count*weights)>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, weight_decay, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, weight_decay=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            group_str = self.group_str[idx]
            stats_dict[f'avg_loss_group:{group_str}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{group_str}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{group_str}'] = self.avg_group_acc[idx].item()
            stats_dict[f'update_batch_count_group:{group_str}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()
        stats_dict['avg_group_acc'] = torch.mean(self.avg_group_acc).item()
        stats_dict['worst_group_acc'] = torch.min(self.avg_group_acc).item()

        # Model stats
        if model is not None:
            assert weight_decay is not None
            stats_dict = self.get_model_stats(model, weight_decay, stats_dict)

        return stats_dict


class GroupDROv2Config(AdaptorConfig):

    __name__: str = "peal.AdaptorConfig"
    model_path: str
    base_dir: str
    data: DataConfig
    unpoisoned_data: DataConfig = None
    training: TrainingConfig
    task: TaskConfig
    alpha: float = 0.2
    generalization_adjustment: Union[float, list[float]] = 0.0
    automatic_adjustment: bool = False
    robust_step_size: float = 0.01
    use_normalized_loss: bool = False
    btl: bool = False
    weight_decay: float = 5e-5
    gamma: float = 0.1
    minimum_variational_weight: float = 0.0
    track_test_acc: int = None
    save_intermediate: bool = True


class GroupDROv2(Adaptor):
    def __init__(self, adaptor_config: GroupDROv2Config):
        self.config = adaptor_config
        torch.manual_seed(self.config.seed)

        if os.path.isdir(adaptor_config.base_dir):
            dest = f"{adaptor_config.base_dir}_old_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            os.rename(adaptor_config.base_dir, dest)
        assert not os.path.exists(adaptor_config.base_dir)
        os.makedirs(adaptor_config.base_dir)

        save_yaml_config(self.config, os.path.join(adaptor_config.base_dir, "config.yaml"))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("running on device: ", self.device)
        self.original_model = torch.load(self.config.model_path, map_location=self.device)
        self.model = copy.deepcopy(self.original_model)

        self.train_dataloader, self.val_dataloader, _ = create_dataloaders_from_datasource(self.config)
        self.train_dataloader.dataset.return_dict = True
        self.train_dataloader.dataset.enable_groups()
        self.train_group_counts = torch.as_tensor(compute_group_sizes(self.train_dataloader), device=self.device)
        self.val_dataloader.dataset.return_dict = True
        self.val_dataloader.dataset.enable_groups()
        self.val_group_counts = torch.as_tensor(compute_group_sizes(self.val_dataloader), device=self.device)

        self.test_data_unpoisoned = None
        self.test_group_counts = None
        if self.config.unpoisoned_data is not None:
            self.test_data_unpoisoned = get_datasets(self.config.unpoisoned_data, return_dict=True)[-1]
            self.test_data_unpoisoned.enable_groups()
            self.test_data_unpoisoned = get_dataloader(self.test_data_unpoisoned, mode="test", batch_size=self.config.training.test_batch_size, task_config=self.config.task)
            self.test_group_counts = torch.as_tensor(compute_group_sizes(self.test_data_unpoisoned), device=self.device)

    def run(self):
        self.train()

    def train(self):
        log_writer = SummaryWriter(log_dir=os.path.join(self.config.base_dir, "logs"))

        adjustments = [self.config.generalization_adjustment] if isinstance(self.config.generalization_adjustment, float) else self.config.generalization_adjustment
        assert len(adjustments) in (1, 4)
        if len(adjustments)==1:
            adjustments = np.array(adjustments * 4)
        else:
            adjustments = np.array(adjustments)

        train_loss_computer = GroupDroLossComputer(
            self.train_group_counts,
            self.config.alpha,
            gamma=self.config.gamma,
            adj=adjustments,
            min_var_weight=self.config.minimum_variational_weight,
            step_size=self.config.robust_step_size,
            normalize_loss=self.config.use_normalized_loss,
            btl=self.config.btl,
            device=self.device)

        self.model.train()
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.training.learning_rate,
            momentum=0.9,
            weight_decay=self.config.weight_decay)

        best_val_acc = 0
        best_epoch = -1
        checkpoint_dir = os.path.join(self.config.base_dir, "checkpoints")
        os.makedirs(checkpoint_dir)

        val_loss_computer = GroupDroLossComputer(self.val_group_counts, self.config.alpha, step_size=self.config.robust_step_size, device=self.device)
        self.run_epoch(-1, self.model, optimizer, self.val_dataloader, val_loss_computer, log_writer, mode="val")

        if self.test_data_unpoisoned is not None:
            test_loss_computer = GroupDroLossComputer(self.test_group_counts, self.config.alpha, step_size=self.config.robust_step_size, device=self.device)
            self.run_epoch(-1, self.model, optimizer, self.test_data_unpoisoned, test_loss_computer, log_writer, mode="test")

        for epoch in range(self.config.training.max_epochs):
            print(f"epoch {epoch}/{self.config.training.max_epochs}")
            self.run_epoch(epoch, self.model, optimizer, self.train_dataloader, train_loss_computer, log_writer=log_writer, mode="train")

            val_loss_computer = GroupDroLossComputer(self.val_group_counts, self.config.alpha, step_size=self.config.robust_step_size, device=self.device)
            self.run_epoch(epoch, self.model, optimizer, self.val_dataloader, val_loss_computer, log_writer=log_writer, mode="val")

            if self.test_data_unpoisoned is not None and self.config.track_test_acc is not None and (epoch+1) % self.config.track_test_acc == 0:
                test_loss_computer = GroupDroLossComputer(self.test_group_counts, self.config.alpha, step_size=self.config.robust_step_size, device=self.device)
                self.run_epoch(epoch, self.model, optimizer, self.test_data_unpoisoned, test_loss_computer, log_writer=log_writer, mode="test")

            if self.config.save_intermediate:
                torch.save(self.model.to("cpu"), os.path.join(checkpoint_dir, f"model_epoch{epoch}.cpl"))

            if self.config.training.early_stopping_goal == "worst_group_accuracy":
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            elif self.config.training.early_stopping_goal == "average_group_accuracy":
                curr_val_acc = torch.mean(val_loss_computer.avg_group_acc).item()
            else:
                raise NotImplementedError

            # if args.reweight_groups:
            #     curr_val_acc = min(val_loss_computer.avg_group_acc)
            # else:
            #     curr_val_acc = val_loss_computer.avg_acc
            print(f'Current validation accuracy: {curr_val_acc} (best={best_val_acc} after epoch {best_epoch})')

            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                best_epoch = epoch
                torch.save(self.model.to("cpu"), os.path.join(self.config.base_dir, "model.cpl"))

            if self.config.automatic_adjustment:
                gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
                adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
                train_loss_computer.adj = adjustments.detach()
                for group_idx in range(train_loss_computer.n_groups):
                    log_writer.add_scalar(f"group_{train_loss_computer.group_str[group_idx]}_adj", train_loss_computer.adj[group_idx], epoch)

            self.model.to(self.device)

        if self.test_data_unpoisoned is not None:
            test_loss_computer = GroupDroLossComputer(self.test_group_counts, self.config.alpha, step_size=self.config.robust_step_size, device=self.device)
            best_model = torch.load(os.path.join(self.config.base_dir, "model.cpl"), map_location=self.device)
            if self.config.track_test_acc is not None:
                self.run_epoch(best_epoch, best_model, optimizer, self.test_data_unpoisoned, test_loss_computer, mode="test")
            else:
                self.run_epoch(best_epoch, self.model, optimizer, self.test_data_unpoisoned, test_loss_computer, log_writer, mode="test")
            stats = test_loss_computer.get_stats()
            log_writer.add_scalar(f"test_final_avg_group_acc", stats["avg_group_acc"], best_epoch)
            log_writer.add_scalar(f"test_final_worst_group_acc", stats["worst_group_acc"], best_epoch)

        log_writer.close()

    def run_epoch(self,
                  epoch: int,
                  model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  loader: DataLoader,
                  loss_computer: GroupDroLossComputer,
                  log_writer: SummaryWriter = None,
                  mode: str = "train"):

        is_training = mode == "train"
        if is_training:
            model.train()
        else:
            model.eval()

        with torch.set_grad_enabled(is_training):
            for batch_idx, batch in enumerate(tqdm(loader)):

                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device).squeeze().long()
                g = batch["has_confounder"].to(self.device).squeeze().long()
                g = 2*y + g

                outputs = model(x)

                loss_main = loss_computer.loss(outputs, y, g)

                if is_training:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()


        if log_writer is not None:
            for key, val in (loss_computer.get_stats(model, self.config.weight_decay) if is_training else loss_computer.get_stats()).items():
                log_writer.add_scalar(f"{mode}_{key}", val, epoch)
        if is_training:
            loss_computer.reset_stats()


def compute_group_sizes(dataloader) -> list[int]:
    group_sizes = [0,0,0,0]
    with tqdm(dataloader) as pbar:
        pbar.set_description(f"determining group sizes...")
        for batch in pbar:
            y = batch['y'].int()
            for i, has_confounder in enumerate(batch['has_confounder'].int()):
                has_confounder = has_confounder.item()
                if y[i].item() == 0:
                    if has_confounder == 0:
                        group_sizes[0] += 1
                    elif has_confounder == 1:
                        group_sizes[1] += 1
                elif y[i].item() == 1:
                    if has_confounder == 0:
                        group_sizes[2] += 1
                    elif has_confounder == 1:
                        group_sizes[3] += 1
    print("group sizes: ", group_sizes)
    return group_sizes