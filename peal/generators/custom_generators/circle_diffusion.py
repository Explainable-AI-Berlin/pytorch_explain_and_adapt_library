import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from typing import Tuple
import logging
from torch.utils.data import DataLoader
import math
from peal.generators.interfaces import EditCapableGenerator

logging.getLogger().setLevel(logging.INFO)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=500):
        super(PositionalEncoding, self).__init__()
        max_len += 1
        self.P = torch.zeros(max_len, embed_dim)
        freqs = torch.arange(max_len)[:, None] / (
            torch.pow(10000, torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim))

        self.P[:, 0::2] = torch.sin(freqs)
        self.P[:, 1::2] = torch.cos(freqs)

        self.P = self.P[1:]

    def forward(self, t):
        return self.P[t]


class ScoreNetwork(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(ScoreNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.layer1 = nn.LazyLinear(embed_dim)
        self.layer2 = nn.LazyLinear(embed_dim)
        self.layer3 = nn.LazyLinear(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.layer4 = nn.LazyLinear(input_dim)

    def forward(self, x, time_embed):
        x = self.layer1(x) + time_embed
        x = F.silu(self.layer2(x))
        x = F.silu(self.layer3(x))
        return self.layer4((self.norm(x)))


class BasicDiscreteTimeModel(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_timesteps: int):
        super(BasicDiscreteTimeModel, self).__init__()

        self.positional_embeddings = PositionalEncoding(
            embed_dim=embed_dim, max_len=num_timesteps
        )
        self.score_network = ScoreNetwork(input_dim=input_dim, embed_dim=embed_dim)
        # self.decoder = ScoreNetwork(input_dim=input_dim, embed_dim=embed_dim)

    def forward(self, x, t):
        time_embed = self.positional_embeddings(t)
        return self.score_network(x, time_embed)


class CircleDiffusionAdaptor(EditCapableGenerator):
    def __init__(self, config, dataset, model_dir=None, device="cpu"):
        super(CircleDiffusionAdaptor, self).__init__()
        # self.config = load_yaml_config(config)
        self.config = config

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            # self.model_dir = config['base_path']
            self.model_dir = config.base_path

        # if not os.path.exists(model_dir):
        #    os.mkdir(model_dir)
        # self.model_dir = model_dir
        # self.input_dim = config['input_dim']
        self.input_dim = config.input_dim

        try:
            # self.num_timesteps = config['num_timesteps']
            self.num_timesteps = config.num_timesteps
        except KeyError:
            pass

        self.dataset = dataset
        self.input_idx = [
            idx
            for idx, element in enumerate(self.dataset.attributes)
            if element not in ["Confounder", "Target"]
        ]
        self.target_idx = [
            idx
            for idx, element in enumerate(self.dataset.attributes)
            if element == "Target"
        ]
        # data = torch.zeros([len(dataset.data),len(dataset.attributes)], dtype=torch.float16)
        # for idx, key in enumerate(dataset.data):
        #    data[idx] = dataset.data[key]
        self.train_and_load_diffusion(model_name=config.model_name)

        def schedules(num_timesteps: int, type: str = "linear"):
            scale = 1000 / num_timesteps
            if type == "linear":
                min_var = scale * 1e-4
                max_var = scale * 1e-2
                return torch.linspace(
                    min_var, max_var, num_timesteps, dtype=torch.float32
                )
            elif type == "cosine":
                steps = num_timesteps + 1
                x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
                alphas_cumprod = (
                        torch.cos(
                            ((x / num_timesteps) + scale) / (1 + scale) * math.pi * 0.5
                        )
                        ** 2
                )
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                return torch.clip(betas, 0, 0.999)

        betas = schedules(num_timesteps=config.num_timesteps, type=config.var_schedule)

        self.register_buffer("beta", betas)
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(0))

    def forward_diffusion(
            self, clean_x: torch.Tensor, noise: torch.tensor, timestep: torch.Tensor
    ):
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep])
            alpha_bar_t = self.alpha_bar[timestep].repeat(clean_x.shape[0])[:, None]
        else:
            alpha_bar_t = self.alpha_bar[timestep][:, None]
        mu = torch.sqrt(alpha_bar_t)
        std = torch.sqrt(1 - alpha_bar_t)
        noisy_x = mu * clean_x + std * noise
        return noisy_x

    def reverse_diffusion_ddpm(
            self, noisy_x: torch.Tensor, model: nn.Module, timestep: torch.Tensor
    ):
        alpha_t = self.alpha[timestep].repeat(noisy_x.shape[0])[:, None]
        alpha_bar_t = self.alpha_bar[timestep].repeat(noisy_x.shape[0])[:, None]
        beta_t = 1 - alpha_t
        eps_hat = model(x=noisy_x, t=timestep)
        posterior_mean = (1 / torch.sqrt(alpha_t)) * (
                noisy_x - (beta_t / torch.sqrt(1 - alpha_bar_t) * eps_hat)
        )
        z = torch.randn_like(noisy_x)

        if timestep > 0:
            denoised_x = (
                    posterior_mean + torch.sqrt(beta_t) * z
            )  # * z * (timestep > 0))  # variance = beta_t
        else:
            denoised_x = posterior_mean

        return denoised_x

    def train_and_load_diffusion(self, model_name="diffusion.pt", mode=None):
        self.model_path = os.path.join(self.model_dir, model_name)
        model = BasicDiscreteTimeModel(
            input_dim=self.config.input_dim,
            embed_dim=self.config.embed_dim,
            num_timesteps=self.config.num_timesteps,
        )
        if model_name in os.listdir(self.model_dir) and not mode == "train":
            model.load_state_dict(torch.load(self.model_path))
            logging.info(f'Model found with path {self.model_path}')
        elif model_name not in os.listdir(self.model_dir) and mode != 'train':
            logging.info(
                'Model not found. Please run train_and_load_diffusion method and set its argument mode="train" ')
        else:
            logging.info(
                f'Training model with path {self.model_path}'
            )

        def diffusion_loss(model: nn.Module, clean_x: torch.Tensor) -> torch.Tensor:
            t = torch.randint(self.num_timesteps, (clean_x.shape[0],))
            eps_t = torch.randn_like(clean_x)
            alpha_bar_t = self.alpha_bar[t][:, None]
            x_t = self.forward_diffusion(clean_x=clean_x, noise=eps_t, timestep=t)
            # x_t = torch.sqrt(alpha_bar_t) * clean_x + torch.sqrt(1 - alpha_bar_t) * eps_t
            eps_hat = model(x=x_t, t=t)
            loss_diff = nn.MSELoss(reduction="sum")(eps_hat, eps_t)

            return loss_diff

        def run_epoch(
                model: nn.Module, dataloader: torch.utils.data.dataloader.DataLoader
        ):
            model.train()
            epoch_loss = 0.0

            for x, _ in dataloader:
                optimizer.zero_grad()
                loss = diffusion_loss(x)
                epoch_loss += loss
                loss.backward()
                optimizer.step()

            return epoch_loss / len(dataloader.dataset)

        if mode == "train":
            model.train()
            # num_epochs = self.config['num_epochs']
            num_epochs = self.config.num_epochs
            dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
            learning_rate = self.config.learning_rate
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            losses = []
            for i in trange(num_epochs):
                epoch_loss = 0.0
                for x, _ in dataloader:
                    optimizer.zero_grad()
                    loss = diffusion_loss(model, x[:, self.input_idx])
                    epoch_loss += loss
                    loss.backward()
                    optimizer.step()

                train_loss = epoch_loss / len(dataloader.dataset)
                print(f"Epoch: {i}, train_loss: {train_loss}")
                losses.append(train_loss.detach().numpy())

            torch.save(model.state_dict(), self.model_path)

        self.model = model

    @torch.no_grad()
    def sample_ddpm(self, model: nn.Module, n_samples: int = 256, label=None):
        """
        iteratively denoises pure noise to produce a list of denoised samples at each timestep
        """
        model.eval()
        x_pred = []
        x = torch.randn(n_samples, self.input_dim)
        x_pred.append(x)

        with torch.no_grad():
            for t in reversed(range(0, self.num_timesteps)):
                x = self.reverse_diffusion_ddpm(noisy_x=x, model=model, timestep=t)

                x_pred.append(x)
        return x_pred

    def sample_x(self, batch_size=1):
        x = self.sample_ddpm(model=self.model, n_samples=batch_size)[-1]
        return x

    def sample_counterfactual_ddpm(
            self,
            clean_batch: torch.Tensor,
            model: nn.Module,
            classifier: nn.Module,
            num_noise_steps: int,
            target_classes: int,
            classifier_grad_weight: float,
    ):
        classifier.eval()
        self.classifier = classifier

        # DEFINE BATCH SIZE AND COUNTERFACTUAL CLASS
        bs = clean_batch.shape[0]

        # COMPUTE CLEAN GRADIENTS FOR THE FIRST STEP

        classifier_criterion = lambda x: F.cross_entropy(classifier(x), target_classes)
        clean_batch_copy = torch.nn.Parameter(clean_batch)
        loss = classifier_criterion(clean_batch_copy)
        loss.backward()
        clean_grad = classifier_grad_weight * clean_batch_copy.grad.detach()

        # REDEFINING VARIABLES AND PERFORMING FORWARD DIFFUSION
        eps_t = torch.randn_like(clean_batch)

        next_z = self.forward_diffusion(
            clean_x=clean_batch, noise=eps_t, timestep=num_noise_steps
        )
        counterfactuals = []  # total counterfactuals
        counterfactuals.append(clean_batch)
        guided_grads = []  # guided grads at the first step
        unconditional_grads = []  # diffusion grads at the first step
        total_series = (
            []
        )  # contains evolution from noisy to cleaned instance for each data point
        for i in tqdm(range(0, num_noise_steps)[::-1]):
            # Denoise z_t to create z_t-1 (next z)
            alpha_i = self.alpha[i].repeat(bs)[:, None]
            alpha_bar_i = self.alpha_bar[i].repeat(bs)[:, None]
            sigma_i = torch.sqrt(1 - self.alpha[i])
            eps_hat = model(next_z, i)

            # Unconditional mean
            unconditional_grad = -eps_hat / torch.sqrt(1 - alpha_bar_i)
            z_t_mean = (next_z + unconditional_grad * (1 - alpha_i)) / torch.sqrt(
                alpha_i
            )

            # Guided mean
            z_t_mean -= sigma_i * (clean_grad / torch.sqrt(alpha_bar_i))

            if i > 0:
                next_z = z_t_mean + (sigma_i * torch.randn_like(clean_batch))
            else:
                next_z = z_t_mean

            next_x = next_z.clone()
            # Denoise to create a cleaned x (next x)
            series = []
            series.append(next_x.detach())
            for t in range(0, i)[::-1]:
                if i == 0:
                    break
                next_x = self.reverse_diffusion_ddpm(
                    noisy_x=next_x, model=model, timestep=t
                )
                series.append(next_x.detach())
            total_series.append(series)
            guided_grads.append(
                -sigma_i * clean_grad.detach() / torch.sqrt(alpha_bar_i)
            )
            unconditional_grads.append(
                unconditional_grad.detach() * (1 - alpha_i) / torch.sqrt(alpha_i)
            )

            if i != 0:
                counterfactuals.append(next_x.detach())

            # Gradient wrt denoised image (next_x)
            next_x_copy = torch.nn.Parameter(next_x.clone())
            loss = classifier_criterion(next_x_copy)
            loss.backward()
            clean_classifier_grad = next_x_copy.grad.detach()
            clean_grad = classifier_grad_weight * clean_classifier_grad

            # self.counterfactuals = counterfactuals
            # self.guided_grads = guided_grads
            # self.diffusion_grads = unconditional_grads
            # self.pointwise_evolution = total_series

        counterfactuals = torch.stack(counterfactuals).permute(1, 0, 2)
        guided_grads = torch.stack(guided_grads).permute(1, 0, 2)
        unguided_grads = torch.stack(unconditional_grads).permute(1, 0, 2)

        self.counterfactuals_series = counterfactuals
        self.guided_grads = guided_grads
        self.unguided_grads = unguided_grads

        return counterfactuals, guided_grads, unguided_grads, total_series


    def discard_counterfactuals(self, counterfactuals, classifier, target_classes, target_confidence,
                                minimal_counterfactuals, tolerance=0.1):

        # compute distance of current minimal_counterefactuals from radius 1.0
        # current_counterfactual_distance_from_manifold = torch.abs((torch.pow(minimal_counterfactuals, 2).sum(dim=-1) - 1.0))

        for i in range(len(counterfactuals)):

            # compute classifier  for all the counterfactuals for each point
            new_counterfactuals_confidence = classifier(counterfactuals[i]).softmax(dim=-1)[:, target_classes[i]]

            # check if new counterfactuals satisfy the confidence constraint

            new_confidence_satisfied_indices = torch.nonzero(new_counterfactuals_confidence > target_confidence)

            current_confidence_satisfied = classifier(minimal_counterfactuals[i:i + 1]).softmax(dim=-1)[0][
                                               target_classes[i]].item() > target_confidence

            # if current counterfactual satisfies confidence and tolerance, maintain status quo

            if current_confidence_satisfied:
                continue

            elif new_confidence_satisfied_indices.nelement() != 0:
                minimal_counterfactuals[i] = counterfactuals[i][new_confidence_satisfied_indices[0].item()]

            else:
                minimal_counterfactuals[i] = counterfactuals[i][-1]

        return minimal_counterfactuals

    def edit(
            self,
            x_in: torch.Tensor,
            target_confidence_goal: float,
            target_classes: torch.Tensor,
            classifier: nn.Module,
            **kwargs,
    ) -> Tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:

        self.original_sample = x_in

        #scales = self.config['grad_scales']
        #noise_steps = self.config['noise_steps_for_counterfactuals']

        scales = self.config.grad_scales
        noise_steps = self.config.noise_steps_for_counterfactuals

        minimal_counterfactuals = x_in.clone()

        for steps in noise_steps:
            for s in scales:
                (
                    counterfactuals,
                    guided_grads,
                    unguided_grads,
                    total_series,
                ) = self.sample_counterfactual_ddpm(
                    clean_batch=minimal_counterfactuals,
                    model=self.model,
                    classifier=classifier,
                    num_noise_steps=steps,
                    target_classes=target_classes,
                    classifier_grad_weight=s,
                )

                minimal_counterfactuals = self.discard_counterfactuals(
                    counterfactuals=counterfactuals,
                    classifier=classifier,
                    target_confidence=target_confidence_goal,
                    target_classes=target_classes,
                    minimal_counterfactuals=minimal_counterfactuals,
                )

        list_counterfactuals = [row_tensor for row_tensor in minimal_counterfactuals]
        diff_latent = x_in - minimal_counterfactuals

        confidences = classifier(minimal_counterfactuals).softmax(dim=-1)
        y_target_end_confidence = [
            confidences[i][target_classes[i]].detach()
            for i in range(len(minimal_counterfactuals))
        ]
        x_list = [row_tensor for row_tensor in x_in]

        self.counterfactuals = minimal_counterfactuals
        self.original_sample = x_in

        return list_counterfactuals, diff_latent, y_target_end_confidence, x_list

    def plot_counterfactuals(self):
        plt.figure(figsize=(5, 5))
        data = torch.zeros(
            [len(self.dataset.data), len(self.dataset.attributes)], dtype=torch.float16
        )
        for idx, key in enumerate(self.dataset.data):
            data[idx] = self.dataset.data[key]
        print(data)
        plt.scatter(
            data[:, self.input_idx[0]],
            data[:, self.input_idx[1]],
            c=np.where(data[:, self.target_idx] == 0, "lightcyan", "lightgray")[0],
        )
        for i, point in enumerate(self.counterfactuals):
            plt.scatter(
                self.original_sample[i, 0], self.original_sample[i, 1], color="green"
            )
            plt.scatter(point[0], point[1])
            plt.arrow(
                self.original_sample[i, 0],
                self.original_sample[i, 1],
                # plot the original point plus arrow until (j+granularity)th point
                point[0] - self.original_sample[i, 0],
                point[1] - self.original_sample[i, 1],
                head_width=0.05,
                head_length=0.05,
                fc="blue",
                ec="blue",
            )
        plt.show()

    __name__ = "CircleDiffusionAdaptor"
