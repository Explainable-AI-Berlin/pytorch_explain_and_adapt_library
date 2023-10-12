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


class VAE(nn.Module):
    def __init__(self, input_dim: int, encoder_dims: list, decoder_dims: list, latent_dim: int):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential()
        for i, dim in enumerate(encoder_dims):
            self.encoder.add_module(f'layer_{i + 1}', nn.Sequential(nn.LazyLinear(dim), nn.SiLU()))
        self.encoder.add_module('norm_encoder', nn.LayerNorm(dim))

        self.latent_mean = nn.Sequential(nn.LazyLinear(latent_dim))
        self.latent_logvar = nn.Sequential(nn.LazyLinear(latent_dim))

        self.decoder = nn.Sequential()
        for i, dim in enumerate(decoder_dims):
            self.decoder.add_module(f'layer_{i + 1}', nn.Sequential(nn.LazyLinear(dim), nn.SiLU()))
        self.decoder.add_module('norm_decoder', nn.LayerNorm(dim))
        self.decoder.add_module(f'to_original', nn.Sequential(nn.LazyLinear(input_dim)))

    def reparameterize(self, mean, logvar):
        if self.training:
            z = mean + torch.exp(logvar * 0.5) * torch.randn_like(logvar)
        else:
            z = mean
        return z

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.latent_mean(x), self.latent_logvar(x)
        return mean, logvar

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def sample(self, num_samples):
        eps = torch.randn([num_samples, self.latent_dim])
        return self.decode(eps).detach()

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


class CircleVAEAdaptor(EditCapableGenerator):
    def __init__(self, config, dataset, model_dir=None, **kwargs):
        super(CircleVAEAdaptor, self).__init__()
        self.model = None
        self.config = config
        self.dataset = dataset

        if not model_dir is None:
            self.model_dir = model_dir
        else:
            self.model_dir = config.base_path

        self.input_dim = config.input_dim
        self.encoder_dims = config.encoder_dims
        self.decoder_dims = config.decoder_dims
        self.latent_dim = config.latent_dim

        self.train_and_load_vae(model_name=config.model_name)

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

    def train_and_load_vae(self, model_name="vae.pt", mode=None):
        self.model_path = os.path.join(self.model_dir, model_name)
        model = VAE(input_dim=self.input_dim, encoder_dims=self.encoder_dims, decoder_dims=self.decoder_dims,
                    latent_dim=self.latent_dim)

        if model_name in os.listdir(self.model_dir) and not mode == "train":
            model.load_state_dict(torch.load(self.model_path))
            logging.info(f'Model found with path {self.model_path}')
        elif model_name not in os.listdir(self.model_dir) and mode != 'train':
            logging.info('Model not found. Please run train_and_load_vae method and set its argument mode="train" ')
        else:
            logging.info(
                f'Training model with path {self.model_path}'
            )

        def VAELoss(x, x_hat, mean, logvar, beta=self.config.beta):
            kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0)
            reconstruction_loss = F.mse_loss(x, x_hat)
            return reconstruction_loss + beta * kl_loss

        def train(model, data_loader, epochs):
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            for epoch in tqdm(range(epochs)):
                total_loss = 0.0
                for x, y in data_loader:
                    x = x
                    optimizer.zero_grad()
                    x_hat, mean, logvar = model(x[:, self.input_idx])
                    loss = VAELoss(x[:, self.input_idx], x_hat=x_hat, mean=mean, logvar=logvar)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                print(f'Epoch: {epoch}, Loss: {loss}')

            return x, x_hat

        if mode == 'train':
            model.train()
            dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
            train(model=model, data_loader=dataloader, epochs=self.config.num_epochs)
            torch.save(model.state_dict(), self.model_path)

        self.model = model

    def sample_x(self, batch_size=1):
        return self.model.sample(num_samples=batch_size).detach()

    def DIVE(self, clean_batch, target_classes, model, classifier):

        classifier.eval()

        lasso_weight = self.config.lasso_weight
        reconstruction_weight = self.config.reconstruction_weight

        batch_size, _ = clean_batch.size()
        latent_dim = self.latent_dim

        mean, logvar = self.model.encode(clean_batch)
        z = model.reparameterize(mean, logvar).detach()

        epsilon = torch.randn_like(z, requires_grad=True)
        epsilon.data *= 0.01
        optimizer = torch.optim.Adam([epsilon], lr=self.config.lr_counterfactual, weight_decay=0)

        for it in range(self.config.num_iterations):
            optimizer.zero_grad()

            z_perturbed = z + epsilon  # no grads required for latents

            decoded = model.decode(z_perturbed)

            classifier_criterion = lambda x: F.cross_entropy(classifier(x), target_classes)
            loss_attack = classifier_criterion(decoded)

            recon_regularizer = reconstruction_weight * torch.abs((clean_batch - decoded).mean(dim=-1)).sum()
            lasso_regularizer = lasso_weight * (torch.abs(z_perturbed - z)).sum()
            regularizer = recon_regularizer + lasso_regularizer

            loss = loss_attack + regularizer

            loss.backward()

            optimizer.step()

        return clean_batch, model.decode(z + epsilon).detach()

    def edit(
            self,
            x_in: torch.Tensor,
            target_confidence_goal: float,
            target_classes: torch.Tensor,
            classifier: torch.nn.Module,
            **kwargs,
    ) -> Tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        """
        Edit a batch of samples to achieve a target confidence goal.
        Args:
            x_in: Batch of samples to edit.
            target_confidence_goal: Target confidence goal.
            target_classes: Target classes for each sample in the batch.
            classifier: Classifier to use for confidence estimation.
            **kwargs: Additional keyword arguments.
        Returns:
            Tuple of (edited samples, confidence estimates, number of iterations, number of queries).
        """

        list_counterfactuals = torch.zeros_like(x_in)
        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        counterfactuals = x_in
        for i in range(len(x_in)):
            #while True:
            _, counterfactual = self.DIVE(x_in[i:i + 1], target_classes[i:i + 1], self.model, classifier)
            current_confidence = classifier(counterfactual).softmax(dim=-1)[0][target_classes[i].item()].item()
            #    if current_confidence > target_confidence_goal:
            #        break
            y_target_end_confidence[i] = current_confidence
            list_counterfactuals[i] = counterfactual

        diff_latent = x_in - list_counterfactuals

        x_list = [row_tensor for row_tensor in x_in]

        return list(list_counterfactuals), diff_latent, y_target_end_confidence, x_list
