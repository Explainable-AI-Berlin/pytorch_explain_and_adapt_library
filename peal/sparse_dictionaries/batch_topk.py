import torch
import copy
import os
from typing import Union
from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig
from peal.dependencies.matryoshka_sae.sae import BatchTopKSAE as InternalBatchTopKSAE
from peal.dependencies.matryoshka_sae.training import train_sae_group_seperate_wandb

class BatchTopKSAEConfig(SparseDictionaryConfig):
    n_components: Union[int, None] = 10016
    sparse_dictionaries_type: str = 'BatchTopKSAE'
    top_k: int = 12
    lr: float = 3e-4
    l1_coeff: float = 0.0
    device: str = 'cuda'
    model_batch_size: int = 32
    n_batches_to_dead: int = 100
    top_k_aux: int = 4
    aux_penalty: float = 1.0/32.0
    dtype: str = "torch.float32"
    seed: int = 42
    input_unit_norm: bool = False
    cfg: dict = {}

class BatchTopKSAE(SparseDictionary):
    def __init__(self, config=BatchTopKSAEConfig()):
        self.config = config
        
        # Prepare internal config for the dependency's SAE class
        if isinstance(self.config.cfg, dict):
            cfg = self.config.cfg
        else:
            cfg = {
                "dict_size": self.config.n_components,
                "act_size": self.config.act_size,
                "top_k": self.config.top_k,
                "lr": self.config.lr,
                "l1_coeff": self.config.l1_coeff,
                "device": self.config.device,
                "model_batch_size": self.config.model_batch_size,
                "n_batches_to_dead": self.config.n_batches_to_dead,
                "top_k_aux": self.config.top_k_aux,
                "aux_penalty": self.config.aux_penalty,
                "dtype": torch.float32, 
                "seed": self.config.seed,
                "input_unit_norm": self.config.input_unit_norm,
            }
        
        self.sae = InternalBatchTopKSAE(cfg)
        self.mu = torch.zeros(self.config.act_size)

    def fit(self, X):
        # We wrap the data in an ActivationStore as expected by train_sae_group_seperate_wandb
        saes = [self.sae.to(self.config.device)]
        cfgs = [self.sae.config]
        
        train_sae_group_seperate_wandb(saes, X, None, cfgs)

    def fit_from_dataloaders(self, dataloaders, feature_extractor=lambda x: x):
        X_list = []
        for dataloader in dataloaders:
            for batch in dataloader:
                # feature_extractor is expected to return tensor on device
                features = feature_extractor(batch[0].to(self.config.device))
                X_list.append(features.detach().cpu())

        X = torch.cat(X_list, dim=0)
        self.mu = torch.mean(X, dim=0)
        # Use ActivationStore from matryoshka_sparse_autoencoder to avoid redefinition if possible,
        # but for simplicity we define it here if needed or import it.
        from peal.sparse_dictionaries.matryoshka_sparse_autoencoder import ActivationStore
        self.fit(ActivationStore(self.sae.config, X - self.mu))

    def get_components(self):
        return self.sae.W_enc

    def save_on_disk(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        torch.save({
            'W_enc': self.sae.W_enc.cpu(),
            'W_dec': self.sae.W_dec.cpu(),
            'b_enc': self.sae.b_enc.cpu(),
            'b_dec': self.sae.b_dec.cpu(),
            'mu': self.mu.cpu(),
            'threshold': self.sae.threshold.cpu() if hasattr(self.sae, 'threshold') else None
        }, path)

    def load_from_disk(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.sae.W_enc.data = checkpoint['W_enc'].to(self.config.device)
        self.sae.W_dec.data = checkpoint['W_dec'].to(self.config.device)
        self.sae.b_enc.data = checkpoint['b_enc'].to(self.config.device)
        self.sae.b_dec.data = checkpoint['b_dec'].to(self.config.device)
        self.mu = checkpoint['mu'].to(self.config.device)
        if 'threshold' in checkpoint and checkpoint['threshold'] is not None:
             self.sae.threshold.data = checkpoint['threshold'].to(self.config.device)

    def encode(self, x):
        # x is (Batch, act_size)
        # Handle normalization if needed
        x_cent = x - self.mu.to(x.device)
        acts = self.sae.encode(x_cent)
        return acts

    def decode(self, acts):
        # acts is (Batch, dict_size)
        recon = self.sae.decode(acts)
        return recon + self.mu.to(recon.device)
