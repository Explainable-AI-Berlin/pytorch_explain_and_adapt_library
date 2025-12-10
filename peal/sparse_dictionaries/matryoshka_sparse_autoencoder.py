import torch
import copy

from typing import Union

from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig
from peal.dependencies.matryoshka_sae.training import train_sae_group_seperate_wandb
from peal.dependencies.matryoshka_sae.sae import BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE
from peal.dependencies.matryoshka_sae.config import get_default_cfg


class MatryoshkaSAEConfig(SparseDictionaryConfig):
    n_components: Union[int, None] = 100
    sparse_dictionaries_type: str = 'MatryoshkaSAE'
    ending: str = '.npz'
    cfg: dict = {}

class MatryoshkaSAE(SparseDictionary):
    def __init__(self, config=MatryoshkaSAEConfig()):
        self.config = config
        self.config.cfg = get_default_cfg()
        self.config.cfg["model_name"] = "gemma-2-2b"
        self.config.cfg["layer"] = 8
        self.config.cfg["site"] = "resid_pre"
        self.config.cfg["dataset_path"] = "Skylion007/openwebtext"
        self.config.cfg["aux_penalty"] = float(1.0/32.0)
        self.config.cfg["lr"] = 3e-4
        self.config.cfg["input_unit_norm"] = False
        self.config.cfg["dict_size"] = 32 # 36864
        self.config.cfg['wandb_project'] = 'batch-topk-matryoshka'
        self.config.cfg['l1_coeff'] = 0.
        self.config.cfg['act_size'] = 1024 #512 #2304
        self.config.cfg['device'] = 'cuda'
        self.config.cfg['bandwidth'] = 0.001
        self.config.cfg["top_k_matryoshka"] = [4] #[10, 10, 10, 10, 10]
        self.config.cfg["top_k"] = 4 #[10, 10, 10, 10, 10]
        self.config.cfg["group_sizes"] = [32] #[2304//4, 2304 // 4 ,2304 // 2, 2304, 2304*2, 2304*4, 2304*8]
        self.config.cfg["num_tokens"] = 5e8
        self.config.cfg["model_batch_size"] = 32
        self.config.cfg["model_dtype"] = "torch.bfloat16"
        self.config.cfg["no_bias"] = True
        self.sae = GlobalBatchTopKMatryoshkaSAE(self.config.cfg)
        self.mu = torch.zeros(self.sae.W_dec.shape[1])

    def fit(self, X):
        # Train the BatchTopK SAEs
        saes = []
        cfgs = []
        """dict_sizes = [8, 16, 32, 512] #[2304, 2304*2, 2304*4, 2304*8, 2304*16]
        topks = [4, 4, 4, 8] #[22, 25, 27, 29, 32]
        cfg = copy.deepcopy(self.config.cfg)



        for i, (dict_size, topk) in enumerate(zip(dict_sizes, topks)):
            cfg = copy.deepcopy(cfg)
            cfg["sae_type"] = 'batch-topk'
            cfg["dict_size"] = dict_size
            cfg["top_k"] = topk

            cfg["name"] = f"{cfg['model_name']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
            sae = BatchTopKSAE(cfg)
            saes.append(sae)
            cfgs.append(cfg)

        # Train the Matryoshka SAE
        dict_size = 512 #2304*16
        topk = 8 #32
        cfg = copy.deepcopy(cfg)
        cfg["sae_type"] = 'global-matryoshka-topk'
        cfg["dict_size"] = dict_size
        cfg["top_k"] = topk
        cfg["group_sizes"] = [dict_size // 16, dict_size // 16, dict_size // 8, dict_size // 4, dict_size // 2]

        cfg["name"] = f"{cfg['model_name']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
        sae = GlobalBatchTopKMatryoshkaSAE(cfg)
        saes.append(sae)"""
        self.sae = GlobalBatchTopKMatryoshkaSAE(self.config.cfg).to("cuda")
        saes.append(self.sae)
        cfgs.append(self.config.cfg)

        train_sae_group_seperate_wandb(saes, X, None, cfgs)

    def fit_from_dataloaders(self, dataloaders, feature_extractor=lambda x: x):
        X_list = []
        # derive which device to use from feature extractor

        for dataloader in dataloaders:
            for batch in dataloader:
                X_list.append(feature_extractor(batch[0].cuda()).cpu())

        X = torch.cat(X_list, dim=0)
        self.mu = torch.mean(X, dim=0)
        self.fit(ActivationStore(self.config, X - self.mu))

    def get_components(self):
        return self.sae.W_enc.transpose(0, 1)

    def save_on_disk(self, path):
        torch.save({
            'W_enc': self.sae.W_enc.cpu(),
            'W_dec': self.sae.W_dec.cpu(),
            'mu': self.mu.cpu(),
        }, path)

    def load_from_disk(self, path):
        checkpoint = torch.load(path)
        self.sae.W_enc.data = checkpoint['W_enc']
        self.sae.W_dec.data = checkpoint['W_dec']
        self.mu = checkpoint['mu'] if 'mu' in checkpoint else torch.zeros(self.sae.W_dec.shape[1])

class ActivationStore:
    def __init__(self, config, X):
        self.config = config
        self.X = X
        self.current_index = 0

    def next_batch(self):
        batch_size = self.config.cfg["model_batch_size"]
        if self.current_index + batch_size > self.X.shape[0]:
            self.current_index = 0  # Reset if we exceed the dataset size
        batch = self.X[self.current_index:self.current_index + batch_size]
        self.current_index += batch_size
        return batch.to("cuda")