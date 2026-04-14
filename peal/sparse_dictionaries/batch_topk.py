import torch
import tqdm
import copy
import os
from typing import Union
from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig
from peal.dependencies.matryoshka_sae.sae import BatchTopKSAE as InternalBatchTopKSAE
from peal.dependencies.matryoshka_sae.training import train_sae_group_seperate_wandb
from peal.sparse_dictionaries.sae_evaluation import compute_component_f1_scores

class BatchTopKSAEConfig(SparseDictionaryConfig):
    n_components: Union[int, None] = 10016
    sparse_dictionaries_type: str = 'BatchTopKSAE'
    top_k: int = 12
    lr: float = 3e-4
    l1_coeff: float = 0.0
    device: str = 'cuda'
    model_batch_size: int = 32
    batch_size: int = 4096
    num_tokens: int = 1000000
    n_batches_to_dead: int = 100
    top_k_aux: int = 4
    aux_penalty: float = 1.0/32.0
    dtype: str = "torch.float32"
    seed: int = 42
    input_unit_norm: bool = False
    eval_epoch_interval: int = 50
    cfg: dict = {}

class BatchTopKSAE(SparseDictionary):
    def __init__(self, config=BatchTopKSAEConfig()):
        self.config = config
        
        # Prepare internal config for the dependency's SAE class
        if isinstance(self.config.cfg, dict) and len(self.config.cfg) > 0:
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
                "batch_size": self.config.batch_size,
                "num_tokens": self.config.num_tokens,
                "n_batches_to_dead": self.config.n_batches_to_dead,
                "top_k_aux": self.config.top_k_aux,
                "aux_penalty": self.config.aux_penalty,
                "dtype": torch.float32, 
                "seed": self.config.seed,
                "beta1": 0.9,
                "beta2": 0.99,
                "max_grad_norm": 100000,
                "perf_log_freq": 1000,
                "sae_type": "topk",
                "checkpoint_freq": 10000,
                "model_name": "unknown_model",
                "name": f"BatchTopKSAE_{self.config.n_components}",
                "wandb_project": "sparse_autoencoders",
                "input_unit_norm": self.config.input_unit_norm,
            }
        
        self.sae = InternalBatchTopKSAE(cfg)
        self.mu = torch.zeros(self.config.act_size)

    def fit(self, X):
        # We wrap the data in an ActivationStore as expected by train_sae_group_seperate_wandb
        saes = [self.sae.to(self.config.device)]
        cfgs = [self.sae.config]
        
        train_sae_group_seperate_wandb(saes, X, None, cfgs)

    def fit_with_evaluation(self, activation_store, ground_truth_labels, log_dir=None):
        """
        Train the SAE with periodic F1 evaluation against ground truth labels.
        Logs F1 scores for the first 10 components to TensorBoard.
        
        Args:
            activation_store: ActivationStore wrapping the centered activations
            ground_truth_labels: Tensor (N, K) of ground truth labels
            log_dir: Directory for TensorBoard logs. If None, uses config base_path.
        """
        from torch.utils.tensorboard import SummaryWriter
        
        if log_dir is None:
            log_dir = os.path.join(self.config.base_path or ".", "sae_logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        sae = self.sae.to(self.config.device)
        cfg = sae.config
        
        # Ensure required keys
        if "num_tokens" not in cfg:
            cfg["num_tokens"] = self.config.num_tokens
        if "batch_size" not in cfg:
            cfg["batch_size"] = self.config.batch_size
        
        num_batches = int(cfg["num_tokens"] // cfg["batch_size"])
        print(f"Number of batches: {num_batches}")
        
        optimizer = torch.optim.Adam(
            sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
        )
        pbar = tqdm.trange(num_batches)
        eval_interval = self.config.eval_epoch_interval
        
        # Keep a copy of the raw (un-centered) activations for F1 eval
        raw_X = activation_store.X + self.mu  # un-center for evaluation
        
        epoch_counter = 0
        for i in pbar:
            batch = activation_store.next_batch()
            sae_output = sae(batch)
            loss = sae_output["loss"]
            
            # Log training metrics
            writer.add_scalar("train/loss", loss.item(), i)
            writer.add_scalar("train/l0_norm", sae_output["l0_norm"], i)
            writer.add_scalar("train/l2_loss", sae_output["l2_loss"], i)
            
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "L0": f"{sae_output['l0_norm']:.4f}",
                "L2": f"{sae_output['l2_loss']:.4f}",
            })
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()
            
            # Periodic F1 evaluation
            if (i + 1) % eval_interval == 0:
                f1_results = compute_component_f1_scores(
                    sae=sae,
                    activation_store_X=raw_X,
                    ground_truth_labels=ground_truth_labels,
                    mu=self.mu,
                    device=self.config.device,
                    n_components=10,
                )
                for key, value in f1_results.items():
                    writer.add_scalar(f"eval/{key}", value, epoch_counter)
                
                mean_f1 = f1_results.get("mean_top10_f1", 0.0)
                print(f"[Eval epoch {epoch_counter}] mean_top10_f1={mean_f1:.4f}")
                epoch_counter += 1
        
        # Final evaluation
        f1_results = compute_component_f1_scores(
            sae=sae,
            activation_store_X=raw_X,
            ground_truth_labels=ground_truth_labels,
            mu=self.mu,
            device=self.config.device,
            n_components=10,
        )
        for key, value in f1_results.items():
            writer.add_scalar(f"eval/{key}", value, epoch_counter)
        
        mean_f1 = f1_results.get("mean_top10_f1", 0.0)
        print(f"[Final eval] mean_top10_f1={mean_f1:.4f}")
        
        writer.close()
        print(f"TensorBoard logs saved to {log_dir}")

    def fit_from_dataloaders(self, dataloaders, feature_extractor=lambda x: x):
        X_list = []
        y_list = []
        for dataloader in dataloaders:
            for batch in dataloader:
                # feature_extractor is expected to return tensor on device
                features = feature_extractor(batch[0].to(self.config.device))
                X_list.append(features.detach().cpu())
                # Collect ground truth labels if available
                if len(batch) > 1:
                    y_list.append(batch[1].detach().cpu())

        X = torch.cat(X_list, dim=0)
        self.mu = torch.mean(X, dim=0)
        
        # Ensure num_tokens and batch_size are correctly set in sae.config
        if "num_tokens" not in self.sae.config:
            self.sae.config["num_tokens"] = self.config.num_tokens
        if "batch_size" not in self.sae.config:
            self.sae.config["batch_size"] = self.config.batch_size
            
        from peal.sparse_dictionaries.matryoshka_sparse_autoencoder import ActivationStore
        activation_store = ActivationStore(self.sae.config, X - self.mu)
        
        # If ground truth labels were collected, use fit_with_evaluation
        if y_list:
            ground_truth_labels = torch.cat(y_list, dim=0)
            self.fit_with_evaluation(activation_store, ground_truth_labels)
        else:
            self.fit(activation_store)

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
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
