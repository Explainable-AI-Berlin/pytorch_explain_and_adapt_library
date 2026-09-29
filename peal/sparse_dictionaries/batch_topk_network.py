"""PEAL's own batch top-K sparse autoencoder, its trainer and an activation source.

This replaces the vendored ``peal.dependencies.matryoshka_sae`` backend, which
was a copy of https://github.com/bartbussmann/matryoshka_sae. That repository
publishes no licence file, so all rights stay with its authors by default and
PEAL could neither redistribute it nor ship as an openly licensed library while
depending on it (see ``THIRD_PARTY_NOTICES.md``).

The method is Bussmann, Leask and Nanda, *BatchTopK Sparse Autoencoders* (2024);
methods are not copyrightable, only a particular expression of them, and this is
PEAL's expression of it, LGPL-3.0 like the rest of PEAL. The two MIT
reimplementations (SAELens ``batchtopk_sae.py`` and dictionary_learning
``batch_top_k.py``) were not usable as dependencies: both declare
``requires-python >= 3.10`` while PEAL runs on 3.9, and both pull a language
model stack (transformer-lens / nnsight, datasets) for what is a 512-dimensional
image-representation dictionary.

Compatibility is deliberate and verified: parameter names, shapes, the
initialisation order and the forward pass match the previous backend, so
checkpoints written before the swap load unchanged and the published Sparse
Numbers dictionary keeps its meaning.

What batch top-K does: a plain top-K autoencoder keeps the K largest codes *per
sample*, which forces every sample to use exactly K atoms. Batch top-K instead
keeps the ``K * batch_size`` largest codes across the whole batch, so the budget
is shared and a sample may use more or fewer than K atoms while the average
stays K. At inference there is no batch to pool over, so a running threshold
learned during training replaces the top-K selection.
"""

import torch
import torch.nn.functional as F
from torch import nn


def get_default_cfg():
    """Return the default trainer/model config dict.

    Only the keys PEAL actually uses are present; the previous backend also
    carried language-model keys (``hook_point``, ``seq_len``, ``dataset_path``,
    ``site``, ``layer``) that no PEAL code path reads.

    Returns
    -------
    dict
        Defaults, to be overridden by ``BatchTopKSAEConfig`` fields.
    """
    return {
        "seed": 49,
        "batch_size": 4096,
        "lr": 3e-4,
        "num_tokens": int(1e9),
        "l1_coeff": 0,
        "orth_coeff": 0.0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "dtype": torch.float32,
        "act_size": 768,
        "dict_size": 12288,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_batch_size": 512,
        "input_unit_norm": True,
        "sae_type": "topk",
        "checkpoint_freq": 10000,
        "n_batches_to_dead": 20,
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": (1 / 32),
        "name": "batch_topk_sae",
    }


class SparseAutoencoderBase(nn.Module):
    """Shared parameters and housekeeping of a tied-init sparse autoencoder.

    Parameters
    ----------
    cfg : dict
        Needs ``seed``, ``act_size``, ``dict_size``, ``device``, ``dtype`` and
        ``input_unit_norm``.

    Attributes
    ----------
    W_enc : torch.nn.Parameter
        ``(act_size, dict_size)`` encoder weight.
    W_dec : torch.nn.Parameter
        ``(dict_size, act_size)`` decoder weight; rows are the dictionary atoms
        and are kept unit-norm during training.
    b_enc, b_dec : torch.nn.Parameter
        Encoder and decoder biases.
    num_batches_not_active : torch.Tensor
        Per-atom count of consecutive batches with no activation, used to find
        dead atoms.
    """

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        torch.manual_seed(self.config["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.config["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.config["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.config["act_size"], self.config["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.config["dict_size"], self.config["act_size"])
            )
        )
        # Tied initialisation: the decoder starts as the unit-norm transpose of
        # the encoder. The kaiming draw above is overwritten, and is kept only so
        # the random stream matches checkpoints fitted before the swap.
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.config["dict_size"],)).to(
            cfg["device"]
        )

        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(self, x):
        """Optionally standardise each row; returns ``(x, mean, std)``."""
        if self.config["input_unit_norm"]:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        """Undo :meth:`preprocess_input` on a reconstruction."""
        if self.config["input_unit_norm"]:
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """Project the decoder rows back to the unit sphere, and remove the
        radial component of their gradient so the optimiser step stays
        tangential."""
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    @torch.no_grad()
    def update_inactive_features(self, acts):
        """Increment the dead-atom counters for atoms unused in this batch."""
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    def encode(self, x):
        raise NotImplementedError

    def decode(self, acts):
        raise NotImplementedError


class BatchTopKNetwork(SparseAutoencoderBase):
    """Batch top-K sparse autoencoder.

    During training the ``top_k * batch_size`` largest codes of the whole batch
    survive; at evaluation a running ``threshold`` stands in for that pooled
    selection, since a single sample has no batch to pool over.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.register_buffer("threshold", torch.tensor(0.0))
        self.x_mean, self.x_std = None, None

    def compute_activations(self, x):
        """Return ``(dense_relu_codes, sparse_codes)`` for a preprocessed ``x``."""
        x_cent = x - self.b_dec
        pre_acts = x_cent @ self.W_enc
        acts = F.relu(pre_acts)

        if self.training:
            acts_topk = torch.topk(
                acts.flatten(), self.config["top_k"] * x.shape[0], dim=-1
            )
            acts_topk = (
                torch.zeros_like(acts.flatten())
                .scatter(-1, acts_topk.indices, acts_topk.values)
                .reshape(acts.shape)
            )
        else:
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))

        return acts, acts_topk

    def forward(self, x):
        """Full training step output: reconstruction plus the loss breakdown."""
        x, x_mean, x_std = self.preprocess_input(x)
        acts, acts_topk = self.compute_activations(x)
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        self.update_threshold(acts_topk, lr=0.01)
        self.update_inactive_features(acts_topk)
        return self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)

    def encode(self, x):
        """Sparse codes of ``x``; the standardisation is cached for :meth:`decode`."""
        x, x_mean, x_std = self.preprocess_input(x)
        self.x_mean = x_mean
        self.x_std = x_std
        acts, acts_topk = self.compute_activations(x)
        return acts_topk

    def decode(self, acts_topk):
        """Reconstruct from sparse codes, undoing the cached standardisation."""
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        return self.postprocess_output(x_reconstruct, self.x_mean, self.x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        """Reconstruction, sparsity, dead-atom revival and orthogonality terms."""
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.config["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        orth_loss = self.config.get("orth_coeff", 0.0) * self.get_orthogonality_loss()
        loss = l2_loss + l1_loss + aux_loss + orth_loss
        num_dead_features = (
            self.num_batches_not_active > self.config["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        return {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "orth_loss": orth_loss,
            "threshold": self.threshold,
        }

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        """Ask the dead atoms to explain the current residual, so they revive."""
        dead_features = self.num_batches_not_active >= self.config["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.config["top_k_aux"], int(dead_features.sum())),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            return (
                self.config["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
        return torch.tensor(0, dtype=x.dtype, device=x.device)

    def get_orthogonality_loss(self):
        """Penalise off-diagonal atom correlation; PEAL's own addition, off by
        default (``orth_coeff`` 0)."""
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        return torch.norm(
            W_dec_normed @ W_dec_normed.t()
            - torch.eye(W_dec_normed.shape[0], device=W_dec_normed.device)
        )

    @torch.no_grad()
    def update_threshold(self, acts_topk, lr=0.01):
        """Track the smallest surviving code, the inference-time cut-off."""
        positive_mask = acts_topk > 0
        if positive_mask.any():
            min_positive = acts_topk[positive_mask].min()
            self.threshold = (1 - lr) * self.threshold + lr * min_positive


class ActivationStore:
    """Minimal ``next_batch()`` source over an in-memory activation tensor.

    The trainer pulls batches from an object with a ``next_batch`` method; this
    adapter serves consecutive slices of ``X`` and wraps around at the end.

    Parameters
    ----------
    config : dict or object
        Where ``batch_size`` (or ``model_batch_size``) and ``device`` are read
        from; a pydantic config is inspected through its ``cfg`` dict.
    X : torch.Tensor
        Activations of shape ``(n, act_size)``.
    """

    def __init__(self, config, X):
        self.config = config
        self.X = X
        self.current_index = 0

    def next_batch(self):
        """Return the next ``(batch_size, act_size)`` slice moved to ``device``."""
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.config, dict):
            source = self.config
        elif hasattr(self.config, "cfg") and isinstance(self.config.cfg, dict):
            source = self.config.cfg
        else:
            source = None

        if source is not None:
            batch_size = source.get("batch_size", source.get("model_batch_size", 32))
            device = source.get("device", default_device)
        else:
            batch_size = getattr(self.config, "batch_size", 32)
            device = getattr(self.config, "device", default_device)

        if self.current_index + batch_size > self.X.shape[0]:
            self.current_index = 0
        batch = self.X[self.current_index : self.current_index + batch_size]
        self.current_index += batch_size
        return batch.to(device)


def train_batch_topk_saes(saes, activation_store, cfgs, progress=True):
    """Train one or more autoencoders on batches drawn from ``activation_store``.

    Replaces the previous backend's ``train_sae_group_seperate_wandb``. That
    function spawned a Weights & Biases subprocess per autoencoder and wrote a
    checkpoint artifact every ``checkpoint_freq`` steps; PEAL disabled the
    former with ``WANDB_MODE=disabled`` and saves through
    ``SparseDictionary.save_on_disk`` instead, so neither is reproduced here. The
    optimiser, the gradient clipping and the unit-norm decoder projection are the
    same.

    Parameters
    ----------
    saes : list of BatchTopKNetwork
        Trained in place.
    activation_store : object
        Anything with ``next_batch()``.
    cfgs : list of dict
        One config per autoencoder; ``num_tokens // batch_size`` sets the number
        of steps.
    progress : bool
        Show a tqdm bar.

    Returns
    -------
    list of dict
        The last loss dict of each autoencoder.
    """
    num_batches = int(cfgs[0]["num_tokens"] // cfgs[0]["batch_size"])
    optimizers = [
        torch.optim.Adam(
            sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
        )
        for sae, cfg in zip(saes, cfgs)
    ]

    steps = range(num_batches)
    bar = None
    if progress:
        import tqdm

        bar = tqdm.trange(num_batches)
        steps = bar

    last = [None] * len(saes)
    for _ in steps:
        batch = activation_store.next_batch()
        for idx, (sae, cfg, optimizer) in enumerate(zip(saes, cfgs, optimizers)):
            sae_output = sae(batch)
            loss = sae_output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()
            last[idx] = sae_output
            if bar is not None:
                bar.set_postfix(
                    {
                        f"loss_{idx}": f"{loss.item():.4f}",
                        f"L0_{idx}": f"{sae_output['l0_norm']:.2f}",
                    }
                )
    return last
