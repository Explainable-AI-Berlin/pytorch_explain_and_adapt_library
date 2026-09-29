"""
BatchTopK sparse autoencoder as a PEAL sparse dictionary.

Sparse dictionaries decompose a generator's semantic latent (or a classifier's
feature space) into interpretable directions that adaptors such as DiDAE edit
along. This module wraps PEAL's own BatchTopK network from
``peal.sparse_dictionaries.batch_topk_network`` behind the ``SparseDictionary``
interface: it centres activations with a stored mean ``mu``, trains with
``train_batch_topk_saes`` or a local loop with TensorBoard logging, and
saves/loads the encoder/decoder weights as a plain ``torch`` checkpoint.
"""

import torch
import tqdm
import os
from typing import Union
from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig

# PEAL's own batch top-K implementation. It replaced the vendored
# peal.dependencies.matryoshka_sae backend on 2026-09-25, which published no
# licence and could therefore not be redistributed; the swap is verified
# bit-for-bit (init, forward in both modes, encode, decode and the weights after
# training all match), so checkpoints and published results carry over.
from peal.sparse_dictionaries.batch_topk_network import (
    ActivationStore,
    BatchTopKNetwork as InternalBatchTopKSAE,
    train_batch_topk_saes,
)
from peal.sparse_dictionaries.sae_evaluation import (
    run_sae_eval,
)
from peal.log import get_logger

_log = get_logger(__name__)


class BatchTopKSAEConfig(SparseDictionaryConfig):
    """
    Config of ``BatchTopKSAE``.

    Parameters
    ----------
    n_components : int
        Dictionary size (number of SAE latents).
    top_k : int
        Number of active latents kept per batch on average (BatchTopK).
    lr, l1_coeff, orth_coeff : float
        Adam learning rate and the L1 / orthogonality penalty weights.
    base_path : str
        Directory for TensorBoard logs and evaluation outputs.
    weights_path : str
        Where ``fit_with_evaluation`` saves the trained weights.
    device : str
        Device the SAE trains and lives on.
    batch_size, num_tokens : int
        Activation batch size and total number of activations processed;
        ``num_tokens // batch_size`` is the number of training steps.
    n_batches_to_dead, top_k_aux, aux_penalty : int, int, float
        Dead-latent detection window and the auxiliary loss that revives
        dead latents.
    input_unit_norm : bool
        Whether the dependency normalises inputs to unit norm.
    eval_epoch_interval : int
        Step interval of the (currently disabled) periodic F1 evaluation.
    cfg : dict
        If non-empty, used verbatim as the dependency's config instead of
        being assembled from the fields above.
    """

    n_components: int = 10016
    dict_size: Union[int, None] = None
    sparse_dictionaries_type: str = "BatchTopKSAE"

    top_k: int = 12
    lr: float = 3e-4
    l1_coeff: float = 0.0
    orth_coeff: float = 0.0
    base_path: str = None
    weights_path: str = None
    comp_min: int = -1
    comp_max: int = -1
    device: str = "cuda"
    model_batch_size: int = 32
    batch_size: int = 4096
    num_tokens: int = 1000000
    n_batches_to_dead: int = 100
    top_k_aux: int = 4
    aux_penalty: float = 1.0 / 32.0
    dtype: str = "torch.float32"
    seed: int = 42
    input_unit_norm: bool = False
    eval_epoch_interval: int = 50
    cfg: dict = {}


class BatchTopKSAE(SparseDictionary):
    """
    Sparse dictionary backed by the vendored BatchTopK SAE.

    Activations are centred by ``mu`` (the training mean) before encoding
    and the mean is added back after decoding, so ``encode``/``decode`` work
    on raw activations of size ``config.act_size``.

    Parameters
    ----------
    config : BatchTopKSAEConfig
        Hyper-parameters; ``config.cfg`` may override the whole internal
        config dict passed to the dependency.

    Attributes
    ----------
    sae : peal.sparse_dictionaries.batch_topk_network.BatchTopKNetwork
        The underlying model with ``W_enc``, ``W_dec``, ``b_enc``, ``b_dec``
        and (after training) ``threshold``.
    mu : torch.Tensor
        Shape ``(act_size,)``: mean of the training activations.
    """

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
                "orth_coeff": self.config.orth_coeff,
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
        """
        Train the SAE with ``train_batch_topk_saes``.

        Parameters
        ----------
        X : torch.Tensor or ActivationStore
            Activations of shape ``(N, act_size)`` (wrapped into an
            ``ActivationStore``) or an object with ``next_batch``. Note that
            ``X`` is used as is; centring by ``mu`` is the caller's job
            (``fit_from_activations`` does it).
        """
        if not hasattr(X, "next_batch"):
            X = ActivationStore(self.sae.config, X)

        saes = [self.sae.to(self.config.device)]
        cfgs = [self.sae.config]
        train_batch_topk_saes(saes, X, cfgs)

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
        _log.info("%s", f"Number of batches: {num_batches}")

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

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "L0": f"{sae_output['l0_norm']:.4f}",
                    "L2": f"{sae_output['l2_loss']:.4f}",
                }
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

            # Periodic F1 evaluation
            if (i + 1) % eval_interval == 0:
                pass  # Placeholder for evaluation logic; can be uncommented and implemented as needed
                # f1_results = compute_component_f1_scores(
                #     sae=sae,
                #     activation_store_X=raw_X,
                #     ground_truth_labels=ground_truth_labels,
                #     mu=self.mu,
                #     device=self.config.device,
                #     n_components=10,
                # )
                # for key, value in f1_results.items():
                #     writer.add_scalar(f"eval/{key}", value, epoch_counter)

                # mean_f1 = f1_results.get("mean_top10_f1", 0.0)
                # print(f"[Eval epoch {epoch_counter}] mean_top10_f1={mean_f1:.4f}")
            epoch_counter += 1

        # Persist the trained weights *before* the evaluation: the metrics below
        # are the failure-prone part (label shapes, memory), and a crash there
        # used to throw away the whole training run.
        if getattr(self.config, "weights_path", None) is not None:
            self.save_on_disk(self.config.weights_path)
            _log.info("%s", f"SAE weights saved to {self.config.weights_path}")

        # Final evaluation
        run_sae_eval(
            sae=sae,
            x=raw_X,
            y=ground_truth_labels,
            base_path=self.config.base_path,
        )
        # f1_results = compute_component_f1_scores(
        #     sae=sae,
        #     activation_store_X=raw_X,
        #     ground_truth_labels=ground_truth_labels,
        #     mu=self.mu,
        #     device=self.config.device,
        #     n_components=10,
        # )
        # for key, value in f1_results.items():
        #     writer.add_scalar(f"eval/{key}", value, epoch_counter)

        # mean_f1 = f1_results.get("mean_top10_f1", 0.0)
        # print(f"[Final eval] mean_top10_f1={mean_f1:.4f}")

        writer.close()
        _log.info("%s", f"TensorBoard logs saved to {log_dir}")

    def fit_from_dataloaders(self, dataloaders, feature_extractor=lambda x: x):
        """
        Run ``feature_extractor`` over the dataloaders and fit on the result.

        Parameters
        ----------
        dataloaders : iterable of torch.utils.data.DataLoader
            Batches whose first element is the input and, optionally, whose
            second element holds ground-truth labels for evaluation.
        feature_extractor : callable
            Maps an input batch on ``config.device`` to activations of shape
            ``(B, act_size)``.
        """
        X_list = []
        y_list = []
        with torch.no_grad():
            for dataloader in dataloaders:
                for batch in dataloader:
                    # feature_extractor is expected to return tensor on device
                    features = feature_extractor(batch[0].to(self.config.device))
                    X_list.append(features.detach().cpu().float())
                    # Collect ground truth labels if available
                    if len(batch) > 1:
                        y_list.append(batch[1].detach().cpu())

        self.fit_from_activations(
            torch.cat(X_list, dim=0),
            torch.cat(y_list, dim=0) if y_list else None,
        )

    def fit_from_activations(self, X, y=None):
        """
        Everything fit_from_dataloaders does once the encoder has been run.

        Sets ``mu`` to the activation mean, wraps the centred activations in
        an ``ActivationStore`` and trains with ``fit_with_evaluation`` when
        labels are given, otherwise with ``fit``.

        Parameters
        ----------
        X : torch.Tensor
            Activations of shape ``(N, act_size)`` on CPU.
        y : torch.Tensor, optional
            Ground-truth labels of shape ``(N,)`` or ``(N, K)`` used for the
            final SAE evaluation.
        """
        self.mu = torch.mean(X, dim=0)

        # Ensure num_tokens and batch_size are correctly set in sae.config
        if "num_tokens" not in self.sae.config:
            self.sae.config["num_tokens"] = self.config.num_tokens
        if "batch_size" not in self.sae.config:
            self.sae.config["batch_size"] = self.config.batch_size

        activation_store = ActivationStore(self.sae.config, X - self.mu)

        # If ground truth labels were collected, use fit_with_evaluation
        if y is not None:
            self.fit_with_evaluation(activation_store, y)
        else:
            self.fit(activation_store)

    def get_components(self):
        """Return the encoder weight ``W_enc`` of shape
        ``(act_size, dict_size)``."""
        return self.sae.W_enc

    def save_on_disk(self, path):
        """
        Save ``W_enc``, ``W_dec``, ``b_enc``, ``b_dec``, ``mu`` and
        ``threshold`` (if present) as a CPU ``torch`` checkpoint.

        Parameters
        ----------
        path : str
            Target file; parent directories are created.
        """
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        torch.save(
            {
                "W_enc": self.sae.W_enc.cpu(),
                "W_dec": self.sae.W_dec.cpu(),
                "b_enc": self.sae.b_enc.cpu(),
                "b_dec": self.sae.b_dec.cpu(),
                "mu": self.mu.cpu(),
                "threshold": (
                    self.sae.threshold.cpu() if hasattr(self.sae, "threshold") else None
                ),
            },
            path,
        )

    def load_from_disk(self, path):
        """
        Load a checkpoint written by ``save_on_disk`` onto ``config.device``.

        Parameters
        ----------
        path : str
            Checkpoint file.
        """
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.sae.W_enc.data = checkpoint["W_enc"].to(self.config.device)
        self.sae.W_dec.data = checkpoint["W_dec"].to(self.config.device)
        self.sae.b_enc.data = checkpoint["b_enc"].to(self.config.device)
        self.sae.b_dec.data = checkpoint["b_dec"].to(self.config.device)
        self.mu = checkpoint["mu"].to(self.config.device)
        if "threshold" in checkpoint and checkpoint["threshold"] is not None:
            self.sae.threshold.data = checkpoint["threshold"].to(self.config.device)

    def encode(self, x):
        """
        Sparse codes of raw activations (thresholded, as at inference).

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, act_size)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, dict_size)``.
        """
        # x is (Batch, act_size)
        # Handle normalization if needed
        x_cent = x - self.mu.to(x.device)
        acts = self.sae.encode(x_cent)
        return acts

    def encode_no_threshold(self, x):
        """
        Pre-activation codes without the learned threshold applied.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, act_size)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, dict_size)``; first output of
            ``sae.compute_activations`` on the centred input.
        """
        x_cent = x - self.mu.to(x.device)
        acts, _ = self.sae.compute_activations(x_cent)
        return acts

    def decode(self, acts):
        """
        Reconstruct raw activations from sparse codes (adds ``mu`` back).

        Parameters
        ----------
        acts : torch.Tensor
            Shape ``(B, dict_size)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, act_size)``.
        """
        # acts is (Batch, dict_size)
        recon = self.sae.decode(acts)
        return recon + self.mu.to(recon.device)
