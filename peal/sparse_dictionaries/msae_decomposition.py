"""Wrapper around the MSAE (Matryoshka Sparse Autoencoder) release for PEAL.

Provides the pretrained CLIP ViT-L/14 MSAE from the WolodjaZ/MSAE Hugging
Face repository, together with its concept-interpreter vocabulary, as a PEAL
``SparseDictionary``; alternatively a plain or Matryoshka autoencoder from
the vendored ``peal/dependencies/MSAE`` code can be trained from scratch on
cached activations. The module also isolates MSAE's bare top-level imports
(``sae``, ``utils``, ...) so they do not collide with RAEv2's ``utils``.
"""

from __future__ import annotations
import os
import sys
import urllib.request
from typing import Union, List, Optional

import numpy as np
import torch

from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig
from peal.log import get_logger

_log = get_logger(__name__)


MSAE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dependencies", "MSAE")
)
# MSAE's files import each other by bare top-level name (`from utils import ...`
# inside sae.py), and so does RAEv2 behind the RAEDiffusionAutoencoder generator
# (`from utils.dist_utils import ...`, a package). Whichever claimed
# sys.modules["utils"] first used to break the other -- this module is imported
# during sparse-dictionary class discovery, so MSAE always won and the RAE
# generator failed to load. The helper below imports MSAE's modules with
# MSAE_DIR at the head of sys.path and then moves them to private keys
# (peal_msae_<name>), restoring whatever was registered under the bare names.
_MSAE_TOP_LEVEL = ("sae", "utils", "loss", "config")


def _msae_module(name):
    """Return peal/dependencies/MSAE/<name>.py as a module without leaving the
    bare names `sae`, `utils`, `loss`, `config` in sys.modules."""
    import importlib

    key = f"peal_msae_{name}"
    if key in sys.modules:
        return sys.modules[key]
    saved = {n: sys.modules.pop(n) for n in _MSAE_TOP_LEVEL if n in sys.modules}
    sys.path.insert(0, MSAE_DIR)
    try:
        importlib.invalidate_caches()
        importlib.import_module(name)
        for n in _MSAE_TOP_LEVEL:
            mod = sys.modules.pop(n, None)
            if (
                mod is not None
                and os.path.dirname(getattr(mod, "__file__", "") or "") == MSAE_DIR
            ):
                sys.modules[f"peal_msae_{n}"] = mod
    finally:
        while MSAE_DIR in sys.path:
            sys.path.remove(MSAE_DIR)
        sys.modules.update(saved)
    return sys.modules[key]


class MSAEDecompositionConfig(SparseDictionaryConfig):
    """Configuration for MSAE (Matryoshka Sparse Autoencoder) decomposition.

    Parameters
    ----------
    clip_model : str, optional
        Encoder the pretrained checkpoint was trained on. Default
        ``"clip:ViT-L/14"``.
    ckpt_url : str or None, optional
        URL of the pretrained MSAE weights; ``None`` skips the download and
        leaves the dictionary in trainable mode.
    interp_url : str or None, optional
        URL of the concept-interpreter similarity matrix used to name each
        latent with a vocabulary entry.
    local_dir : str, optional
        Download cache directory. Default ``"/tmp/msae_weights"``.
    vocab_filename : str, optional
        Vocabulary file under ``peal/dependencies/MSAE/vocab``.
    n_components : int or None, optional
        Number of latents; overwritten by the checkpoint's ``latent_dim`` when
        a pretrained model is loaded. Default 6144.
    activation : str, optional
        Activation spec for training from scratch. Default ``"TopKReLU_64"``.
    use_matryoshka : bool, optional
        Train a ``MatryoshkaAutoencoder`` instead of a plain ``Autoencoder``.
    lr, epochs, batch_size
        Hyper-parameters for :meth:`MSAEDecomposition.fit`.
    """

    sparse_dictionaries_type: str = "MSAEDecomposition"
    clip_model: str = "clip:ViT-L/14"
    ckpt_url: Optional[str] = (
        "https://huggingface.co/WolodjaZ/MSAE/resolve/main/ViT-L_14/centered/"
        "6144_768_TopKReLU_64_RW_False_False_0.0_cc3m_ViT-L~14_train_image_2905936_768.pth"
    )
    interp_url: Optional[str] = (
        "https://huggingface.co/WolodjaZ/MSAE/resolve/main/ViT-L_14/centered/"
        "Concept_Interpreter_6144_768_TopKReLU_64_RW_False_False_0.0_cc3m_ViT-L~14_train_image_2905936_768_disect_ViT-L~14_-1_text_20000_768.npy"
    )
    local_dir: str = "/tmp/msae_weights"
    vocab_filename: str = "clip_disect_20k.txt"
    n_components: Union[int, None] = 6144
    component_strings: Union[list, None] = None
    activation: str = "TopKReLU_64"
    use_matryoshka: bool = False
    lr: float = 5e-4
    epochs: int = 20
    batch_size: int = 1024


class MSAEDecomposition(SparseDictionary):
    """Sparse dictionary wrapper around MSAE (Matryoshka Sparse Autoencoder).

    Supports pretrained MSAE checkpoints as well as fitting from scratch on
    dataloaders. When ``config.ckpt_url`` is set the constructor downloads and
    loads the pretrained model immediately, falling back to trainable mode
    (with a printed message) if that fails.

    Parameters
    ----------
    config : MSAEDecompositionConfig, optional
        Checkpoint URLs and training hyper-parameters.

    Attributes
    ----------
    sae : torch.nn.Module or None
        The MSAE ``SAE`` wrapper (pretrained) or a trained ``Autoencoder``.
    vocabulary : list of str or None
        One concept name per latent, from the concept interpreter.
    mu : torch.Tensor or None
        Activation mean subtracted before encoding a self-trained model.
    """

    def __init__(self, config=MSAEDecompositionConfig()):
        """Pick a device and, if ``config.ckpt_url`` is set, load the pretrained MSAE.

        A failure to download or load the checkpoint is caught and reported;
        the instance then stays in trainable mode with ``sae`` unset.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sae = None
        self.vocabulary = None
        self.mu = None
        if getattr(self.config, "ckpt_url", None):
            try:
                self._load_msae()
            except Exception as e:
                _log.info(
                    "%s",
                    f"[MSAE] Could not load pretrained checkpoint ({e}). Operating in trainable mode.",
                )

    def _ensure_file_downloaded(self, url: str, local_path: str, desc: str):
        """Download ``url`` to ``local_path`` unless it is already there."""
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            _log.info("%s", f"[MSAE] Downloading {desc} from {url}...")
            urllib.request.urlretrieve(url, local_path)
            _log.info("%s", f"[MSAE] Saved to {local_path}")

    def _load_msae(self):
        """Fetch the pretrained checkpoint and concept vocabulary and load them."""
        local_dir = getattr(self.config, "local_dir", "/tmp/msae_weights")
        ckpt_filename = os.path.basename(self.config.ckpt_url)
        interp_filename = (
            os.path.basename(self.config.interp_url) if self.config.interp_url else None
        )

        ckpt_path = os.path.join(local_dir, ckpt_filename)
        self._ensure_file_downloaded(self.config.ckpt_url, ckpt_path, "MSAE checkpoint")

        SAE = _msae_module("sae").SAE
        self.sae = SAE(ckpt_path).to(self.device)
        self.sae.eval()

        self.config.n_components = self.sae.latent_dim

        if self.config.interp_url:
            interp_path = os.path.join(local_dir, interp_filename)
            self._ensure_file_downloaded(
                self.config.interp_url, interp_path, "Concept Interpreter"
            )
            vocab_path = os.path.join(
                MSAE_DIR,
                "vocab",
                getattr(self.config, "vocab_filename", "clip_disect_20k.txt"),
            )
            if not os.path.exists(vocab_path):
                vocab_path = os.path.join(MSAE_DIR, "vocab", "clip_disect_20k.txt")

            if os.path.exists(vocab_path):
                with open(vocab_path, "r", encoding="utf-8") as f:
                    all_concepts = [line.strip() for line in f if line.strip()]
                sim_matrix = np.load(interp_path)
                best_concept_indices = np.argmax(sim_matrix, axis=0)
                self.vocabulary = [all_concepts[idx] for idx in best_concept_indices]
                _log.info(
                    "%s",
                    f"[MSAE] Loaded MSAE with {self.sae.latent_dim} directions mapped to concepts.",
                )

    def get_components(self) -> torch.Tensor:
        """Returns decoder weights of shape [n_inputs, n_components]."""
        if self.sae is None:
            raise RuntimeError("MSAE model is not initialized or trained.")
        if hasattr(self.sae, "model"):
            model = self.sae.model
        else:
            model = self.sae

        if hasattr(model, "decoder") and model.decoder is not None:
            return model.decoder.data.T.cpu()
        elif hasattr(model, "encoder") and model.encoder is not None:
            return model.encoder.data.cpu()
        raise AttributeError("Cannot extract components from MSAE model.")

    def get_vocabulary(self) -> List[str]:
        """Return the per-latent concept names, or ``None`` when not loaded."""
        return self.vocabulary

    def encode(self, x: torch.Tensor, topk: int = 64) -> torch.Tensor:
        """Encodes activations into sparse MSAE activations.

        Parameters
        ----------
        x : torch.Tensor
            Activations of shape ``(n, act_size)``.
        topk : int, optional
            Number of active latents for the pretrained ``SAE.encode``;
            ignored by a self-trained autoencoder, which instead subtracts
            ``mu`` first. Default 64.

        Returns
        -------
        torch.Tensor
            Sparse latents of shape ``(n, n_components)``.
        """
        if self.sae is None:
            raise RuntimeError("MSAE model is not initialized or trained.")
        x_dev = x.to(self.device)
        if hasattr(self.sae, "encode"):
            latents, _ = self.sae.encode(x_dev, topk=topk)
            return latents
        else:
            # Custom trained Autoencoder
            if self.mu is not None:
                x_dev = x_dev - self.mu.to(self.device)
            return self.sae.encode(x_dev)

    def fit(self, X: torch.Tensor, y: torch.Tensor = None):
        """Train MSAE from raw activation tensor X [N, act_size].

        Centres ``X`` (storing the mean in ``mu``), builds an ``Autoencoder``
        or ``MatryoshkaAutoencoder`` sized by ``config.n_components`` (default
        ``2 * act_size``) and trains it with AdamW and an MSE ``SAELoss`` for
        ``config.epochs`` epochs, printing the loss every five epochs.

        Parameters
        ----------
        X : torch.Tensor
            Activations of shape ``(N, act_size)``.
        y : torch.Tensor, optional
            Ignored; accepted for interface symmetry.
        """
        _sae = _msae_module("sae")
        Autoencoder, MatryoshkaAutoencoder = (
            _sae.Autoencoder,
            _sae.MatryoshkaAutoencoder,
        )
        SAELoss = _msae_module("loss").SAELoss

        N, act_size = X.shape
        self.mu = torch.mean(X, dim=0).to(self.device)
        X_centered = (X - self.mu.cpu()).to(self.device)

        n_latents = (
            self.config.n_components if self.config.n_components else act_size * 2
        )
        activation_str = getattr(self.config, "activation", "TopKReLU_64")
        use_matryoshka = getattr(self.config, "use_matryoshka", False)

        if use_matryoshka:
            model = MatryoshkaAutoencoder(
                n_inputs=act_size,
                n_latents=n_latents,
                activation=activation_str,
                nesting_list=[n_latents // 4, n_latents // 2, n_latents],
                relative_importance="UW",
            ).to(self.device)
        else:
            model = Autoencoder(
                n_inputs=act_size,
                n_latents=n_latents,
                activation=activation_str,
            ).to(self.device)

        loss_fn = SAELoss(reconstruction_loss="mse", sparse_weight=0.0)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=getattr(self.config, "lr", 5e-4)
        )

        batch_size = getattr(self.config, "batch_size", 1024)
        epochs = getattr(self.config, "epochs", 20)

        model.train()
        _log.info(
            "%s",
            f"[MSAE] Training {model.__class__.__name__} (latents={n_latents}, act={activation_str}) for {epochs} epochs...",
        )
        for epoch in range(epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, N, batch_size):
                idx = perm[i : i + batch_size]
                batch_x = X_centered[idx]

                optimizer.zero_grad()
                recons, repr_latents = model(batch_x)
                if use_matryoshka:
                    recons = recons[-1]
                    repr_latents = repr_latents[-1]
                loss, _, _ = loss_fn(recons, batch_x, repr_latents)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                _log.info(
                    "%s",
                    f"[MSAE Epoch {epoch+1}/{epochs}] Loss: {epoch_loss / max(1, num_batches):.6f}",
                )

        model.eval()
        self.sae = model

    def fit_from_dataloaders(self, dataloaders, feature_extractor=lambda x: x):
        """Encode every batch of the loaders with ``feature_extractor``, then fit.

        Parameters
        ----------
        dataloaders : iterable of torch.utils.data.DataLoader
            Loaders whose first batch element is the input.
        feature_extractor : callable, optional
            Maps a batch to ``(batch, act_size)`` activations; identity by
            default.
        """
        X_list = []
        for dataloader in dataloaders:
            for batch in dataloader:
                features = feature_extractor(batch[0].to(self.device))
                X_list.append(features.detach().cpu())

        X = torch.cat(X_list, dim=0)
        self.fit(X)

    def save_on_disk(self, path: str):
        """Write ``state_dict``, ``mu`` and the config to ``path`` with ``torch.save``."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            "state_dict": self.sae.state_dict() if self.sae is not None else None,
            "mu": self.mu.cpu() if self.mu is not None else None,
            "config": self.config,
        }
        torch.save(state_dict, path)
        _log.info("%s", f"[MSAE] Saved weights to {path}")

    def load_from_disk(self, path: str):
        """Rebuild a self-trained autoencoder from a :meth:`save_on_disk` file.

        Does nothing when ``path`` does not exist. The architecture is
        re-derived from the current config (``act_size``, ``n_components``,
        ``activation``, ``use_matryoshka``), not from the saved config.
        """
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.mu = checkpoint.get("mu", None)
        if "state_dict" in checkpoint and checkpoint["state_dict"] is not None:
            _sae = _msae_module("sae")
            Autoencoder, MatryoshkaAutoencoder = (
                _sae.Autoencoder,
                _sae.MatryoshkaAutoencoder,
            )
            act_size = self.config.act_size if hasattr(self.config, "act_size") else 512
            n_latents = (
                self.config.n_components if self.config.n_components else act_size * 2
            )
            activation_str = getattr(self.config, "activation", "TopKReLU_64")
            use_matryoshka = getattr(self.config, "use_matryoshka", False)

            if use_matryoshka:
                model = MatryoshkaAutoencoder(
                    n_inputs=act_size,
                    n_latents=n_latents,
                    activation=activation_str,
                    nesting_list=[n_latents // 4, n_latents // 2, n_latents],
                )
            else:
                model = Autoencoder(
                    n_inputs=act_size, n_latents=n_latents, activation=activation_str
                )
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to(self.device)
            model.eval()
            self.sae = model
            _log.info("%s", f"[MSAE] Loaded model state from {path}")
