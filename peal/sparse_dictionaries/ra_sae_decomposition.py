"""Pretrained RA-SAE (Relaxed Archetypal SAE) dictionaries of DINOv2 token space.

Wraps https://huggingface.co/matybohacek/RA-SAE-DINOv2-32k -- a 32000-concept
BatchTopK SAE with an archetypal dictionary (Fel et al. 2025, arXiv:2502.12892),
trained on the ~261 tokens per image that `dinov2_vitb14_reg` produces.

The published checkpoint is a 4.4 GB *pickled object graph*
(`torch.load(..., weights_only=False)` of a `DataParallel(BatchTopKSAE)`), which
would drag the `overcomplete` package in as an import-time dependency just to
unpickle. Instead the loader below reads it once behind stub classes, fuses the
archetypal dictionary, and caches ~300 MB of plain tensors; every later run
loads only that cache and never touches the original file or `overcomplete`.

Inference reproduces overcomplete 0.3.0 exactly:

    pre_z = LayerNorm(Linear(x))            # MLPEncoder.final_block
    z     = ReLU(pre_z)                     # MLPEncoder.final_activation
    codes = z * (z >= running_threshold)    # BatchTopKSAE.encode, eval branch
    x_hat = codes @ D                       # RelaxedArchetypalDictionary.forward
    D     = exp(multiplier) * (rowstochastic(relu(W)) @ C + Relax)

`running_threshold` is not stored in the checkpoint (it is None there); the
model card's config.json supplies it, which is also what the HF wrapper sets
before calling `.eval()`.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import List, Optional, Union

import torch
import torch.nn as nn

from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig
from peal.log import get_logger

_log = get_logger(__name__)


class RASAEDecompositionConfig(SparseDictionaryConfig):
    """Configuration for a pretrained RA-SAE decomposition."""

    sparse_dictionaries_type: str = "RASAEDecomposition"
    encoder_model: str = "dino_v2_base_reg"
    """Backbone whose tokens this dictionary decomposes. Kept for the record and
    checked against the generator's encoder tag by consumers that edit z_sem."""
    repo_id: str = "matybohacek/RA-SAE-DINOv2-32k"
    ckpt_url: str = (
        "https://huggingface.co/matybohacek/RA-SAE-DINOv2-32k/resolve/main/"
        "RA-SAE-DINOv2-32k.pth"
    )
    config_url: str = (
        "https://huggingface.co/matybohacek/RA-SAE-DINOv2-32k/raw/main/config.json"
    )
    local_dir: str = "/tmp/ra_sae_weights"
    """Writable scratch directory. The GPU nodes mount ~/.cache read-only, so
    this must not default anywhere under it."""
    fused_filename: str = "ra_sae_dinov2_32k_fused.pt"
    """Name of the fused-tensor cache written next to the raw checkpoint."""
    running_threshold: Optional[float] = None
    """Activation threshold of the BatchTopK eval branch. None -> read from the
    published config.json (0.829 for this checkpoint)."""
    drop_raw_checkpoint_after_fuse: bool = False
    """Delete the 4.4 GB raw checkpoint once the fused cache is written."""
    n_components: Union[int, None] = 32000
    act_size: int = 768


class RASAEDecomposition(SparseDictionary):
    """Sparse dictionary wrapper around a pretrained RA-SAE checkpoint.

    Pretrained-only: `fit`/`fit_from_dataloaders` are deliberately not
    implemented, since the point of this dictionary is that it is *not* fitted
    on our activations.
    """

    def __init__(self, config=RASAEDecompositionConfig()):
        """Pick the device, allocate the tensor slots and load the fused weights.

        Parameters
        ----------
        config : RASAEDecompositionConfig, optional
            Download/cache locations and the activation threshold. Its
            ``n_components`` and ``act_size`` are overwritten by the values
            read from the checkpoint.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lin_weight = None  # [K, D]
        self.lin_bias = None  # [K]
        self.ln_weight = None  # [K]
        self.ln_bias = None  # [K]
        self.dictionary = None  # [K, D]
        self.threshold = None  # float
        self.vocabulary = None  # RA-SAE ships no concept names
        self._load()

    # ------------------------------------------------------------------ load

    def _ensure_file_downloaded(self, url: str, local_path: str, desc: str):
        """Download ``url`` to ``local_path`` once, checking the advertised length."""
        if os.path.exists(local_path):
            return
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        _log.info("%s", f"[RA-SAE] Downloading {desc} from {url}...")
        # urlretrieve can return a truncated file on a stalled connection, so
        # verify the length the server advertises before accepting it.
        tmp_path = local_path + ".part"
        with urllib.request.urlopen(url, timeout=60) as response:
            expected = response.headers.get("Content-Length")
            expected = int(expected) if expected is not None else None
            with open(tmp_path, "wb") as handle:
                while True:
                    chunk = response.read(1 << 22)
                    if not chunk:
                        break
                    handle.write(chunk)
        got = os.path.getsize(tmp_path)
        if expected is not None and got != expected:
            os.remove(tmp_path)
            raise IOError(
                f"[RA-SAE] {desc} download truncated: got {got} of {expected} bytes."
            )
        os.rename(tmp_path, local_path)
        _log.info("%s", f"[RA-SAE] Saved to {local_path} ({got} bytes)")

    def _resolve_threshold(self, local_dir: str) -> float:
        """Configured ``running_threshold``, else the one in the published config."""
        if self.config.running_threshold is not None:
            return float(self.config.running_threshold)
        config_path = os.path.join(local_dir, "ra_sae_config.json")
        self._ensure_file_downloaded(
            self.config.config_url, config_path, "RA-SAE config.json"
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            published = json.load(handle)
        threshold = published.get("running_threshold")
        if threshold is None:
            raise ValueError(
                "[RA-SAE] The published config.json carries no running_threshold and "
                "none was configured; the BatchTopK eval branch has no threshold to "
                "apply."
            )
        return float(threshold)

    def _fuse_from_raw_checkpoint(self, ckpt_path: str) -> dict:
        """Unpickle the published checkpoint and reduce it to plain tensors.

        The pickle names three `overcomplete` classes. Rather than depend on
        that package, they are stubbed with permissive `nn.Module`s: the pickle
        only ever restores their `__dict__`, so the stubs recover every buffer
        and parameter, and the arithmetic below is re-implemented here.
        """
        import sys
        import types

        class _Stub(nn.Module):
            """Permissive stand-in for an `overcomplete` class in the pickle.

            Accepts any constructor arguments and, on unpickling, copies the
            pickled ``__dict__`` verbatim so buffers and parameters survive
            without the real package being installed.
            """

            def __init__(self, *args, **kwargs):
                """Ignore every argument; only ``__setstate__`` matters here."""
                super().__init__()

            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)

        stubbed = {
            "overcomplete.sae.archetypal_dictionary": ["RelaxedArchetypalDictionary"],
            "overcomplete.sae.batchtopk_sae": ["BatchTopKSAE"],
            "overcomplete.sae.modules": ["MLPEncoder"],
        }
        installed = []
        for package in ("overcomplete", "overcomplete.sae"):
            if package not in sys.modules:
                sys.modules[package] = types.ModuleType(package)
                installed.append(package)
        for module_name, class_names in stubbed.items():
            if module_name in sys.modules:
                continue
            module = types.ModuleType(module_name)
            for class_name in class_names:
                setattr(module, class_name, type(class_name, (_Stub,), {}))
            sys.modules[module_name] = module
            installed.append(module_name)

        try:
            _log.info(
                "%s", f"[RA-SAE] Fusing {ckpt_path} (one-off; ~5 GB of host RAM)..."
            )
            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sae = getattr(loaded, "module", loaded)
            state = sae.state_dict()

            # RelaxedArchetypalDictionary.get_dictionary(), training branch:
            # W is renormalised row-stochastic, Relax is norm-clamped to delta,
            # then D = (W @ C + Relax) * exp(multiplier). The published W and
            # Relax already satisfy both constraints, but reapplying them costs
            # nothing and makes the fusion independent of that assumption.
            # W is [32000, 32000] (4.1 GB), so walk it in row blocks rather than
            # materialising relu(W) and its normalised copy whole.
            w_all = state["dictionary.W"]
            candidates = state["dictionary.C"]
            relax = state["dictionary.Relax"]
            delta = float(getattr(sae.dictionary, "delta", 1.0))
            relax = relax * torch.clamp(
                delta / (relax.norm(dim=-1, keepdim=True) + 1e-12), max=1.0
            )
            multiplier = torch.exp(state["dictionary.multiplier"])

            dictionary = torch.empty(
                w_all.shape[0], candidates.shape[1], dtype=candidates.dtype
            )
            block = 2048
            for start in range(0, w_all.shape[0], block):
                stop = min(start + block, w_all.shape[0])
                w_block = torch.relu(w_all[start:stop])
                w_block /= w_block.sum(dim=-1, keepdim=True) + 1e-8
                dictionary[start:stop] = (
                    w_block @ candidates + relax[start:stop]
                ) * multiplier
                del w_block

            fused = {
                "lin_weight": state["encoder.final_block.0.weight"].clone(),
                "lin_bias": state["encoder.final_block.0.bias"].clone(),
                "ln_weight": state["encoder.final_block.1.weight"].clone(),
                "ln_bias": state["encoder.final_block.1.bias"].clone(),
                "dictionary": dictionary.contiguous(),
                "nb_concepts": int(getattr(sae, "nb_concepts", dictionary.shape[0])),
                "in_dimensions": int(dictionary.shape[1]),
            }
            del loaded, sae, state, w_all, candidates, relax, dictionary
            return fused
        finally:
            for module_name in installed:
                sys.modules.pop(module_name, None)

    def _load(self):
        """Load the fused cache, fusing the raw 4.4 GB checkpoint on first use."""
        local_dir = getattr(self.config, "local_dir", "/tmp/ra_sae_weights")
        fused_path = os.path.join(local_dir, self.config.fused_filename)

        if os.path.exists(fused_path):
            fused = torch.load(fused_path, map_location="cpu")
        else:
            ckpt_path = os.path.join(local_dir, os.path.basename(self.config.ckpt_url))
            self._ensure_file_downloaded(
                self.config.ckpt_url, ckpt_path, "RA-SAE checkpoint (4.4 GB)"
            )
            fused = self._fuse_from_raw_checkpoint(ckpt_path)
            os.makedirs(local_dir, exist_ok=True)
            torch.save(fused, fused_path)
            _log.info("%s", f"[RA-SAE] Cached fused tensors to {fused_path}")
            if self.config.drop_raw_checkpoint_after_fuse:
                os.remove(ckpt_path)
                _log.info("%s", f"[RA-SAE] Removed raw checkpoint {ckpt_path}")

        self.lin_weight = fused["lin_weight"].to(self.device)
        self.lin_bias = fused["lin_bias"].to(self.device)
        self.ln_weight = fused["ln_weight"].to(self.device)
        self.ln_bias = fused["ln_bias"].to(self.device)
        self.dictionary = fused["dictionary"].to(self.device)
        self.threshold = self._resolve_threshold(local_dir)
        self.config.n_components = int(fused["nb_concepts"])
        self.config.act_size = int(fused["in_dimensions"])
        _log.info(
            "%s",
            f"[RA-SAE] Loaded {self.config.n_components} concepts of "
            f"{self.config.act_size}-d {self.config.encoder_model} token space "
            f"(threshold={self.threshold}).",
        )

    # ------------------------------------------------------------- interface

    def get_components(self) -> torch.Tensor:
        """Decoder atoms as [n_inputs, n_components], the layout PEAL expects."""
        return self.dictionary.T.cpu()

    def get_vocabulary(self) -> Optional[List[str]]:
        """RA-SAE ships no concept names -- unlike MSAE there is no text-space
        interpreter, so concepts are identified by ground-truth matching or by
        their maximally-activating images, never by a vocabulary lookup."""
        return self.vocabulary

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse codes for [N, 768] tokens, as overcomplete computes them in eval."""
        x_dev = x.to(device=self.device, dtype=self.lin_weight.dtype)
        pre_z = torch.nn.functional.linear(x_dev, self.lin_weight, self.lin_bias)
        pre_z = torch.nn.functional.layer_norm(
            pre_z, (pre_z.shape[-1],), self.ln_weight, self.ln_bias
        )
        codes = torch.relu(pre_z)
        return codes * (codes >= self.threshold).to(codes.dtype)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct token activations from sparse codes."""
        return torch.matmul(
            codes.to(device=self.device, dtype=self.dictionary.dtype), self.dictionary
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor = None):
        """Not supported: the dictionary is pretrained (raises ``NotImplementedError``)."""
        raise NotImplementedError(
            "RASAEDecomposition wraps a pretrained dictionary of DINOv2 token space; "
            "fitting it here would discard exactly the pretrained concepts it exists "
            "to provide."
        )

    def fit_from_dataloaders(self, dataloaders, feature_extractor=lambda x: x):
        """Not supported: the dictionary is pretrained (raises ``NotImplementedError``)."""
        raise NotImplementedError(
            "RASAEDecomposition wraps a pretrained dictionary of DINOv2 token space; "
            "fitting it here would discard exactly the pretrained concepts it exists "
            "to provide."
        )

    def save_on_disk(self, path: str):
        """Save the fused tensors and the threshold to ``path`` with ``torch.save``.

        Parameters
        ----------
        path : str
            Destination file; parent directories are created.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "lin_weight": self.lin_weight.cpu(),
                "lin_bias": self.lin_bias.cpu(),
                "ln_weight": self.ln_weight.cpu(),
                "ln_bias": self.ln_bias.cpu(),
                "dictionary": self.dictionary.cpu(),
                "nb_concepts": self.config.n_components,
                "in_dimensions": self.config.act_size,
                "threshold": self.threshold,
            },
            path,
        )
        _log.info("%s", f"[RA-SAE] Saved fused weights to {path}")

    def load_from_disk(self, path: str):
        """Restore tensors written by :meth:`save_on_disk`; a missing path is a no-op.

        Parameters
        ----------
        path : str
            File written by :meth:`save_on_disk`. Updates ``config.n_components``,
            ``config.act_size`` and, when stored, the threshold.
        """
        if not os.path.exists(path):
            return
        fused = torch.load(path, map_location=self.device)
        self.lin_weight = fused["lin_weight"].to(self.device)
        self.lin_bias = fused["lin_bias"].to(self.device)
        self.ln_weight = fused["ln_weight"].to(self.device)
        self.ln_bias = fused["ln_bias"].to(self.device)
        self.dictionary = fused["dictionary"].to(self.device)
        if fused.get("threshold") is not None:
            self.threshold = float(fused["threshold"])
        self.config.n_components = int(fused["nb_concepts"])
        self.config.act_size = int(fused["in_dimensions"])
        _log.info("%s", f"[RA-SAE] Loaded fused weights from {path}")
