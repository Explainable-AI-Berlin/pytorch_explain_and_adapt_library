"""SVD-prefiltered sparse autoencoders.

Encoder activations are usually dominated by a few high-variance directions
that an SAE would spend many dictionary atoms on. The wrapper in this module
centres the activations, removes the leading ``k`` principal directions found
by an SVD and fits an inner SAE (BatchTopK or MSAE) on the
residual. The resulting dictionary exposes the ``k`` SVD directions followed by
the inner SAE's atoms as one component matrix, so DiDAE can sweep both kinds of
direction. ``k`` is chosen by a fixed number, a cumulative-variance target or a
relative-mean-ratio rule, and a diagnostic plot is written to ``base_path``.
"""

import os
import torch
from typing import Union
import numpy as np

from peal.sparse_dictionaries.interfaces import SparseDictionary, SparseDictionaryConfig
from peal.sparse_dictionaries.batch_topk import BatchTopKSAE, BatchTopKSAEConfig
from peal.sparse_dictionaries.msae_decomposition import (
    MSAEDecomposition,
    MSAEDecompositionConfig,
)
from peal.log import get_logger

_log = get_logger(__name__)


class SVDFilteredSAEConfig(SparseDictionaryConfig):
    """Configuration of the SVD-prefiltered SAE wrapper.

    Parameters
    ----------
    svd_mode : str
        How ``k`` is chosen: ``"fixed_k"`` uses ``svd_k``,
        ``"variance_threshold"`` the smallest ``k`` whose cumulative explained
        variance reaches ``svd_variance_threshold``, ``"relative_mean_ratio"``
        the first component whose variance ratio drops below
        ``svd_ratio_multiplier`` times the mean of the remaining variance.
    svd_k : int
        Number of SVD components in ``"fixed_k"`` mode.
    svd_variance_threshold : float
        Target cumulative variance in ``"variance_threshold"`` mode.
    svd_ratio_multiplier : float
        Multiplier ``M`` of the relative-mean-ratio rule.
    inner_sae_type : str
        ``"BatchTopKSAE"`` or ``"MSAEDecomposition"``.
    dict_size, top_k, act_size, n_components : int
        Forwarded to the inner SAE config; ``dict_size`` and
        ``n_components`` are kept in sync with each other.
    num_tokens, lr, seed : int, float, int
        Training budget, learning rate and seed forwarded to the inner SAE so
        a yaml's settings are honoured instead of the inner class defaults.
    """

    sparse_dictionaries_type: str = "SVDFilteredBatchTopKSAE"
    svd_mode: str = (
        "fixed_k"  # "fixed_k", "variance_threshold", or "relative_mean_ratio"
    )
    svd_k: int = 8
    svd_variance_threshold: float = 0.95
    svd_ratio_multiplier: float = 2.0
    inner_sae_type: str = "BatchTopKSAE"  # "BatchTopKSAE", "MSAEDecomposition"
    dict_size: int = 1000
    top_k: int = 32
    act_size: int = 512
    n_components: Union[int, None] = None
    # Declared so the yaml's training budget reaches the inner SAE. Without these
    # the wrapper silently fell back to the inner class defaults (1e6 tokens,
    # lr 3e-4), which is not comparable to an unwrapped run of the same config.
    num_tokens: int = 1000000
    lr: float = 3e-4
    seed: int = 42


class SVDFilteredBatchTopKSAEConfig(SVDFilteredSAEConfig):
    """``SVDFilteredSAEConfig`` pinned to a BatchTopK inner SAE."""

    sparse_dictionaries_type: str = "SVDFilteredBatchTopKSAE"
    inner_sae_type: str = "BatchTopKSAE"


class SVDFilteredBatchTopKSAE(SparseDictionary):
    """
    SVD Pre-filtered SAE Wrapper.
    Performs SVD decomposition on raw centered activations, projects out top-K SVD components,
    and fits the inner SAE on the residual activations.

    Despite the name the inner SAE is chosen by ``config.inner_sae_type`` and
    may be a BatchTopK or MSAE dictionary.

    Parameters
    ----------
    config : SVDFilteredSAEConfig
        Selection rule for ``k`` and the settings copied into the inner SAE.

    Attributes
    ----------
    mu : torch.Tensor or None
        Mean activation ``[D]`` subtracted before the SVD; set by :meth:`fit`.
    svd_components : torch.Tensor or None
        Leading right singular vectors ``[D, k]``.
    svd_explained_variance_ratio : numpy.ndarray or None
        Explained variance ratio of every singular direction, length ``D``.
    inner_sae : SparseDictionary
        The SAE fitted on the residual activations.
    """

    def __init__(self, config=SVDFilteredSAEConfig()):
        """Store the config and instantiate the (unfitted) inner SAE."""
        self.config = config
        self.device = getattr(
            config, "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mu = None
        self.svd_components = None  # [n_inputs, k]
        self.svd_explained_variance_ratio = None
        self.inner_sae = None

        self._init_inner_sae()

    def _init_inner_sae(self):
        """Build the inner SAE config from this config's shared fields."""
        inner_type = getattr(self.config, "inner_sae_type", "BatchTopKSAE")
        if inner_type == "MSAEDecomposition":
            inner_cfg = MSAEDecompositionConfig()
        else:
            inner_cfg = BatchTopKSAEConfig()

        # Copy over relevant settings
        for attr in [
            "dict_size",
            "top_k",
            "act_size",
            "base_path",
            "weights_path",
            "run_name",
            "n_components",
            "num_tokens",
            "lr",
            "seed",
            "batch_size",
        ]:
            if hasattr(self.config, attr) and getattr(self.config, attr) is not None:
                val = getattr(self.config, attr)
                if hasattr(inner_cfg, attr):
                    setattr(inner_cfg, attr, val)
                if attr == "dict_size" and hasattr(inner_cfg, "n_components"):
                    inner_cfg.n_components = val
                if attr == "n_components" and hasattr(inner_cfg, "dict_size"):
                    inner_cfg.dict_size = val

        if inner_type == "MSAEDecomposition":
            self.inner_sae = MSAEDecomposition(config=inner_cfg)
        else:
            self.inner_sae = BatchTopKSAE(config=inner_cfg)

    def fit_from_dataloaders(self, dataloaders, feature_extractor=lambda x: x):
        """Run the encoder over every loader and fit on the stacked activations.

        Parameters
        ----------
        dataloaders : list of torch.utils.data.DataLoader
            Loaders yielding batches whose first element is the input.
        feature_extractor : callable
            Maps a batch on ``self.device`` to activations ``[B, D]``;
            identity by default.
        """
        X_list = []
        for dataloader in dataloaders:
            for batch in dataloader:
                features = feature_extractor(batch[0].to(self.device))
                X_list.append(features.detach().cpu())

        self.fit_from_activations(torch.cat(X_list, dim=0))

    def fit_from_activations(self, X, y=None):
        """Everything fit_from_dataloaders does once the encoder has been run."""
        self.fit(X, y)

    def fit(self, X: torch.Tensor, y: torch.Tensor = None):
        """Choose ``k``, store the SVD subspace and fit the inner SAE on the residual.

        Computes a thin SVD of the centred activations, selects ``k`` according
        to ``config.svd_mode``, writes the variance plot via
        :meth:`plot_and_log_svd_variance`, projects the top-``k`` subspace out
        of the centred activations and calls ``inner_sae.fit`` on what is left.
        Finally ``config.n_components`` is set to ``k`` plus the inner SAE's
        component count.

        Parameters
        ----------
        X : torch.Tensor
            Raw activations ``[N, act_size]``.
        y : torch.Tensor, optional
            Unused; accepted for interface compatibility.
        """
        N, D = X.shape
        self.mu = torch.mean(X, dim=0)
        X_centered = X - self.mu

        # SVD computation
        _log.info(
            "%s",
            f"[SVD Filtered SAE] Computing SVD on activation tensor shape {X.shape}...",
        )
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        # Vh is [D, D], rows are right singular vectors. Transpose to [D, D] columns
        V = Vh.T  # [D, D]
        var_explained = (S**2) / (S**2).sum()
        self.svd_explained_variance_ratio = var_explained.cpu().numpy()

        mode = getattr(self.config, "svd_mode", "variance_threshold")
        multiplier = getattr(self.config, "svd_ratio_multiplier", 2.0)

        thresholds = np.zeros(D)
        cum_var_so_far = 0.0
        k_rule = D
        k_found = False

        for i in range(D):
            var_remaining = 1.0 - cum_var_so_far
            n_remaining = D - i
            thresholds[i] = multiplier * (var_remaining / n_remaining)
            v_i = self.svd_explained_variance_ratio[i]

            if not k_found and v_i <= thresholds[i]:
                k_rule = i
                k_found = True

            cum_var_so_far += v_i

        if mode == "variance_threshold":
            # Smallest k whose leading components already explain svd_variance_threshold
            # of the total variance. Unlike the relative-mean-ratio rule below this is
            # monotone in k and comparable across activation spaces of different
            # dimensionality, so one threshold transfers between datasets.
            target = float(getattr(self.config, "svd_variance_threshold", 0.95))
            cum_var = np.cumsum(self.svd_explained_variance_ratio)
            k = int(np.searchsorted(cum_var, target) + 1)
            k = int(np.clip(k, 1, D))
            cum_var_k = float(cum_var[k - 1])
            _log.info(
                "%s",
                f"[SVD Filtered SAE] Selected {k} SVD components using cumulative-variance rule: smallest k with cumvar >= {target}. (Explains {cum_var_k*100:.2f}% variance).",
            )
        elif mode == "relative_mean_ratio":
            k = k_rule
            cum_var_k = np.sum(self.svd_explained_variance_ratio[:k]) if k > 0 else 0.0
            _log.info(
                "%s",
                f"[SVD Filtered SAE] Selected {k} SVD components using rule: v_i <= {multiplier} * (var_remaining / N_remaining). (Explains {cum_var_k*100:.2f}% variance).",
            )
        else:
            k = getattr(self.config, "svd_k", 8)
            cum_var_k = np.sum(self.svd_explained_variance_ratio[:k]) if k > 0 else 0.0
            _log.info(
                "%s",
                f"[SVD Filtered SAE] Selected top {k} SVD components ({cum_var_k*100:.2f}% variance explained).",
            )

        self.svd_components = V[:, :k]  # [D, k]
        self.svd_k_selected = k
        self.svd_k_rule = k_rule
        self.svd_thresholds = thresholds

        base_path = getattr(self.config, "base_path", None)
        self.plot_and_log_svd_variance(base_path)

        # Project out SVD components: Residual = X_c - X_c @ V_k @ V_k^T
        X_svd_proj = X_centered @ self.svd_components @ self.svd_components.T
        X_residual = X_centered - X_svd_proj

        _log.info(
            "%s",
            f"[SVD Filtered SAE] Fitting inner {self.inner_sae.__class__.__name__} on residual activations...",
        )
        self.inner_sae.fit(X_residual)

        # Update n_components
        sae_comp_count = (
            self.inner_sae.get_components().shape[1]
            if hasattr(self.inner_sae, "get_components")
            else 0
        )
        self.config.n_components = k + sae_comp_count

    def plot_and_log_svd_variance(self, base_path: str = None):
        """Plot the relative variance ratio and cumulative variance per component.

        The upper panel shows ``R_i = v_i / mean(remaining variance)`` on a log
        scale with the selected cutoff (and the ratio-rule cutoff when it
        differs), the lower panel the cumulative explained variance. Does
        nothing before :meth:`fit`.

        Parameters
        ----------
        base_path : str, optional
            If given, the figure is saved as
            ``<base_path>/svd_variance_explained.png`` and additionally logged
            to TensorBoard (``<base_path>/sae_logs``) and to W&B when a run is
            active; logging failures are printed and ignored.
        """
        if self.svd_explained_variance_ratio is None:
            return

        import matplotlib.pyplot as plt

        var_ratio = self.svd_explained_variance_ratio
        D = len(var_ratio)
        cum_var = np.cumsum(var_ratio)
        k = getattr(
            self,
            "svd_k_selected",
            self.svd_components.shape[1] if self.svd_components is not None else 8,
        )
        thresholds = getattr(self, "svd_thresholds", None)

        multiplier = getattr(self.config, "svd_ratio_multiplier", 2.0)

        # Compute Relative Variance Ratio R_i = v_i / (Var_rem / N_rem)
        relative_ratios = np.zeros(D)
        cum_var_so_far = 0.0
        for i in range(D):
            var_remaining = 1.0 - cum_var_so_far
            n_remaining = D - i
            avg_rem = var_remaining / n_remaining
            relative_ratios[i] = var_ratio[i] / avg_rem if avg_rem > 0 else 1.0
            cum_var_so_far += var_ratio[i]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        x_indices = np.arange(1, D + 1)
        mode = getattr(self.config, "svd_mode", "variance_threshold")
        k_rule = getattr(self, "svd_k_rule", None)

        if mode == "fixed_k" and k > 0 and k <= D:
            r_at_k = relative_ratios[k - 1]
            ax1.axhline(
                y=r_at_k,
                color="tab:purple",
                linestyle="--",
                linewidth=1.5,
                label=f"Implied Threshold at k={k} ($M = {r_at_k:.1f}$)",
            )
            ax1.plot(
                k,
                r_at_k,
                "o",
                color="red",
                markersize=8,
                label=f"Intersection Point ($k={k}, R_{{{k}}}={r_at_k:.1f}$)",
            )
            ax1.axvline(
                x=k,
                color="red",
                linestyle="-",
                linewidth=2,
                label=f"Fixed SVD Cutoff (k={k})",
            )
            if k_rule is not None and k_rule != k:
                ax1.axvline(
                    x=k_rule,
                    color="orange",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"Ratio Rule Cutoff (k={k_rule} @ M={multiplier:g})",
                )
        else:
            ax1.axhline(
                y=multiplier,
                color="tab:orange",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold Line ($M = {multiplier:g}$)",
            )
            ax1.axvline(
                x=k,
                color="red",
                linestyle="-",
                linewidth=2,
                label=f"Selected SVD Cutoff (k={k})",
            )
            if k > 0 and k <= D:
                r_at_k = relative_ratios[k - 1]
                ax1.plot(
                    k,
                    r_at_k,
                    "o",
                    color="red",
                    markersize=8,
                    label=f"Intersection ($k={k}, R_{{{k}}}={r_at_k:.1f}$)",
                )

        ax1.axhline(
            y=1.0,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            label="Theoretical Min ($R_i = 1.0$)",
        )

        ax1.set_ylabel(r"Relative Ratio $R_i$ (Log Scale)")
        ax1.set_yscale("log")
        ax1.set_title(
            f"SVD Component Relative Variance Ratio & Cutoff Threshold (k = {k})"
        )
        ax1.legend(loc="upper right")
        ax1.grid(True, which="both", alpha=0.3)

        ax2.plot(
            x_indices,
            cum_var,
            color="tab:green",
            linewidth=2,
            label="Cumulative Variance",
        )
        ax2.axvline(
            x=k,
            color="red",
            linestyle="-",
            linewidth=2,
            label=(
                f"Selected SVD Subspace ({cum_var[k-1]*100:.1f}%)" if k > 0 else "k=0"
            ),
        )
        if mode == "fixed_k" and k_rule is not None and k_rule != k:
            ax2.axvline(
                x=k_rule,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label=f"Ratio Rule Cutoff (k={k_rule})",
            )

        if k > 0:
            ax2.axhline(y=cum_var[k - 1], color="red", linestyle=":", alpha=0.7)
        ax2.set_xlabel("Ordered SVD Component Index")
        ax2.set_ylabel("Cumulative Variance")
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        if base_path:
            save_path = os.path.join(base_path, "svd_variance_explained.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            _log.info(
                "%s", f"[SVD Filtered SAE] Saved SVD variance plot to {save_path}"
            )

            # Log to TensorBoard
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_writer = SummaryWriter(os.path.join(base_path, "sae_logs"))
                tb_writer.add_figure("svd/variance_explained", fig, 0)
                tb_writer.add_scalar("svd/selected_k", k, 0)
                tb_writer.add_scalar(
                    "svd/cum_variance_selected_k",
                    float(cum_var[k - 1]) if k > 0 else 0.0,
                    0,
                )
                tb_writer.close()
            except Exception as e:
                _log.info("%s", f"[TensorBoard] Could not log SVD variance plot: {e}")

            # Log to W&B
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "svd/variance_explained_plot": wandb.Image(fig),
                            "svd/cum_variance_selected_k": (
                                float(cum_var[k - 1]) if k > 0 else 0.0
                            ),
                            "svd/selected_k": k,
                        }
                    )
            except Exception as e:
                _log.info("%s", f"[W&B] Could not log SVD variance plot: {e}")

        plt.close(fig)

    def get_components(self) -> torch.Tensor:
        """Concatenated component matrix of the SVD subspace and the inner SAE.

        Returns
        -------
        torch.Tensor
            CPU tensor ``[D, k + n_sae]``; the first ``k`` columns are the SVD
            directions, the remaining ones the inner SAE's components.

        Raises
        ------
        RuntimeError
            If :meth:`fit` (or :meth:`load_from_disk`) has not run yet.
        """
        if self.svd_components is None:
            raise RuntimeError("SVD components have not been computed yet.")
        sae_comps = self.inner_sae.get_components()  # [D, N_sae]
        return torch.cat([self.svd_components.cpu(), sae_comps.cpu()], dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse code of raw activations.

        Parameters
        ----------
        x : torch.Tensor
            Raw activations ``[N, D]``.

        Returns
        -------
        torch.Tensor
            ``[N, k + n_sae]`` on ``self.device``: the projections of the
            centred input onto the ``k`` SVD directions, followed by the inner
            SAE's code of the residual.
        """
        x_dev = x.to(self.device)
        mu_dev = self.mu.to(self.device)
        svd_comp_dev = self.svd_components.to(self.device)

        x_c = x_dev - mu_dev
        z_svd = x_c @ svd_comp_dev  # [N, k]

        x_svd_proj = z_svd @ svd_comp_dev.T
        x_residual = x_c - x_svd_proj

        z_sae = self.inner_sae.encode(x_residual)  # [N, N_sae]
        return torch.cat([z_svd, z_sae], dim=1)

    def save_on_disk(self, path: str):
        """Save ``mu``, the SVD subspace, variance ratios and config to ``path``.

        The inner SAE is saved separately to ``path + ".inner"`` when it
        supports ``save_on_disk``.

        Parameters
        ----------
        path : str
            Target file; parent directories are created.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "mu": self.mu.cpu() if self.mu is not None else None,
                "svd_components": (
                    self.svd_components.cpu()
                    if self.svd_components is not None
                    else None
                ),
                "svd_explained_variance_ratio": self.svd_explained_variance_ratio,
                "config": self.config,
            },
            path,
        )
        if hasattr(self.inner_sae, "save_on_disk"):
            self.inner_sae.save_on_disk(path + ".inner")
        _log.info("%s", f"[SVD Filtered SAE] Saved to {path}")

    def load_from_disk(self, path: str):
        """Restore the state written by :meth:`save_on_disk`.

        Silently does nothing when ``path`` does not exist. The inner SAE is
        restored from ``path + ".inner"`` if that file exists.

        Parameters
        ----------
        path : str
            File written by :meth:`save_on_disk`.
        """
        if not os.path.exists(path):
            return
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.mu = checkpoint.get("mu")
        self.svd_components = checkpoint.get("svd_components")
        self.svd_explained_variance_ratio = checkpoint.get(
            "svd_explained_variance_ratio"
        )
        if hasattr(self.inner_sae, "load_from_disk") and os.path.exists(
            path + ".inner"
        ):
            self.inner_sae.load_from_disk(path + ".inner")
        _log.info("%s", f"[SVD Filtered SAE] Loaded from {path}")

    def eval(self):
        """Put the inner SAE (or its wrapped ``sae`` module) into eval mode.

        Returns
        -------
        SVDFilteredBatchTopKSAE
            ``self``, for chaining like ``nn.Module.eval``.
        """
        if hasattr(self.inner_sae, "eval") and callable(self.inner_sae.eval):
            self.inner_sae.eval()
        elif hasattr(self.inner_sae, "sae") and hasattr(self.inner_sae.sae, "eval"):
            self.inner_sae.sae.eval()
        return self

    def train(self, mode: bool = True):
        """Forward ``train(mode)`` to the inner SAE when it has one.

        Returns
        -------
        SVDFilteredBatchTopKSAE
            ``self``.
        """
        if hasattr(self.inner_sae, "train") and callable(self.inner_sae.train):
            self.inner_sae.train(mode)
        return self
