"""Plotting helpers for evaluating sparse dictionaries against known attributes.

Used by the SAE / MSAE evaluation scripts to visualise how strongly each learned
component correlates with ground-truth attributes (e.g. CelebA labels) and how
much variance of the underlying activations the leading components explain.
The commented-out block at the end is an earlier version kept for reference.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from peal.log import get_logger

_log = get_logger(__name__)


def plot_component_ground_truth_correlations(
    filename,
    components,
    ground_truth_attributes,
    data,
    component_names=None,
    attribute_names=None,
    text_cutoff=10,  # New parameter: threshold for hiding text
):
    """
    Plots a correlation heatmap (Top) and a Full Cumulative Explained Variance plot (Bottom).

    Args:
        filename (str): Path to save the image.
        components (array-like): Shape (n_samples, M). Extracted component scores (X-axis).
        ground_truth_attributes (array-like): Shape (n_samples, K). Ground truth values (Y-axis).
        data (array-like): Shape (n_samples, D). Original data used for full variance calculation.
        component_names (list): Optional labels for components.
        attribute_names (list): Optional labels for attributes.
        text_cutoff (int): If number of component columns > text_cutoff, values won't be printed on the heatmap.
    """

    # --- 1. Data Standardization ---
    def process_input(x):
        arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        return arr

    comps = process_input(components)
    attrs = process_input(ground_truth_attributes)
    data_raw = process_input(data)

    n_samples = data_raw.shape[0]

    # Basic shape checks
    if comps.shape[0] != n_samples:
        raise ValueError(
            f"Shape Mismatch: Data has {n_samples} samples, Components has {comps.shape[0]}."
        )
    if attrs.shape[0] != n_samples:
        raise ValueError(
            f"Shape Mismatch: Data has {n_samples} samples, Attributes has {attrs.shape[0]}."
        )

    M = comps.shape[1]  # Number of Components to Plot
    K = attrs.shape[1]  # Number of Attributes
    D = data_raw.shape[1]  # Total Data Dimensions

    # --- 2. Calculations ---

    # A. Correlation Matrix
    corr_matrix = np.zeros((K, M))
    for i in range(K):
        for j in range(M):
            # Handle constant columns to avoid NaN
            if np.std(attrs[:, i]) == 0 or np.std(comps[:, j]) == 0:
                c = 0.0
            else:
                c = np.corrcoef(attrs[:, i], comps[:, j])[0, 1]
            corr_matrix[i, j] = c

    # B. True Variance Spectrum (SVD on full data)
    data_centered = data_raw - np.mean(data_raw, axis=0)
    try:
        # Use simple numpy svd (S are singular values)
        S = np.linalg.svd(data_centered, full_matrices=False, compute_uv=False)
        eigenvalues = S**2 / (n_samples - 1)
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
    except np.linalg.LinAlgError:
        _log.info("%s", "SVD Convergence failed. Plotting flat variance.")
        cumulative_variance = np.zeros(D)

    # --- 3. Plotting (Vertical Stack) ---
    fig, (ax_corr, ax_var) = plt.subplots(
        2, 1, figsize=(12, 16), gridspec_kw={"height_ratios": [2, 1]}
    )

    # === TOP PLOT: Heatmap ===
    im = ax_corr.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    # Labels
    x_labels = component_names if component_names else [f"Comp {j+1}" for j in range(M)]
    y_labels = attribute_names if attribute_names else [f"Attr {i+1}" for i in range(K)]

    ax_corr.set_xticks(np.arange(M))
    ax_corr.set_yticks(np.arange(K))
    ax_corr.set_xticklabels(x_labels, rotation=45, ha="right")
    ax_corr.set_yticklabels(y_labels)

    title_suffix = "" if M <= text_cutoff else " (Values hidden due to size)"
    ax_corr.set_xlabel(f"Components (Top {M} Visualized)")
    ax_corr.set_ylabel("Ground Truth Attributes")
    ax_corr.set_title(f"Correlation Heatmap: Attributes vs Components{title_suffix}")

    # Annotate Heatmap (CONDITIONAL)
    # Only run this loop if columns are within the cutoff
    if M <= text_cutoff:
        for i in range(K):
            for j in range(M):
                val = corr_matrix[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax_corr.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_corr, pad=0.02)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

    # === BOTTOM PLOT: Variance ===
    x_indices = np.arange(1, len(cumulative_variance) + 1)

    ax_var.plot(
        x_indices,
        cumulative_variance,
        marker=".",
        linestyle="-",
        color="b",
        linewidth=1.5,
    )
    ax_var.fill_between(x_indices, cumulative_variance, alpha=0.1, color="b")

    # Highlight the specific cutoff M used in the heatmap
    if M <= len(cumulative_variance):
        current_explained = cumulative_variance[M - 1]
        ax_var.axvline(x=M, color="red", linestyle="--", alpha=0.7)
        ax_var.scatter(
            [M],
            [current_explained],
            color="red",
            s=100,
            zorder=5,
            label=f"Heatmap Cutoff (Comp {M}: {current_explained:.2%} var)",
        )
        ax_var.legend(loc="lower right")

    ax_var.set_title(f"True Cumulative Variance Spectrum (Total Dimensions: {D})")
    ax_var.set_xlabel("Component Index")
    ax_var.set_ylabel("Fraction of Total Data Variance")
    ax_var.set_ylim(0, 1.05)
    ax_var.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    _log.info("%s", f"Analysis saved to {filename}")


def select_dictionary_components(sparse_dictionary, explainer_config, num_attempts):
    """Return (W, selected): the dictionary columns the counterfactual attempts steer along.

    W has shape [Dim, num_attempts]; selected is the list of column indices.
    If explainer_config.component_indices is set, attempt k uses column
    component_indices[k]. Otherwise attempt k uses column k, which is the original
    behaviour (get_components()[:, :num_attempts]).
    """
    custom = getattr(explainer_config, "component_indices", None)
    if custom:
        selected = [int(i) for i in custom]
        if len(selected) != num_attempts:
            raise ValueError(
                f"component_indices has {len(selected)} entries but num_attempts is "
                f"{num_attempts}; they must match (attempt k uses component_indices[k])."
            )
    else:
        selected = list(range(num_attempts))
    W_all = sparse_dictionary.get_components()
    if isinstance(W_all, torch.Tensor):
        W_all = W_all.detach()
    if max(selected) >= W_all.shape[1]:
        raise ValueError(
            f"component_indices {selected} out of range: the dictionary has "
            f"{W_all.shape[1]} components."
        )
    return W_all[:, selected], selected


def read_component_bounds(c_min_max_path, selected, device, bounds_scale=1.0):
    """Read c_min_and_maxes.txt and return (c_mins, c_maxs) for the selected columns.

    bounds_scale (optional, default 1.0) widens each [c_min, c_max] range around its centre,
    as component_bounds_scale on the dev branch. At 1.0 the bounds are returned unchanged.
    """
    c_mins, c_maxs = [], []
    with open(c_min_max_path, "r") as file:
        for line in file:
            parts = line.strip().split("min=")
            if len(parts) > 1:
                c_mins.append(float(parts[1].split(",")[0]))
                c_maxs.append(float(line.strip().split("max=")[1]))
    c_mins = torch.tensor(c_mins, device=device)[selected]
    c_maxs = torch.tensor(c_maxs, device=device)[selected]
    bounds_scale = float(bounds_scale if bounds_scale is not None else 1.0)
    if bounds_scale != 1.0:
        mid = 0.5 * (c_mins + c_maxs)
        half = 0.5 * (c_maxs - c_mins) * bounds_scale
        c_mins, c_maxs = mid - half, mid + half
    return c_mins, c_maxs
