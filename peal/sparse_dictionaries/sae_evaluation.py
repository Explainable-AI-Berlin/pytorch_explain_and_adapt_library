"""Ground-truth evaluation of sparse dictionaries (SAE / MSAE / SVD-filtered).

Given a fitted dictionary with an ``encode`` method, activations ``x`` and
attribute labels ``y``, :func:`run_sae_eval` matches every binary attribute to
one latent (Hungarian assignment on Pearson correlation, optionally also on the
achievable F1), picks a threshold per attribute, and reports coverage, F1,
AUC and average precision, in-sample and on an optional held-out split.
Continuous ("dense") factors are scored by linear regression from their
matched latent. Results go to ``sae_evaluation_report.txt``,
``sae_evaluation_results.pkl`` and ``matching_matrix_diagonalized.png`` in
``base_path`` and can be forwarded to W&B or TensorBoard. The module also
carries the CelebA attribute consistency table used to define the
"well-defined" attribute subsets.
"""

import os
import torch
import numpy as np
import pickle
from pathlib import Path

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    average_precision_score,
    roc_auc_score,
)
from peal.log import get_logger

_log = get_logger(__name__)


# Table 1 of Lingenfelter et al., "A Quantitative Analysis of Labeling Issues in
# the CelebA Dataset" (arXiv:2210.07356): the number of images, out of 1,000, on
# which two independent manual relabellings disagree about each attribute. Low
# N_d means the attribute is objectively defined; high N_d means two careful
# annotators simply do not mean the same thing by it.
#
# The paper's own banding is >= 95% consistency (N_d <= 50, 12 attributes),
# 85-95% (yellow), and <= 85% (pink). Keys match the CelebA data.csv header.
CELEBA_ATTRIBUTE_ND = {
    "Eyeglasses": 3,
    "Wearing_Hat": 14,
    "Goatee": 16,
    "Mouth_Slightly_Open": 17,
    "Bald": 18,
    "Mustache": 22,
    "Male": 23,
    "Wearing_Necktie": 27,
    "No_Beard": 34,
    "Gray_Hair": 37,
    "Wearing_Necklace": 49,
    "Wearing_Earrings": 50,
    "Double_Chin": 58,
    "Blond_Hair": 101,
    "Sideburns": 105,
    "Chubby": 110,
    "5_o_Clock_Shadow": 110,
    "Blurry": 118,
    "Wearing_Lipstick": 133,
    "Smiling": 141,
    "Bangs": 150,
    "Bushy_Eyebrows": 158,
    "Young": 169,
    "Receding_Hairline": 178,
    "Heavy_Makeup": 181,
    "Big_Lips": 206,
    "Rosy_Cheeks": 217,
    "Black_Hair": 232,
    "Pale_Skin": 237,
    "Bags_Under_Eyes": 284,
    "Brown_Hair": 284,
    "Big_Nose": 286,
    "Wavy_Hair": 299,
    "Straight_Hair": 369,
    "Attractive": 377,
    "Narrow_Eyes": 394,
    "Arched_Eyebrows": 406,
    "Oval_Face": 466,
    "Pointy_Nose": 490,
    "High_Cheekbones": 512,
}

# Ordered most-objective-first.
CELEBA_ATTRIBUTES_BY_CONSISTENCY = sorted(
    CELEBA_ATTRIBUTE_ND, key=CELEBA_ATTRIBUTE_ND.get
)

# The 12 the paper puts in its >= 95% consistency band.
CELEBA_CONSISTENT_12 = [
    a for a in CELEBA_ATTRIBUTES_BY_CONSISTENCY if CELEBA_ATTRIBUTE_ND[a] <= 50
]

# The 20 best-defined attributes: the 12 above plus the top of the 85-95% band.
# The cut falls between Smiling (N_d = 141) and Bangs (N_d = 150).
CELEBA_WELL_DEFINED_20 = CELEBA_ATTRIBUTES_BY_CONSISTENCY[:20]


# OLD maybe no longer needed
def compute_component_f1_scores(
    sae,
    activation_store_X,
    ground_truth_labels,
    mu,
    device="cuda",
    n_components=10,
):
    """
    Compute F1 scores for the first n_components SAE components against
    best-matching ground truth binary attributes.

    Args:
        sae: The SAE model (InternalBatchTopKSAE or similar with encode method)
        activation_store_X: Tensor of shape (N, act_size) — raw activations (before centering)
        ground_truth_labels: Tensor of shape (N, K) — ground truth binary labels
                             (first output_split columns are binary digit presence)
        mu: Tensor of shape (act_size,) — mean used for centering
        device: Device to run computation on
        n_components: Number of SAE components to evaluate (default 10)

    Returns:
        dict: Mapping of metric names to values, e.g.:
            {"component_0/f1": 0.85, "component_0/best_gt_idx": 42, ...}
    """
    sae.eval()

    # Center the activations
    X_centered = activation_store_X - mu.to(activation_store_X.device)

    # Compute SAE activations in batches to avoid OOM
    batch_size = 256
    all_acts = []
    with torch.no_grad():
        for i in range(0, X_centered.shape[0], batch_size):
            batch = X_centered[i : i + batch_size].to(device)
            # Forward through the encoder part of the SAE
            x_cent = batch - sae.b_dec
            pre_acts = x_cent @ sae.W_enc
            acts = torch.relu(pre_acts)
            all_acts.append(acts.cpu())

    all_acts = torch.cat(all_acts, dim=0)  # (N, dict_size)

    # Binarize SAE activations: active if > 0
    sae_binary = (all_acts > 0).float().numpy()  # (N, dict_size)

    # Get ground truth binary labels
    gt = ground_truth_labels.cpu().numpy()  # (N, K)
    n_gt_features = gt.shape[1]

    # Limit to first n_components
    n_components = min(n_components, sae_binary.shape[1])

    results = {}
    best_f1_scores = []

    for comp_idx in range(n_components):
        comp_activations = sae_binary[:, comp_idx]

        best_f1 = 0.0
        best_gt_idx = -1

        # Only check binary ground truth features (skip if all 0 or all 1)
        for gt_idx in range(n_gt_features):
            gt_col = gt[:, gt_idx]

            # Skip constant columns (no meaningful F1)
            if np.std(gt_col) < 1e-6:
                continue

            # Binarize ground truth if not already binary
            gt_binary = (gt_col > 0.5).astype(float)

            f1 = f1_score(gt_binary, comp_activations, zero_division=0.0)

            if f1 > best_f1:
                best_f1 = f1
                best_gt_idx = gt_idx

        results[f"component_{comp_idx}/f1"] = best_f1
        results[f"component_{comp_idx}/best_gt_idx"] = best_gt_idx
        best_f1_scores.append(best_f1)

    # Also log the average F1 across all evaluated components
    if best_f1_scores:
        results["mean_top10_f1"] = float(np.mean(best_f1_scores))

    sae.train()
    return results


def run_sae_eval(
    sae,
    x: torch.Tensor,
    y: torch.Tensor,
    base_path: str = None,
    label_names: list = None,
    matching_method: str = "correlation",
    verbose: bool = True,
    encode_batch_size: int = 8192,
    store_latents: bool = False,
    n_binary: int = None,
    x_holdout: torch.Tensor = None,
    y_holdout: torch.Tensor = None,
    f1_matching: bool = True,
    subset_names: list = None,
    subset_key: str = "subset",
) -> dict:
    """
    Full SAE evaluation for superposition datasets.

    Args:
        sae:             SAE model with a .encode() method
        x:               Input tensors, shape (N, 512)
        y:               Labels, shape (N, num_labels). May mix binary
                         (sparse) columns with continuous (dense) columns,
                         as in SparseNumbersDenseZipf: 1000 binary "Num"
                         columns followed by 8 continuous "Red" intensities.
        label_names:     Optional names for the labels
        matching_method: "correlation" (optimal) or "greedy" (fast)
        verbose:         Print and save a report
        n_binary:        Number of leading binary columns. If None, columns
                         are classified automatically (a column is binary iff
                         all its values are 0 or 1).
        x_holdout:       Optional second input tensor, never used for matching
                         or threshold selection. Since both the latent->label
                         assignment and the per-label threshold are chosen to
                         maximise agreement on ``x``, the metrics on ``x`` are
                         in-sample. Passing a disjoint split here gives the
                         honest number under the same protocol.
        y_holdout:       Labels for ``x_holdout``.
        subset_names:    Optional subset of ``label_names`` to additionally
                         aggregate over (e.g. CELEBA_WELL_DEFINED_20). Reported
                         as ``<subset_key>_f1_macro`` etc.
        subset_key:      Metric-name prefix for that subset.

    Returns:
        dict with all results
    """
    if hasattr(sae, "eval") and callable(sae.eval):

        sae.eval()
    elif hasattr(sae, "sae") and hasattr(sae.sae, "eval") and callable(sae.sae.eval):
        sae.sae.eval()
    elif (
        hasattr(sae, "inner_sae")
        and hasattr(sae.inner_sae, "eval")
        and callable(sae.inner_sae.eval)
    ):
        sae.inner_sae.eval()

    if hasattr(sae, "device"):
        device = sae.device
    elif hasattr(sae, "parameters") and len(list(sae.parameters())) > 0:
        device = next(sae.parameters()).device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x = x.to(device)

    def _binarize_labels(labels):
        """Map the categorical label columns to {0, 1}, leave continuous ones.

        CelebA columns arrive as {-1, 1} or {0, 1}, and sklearn's binary metrics
        need {0, 1} or the {-1, 0, 1} union of targets and predictions is read as
        a multiclass problem and f1_score raises.

        Only columns that are already two-valued are touched. Thresholding
        everything at 0.5 would also flatten genuinely continuous factors — on
        the numbers dataset that silently turned the 8 `Red*` intensities into
        binary columns, so `col_is_binary` saw 1008 binary and 0 continuous
        columns and the whole dense-factor section of the report disappeared.
        """
        arr = (
            labels.cpu().numpy()
            if isinstance(labels, torch.Tensor)
            else np.asarray(labels)
        )
        arr = arr.astype(np.float64, copy=True)
        for j in range(arr.shape[1]):
            uniq = np.unique(arr[:, j])
            if len(uniq) <= 2 and np.all(np.isin(uniq, (-1.0, 0.0, 1.0))):
                arr[:, j] = (arr[:, j] > 0.5).astype(np.float64)
        return arr

    y = _binarize_labels(y)

    def _encode_all(inputs):
        # Encode in chunks: with a wide dictionary (e.g. 16384) the dense
        # (N, dict) activation matrix does not fit on the GPU in one go. The
        # result always comes back on the CPU, so every consumer — including
        # the held-out split below — agrees on where the tensors live.
        chunks = []
        with torch.no_grad():
            for start in range(0, inputs.shape[0], encode_batch_size):
                chunk = inputs[start : start + encode_batch_size].to(device)
                chunks.append(sae.encode(chunk).cpu())
        return torch.cat(chunks, dim=0)

    _log.info("%s", "► Encoding samples...")
    latents = _encode_all(x)

    # Normalize latents to [0, 1] for stable thresholding
    lat_min = latents.min(dim=0).values
    lat_max = latents.max(dim=0).values
    latents_norm = (latents - lat_min) / (lat_max - lat_min + 1e-8)
    # latents_norm = latents

    y_full = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    num_labels_full = y_full.shape[1]

    if label_names is None:
        label_names = [f"label_{j}" for j in range(num_labels_full)]

    # --- Split binary (sparse) from continuous (dense) label columns -------
    # f1_score cannot handle continuous targets, so the dense factors are
    # scored separately as a regression problem (see _dense_analysis).
    if n_binary is None:
        col_is_binary = np.all((y_full == 0) | (y_full == 1), axis=0)
    else:
        col_is_binary = np.zeros(num_labels_full, dtype=bool)
        col_is_binary[:n_binary] = True

    bin_idx = np.flatnonzero(col_is_binary)
    dense_idx = np.flatnonzero(~col_is_binary)
    _log.info(
        "%s",
        f"  {len(bin_idx)} binary (sparse) label columns, "
        f"{len(dense_idx)} continuous (dense) label columns",
    )

    y_bin = y_full[:, bin_idx]
    bin_names = [label_names[j] for j in bin_idx]

    _log.info("%s", "► Matching latents to labels...")
    latent_to_label, label_to_latent, corr_matrix, signs = _match_latents_to_labels(
        latents_norm, y_bin, method=matching_method
    )

    thresholds = _find_optimal_thresholds(latents_norm, y_bin, label_to_latent, signs)
    # thresholds = np.full(y_bin.shape[1], 0.1)

    _log.info("%s", "► Computing predictions...")
    preds, probs = _predict_labels(latents_norm, label_to_latent, thresholds, signs)

    _log.info("%s", "► Computing metrics...")
    metrics = _compute_metrics(y_bin, preds, probs, label_to_latent)
    per_label = _per_label_analysis(
        y_bin, preds, probs, label_to_latent, corr_matrix, bin_names
    )

    if subset_names:
        metrics.update(
            _subset_metrics(
                y_bin,
                preds,
                probs,
                label_to_latent,
                bin_names,
                subset_names,
                subset_key,
            )
        )

    # --- An alternative matcher, scored on the quantity being reported -----
    # Reported under an `f1match_` prefix and never mixed into the primary
    # numbers: it answers "does the dictionary contain a latent that detects
    # this attribute at all", where the correlation matcher answers "does it
    # contain one that is linearly related to it". They differ most on the rare
    # attributes, which is most of the well-defined ones.
    # Cost is O(latents x labels) sorted passes over the samples. That is a few
    # seconds for CelebA's 40 labels and minutes-to-hours for the numbers
    # dataset's 1000, where the correlation matcher already reaches 0.999
    # coverage and there is nothing for this to diagnose.
    f1_match_budget = 1_000_000
    if f1_matching and latents_norm.shape[1] * y_bin.shape[1] > f1_match_budget:
        _log.info(
            "%s",
            f"  Skipping F1 matching: {latents_norm.shape[1]} latents x "
            f"{y_bin.shape[1]} labels exceeds the {f1_match_budget} budget.",
        )
        f1_matching = False

    if f1_matching:
        _log.info("%s", "► Matching latents to labels by achievable F1...")
        f1mat, signmat = _best_f1_matrix(latents_norm, y_bin)
        row_ind, col_ind = linear_sum_assignment(-f1mat)
        l2l_f1 = np.full(y_bin.shape[1], -1, dtype=int)
        signs_f1 = np.ones(y_bin.shape[1])
        for lat_i, lab_j in zip(row_ind, col_ind):
            # A latent whose best F1 does not beat "predict everything positive"
            # has not detected anything, so it does not count as a match.
            prevalence = y_bin[:, lab_j].mean()
            trivial = 2 * prevalence / (1 + prevalence) if prevalence > 0 else 0.0
            if f1mat[lat_i, lab_j] > trivial:
                l2l_f1[lab_j] = lat_i
                signs_f1[lab_j] = signmat[lat_i, lab_j]
        th_f1 = _find_optimal_thresholds(latents_norm, y_bin, l2l_f1, signs_f1)
        preds_f1, probs_f1 = _predict_labels(latents_norm, l2l_f1, th_f1, signs_f1)
        m_f1 = _compute_metrics(y_bin, preds_f1, probs_f1, l2l_f1)
        metrics.update({f"f1match_{k}": v for k, v in m_f1.items()})
        if subset_names:
            metrics.update(
                {
                    f"f1match_{k}": v
                    for k, v in _subset_metrics(
                        y_bin,
                        preds_f1,
                        probs_f1,
                        l2l_f1,
                        bin_names,
                        subset_names,
                        subset_key,
                    ).items()
                }
            )
    else:
        l2l_f1 = th_f1 = signs_f1 = None

    # --- The subset on its own terms ---------------------------------------
    # The matching above is one-to-one over all 40 attributes, so a latent that
    # is the best match for a well-defined attribute can be assigned away to an
    # ill-defined one whenever that raises the total correlation. Re-running the
    # assignment against the subset alone answers the narrower question: if only
    # these attributes are of interest, how well does the dictionary carry them?
    if subset_names:
        name_to_idx = {n: j for j, n in enumerate(bin_names)}
        sub_idx = np.asarray([name_to_idx[n] for n in subset_names if n in name_to_idx])
        if len(sub_idx) > 0:
            y_sub = y_bin[:, sub_idx]
            _, l2l_sub, corr_sub, signs_sub = _match_latents_to_labels(
                latents_norm, y_sub, method=matching_method
            )
            th_sub = _find_optimal_thresholds(latents_norm, y_sub, l2l_sub, signs_sub)
            preds_sub, probs_sub = _predict_labels(
                latents_norm, l2l_sub, th_sub, signs_sub
            )
            m_sub = _compute_metrics(y_sub, preds_sub, probs_sub, l2l_sub)
            metrics.update({f"{subset_key}only_{k}": v for k, v in m_sub.items()})
            results_subset_only = _per_label_analysis(
                y_sub,
                preds_sub,
                probs_sub,
                l2l_sub,
                corr_sub,
                [bin_names[j] for j in sub_idx],
            )
        else:
            results_subset_only = None
    else:
        results_subset_only = None

    # --- Same matching and thresholds, applied to an unseen split -----------
    per_label_holdout = None
    if x_holdout is not None and y_holdout is not None:
        _log.info("%s", "► Re-scoring on the held-out split...")
        latents_ho = _encode_all(x_holdout)
        latents_ho_norm = (latents_ho - lat_min) / (lat_max - lat_min + 1e-8)
        # Same treatment as the primary labels, or the columns disagree on
        # encoding and sklearn refuses to score them together.
        y_ho_full = _binarize_labels(y_holdout)
        y_ho_bin = y_ho_full[:, bin_idx].astype(np.int64)
        preds_ho, probs_ho = _predict_labels(
            latents_ho_norm, label_to_latent, thresholds, signs
        )
        metrics_ho = _compute_metrics(y_ho_bin, preds_ho, probs_ho, label_to_latent)
        if subset_names:
            metrics_ho.update(
                _subset_metrics(
                    y_ho_bin,
                    preds_ho,
                    probs_ho,
                    label_to_latent,
                    bin_names,
                    subset_names,
                    subset_key,
                )
            )
        if l2l_f1 is not None:
            preds_hf, probs_hf = _predict_labels(
                latents_ho_norm, l2l_f1, th_f1, signs_f1
            )
            metrics_ho.update(
                {
                    f"f1match_{k}": v
                    for k, v in _compute_metrics(
                        y_ho_bin, preds_hf, probs_hf, l2l_f1
                    ).items()
                }
            )
            if subset_names:
                metrics_ho.update(
                    {
                        f"f1match_{k}": v
                        for k, v in _subset_metrics(
                            y_ho_bin,
                            preds_hf,
                            probs_hf,
                            l2l_f1,
                            bin_names,
                            subset_names,
                            subset_key,
                        ).items()
                    }
                )
        metrics.update({f"holdout_{k}": v for k, v in metrics_ho.items()})
        per_label_holdout = _per_label_analysis(
            y_ho_bin, preds_ho, probs_ho, label_to_latent, corr_matrix, bin_names
        )

    _log.info("%s", "► Analysing dense factors...")
    per_dense, dense_metrics = _dense_analysis(
        latents_norm,
        y_full[:, dense_idx],
        [label_names[j] for j in dense_idx],
        sae=sae,
    )
    metrics.update(dense_metrics)

    # Re-express the matching in terms of the ORIGINAL label columns so that
    # downstream consumers (e.g. find_counterfactual) can index y directly.
    latent_to_label_full = np.where(
        latent_to_label >= 0, bin_idx[latent_to_label.clip(min=0)], -1
    )
    label_to_latent_full = np.full(num_labels_full, -1, dtype=int)
    label_to_latent_full[bin_idx] = label_to_latent

    results = {
        "metrics": metrics,
        "per_label": per_label,
        "per_label_holdout": per_label_holdout,
        "per_label_subset_only": results_subset_only,
        "signs": signs,
        "subset_names": list(subset_names) if subset_names else None,
        "subset_key": subset_key,
        "per_dense": per_dense,
        "latent_to_label": latent_to_label_full,
        "label_to_latent": label_to_latent_full,
        "corr_matrix": corr_matrix,
        "thresholds": thresholds,
        "binary_label_indices": bin_idx,
        "dense_label_indices": dense_idx,
        "binary_label_names": bin_names,
        "binary_label_to_latent": label_to_latent,
    }
    if store_latents:
        # (N, dict_size) float32 - several GB for a wide dictionary, so opt-in.
        results["latents"] = latents

    if verbose and base_path:
        _print_report(results)
        _save_report(results, os.path.join(base_path, "sae_evaluation_report.txt"))
        _save_results(results, os.path.join(base_path, "sae_evaluation_results.pkl"))
        plot_diagonalized_matching_matrix(
            corr_matrix,
            label_names=bin_names,
            label_to_latent=label_to_latent,
            save_path=os.path.join(base_path, "matching_matrix_diagonalized.png"),
        )

    return results


def _dense_analysis(
    latents,
    y_dense: np.ndarray,
    dense_names: list,
    sae=None,
) -> tuple[list, dict]:
    """
    Evaluate continuous ("dense") ground-truth factors against SAE latents.

    Each dense factor is assigned exactly one latent via Hungarian matching on
    the absolute Pearson correlation, and is then predicted from that latent
    with a least-squares linear fit  y ≈ a * z + b.

    Reported per factor: correlation, MAE, the MAE of the mean-predictor
    baseline (so the MAE is interpretable), R², and whether the assigned
    latent lies in the SVD block of an SVDFiltered* dictionary — i.e. whether
    the dense factor was captured by the dense subspace as intended, or leaked
    into the sparse SAE part.
    """
    if y_dense.shape[1] == 0:
        return [], {}

    L = (
        latents.cpu().numpy()
        if isinstance(latents, torch.Tensor)
        else np.asarray(latents)
    )

    L_c = L - L.mean(axis=0, keepdims=True)
    Y_c = y_dense - y_dense.mean(axis=0, keepdims=True)
    L_c = L_c / L_c.std(axis=0, keepdims=True).clip(1e-8)
    Y_c = Y_c / Y_c.std(axis=0, keepdims=True).clip(1e-8)
    corr = np.abs((L_c.T @ Y_c) / L.shape[0])  # (latent_dim, n_dense)

    row_ind, col_ind = linear_sum_assignment(-corr)
    assignment = {int(c): int(r) for r, c in zip(row_ind, col_ind)}

    svd_k = 0
    if sae is not None and getattr(sae, "svd_components", None) is not None:
        svd_k = int(sae.svd_components.shape[1])

    per_dense = []
    for j, name in enumerate(dense_names):
        y_j = y_dense[:, j]
        baseline_mae = float(np.mean(np.abs(y_j - y_j.mean())))
        lat_i = assignment.get(j, -1)

        if lat_i < 0:
            per_dense.append(
                {
                    "label": name,
                    "latent_idx": -1,
                    "corr": 0.0,
                    "mae": baseline_mae,
                    "baseline_mae": baseline_mae,
                    "r2": 0.0,
                    "in_svd_block": False,
                }
            )
            continue

        z = L[:, lat_i]
        design = np.stack([z, np.ones_like(z)], axis=1)
        coef, *_ = np.linalg.lstsq(design, y_j, rcond=None)
        pred = design @ coef

        ss_res = float(np.sum((y_j - pred) ** 2))
        ss_tot = float(np.sum((y_j - y_j.mean()) ** 2))

        per_dense.append(
            {
                "label": name,
                "latent_idx": lat_i,
                "corr": float(corr[lat_i, j]),
                "mae": float(np.mean(np.abs(y_j - pred))),
                "baseline_mae": baseline_mae,
                "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                "in_svd_block": bool(lat_i < svd_k),
            }
        )

    metrics = {
        "dense_total": len(per_dense),
        "dense_mae_mean": float(np.mean([e["mae"] for e in per_dense])),
        "dense_baseline_mae_mean": float(
            np.mean([e["baseline_mae"] for e in per_dense])
        ),
        "dense_r2_mean": float(np.nanmean([e["r2"] for e in per_dense])),
        "dense_corr_mean": float(np.mean([e["corr"] for e in per_dense])),
        "dense_in_svd_block": int(sum(e["in_svd_block"] for e in per_dense)),
        "svd_k": svd_k,
    }
    return per_dense, metrics


def _dense_report_lines(results: dict) -> list:
    """Report section for the continuous ("dense") ground-truth factors."""
    per_dense = results.get("per_dense") or []
    if not per_dense:
        return []

    m = results.get("metrics", {})
    svd_k = m.get("svd_k", 0)

    lines = ["\n[6] DENSE FACTORS (regression — lower MAE is better)"]
    lines.append(
        f"    {'Factor':<20} {'Latent':>8} {'Block':>8} "
        f"{'Corr':>6} {'MAE':>8} {'Base':>8} {'R2':>7}"
    )
    lines.append(f"    {'-'*68}")
    for e in per_dense:
        block = "SVD" if e["in_svd_block"] else "SAE"
        lines.append(
            f"    {e['label']:<20} {str(e['latent_idx']):>8} {block:>8} "
            f"{e['corr']:>6.3f} {e['mae']:>8.4f} {e['baseline_mae']:>8.4f} "
            f"{e['r2']:>7.3f}"
        )

    lines.append(
        f"\n    Mean MAE {m.get('dense_mae_mean', float('nan')):.4f} vs. "
        f"mean-predictor baseline {m.get('dense_baseline_mae_mean', float('nan')):.4f}"
        f"  (mean R² {m.get('dense_r2_mean', float('nan')):.3f})"
    )
    lines.append(
        f"    {m.get('dense_in_svd_block', 0)} / {len(per_dense)} dense factors "
        f"matched a latent inside the SVD block (k = {svd_k})"
    )
    return lines


def _match_latents_to_labels(
    latents: torch.Tensor,
    y: torch.Tensor,
    method: str = "correlation",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the best assignment: latent i → label j.

    Args:
        latents:  SAE activations, shape (N, latent_dim)
        y:        Binary labels,    shape (N, num_labels)
        method:   "correlation"  – Pearson correlation + Hungarian matching
                  "greedy"       – Faster but suboptimal

    Returns:
        latent_to_label: Array of length latent_dim.
                         latent_to_label[i] = j  →  latent i matches label j
                         latent_to_label[i] = -1 →  latent i has no match
        label_to_latent: Array of length num_labels.
                         label_to_latent[j] = i  →  label j is encoded by latent i
                         label_to_latent[j] = -1 →  label j was not found
    """
    L = latents.cpu().numpy() if isinstance(latents, torch.Tensor) else latents
    y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    latent_dim = L.shape[1]
    num_labels = y.shape[1]

    _log.info("%s", f"  Computing correlation matrix ({latent_dim} × {num_labels})...")

    # Pearson correlation between every latent and every label
    # shaped as a (latent_dim, num_labels) cost matrix
    y_norm = y - y.mean(axis=0, keepdims=True)
    y_norm = y_norm / y_norm.std(axis=0, keepdims=True).clip(1e-8)

    # correlation matrix (latent_dim, num_labels), computed column by column
    # so that a wide dictionary matrix is never held in RAM more than once.
    corr_matrix = np.empty((latent_dim, num_labels), dtype=np.float32)
    col_chunk = max(1, int(2**24 // max(1, L.shape[0])))
    for start in range(0, latent_dim, col_chunk):
        block = L[:, start : start + col_chunk].astype(np.float32, copy=True)
        block -= block.mean(axis=0, keepdims=True)
        block /= block.std(axis=0, keepdims=True).clip(1e-8)
        corr_matrix[start : start + col_chunk] = (block.T @ y_norm) / L.shape[0]

    # The sign is kept: a latent that fires when a label is ABSENT encodes that
    # label just as well as one that fires when it is present, but predicting
    # from it requires flipping the comparison. Matching is done on |r|, the
    # sign is handed back so thresholding and the scores fed to AUC/mAP can be
    # oriented accordingly. Without this, every negatively matched label scored
    # near-zero F1 and below-chance AUC, which dragged the macro averages down.
    corr_signs = np.sign(corr_matrix)
    corr_signs[corr_signs == 0] = 1.0
    corr_matrix = np.abs(corr_matrix)  # sign does not matter here

    if method == "correlation":
        # Hungarian algorithm: finds the global optimum
        # works only for square or smaller matrices,
        # hence limited to min(latent_dim, num_labels)
        _log.info("%s", "  Hungarian matching...")
        cost = -corr_matrix  # Minimierungsproblem → negieren
        row_ind, col_ind = linear_sum_assignment(cost)
        # row_ind: Latent-Indizes, col_ind: Label-Indizes

        latent_to_label = np.full(latent_dim, -1, dtype=int)
        label_to_latent = np.full(num_labels, -1, dtype=int)

        for lat_i, lab_j in zip(row_ind, col_ind):
            corr_val = corr_matrix[lat_i, lab_j]
            # Only match if correlation is significant (> 0.1)
            if corr_val > 0.1:
                latent_to_label[lat_i] = lab_j
                label_to_latent[lab_j] = lat_i

    elif method == "greedy":
        # fast: every latent gets its most strongly correlated label
        latent_to_label = np.full(latent_dim, -1, dtype=int)
        label_to_latent = np.full(num_labels, -1, dtype=int)
        used_labels = set()

        best_latents = np.argsort(-corr_matrix.max(axis=1))
        for lat_i in best_latents:
            best_label = np.argmax(corr_matrix[lat_i])
            if corr_matrix[lat_i, best_label] > 0.1 and best_label not in used_labels:
                latent_to_label[lat_i] = best_label
                label_to_latent[best_label] = lat_i
                used_labels.add(best_label)

    # Per-label orientation of its matched latent (+1 / -1).
    signs = np.ones(num_labels)
    for lab_j, lat_i in enumerate(label_to_latent):
        if lat_i >= 0:
            signs[lab_j] = corr_signs[lat_i, lab_j]

    return latent_to_label, label_to_latent, corr_matrix, signs


def _find_optimal_thresholds(
    latents: torch.Tensor,
    y: np.ndarray,
    label_to_latent: np.ndarray,
    signs: np.ndarray = None,
) -> np.ndarray:
    """
    Find the F1-optimal activation threshold per label, exactly.

    The candidate thresholds are the latent's own activation values rather than
    a fixed grid over [0, 1]. That matters: SAE latents are sparse with a long
    right tail, so after min-max normalisation almost all of their mass sits
    just above 0. A 50-point ``linspace(0, 1)`` therefore had its first non-zero
    candidate above nearly every non-zero activation, and the sweep could only
    choose between "predict everything" and "predict almost nothing" — which
    capped the reported F1 far below what the latent actually supports.

    Sorting the activations descending and walking the prefix gives the optimum
    over *all* thresholds in one O(n log n) pass:  cutting after the i-th sample
    predicts exactly those i samples positive, so with ``tp = cumsum(y_sorted)``
    the F1 of that cut is ``2 * tp / (i + P)``. Only cuts that fall between two
    distinct activation values are admissible, otherwise ``>= threshold`` would
    also pull in the tied samples on the other side of the cut.

    Returns:
        thresholds: Array (num_labels,) – optimal threshold per label
    """
    L = latents.cpu().numpy() if isinstance(latents, torch.Tensor) else latents
    y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    num_labels = y.shape[1]
    thresholds = np.full(num_labels, 0.5)
    if signs is None:
        signs = np.ones(num_labels)

    for label_j, lat_i in enumerate(label_to_latent):
        if lat_i == -1:
            continue  # Label not found — threshold irrelevant

        activations = signs[label_j] * L[:, lat_i]
        y_j = (y[:, label_j] > 0.5).astype(np.float64)
        n_pos = y_j.sum()
        if n_pos == 0 or n_pos == len(y_j):
            thresholds[label_j] = float(activations.min())
            continue

        order = np.argsort(-activations, kind="mergesort")
        a_sorted = activations[order]
        tp = np.cumsum(y_j[order])
        k = np.arange(1, len(a_sorted) + 1)
        f1 = 2.0 * tp / (k + n_pos)

        # A cut after position i is only realisable as a threshold if the next
        # activation is strictly smaller; ties must stay on the same side.
        admissible = np.empty(len(a_sorted), dtype=bool)
        admissible[:-1] = a_sorted[:-1] > a_sorted[1:]
        admissible[-1] = True

        f1_masked = np.where(admissible, f1, -1.0)
        best_i = int(np.argmax(f1_masked))
        thresholds[label_j] = float(a_sorted[best_i])

    return thresholds


def _predict_labels(
    latents: torch.Tensor,
    label_to_latent: np.ndarray,
    thresholds: np.ndarray,
    signs: np.ndarray = None,
) -> np.ndarray:
    """
    Produce binary label predictions from SAE latents after matching.

    Returns:
        preds: Shape (N, num_labels), binary predictions
        probs: Shape (N, num_labels), raw activations (for AUC/mAP)
    """
    L = latents.cpu().numpy() if isinstance(latents, torch.Tensor) else latents
    N = L.shape[0]
    num_labels = len(label_to_latent)

    probs = np.zeros((N, num_labels))
    preds = np.zeros((N, num_labels), dtype=int)
    if signs is None:
        signs = np.ones(num_labels)

    for label_j, lat_i in enumerate(label_to_latent):
        if lat_i == -1:
            continue  # Not found -> remains 0
        # Orient the latent so that "more activation" always means "label more
        # likely present", so a single threshold direction works for both signs
        # and so the scores handed to ROC-AUC / mAP are not inverted.
        probs[:, label_j] = signs[label_j] * L[:, lat_i]
        preds[:, label_j] = (probs[:, label_j] >= thresholds[label_j]).astype(int)

    return preds, probs


def _compute_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    label_to_latent: np.ndarray,
) -> dict:
    """Compute all relevant multilabel metrics."""

    found_mask = label_to_latent != -1  # Labels that have a matching latent

    metrics = {
        # --- Coverage ---
        "labels_found": found_mask.sum(),
        "labels_total": len(label_to_latent),
        "coverage": found_mask.mean(),
        # --- Overall performance (all labels) ---
        "f1_micro_all": f1_score(y_true, preds, average="micro", zero_division=0),
        "f1_macro_all": f1_score(y_true, preds, average="macro", zero_division=0),
        "f1_samples_all": f1_score(y_true, preds, average="samples", zero_division=0),
        "hamming_loss_all": hamming_loss(y_true, preds),
        # --- Performance only on found labels ---
        "f1_micro_found": f1_score(
            y_true[:, found_mask],
            preds[:, found_mask],
            average="micro",
            zero_division=0,
        ),
        "f1_macro_found": f1_score(
            y_true[:, found_mask],
            preds[:, found_mask],
            average="macro",
            zero_division=0,
        ),
        "hamming_loss_found": hamming_loss(y_true[:, found_mask], preds[:, found_mask]),
    }

    # mAP and AUC only on found labels (require real scores)
    if found_mask.sum() > 0:
        try:
            metrics["mAP_found"] = average_precision_score(
                y_true[:, found_mask], probs[:, found_mask], average="macro"
            )
            metrics["roc_auc_found"] = roc_auc_score(
                y_true[:, found_mask], probs[:, found_mask], average="macro"
            )
        except ValueError:
            metrics["mAP_found"] = float("nan")
            metrics["roc_auc_found"] = float("nan")

    return metrics


def _best_f1_matrix(latents, y, device=None, chunk=64):
    """Best achievable F1 of every (latent, label) pair, and the sign to use.

    Correlation is a poor proxy for "does this latent detect this attribute"
    when the attribute is rare: a latent that fires on exactly the 2% of images
    that are Bald still only reaches a modest Pearson r, so a fixed r > 0.1 gate
    systematically discards precisely the rare attributes. This scores each pair
    by the quantity actually being reported instead.

    Both orientations are tried and the better kept, so a latent that fires on
    the complement of an attribute counts as detecting it.

    Returns (F1 matrix (latents x labels), sign matrix), computed exactly by the
    same descending-prefix sweep as _find_optimal_thresholds.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    L = latents if isinstance(latents, torch.Tensor) else torch.as_tensor(latents)
    L = L.to(device=device, dtype=torch.float32)
    Y = torch.as_tensor(y, device=device, dtype=torch.float32)
    n, d = L.shape
    n_labels = Y.shape[1]

    best = torch.zeros(d, n_labels, device=device)
    sign = torch.ones(d, n_labels, device=device)
    k = torch.arange(1, n + 1, device=device, dtype=torch.float32).unsqueeze(1)

    for sgn in (1.0, -1.0):
        S = sgn * L
        # One sort per latent, reused for every label.
        for a in range(0, d, chunk):
            b = min(a + chunk, d)
            order = torch.argsort(S[:, a:b], dim=0, descending=True, stable=True)
            s_sorted = torch.gather(S[:, a:b], 0, order)
            # Ties must not be split by a threshold, so only cuts before a
            # strict decrease are admissible.
            adm = torch.ones_like(s_sorted, dtype=torch.bool)
            adm[:-1] = s_sorted[:-1] > s_sorted[1:]
            for j in range(n_labels):
                n_pos = Y[:, j].sum()
                if n_pos == 0:
                    continue
                tp = torch.cumsum(Y[:, j][order], dim=0)
                f1 = 2.0 * tp / (k + n_pos)
                f1 = torch.where(adm, f1, torch.full_like(f1, -1.0))
                v = f1.max(dim=0).values
                upd = v > best[a:b, j]
                best[a:b, j] = torch.where(upd, v, best[a:b, j])
                sign[a:b, j] = torch.where(upd, torch.full_like(v, sgn), sign[a:b, j])

    return best.cpu().numpy(), sign.cpu().numpy()


def _subset_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    label_to_latent: np.ndarray,
    label_names: list,
    subset_names: list,
    key: str = "subset",
) -> dict:
    """
    Aggregate the same metrics over a named subset of the labels.

    Motivation on CelebA: the 40 attributes are not equally well defined. Roughly
    half of them are annotated so inconsistently that two human passes over the
    same 1,000 images disagree on 15-50% of them (arXiv:2210.07356, Table 1), so
    a macro average over all 40 is bounded well below 1 by label noise alone and
    mostly measures how much of that noise the SAE happened to fit. Averaging
    over the well-defined subset instead is the interpretable number.

    Unmatched labels count as F1 = 0, exactly as in the all-label macro average,
    so ``<key>_f1_macro = <key>_coverage x <key>_f1_macro_found`` still holds.
    """
    name_to_idx = {name: j for j, name in enumerate(label_names)}
    idx = [name_to_idx[n] for n in subset_names if n in name_to_idx]
    missing = [n for n in subset_names if n not in name_to_idx]
    if missing:
        _log.info("%s", f"  [subset] not present in this dataset, ignored: {missing}")
    if not idx:
        return {}

    idx = np.asarray(idx)
    found = label_to_latent[idx] != -1

    out = {
        f"{key}_n": int(len(idx)),
        f"{key}_labels_found": int(found.sum()),
        f"{key}_coverage": float(found.mean()),
        f"{key}_f1_macro": float(
            f1_score(y_true[:, idx], preds[:, idx], average="macro", zero_division=0)
        ),
        f"{key}_f1_micro": float(
            f1_score(y_true[:, idx], preds[:, idx], average="micro", zero_division=0)
        ),
    }
    if found.any():
        fidx = idx[found]
        out[f"{key}_f1_macro_found"] = float(
            f1_score(y_true[:, fidx], preds[:, fidx], average="macro", zero_division=0)
        )
        try:
            out[f"{key}_mAP_found"] = float(
                average_precision_score(
                    y_true[:, fidx], probs[:, fidx], average="macro"
                )
            )
            out[f"{key}_roc_auc_found"] = float(
                roc_auc_score(y_true[:, fidx], probs[:, fidx], average="macro")
            )
        except ValueError:
            out[f"{key}_mAP_found"] = float("nan")
            out[f"{key}_roc_auc_found"] = float("nan")
    return out


def _per_label_analysis(
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    label_to_latent: np.ndarray,
    corr_matrix: np.ndarray,
    label_names: list = None,
) -> list[dict]:
    """
    Detailed per-label analysis: F1, correlation, status.

    Returns:
        List of dicts, sorted by F1 (ascending — worst first)
    """
    num_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"label_{j}" for j in range(num_labels)]

    results = []
    for j in range(num_labels):
        lat_i = label_to_latent[j]

        if lat_i == -1:
            status = "not found"
            corr = 0.0
            f1 = 0.0
            ap = 0.0
        else:
            status = "found"
            corr = corr_matrix[lat_i, j]
            f1 = f1_score(y_true[:, j], preds[:, j], zero_division=0)
            try:
                ap = average_precision_score(y_true[:, j], probs[:, j])
            except ValueError:
                ap = float("nan")

        results.append(
            {
                "label": label_names[j],
                "label_idx": j,
                "latent_idx": lat_i,
                "status": status,
                "corr": corr,
                "f1": f1,
                "ap": ap,
                "prevalence": y_true[:, j].mean(),
            }
        )

    return sorted(results, key=lambda d: d["f1"])


def _subset_report_lines(results: dict) -> list:
    """Report section for the named label subset and for the held-out split."""
    m = results.get("metrics", {})
    key = results.get("subset_key", "subset")
    names = results.get("subset_names")
    lines = []

    if names and f"{key}_f1_macro" in m:
        lines.append(
            f"\n[7] WELL-DEFINED SUBSET ({m.get(f'{key}_n', len(names))} labels)"
        )
        lines.append(f"    {'Metric':<25} {'Value':>12}")
        lines.append(f"    {'-'*39}")
        lines.append(
            f"    {'Coverage':<25} "
            f"{m.get(f'{key}_labels_found', 0):>5} / {m.get(f'{key}_n', 0):<6}"
        )
        for label, mk in [
            ("F1 Macro (all)", f"{key}_f1_macro"),
            ("F1 Macro (found)", f"{key}_f1_macro_found"),
            ("F1 Micro", f"{key}_f1_micro"),
            ("mAP (found)", f"{key}_mAP_found"),
            ("ROC-AUC (found)", f"{key}_roc_auc_found"),
        ]:
            if mk in m:
                lines.append(f"    {label:<25} {m[mk]:>12.4f}")

        per_label = {e["label"]: e for e in results.get("per_label", [])}
        lines.append(
            f"\n    {'Attribute':<22} {'Latent':>8} {'Corr':>6} {'F1':>6} {'AP':>6} {'Prev':>6}"
        )
        lines.append(f"    {'-'*58}")
        for name in names:
            e = per_label.get(name)
            if e is None:
                continue
            lines.append(
                f"    {e['label']:<22} {str(e['latent_idx']):>8} {e['corr']:>6.3f} "
                f"{e['f1']:>6.3f} {e['ap']:>6.3f} {e['prevalence']:>6.3f}"
            )

    if f"f1match_{key}_f1_macro" in m:
        lines.append(
            "\n[9] ALTERNATIVE MATCHER (latents matched by achievable F1 rather"
        )
        lines.append("    than by correlation; diagnostic, not the headline number)")
        lines.append(f"    {'Metric':<25} {'val':>10} {'held-out':>10}")
        lines.append(f"    {'-'*47}")
        for label, mk in [
            ("Coverage (subset)", f"f1match_{key}_labels_found"),
            ("F1 Macro (subset)", f"f1match_{key}_f1_macro"),
            ("F1 Macro (all 40)", "f1match_f1_macro_all"),
        ]:
            v = m.get(mk)
            h = m.get(f"holdout_{mk}")
            if v is not None:
                fmt = "{:>10.0f}" if mk.endswith("labels_found") else "{:>10.4f}"
                lines.append(
                    f"    {label:<25} "
                    + fmt.format(v)
                    + (" " + fmt.format(h).strip().rjust(9) if h is not None else "")
                )

    if "holdout_f1_macro_all" in m:
        lines.append(
            "\n[8] HELD-OUT SPLIT (same latent->label matching and thresholds,"
        )
        lines.append("    neither of which was chosen on this data)")
        lines.append(f"    {'Metric':<25} {'Value':>12}")
        lines.append(f"    {'-'*39}")
        for label, mk in [
            ("F1 Macro (all 40)", "holdout_f1_macro_all"),
            ("F1 Micro (all 40)", "holdout_f1_micro_all"),
            ("ROC-AUC (found)", "holdout_roc_auc_found"),
            ("mAP (found)", "holdout_mAP_found"),
            ("F1 Macro (subset)", f"holdout_{key}_f1_macro"),
            ("F1 Macro (subset, found)", f"holdout_{key}_f1_macro_found"),
            ("ROC-AUC (subset)", f"holdout_{key}_roc_auc_found"),
            ("mAP (subset)", f"holdout_{key}_mAP_found"),
        ]:
            if mk in m:
                lines.append(f"    {label:<25} {m[mk]:>12.4f}")

    return lines


def _print_report(results: dict):
    """Print the human-readable evaluation report of :func:`run_sae_eval`."""
    sep = "=" * 62
    m = results["metrics"]
    pl = results["per_label"]

    _log.info("%s", f"\n{sep}")
    _log.info("%s", "  SAE EVALUATION – SUPERPOSITION DATASET")
    _log.info("%s", sep)

    # --- Coverage ---
    found = m["labels_found"]
    total = m["labels_total"]
    pct = m["coverage"] * 100
    _log.info("%s", "\n[1] LABEL COVERAGE")
    _log.info("%s", f"    Found:  {found} / {total}  ({pct:.1f}%)")
    _log.info(
        "%s",
        f"    Missed:  {total - found} Labels → this information is lost in the SAE!",
    )

    # --- Overall metrics ---
    _log.info("%s", "\n[2] CLASSIFICATION PERFORMANCE")
    _log.info("%s", f"    {'Metric':<25} {'All Labels':>12} {'Found Only':>14}")
    _log.info("%s", f"    {'-'*53}")
    _log.info(
        "%s",
        f"    {'F1 Micro':<25} {m['f1_micro_all']:>12.4f} {m['f1_micro_found']:>14.4f}",
    )
    _log.info(
        "%s",
        f"    {'F1 Macro':<25} {m['f1_macro_all']:>12.4f} {m['f1_macro_found']:>14.4f}",
    )
    _log.info("%s", f"    {'F1 Samples':<25} {m['f1_samples_all']:>12.4f} {'—':>14}")
    _log.info(
        "%s",
        f"    {'Hamming Loss':<25} {m['hamming_loss_all']:>12.4f} {m['hamming_loss_found']:>14.4f}",
    )
    if "mAP_found" in m:
        _log.info("%s", f"    {'mAP (found)':<25} {'—':>12} {m['mAP_found']:>14.4f}")
    if "roc_auc_found" in m:
        _log.info(
            "%s", f"    {'ROC-AUC (found)':<25} {'—':>12} {m['roc_auc_found']:>14.4f}"
        )

    # --- Worst Labels ---
    _log.info("%s", "\n[3] WORST 15 LABELS (potential information loss)")
    _log.info(
        "%s", f"    {'Label':<20} {'Status':<18} {'Corr':>6} {'F1':>6} {'Prev':>6}"
    )
    _log.info("%s", f"    {'-'*60}")
    for entry in pl[:15]:
        status_short = "✗ missing" if entry["status"] == "not found" else "~ weak"
        _log.info(
            "%s",
            f"    {entry['label']:<20} {status_short:<18} "
            f"{entry['corr']:>6.3f} {entry['f1']:>6.3f} {entry['prevalence']:>6.3f}",
        )

    # --- Best Labels ---
    _log.info("%s", "\n[4] TOP 10 LABELS (well reconstructed)")
    _log.info("%s", f"    {'Label':<20} {'Latent':>8} {'Corr':>6} {'F1':>6} {'AP':>6}")
    _log.info("%s", f"    {'-'*52}")
    for entry in sorted(pl, key=lambda d: -d["f1"])[:10]:
        _log.info(
            "%s",
            f"    {entry['label']:<20} {str(entry['latent_idx']):>8} "
            f"{entry['corr']:>6.3f} {entry['f1']:>6.3f} {entry['ap']:>6.3f}",
        )

    # --- Summary ---
    good = sum(1 for e in pl if e["f1"] >= 0.8)
    medium = sum(1 for e in pl if 0.5 <= e["f1"] < 0.8)
    bad = sum(1 for e in pl if e["f1"] < 0.5)
    _log.info("%s", "\n[5] SUMMARY")
    _log.info("%s", f"    F1 ≥ 0.8  (good):     {good:>4} labels")
    _log.info("%s", f"    F1 0.5–0.8 (okay):   {medium:>4} labels")
    _log.info(
        "%s", f"    F1 < 0.5  (poor):    {bad:>4} labels  ← information is lost here"
    )

    for line in _dense_report_lines(results):
        _log.info("%s", line)

    for line in _subset_report_lines(results):
        _log.info("%s", line)

    _log.info("%s", f"\n{sep}\n")


def _save_report(results: dict, filepath: str = "report.txt"):
    """Write the same report as :func:`_print_report` to ``filepath`` (utf-8)."""
    sep = "=" * 62
    m = results["metrics"]
    pl = results["per_label"]

    lines = []

    lines.append(f"\n{sep}")
    lines.append("  SAE EVALUATION – SUPERPOSITION DATASET")
    lines.append(sep)

    # --- Coverage ---
    found = m["labels_found"]
    total = m["labels_total"]
    pct = m["coverage"] * 100

    lines.append("\n[1] LABEL COVERAGE")
    lines.append(f"    Found:  {found} / {total}  ({pct:.1f}%)")
    lines.append(
        f"    Missed:  {total - found} Labels → this information is lost in the SAE!"
    )

    # --- Overall metrics ---
    lines.append("\n[2] CLASSIFICATION PERFORMANCE")
    lines.append(f"    {'Metric':<25} {'All Labels':>12} {'Found Only':>14}")
    lines.append(f"    {'-'*53}")

    lines.append(
        f"    {'F1 Micro':<25} "
        f"{m['f1_micro_all']:>12.4f} {m['f1_micro_found']:>14.4f}"
    )
    lines.append(
        f"    {'F1 Macro':<25} "
        f"{m['f1_macro_all']:>12.4f} {m['f1_macro_found']:>14.4f}"
    )
    lines.append(f"    {'F1 Samples':<25} " f"{m['f1_samples_all']:>12.4f} {'—':>14}")
    lines.append(
        f"    {'Hamming Loss':<25} "
        f"{m['hamming_loss_all']:>12.4f} {m['hamming_loss_found']:>14.4f}"
    )

    if "mAP_found" in m:
        lines.append(f"    {'mAP (found)':<25} {'—':>12} {m['mAP_found']:>14.4f}")

    if "roc_auc_found" in m:
        lines.append(
            f"    {'ROC-AUC (found)':<25} {'—':>12} {m['roc_auc_found']:>14.4f}"
        )

    # --- Worst Labels ---
    lines.append("\n[3] WORST 15 LABELS (potential information loss)")
    lines.append(f"    {'Label':<20} {'Status':<18} {'Corr':>6} {'F1':>6} {'Prev':>6}")
    lines.append(f"    {'-'*60}")

    for entry in pl[:15]:
        status_short = "✗ missing" if entry["status"] == "not found" else "~ weak"

        lines.append(
            f"    {entry['label']:<20} {status_short:<18} "
            f"{entry['corr']:>6.3f} {entry['f1']:>6.3f} "
            f"{entry['prevalence']:>6.3f}"
        )

    # --- Best Labels ---
    lines.append("\n[4] TOP 10 LABELS (well reconstructed)")
    lines.append(f"    {'Label':<20} {'Latent':>8} {'Corr':>6} {'F1':>6} {'AP':>6}")
    lines.append(f"    {'-'*52}")

    for entry in sorted(pl, key=lambda d: -d["f1"])[:10]:
        lines.append(
            f"    {entry['label']:<20} "
            f"{str(entry['latent_idx']):>8} "
            f"{entry['corr']:>6.3f} "
            f"{entry['f1']:>6.3f} "
            f"{entry['ap']:>6.3f}"
        )

    # --- Summary ---
    good = sum(1 for e in pl if e["f1"] >= 0.8)
    medium = sum(1 for e in pl if 0.5 <= e["f1"] < 0.8)
    bad = sum(1 for e in pl if e["f1"] < 0.5)

    lines.append("\n[5] SUMMARY")
    lines.append(f"    F1 ≥ 0.8  (good):     {good:>4} labels")
    lines.append(f"    F1 0.5–0.8 (okay):   {medium:>4} labels")
    lines.append(
        f"    F1 < 0.5  (poor):    {bad:>4} labels  ← information is lost here"
    )

    lines.extend(_dense_report_lines(results))
    lines.extend(_subset_report_lines(results))

    lines.append(f"\n{sep}\n")

    # Write file
    filepath = Path(filepath)
    filepath.write_text("\n".join(lines), encoding="utf-8")

    _log.info("%s", f"Report saved to: {filepath}")


def _save_results(results: dict, filepath: str = "report.txt"):
    """Pickle the full results dict (may include the latents) to ``filepath``."""
    with open(filepath, "wb") as f:
        pickle.dump(results, f)
    _log.info("%s", f"Results (including latents) saved to: {filepath}")


def plot_diagonalized_matching_matrix(
    corr_matrix: np.ndarray,
    label_names: list = None,
    label_to_latent: np.ndarray = None,
    save_path: str = None,
):
    """
    Plots bipartite correlation matrix ordered by best match on X axis so the primary
    matches form a clean diagonal.
    """
    import matplotlib.pyplot as plt

    n_latents, n_labels = corr_matrix.shape

    matched_indices = []
    unmatched_indices = []
    if label_to_latent is not None:
        for j in range(n_labels):
            lat_i = label_to_latent[j]
            if lat_i != -1:
                matched_indices.append((lat_i, j))
            else:
                unmatched_indices.append(j)
    else:
        for j in range(n_labels):
            lat_i = np.argmax(corr_matrix[:, j])
            matched_indices.append((lat_i, j))

    matched_indices.sort(key=lambda item: item[0])
    ordered_gt_indices = [j for _, j in matched_indices] + unmatched_indices
    ordered_lat_indices = [lat for lat, _ in matched_indices]
    used_lats = set(ordered_lat_indices)
    remaining_lats = [i for i in range(n_latents) if i not in used_lats]
    ordered_lat_indices = ordered_lat_indices + remaining_lats

    sub_gt = ordered_gt_indices[: min(100, len(ordered_gt_indices))]
    sub_lat = ordered_lat_indices[: min(100, len(ordered_lat_indices))]

    sub_matrix = corr_matrix[np.ix_(sub_lat, sub_gt)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Pearson Correlation / Match Strength")

    ax.set_title(
        "Diagonalized Bipartite Matching Matrix (Ground Truth vs SAE Features)"
    )
    ax.set_xlabel("Ground Truth Attributes (Ordered by Matched SAE Feature)")
    ax.set_ylabel("Matched SAE Latent Features")

    if label_names and len(sub_gt) <= 40:
        ax.set_xticks(range(len(sub_gt)))
        ax.set_xticklabels([label_names[j] for j in sub_gt], rotation=90, fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def log_evaluation_to_wandb(
    results: dict,
    run_name: str = None,
    group: str = None,
    project: str = "peal_sae_analysis",
):
    """
    Logs all ground-truth attribute matching metrics, per-attribute F1 scores, and
    the diagonalized bipartite matching matrix plot to Weights & Biases.
    """
    try:
        import wandb
    except ImportError:
        _log.info("%s", "[W&B] wandb package not found, skipping W&B logging.")
        return

    if wandb.run is None:
        try:
            wandb.init(project=project, name=run_name, group=group)
        except Exception as err:
            # No API key on the GPU nodes. Losing the W&B copy of the metrics is
            # not a reason to lose the run: the report, the pickle and the
            # TensorBoard logs are all written independently of this.
            _log.info(
                "%s", f"[W&B] Could not start a run ({err}); skipping W&B logging."
            )
            return

    metrics = results.get("metrics", {})
    per_label = results.get("per_label", [])

    log_dict = {}
    for key, val in metrics.items():
        if isinstance(val, (int, float, np.number)):
            log_dict[f"eval/{key}"] = float(val)

    f1_scores = []
    for item in per_label:
        attr_name = item["label"]
        f1_val = item["f1"]
        f1_scores.append(f1_val)
        log_dict[f"attributes_f1/{attr_name}"] = float(f1_val)

    if f1_scores:
        log_dict["eval/mean_attribute_f1"] = float(np.mean(f1_scores))

    if "corr_matrix" in results and "label_to_latent" in results:
        label_names = [item["label"] for item in per_label] if per_label else None
        fig = plot_diagonalized_matching_matrix(
            results["corr_matrix"],
            label_names,
            results["label_to_latent"],
        )
        log_dict["eval/matching_matrix_diagonalized"] = wandb.Image(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    wandb.log(log_dict)
    _log.info(
        "%s",
        f"[W&B] Successfully logged SAE evaluation metrics to W&B run '{wandb.run.name}'.",
    )


def log_evaluation_to_tensorboard(
    results: dict,
    log_dir: str,
    step: int = 0,
):
    """
    Logs all ground-truth attribute matching metrics and the diagonalized bipartite
    matching matrix plot to TensorBoard.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        _log.info(
            "%s",
            "[TensorBoard] SummaryWriter not available, skipping TensorBoard logging.",
        )
        return

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    metrics = results.get("metrics", {})
    per_label = results.get("per_label", [])

    for key, val in metrics.items():
        if isinstance(val, (int, float, np.number)) and not np.isnan(val):
            writer.add_scalar(f"eval/{key}", float(val), step)

    f1_scores = []
    for item in per_label:
        attr_name = item["label"]
        f1_val = item["f1"]
        f1_scores.append(f1_val)
        writer.add_scalar(f"attributes_f1/{attr_name}", float(f1_val), step)

    if f1_scores:
        writer.add_scalar("eval/mean_attribute_f1", float(np.mean(f1_scores)), step)

    for item in results.get("per_dense") or []:
        writer.add_scalar(f"dense_mae/{item['label']}", float(item["mae"]), step)
        writer.add_scalar(f"dense_r2/{item['label']}", float(item["r2"]), step)
        writer.add_scalar(f"dense_corr/{item['label']}", float(item["corr"]), step)

    if "corr_matrix" in results and "label_to_latent" in results:
        label_names = [item["label"] for item in per_label] if per_label else None
        fig = plot_diagonalized_matching_matrix(
            results["corr_matrix"],
            label_names,
            results.get("binary_label_to_latent", results["label_to_latent"]),
        )
        writer.add_figure("eval/matching_matrix_diagonalized", fig, step)
        import matplotlib.pyplot as plt

        plt.close(fig)

    writer.close()
    _log.info(
        "%s",
        f"[TensorBoard] Successfully logged SAE evaluation metrics to TensorBoard in '{log_dir}'.",
    )
