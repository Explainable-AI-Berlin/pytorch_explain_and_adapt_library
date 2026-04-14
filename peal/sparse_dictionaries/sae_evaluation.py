import torch
import numpy as np
from sklearn.metrics import f1_score


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
            batch = X_centered[i:i + batch_size].to(device)
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
