import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot_svd_correlations(data, ground_truth, feature_names=None):
    """
    Args:
        data: torch.Tensor of shape (n_samples, n_features)
        ground_truth: torch.Tensor of shape (n_samples, N_sources_of_variation)
                      (Values should be between 0 and 1)
        feature_names: List of strings (optional) names for your ground truths
    """

    # 1. Pre-processing: Center the data (Standard for PCA/SVD analysis)
    # SVD on centered data is equivalent to PCA
    data_mean = torch.mean(data, dim=0)
    data_centered = data - data_mean

    # 2. Perform SVD
    # torch.linalg.svd is preferred over torch.svd in newer PyTorch versions
    # U: Unitary matrix, S: Singular values, Vh: Conjugate transpose of V (Vt)
    U, S, Vh = torch.linalg.svd(data_centered, full_matrices=False)

    # Transpose Vh to get V (columns are principal components)
    V = Vh.T

    # 3. Get Component Values (Projections/Scores)
    # We project the data onto the singular vectors.
    # This gives us the coordinate of each data point in the "SVD space".
    # We only care about the first N components corresponding to the N ground truths
    num_vars = ground_truth.shape[1]

    # Select top N components. Shape: (n_samples, num_vars)
    component_scores = data_centered @ V[:, :num_vars]

    # Convert to numpy for easier plotting/math
    scores_np = component_scores.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()

    # 4. Correlation Analysis
    # We calculate a correlation matrix between every Component and every Ground Truth
    n_components = scores_np.shape[1]
    n_gt = gt_np.shape[1]

    # Matrix to store correlation coefficients
    corr_matrix = np.zeros((n_gt, n_components))

    for i in range(n_gt): # For each ground truth
        for j in range(n_components): # For each SVD component
            # Calculate Pearson correlation
            corr = np.corrcoef(gt_np[:, i], scores_np[:, j])[0, 1]
            corr_matrix[i, j] = corr

    # 5. Plotting
    fig, axes = plt.subplots(1, n_gt, figsize=(5 * n_gt, 5), constrained_layout=True)
    if n_gt == 1: axes = [axes] # Handle single plot case

    print(f"{'Ground Truth':<20} | {'Best Component':<15} | {'Correlation':<10}")
    print("-" * 55)

    for i in range(n_gt):
        gt_name = feature_names[i] if feature_names else f"GT Annotation {i+1}"

        # Find the component with the highest MAGNITUDE correlation
        # (SVD sign is arbitrary, so -0.9 is just as good as 0.9)
        best_comp_idx = np.argmax(np.abs(corr_matrix[i, :]))
        best_corr = corr_matrix[i, best_comp_idx]

        # Print stats
        print(f"{gt_name:<20} | Component {best_comp_idx+1:<14} | {best_corr:.4f}")

        # Scatter Plot
        ax = axes[i]

        # Optional: Add a regression line for visual aid
        sns.regplot(x=gt_np[:, i], y=scores_np[:, best_comp_idx],
                    ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

        ax.set_title(f"Correlation: {best_corr:.2f}")
        ax.set_xlabel(f"{gt_name} (0-1)")
        ax.set_ylabel(f"SVD Component {best_comp_idx+1} Score")
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Analysis: Ground Truth vs. Strongest SVD Components", fontsize=16)
    plt.show()

# ==========================================
# MOCK DATA GENERATION (To test the script)
# ==========================================
if __name__ == "__main__":
    # Create synthetic data with N=3 sources of variation
    N_SAMPLES = 500
    N_FEATURES = 50
    N_SOURCES = 3

    # 1. Generate Ground Truths (latent factors) between 0 and 1
    # Example: Size, Opacity, Rotation
    gt = torch.rand(N_SAMPLES, N_SOURCES)

    # 2. Create random projection matrix (mixing matrix)
    # This simulates how latent factors manifest in high-dim features
    mixing_matrix = torch.randn(N_SOURCES, N_FEATURES)

    # 3. Create Data = GroundTruth * Mixing + Noise
    data_clean = gt @ mixing_matrix
    noise = torch.randn(N_SAMPLES, N_FEATURES) * 0.1
    data = data_clean + noise

    # Run the analysis
    analyze_and_plot_svd_correlations(
        data,
        gt,
        feature_names=["Size", "Opacity", "Rotation"]
    )