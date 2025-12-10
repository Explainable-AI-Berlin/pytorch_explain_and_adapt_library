import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_component_ground_truth_correlations(filename, components, ground_truth_attributes):
    """
    Calculates the correlation between extracted components and ground truth attributes
    and saves a heatmap visualization to the specified filename.

    Args:
        filename (str): Path to save the image (e.g., 'correlation_matrix.png')
        components (array-like): Shape (n_samples, n_components). The extracted component values.
        ground_truth_attributes (array-like): Shape (n_samples, n_attributes). The ground truth metadata.
    """

    # 1. Convert Inputs to Numpy if they are PyTorch Tensors
    if torch.is_tensor(components):
        components = components.detach().cpu().numpy()
    if torch.is_tensor(ground_truth_attributes):
        ground_truth_attributes = ground_truth_attributes.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    components = np.array(components)
    ground_truth_attributes = np.array(ground_truth_attributes)

    n_features = ground_truth_attributes.shape[1]
    n_components = components.shape[1]

    # 2. Compute Correlation Matrix
    # Matrix shape: (n_ground_truths, n_components)
    corr_matrix = np.zeros((n_features, n_components))

    for i in range(n_features):
        for j in range(n_components):
            # Calculate Pearson correlation
            # Handle edge case where standard deviation is 0 (constant value)
            if np.std(ground_truth_attributes[:, i]) == 0 or np.std(components[:, j]) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(ground_truth_attributes[:, i], components[:, j])[0, 1]
            corr_matrix[i, j] = corr

    # 3. Plotting with Matplotlib (No Seaborn)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the heatmap
    # cmap='coolwarm' or 'bwr' gives a nice Blue-White-Red divergence
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Add Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

    # Show all ticks and label them
    ax.set_xticks(np.arange(n_components))
    ax.set_yticks(np.arange(n_features))

    ax.set_xticklabels([f"Comp {j+1}" for j in range(n_components)])
    ax.set_yticklabels([f"Attr {i+1}" for i in range(n_features)])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # This manually recreates the 'annot=True' feature of seaborn
    for i in range(n_features):
        for j in range(n_components):
            val = corr_matrix[i, j]
            # Choose text color based on background intensity for readability
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center", color=text_color, fontsize=9)

    ax.set_title("Correlation: Components vs Ground Truth Attributes")
    fig.tight_layout()

    # 4. Save and Close
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Correlation plot saved to {filename}")

# --- Example Usage ---
if __name__ == "__main__":
    # Mock data
    N_SAMPLES = 100
    N_COMPS = 5
    N_ATTRS = 3

    comps = np.random.randn(N_SAMPLES, N_COMPS)
    attrs = np.random.rand(N_SAMPLES, N_ATTRS)

    # Force a correlation for demonstration
    comps[:, 0] = attrs[:, 0] * 0.9 + np.random.normal(0, 0.1, N_SAMPLES)

    plot_component_ground_truth_correlations("my_analysis.png", comps, attrs)