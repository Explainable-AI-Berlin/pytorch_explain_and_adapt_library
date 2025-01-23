import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def create_bar_diagram(data, title, ax, base_colors, bar_width=0.6, group_offset=1.5):
    """
    Creates a grouped and stacked bar diagram for each category with bars touching within groups.
    Args:
        data: A 2D numpy array with shape (n_methods, 2), where each row contains [low, high] values.
        title: The title of the bar diagram.
        ax: The matplotlib axes object.
        base_colors: A list of base colors for each method. Low and high bars will be shades of these colors.
        bar_width: The width of each bar.
        group_offset: The offset to add between groups for separation.
    """
    n_methods = data.shape[0]
    x = np.arange(n_methods)  # Positions for bars within a group

    # Generate lighter colors for high bars
    high_colors = [
        tuple(min(1.0, c + 0.4) for c in mcolors.to_rgb(color)) for color in base_colors
    ]

    # Plot stacked bars for each method
    rects_low = ax.bar(x, data[:, 0], bar_width, color=base_colors, label='Low')
    rects_high = ax.bar(x, data[:, 1] - data[:, 0], bar_width, bottom=data[:, 0], color=high_colors, label='High')

    # Add labels to bars
    ax.bar_label(rects_low, fmt="%.2f", label_type="center", fontsize=12, padding=1)
    ax.bar_label(rects_high, fmt="%.2f", label_type="center", fontsize=12, padding=1)

    # Set y-axis limits to [0, 1]
    ax.set_ylim(0, 1)

    # Set x-ticks for methods
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_yticks([])

    # Hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add title below the plot
    ax.text(0.5, -0.1, title, transform=ax.transAxes, ha="center", va="top", fontsize=12)

if __name__ == "__main__":
    data = np.array([
        [[0.97, 0.99], [0.91, 1.0], [0.78, 1.0], [0.75, 0.81]],
        [[0.41, 0.51], [0.59, 0.92], [0.39, 0.75], [0.89, 0.91]],
        [[0.2, 0.59], [0.29, 0.74], [0.3, 0.67], [0.78, 0.94]],
        [[0.03, 0.04], [0.21, 0.35], [0.06, 0.26], [0.93, 0.94]],
    ])
    labels = ["ACE", "DiME", "FastDiME", "PDC (ours)"]
    titles = ["Smiling", "Blond_Hair", "Waterbirds", "Square"]
    base_colors = ["#1f44b4", "#1f77b4", "#1faab4", "#d62728"]

    fig, axs = plt.subplots(1, len(titles), figsize=(16, 6), sharey=True)

    for i, ax in enumerate(axs):
        create_bar_diagram(data[i], titles[i], ax=ax, base_colors=base_colors, bar_width=0.8)

    # Add a global legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none", label=label)
        for color, label in zip(base_colors, labels)
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(labels),
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
