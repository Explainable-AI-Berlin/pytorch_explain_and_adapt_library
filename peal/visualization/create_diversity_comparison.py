import matplotlib.pyplot as plt
import numpy as np


def create_bar_diagram(data, title, ax, colors):
    """
    Creates a bar diagram with the given data and title.
    Args:
        data: A 1D numpy array containing the data for the bar diagram.
        title: The title of the bar diagram.
        ax: The matplotlib axes object.
        colors: A list of colors for the bars.
    """
    x = np.arange(len(data))
    width = 1.0  # Make bars touch each other
    rects = ax.bar(x, data, width, color=colors)
    ax.bar_label(rects, fmt="%.1f", label_type="edge", fontsize=6, padding=1)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add title below the plot
    ax.text(0.5, -0.1, title, transform=ax.transAxes, ha="center", va="top", fontsize=8)

if __name__ == "__main__":
    # Example usage:
    data = np.array(
        [
            [-0.0, -0.0, -0.0, 1.01],
            [-0.0, -0.0, -0.0, 1.19],
        ]
    )
    labels = ["DiME", "ACE", "FastDiME", "PDC"]
    titles = ["Waterbirds", "Square"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axs = plt.subplots(1, data.shape[0], figsize=(25, 4))  # Reduced height from 6 to 4
    plt.subplots_adjust(wspace=0.3, bottom=0.2)  # Adjusted bottom margin

    for i in range(data.shape[0]):
        create_bar_diagram(data[i], titles[i], ax=axs[i], colors=colors)

    # Add a global color legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none") for color in colors
    ]
    fig.legend(
        legend_elements,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        fontsize=8,
        title_fontsize=10,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust top margin for legend
    plt.show()
