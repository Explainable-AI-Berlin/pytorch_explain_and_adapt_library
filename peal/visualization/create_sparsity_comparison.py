import matplotlib.pyplot as plt
import numpy as np


def create_bar_diagram(data, title, ax, colors, bar_width=0.9):
    """
    Creates a bar diagram with the given data and title.
    Args:
        data: A 1D numpy array containing the data for the bar diagram.
        title: The title of the bar diagram.
        ax: The matplotlib axes object.
        colors: A list of colors for the bars.
    """
    x = np.arange(len(data))
    rects = ax.bar(x, data, bar_width, color=colors)
    ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12, padding=3)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_xticklabels([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add title below the plot
    ax.text(
        0.5, -0.1, title, transform=ax.transAxes, ha="center", va="top", fontsize=12
    )


if __name__ == "__main__":
    # Example usage:
    data = np.array(
        [
            [0.79, 0.73, 0.71, 0.76],
            [0.76, 0.72, 0.72, 0.79],
            [0.76, 0.27, 0.32, 0.44],
            [1.0, 0.85, 0.85, 0.87],
        ]
    )
    labels = ["ACE", "DiME", "FastDiME", "PDC (ours)"]
    titles = ["Smiling", "Blond_Hair", "Waterbirds", "Square"]
    colors = ["#1f44b4", "#1f77b4", "#1faab4", "#d62728"]

    fig, axs = plt.subplots(
        1, data.shape[0], figsize=(25, 4), sharey=True
    )  # Reduced height from 6 to 4
    # plt.subplots_adjust(wspace=0.3, bottom=0.4)  # Adjusted bottom margin

    for i in range(data.shape[0]):
        create_bar_diagram(data[i], titles[i], ax=axs[i], colors=colors, bar_width=0.9)

    # Add a global color legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
        for color in colors
    ]
    fig.legend(
        legend_elements,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.9),  # Positioned legend below the plot
        ncol=4,
        fontsize=12,  # Increased font size
        # title_fontsize=14,  # Increased title font size
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust top margin for overall layout
    plt.show()
