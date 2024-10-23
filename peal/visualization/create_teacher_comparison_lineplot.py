# Hi, I have a 6 x 3 x N numpy array containing values between 0 and 1. The out 6 dimensions are called "Human", "Mask", "Oracle", "Random", "Positive", "Negative". The inner 3 dimensions are called "Empirical Accuracy", "Worst Group Accuracy" and "Feedback Accuracy". Please write me a Python function that creates a line plot for all the 3 inner dimensions each containing one line for each of the 6 outer dimensions consisting of N points. Visualize the 3 diagrams next to each other. Write a legend for the used colors of the lines over the plot and a legend for the inner dimension under each diagram. Directly give a 6 x 3 x 2 array of random floats between 0 and 1 for testing purposes.
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy_line(data):
    """
    Plots a line chart of accuracy metrics for different models and groups.

    Args:
        data: A 6x3xN numpy array containing accuracy metrics.

    Returns:
        None
    """

    # Create a figure with adjusted spacing
    fig, axs = plt.subplots(
        1,
        4,
        figsize=(15, 6),
        layout="constrained",
        gridspec_kw={"hspace": 0.4, "wspace": 0.5},
    )  # Increase horizontal spacing

    # Define color and label mappings
    colors = ["red", "blue", "grey", "green", "purple", "black"]
    labels = ["Human", "Mask", "Oracle", "Random", "Positive", "Negative"]
    inner_labels = ["Empirical Accuracy", "Worst Group Accuracy", "Group Average Accuracy", "Feedback Accuracy"]

    # Create a dictionary to map colors to labels for legend clarity
    color_label_map = dict(zip(colors, labels))

    # Plot for each inner dimension
    for i in range(len(inner_labels)):
        for j in range(len(labels)):
            if j < len(labels) - 3:
                linestyle = "solid"

            else:
                linestyle = "dashed"

            axs[i].plot(data[j, i, :], label=labels[j], color=colors[j], linestyle=linestyle)

        # Set labels and title
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel(f"{inner_labels[i]}")
        # Set integer x-axis ticks
        axs[i].set_xticks(np.arange(data.shape[2]))

    # Add overall legend outside the subplots with clear label mapping
    handles, labels = zip(
        *[(plt.Line2D([], [], color=c), color_label_map[c]) for c in colors]
    )
    fig.legend(
        handles, labels=labels, loc="upper left", bbox_to_anchor=(0.05, 1.0), ncol=len(labels)
    )  # Adjust location

    # Add space above the subplots
    plt.subplots_adjust(top=0.9)  # Adjust the value as needed

    # Show the plot
    plt.show()


# Generate random data for testing
# test_data = np.random.rand(6, 3, 2)
test_data = np.array(
    [
        [
            [1.0, 0.98],
            [0.675, 0.774],
            [0.835, 0.873],
            [0.405, 0.7],
        ],
        [
            [1.0, 0.98],
            [0.675, 0.799],
            [0.835, 0.862],
            [0.237, 0.425],
        ],
        [
            [1.0, 0.98],
            [0.675, 0.796],
            [0.835, 0.876],
            [0.679, 0.893],
        ],
        [
            [1.0, 0.93],
            [0.675, 0.697],
            [0.835, 0.822],
            [0.0, 0.0],
        ],
        [
            [1.0, 1.0],
            [0.675, 0.647],
            [0.835, 0.828],
            [1.0, 1.0],
        ],
        [
            [1.0, 0.96],
            [0.675, 0.708],
            [0.835, 0.838],
            [0.5, 0.5],
        ],
    ]
)

# Plot the test data
plot_accuracy_line(test_data)
