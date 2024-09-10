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
        3,
        figsize=(15, 6),
        layout="constrained",
        gridspec_kw={"hspace": 0.4, "wspace": 0.5},
    )  # Increase horizontal spacing

    # Define color and label mappings
    colors = ["red", "blue", "green", "purple", "orange", "black"]
    labels = ["Human", "Mask", "Oracle", "Random", "Positive", "Negative"]
    inner_labels = ["Empirical Accuracy", "Worst Group Accuracy", "Feedback Accuracy"]

    # Create a dictionary to map colors to labels for legend clarity
    color_label_map = dict(zip(colors, labels))

    # Plot for each inner dimension
    for i in range(3):
        for j in range(6):
            if j < 3:
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
        handles, labels=labels, loc="upper left", bbox_to_anchor=(0.05, 1.0), ncol=6
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
            [1.0, 0.87],
            [0.263, 0.785],
            [0.4, 0.917],
        ],
        [
            [1.0, 0.85],
            [0.263, 0.8],
            [0.2, 0.85],
        ],
        [
            [1.0, 0.9],
            [0.263, 0.77],
            [0.3, 0.87],
        ],
        [
            [1.0, 0.67],
            [0.263, 0.46],
            [0.5, 0.5],
        ],
        [
            [1.0, 1.0],
            [0.263, 0.263],
            [1.0, 1.0],
        ],
        [
            [1.0, 0.7],
            [0.263, 0.45],
            [0.0, 0.0],
        ],
    ]
)

# Plot the test data
plot_accuracy_line(test_data)
