import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(
        description="Compare two images and generate a heatmap of differences."
    )

    # Add arguments
    parser.add_argument("base", type=str, help="Path to the base image")
    parser.add_argument("comp", type=str, help="Path to the comparison image")

    # Parse arguments
    args = parser.parse_args()

    # Load images
    image1 = Image.open(args.comp)
    image2 = Image.open(args.base)

    # Convert images to numpy arrays
    data1 = np.array(image1, dtype=np.float32) / 255.0
    data2 = np.array(image2, dtype=np.float32) / 255.0

    # Calculate the absolute difference between the two images
    difference = data1 - data2

    print("Mean Diff for channels: ", np.mean(difference, axis=(0, 1)))

    # Create subplots for each color channel
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Create heatmap of differences for each color channel
    for i, color in enumerate(["Red", "Green", "Blue"]):
        im = axs[i].imshow(
            difference[:, :, i], cmap="bwr", interpolation="nearest", vmin=-1, vmax=1
        )
        axs[i].set_title(f"Diff for {color}")
        axs[i].axis("off")  # Hide axes
        fig.colorbar(
            im,
            ax=axs[i],
        )

    plt.tight_layout()
    plt.savefig("heatmap.png")
