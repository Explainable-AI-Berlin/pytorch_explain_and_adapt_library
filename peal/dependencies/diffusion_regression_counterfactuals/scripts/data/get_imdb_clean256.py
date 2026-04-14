from matplotlib import pyplot as plt
import pandas as pd


if __name__ == "__main__":
    # Load the dataset
    df: pd.DataFrame = pd.read_csv(
        "/home/space/datasets/imdb-wiki-clean/imdb-clean/data/clean/imdb-clean-1024-cropped/imdb_valid_new_1024.csv"
    )
    df["x_crop_size"] = df["x_max"] - df["x_min"]
    df["y_crop_size"] = df["y_max"] - df["y_min"]

    is_atleast_256 = (df["x_crop_size"] >= 256) & (df["y_crop_size"] >= 256)

    df = df[is_atleast_256]
    df[["filename", "age"]].to_csv(
        "imdb_clean_atleast256.csv", index=False, header=False
    )

    bins = range(0, 101, 1)
    df["age"].hist(bins=bins, label="Predictions")

    # Add labels and legend
    plt.xlabel("Predicted Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"imdb-clean_atleast256_distribution.png", dpi=200)
