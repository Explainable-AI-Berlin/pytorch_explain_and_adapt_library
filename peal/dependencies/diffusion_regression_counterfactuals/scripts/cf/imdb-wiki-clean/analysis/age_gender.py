import os
import shutil
from matplotlib import pyplot as plt
import pandas as pd
import argparse


def plot_age_vs_gender(df: pd.DataFrame):
    plt.figure(figsize=(5, 5))

    df = df.rename(columns={"age": "Age", "gender": "Gender"})

    df_female = df[df["Gender"] == "F"]
    df_male = df[df["Gender"] == "M"]

    plt.hist(
        df_female["Age"],
        bins=range(1, 101),
        alpha=0.5,
        label="Age Female",
        color="orange",
        edgecolor="black",
    )
    plt.hist(
        df_male["Age"],
        bins=range(1, 101),
        alpha=0.5,
        label="Age Male",
        color="blue",
        edgecolor="black",
    )

    mean_age_female = df[df["Gender"] == "F"]["Age"].mean()
    mean_age_male = df[df["Gender"] == "M"]["Age"].mean()

    plt.axvline(
        mean_age_female,
        color="orange",
        linestyle="--",
        label=f"Age Female (Mean: {mean_age_female:.2f})",
    )
    plt.axvline(
        mean_age_male,
        color="blue",
        linestyle="--",
        label=f"Age Male (Mean: {mean_age_male:.2f})",
    )

    plt.xlabel("Age")
    plt.xticks(range(0, 101, 10))
    plt.ylabel("Frequency")
    plt.title("Age Distribution by Gender")
    plt.tight_layout()
    plt.legend()
    plt.savefig("age_vs_gender.png", dpi=200)


def sample_from_age_gender(
    root_folder: str, df: pd.DataFrame, step: int, num_samples: int
):
    samples = []
    random_state = 0

    for start_age in range(0, 101, step):
        end_age = start_age + step
        for gender in ["F", "M"]:
            subset = df[
                (df["age"] >= start_age)
                & (df["age"] < end_age)
                & (df["gender"] == gender)
            ]
            n_sample = len(subset) if len(subset) < num_samples else num_samples
            sampled_subset = subset.sample(n=n_sample, random_state=random_state)
            samples.append(sampled_subset)

            age_range_folder = f"samples/g={gender}_a={start_age:03d}-{end_age:03d}"
            save_samples(root_folder, sampled_subset, age_range_folder)

    sampled_df = pd.concat(samples)
    return sampled_df


def save_samples(root_folder: str, sampled_subset: pd.DataFrame, age_range_folder):
    os.makedirs(age_range_folder, exist_ok=True)
    image_folder = "data/imdb-clean-1024-cropped"
    for i, row in sampled_subset.iterrows():
        source_path = os.path.join(root_folder, image_folder, row["filename"])
        save_filename = f"a={row['age']}_i={i}.jpg"
        destination_path = os.path.join(age_range_folder, save_filename)
        shutil.copyfile(source_path, destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots and samples from the imdb-wiki-clean dataset."
    )
    parser.add_argument(
        "root_folder", type=str, help="Root folder containing the dataset"
    )
    parser.add_argument("--step", type=int, default=10, help="Step size for age range")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=25,
        help="Number of samples per age range and gender",
    )

    args = parser.parse_args()

    root_folder = args.root_folder
    step = args.step
    num_samples = args.num_samples

    # Load the dataset
    df: pd.DataFrame = pd.read_csv(
        os.path.join(root_folder, "csvs/imdb_train_new_1024.csv"), index_col=False
    )
    df_val: pd.DataFrame = pd.read_csv(
        os.path.join(root_folder, "csvs/imdb_valid_new_1024.csv"), index_col=False
    )
    df_test: pd.DataFrame = pd.read_csv(
        os.path.join(root_folder, "csvs/imdb_test_new_1024.csv"), index_col=False
    )

    df = pd.concat([df, df_val, df_test])

    # Plot
    plot_age_vs_gender(df)

    # Sample
    sample_from_age_gender(root_folder, df, step, num_samples)
