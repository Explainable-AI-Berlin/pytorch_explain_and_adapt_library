import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from diff_cf_ir.image_folder_dataset import ImageFolderDataset
from diff_cf_ir.models import load_model
from diff_cf_ir.file_utils import assert_paths_exist
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

MAX_AGE = 100


def predict(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[list[str], list[float]]:
    model.to(device)
    predictions = []
    filenames = []

    with torch.no_grad():
        for paths, inputs in tqdm(dataloader, desc="Predicting Batches"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs * MAX_AGE
            predictions.extend(outputs.cpu().numpy().reshape(-1))
            filenames.extend(paths)

    return filenames, predictions


def main(args: argparse.Namespace) -> None:
    dataset = ImageFolderDataset(args.folder_path, size=args.size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    assert len(dataloader) > 0, "No images found in the folder"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.predictor_path)

    filenames, predictions = predict(model, dataloader, device)

    results = pd.DataFrame({"file": filenames, "prediction": predictions})

    output_name = os.path.basename(os.path.dirname(args.folder_path))
    print(f"Saving results to {output_name}.csv")
    results.to_csv(f"{output_name}.csv", index=False)

    print(f"Saving distribution to {output_name}_distribution.png")

    bins = range(0, 101, 1)
    results["prediction"].hist(bins=bins, label="Predictions")

    # Add labels and legend
    plt.xlabel("Predicted Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{output_name}_distribution.png", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict images in a folder using a regression model."
    )
    parser.add_argument("folder_path", type=str, help="Path to the image folder.")
    parser.add_argument("--size", type=int, help="Size of the images", required=True)
    parser.add_argument(
        "--predictor_path",
        type=str,
        help="Path to the regression model checkpoint.",
        required=True,
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for prediction.", default=512
    )

    args = parser.parse_args()
    assert_paths_exist([args.folder_path, args.predictor_path])

    main(args)
