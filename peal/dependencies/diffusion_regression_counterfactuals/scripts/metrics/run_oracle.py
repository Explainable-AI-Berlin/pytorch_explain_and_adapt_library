import sys
import os

from scripts.metrics.run_metrics import ImageFolderDataset


import numpy as np
from diff_cf_ir.models import load_resnet
from diff_cf_ir.file_utils import rename_if_exists
import torch
import argparse
import os
from PIL import Image
from torchvision import transforms
import json

from tqdm import tqdm


def get_dataloader(dataset, batch_size) -> torch.Tensor:
    if len(dataset) == 0:
        raise ValueError("Empty dataset")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
    )

    return data_loader


def run_oracle(args, model: torch.nn.Module, oracle: torch.nn.Module):
    l1_dist = torch.nn.L1Loss(reduction="none")

    with tqdm(args.folder, desc="Processing Folders") as pbar_folder:
        for folder in pbar_folder:
            result_dict = {}
            pbar_folder.set_postfix_str(folder)

            image_folder = ImageFolderDataset(folder, args.size)
            data_loader = get_dataloader(image_folder, args.batch_size)
            for img_path, img in tqdm(data_loader, desc="Processing Image Batches"):
                img = img.to(device)

                model_out = model(img)
                oracle_out = oracle(img)
                dist = l1_dist(model_out, oracle_out)

                for i, path in enumerate(img_path):
                    result_dict[path] = [
                        model_out[i].item(),
                        oracle_out[i].item(),
                        dist[i].item(),
                    ]

            # Calculate statistcs of all distances
            distances = [v[2] for v in result_dict.values()]
            mean = np.mean(distances)
            std_dist = np.std(distances)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            result_dict["MEAN"] = mean
            result_dict["STD"] = std_dist
            result_dict["MIN"] = min_dist
            result_dict["MAX"] = max_dist

            save_result(args, folder, result_dict)
            pbar_folder.update()


def base_name(path):
    return os.path.basename(os.path.abspath(path))


def save_result(args, folder, score_dict):
    folder_name = base_name(folder)
    result_folder = f"results_oracle/{folder_name}"
    os.makedirs(result_folder, exist_ok=True)

    out_file = f"{folder_name}.json"
    out_path = os.path.join(result_folder, out_file)
    rename_if_exists(out_path)

    print("Saving results to", out_path, "\n")

    with open(out_path, "w") as f:
        json.dump(score_dict, f, indent=2)

    print(json.dumps(score_dict, indent=2))
    return result_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Calculates the prediction and oracle prediction for a folder of images."
            "Two models need to be provided."
        )
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder with images. Multiple can be specified.",
        action="append",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="Images are resized to this size"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of the model (in this repo) to load. Used for loading both models.",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The main predictor used for the counterfactuals.",
        required=True,
    )
    parser.add_argument(
        "--oracle", type=str, help="The oracle model for comparison", required=True
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size during processing"
    )

    args = parser.parse_args()
    # Check all dirs exist
    for folder in args.folder:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # Load models
    model = load_resnet(args.model_type, args.model)
    oracle = load_resnet(args.model_type, args.oracle)

    run_oracle(args, model, oracle)
    print("Done!")
