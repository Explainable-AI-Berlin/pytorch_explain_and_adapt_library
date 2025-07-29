import os
from typing import Union
import numpy as np
from diff_cf_ir.file_utils import rename_if_exists
from diff_cf_ir.image_folder_dataset import (
    ImageFolderDataset,
    PairedImageFolderDataset,
)
from diff_cf_ir.metrics import (
    FIDScorer,
    PeakSNR,
    LPIPS,
    SSIM,
    ReferenceScorer,
    AceFlipRateYoungOld,
    AceFVA,
    AceMNAC,
)
from diff_cf_ir.squares_dataset import background_color
import torch
import argparse
import os
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
import json

from tqdm import tqdm
import csv


def get_dataloader(dataset, batch_size) -> torch.utils.data.DataLoader:
    if len(dataset) == 0:
        raise ValueError("Empty dataset")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
    )

    return data_loader


def run_fid(args) -> dict[str, float]:
    fid_scorer: Union[None, FIDScorer] = None

    scores: list[tuple[str, float]] = []
    for fake_folder in tqdm(args.fake_folder, "FID: Fake folders"):
        paired_dataset = PairedImageFolderDataset(args.real_folder, fake_folder, 64)
        if fid_scorer is None:
            fid_scorer = FIDScorer(paired_dataset, batch_size=args.batch_size)

        fake_images = get_dataloader(paired_dataset, args.batch_size)
        for _, fake_imgs, _ in tqdm(
            fake_images, desc=f"FID: Fake images {fake_folder}"
        ):
            fid_scorer.update_fake_images(fake_images=fake_imgs)

        fid_score = fid_scorer.compute_score().item()
        fid_scorer.reset()
        # save_result(args, fake_folder, {f"FID_({args.limit})": fid_score})

        scores.append((fake_folder, fid_score))

    return dict(scores)


def run_background_mae(args) -> dict[str, dict[str, float]]:
    def compute_background_mae(paired_dataset):
        backgrounds_real = []
        backgrounds_fake = []
        for real_img, fake_img, img_name in tqdm(
            paired_dataset, desc="Background MAE: Real and Fake images"
        ):
            mask_path = os.path.join(
                os.path.dirname(args.real_folder), "masks", img_name
            )
            mask = Image.open(mask_path).convert("RGB")
            mask = transforms.ToTensor()(mask)

            bg_real = background_color(real_img, mask).item()
            bg_fake = background_color(fake_img, mask).item()

            backgrounds_real.append(bg_real)
            backgrounds_fake.append(bg_fake)

        backgrounds_real = np.array(backgrounds_real)
        backgrounds_fake = np.array(backgrounds_fake)

        mean = np.abs(backgrounds_real - backgrounds_fake).mean()
        std = np.abs(backgrounds_real - backgrounds_fake).std()

        return {
            "MEAN": mean,
            "STD": std,
        }

    scores: list[tuple[str, str, float]] = []
    for fake_folder in tqdm(args.fake_folder, "Background MAE: Fake folders"):
        paired_dataset = PairedImageFolderDataset(args.real_folder, fake_folder, 64)

        score_dict = {}
        scorer_name = "BackgroundMAE"

        background_mae = compute_background_mae(paired_dataset)
        score_dict[scorer_name] = background_mae

        scores.append((fake_folder, scorer_name + "_MEAN", background_mae["MEAN"]))
        scores.append((fake_folder, scorer_name + "_STD", background_mae["STD"]))

        save_ref_result(args, fake_folder, score_dict)

    scores_dict: dict[str, dict[str, float]] = {}
    for folder, metric, background_mae in scores:
        scores_dict.setdefault(folder, {})[metric] = background_mae

    return scores_dict


def base_name(path):
    abs_path = os.path.abspath(path)
    path_parts = abs_path.split(os.sep)
    return "_".join(path_parts[-2:])


def save_ref_result(args, fake_folder, score_dict):
    base_name_real = base_name(args.real_folder)
    base_name_fake = base_name(fake_folder)

    res_id = f"real={base_name_real}-fake={base_name_fake}"
    # result_folder = f"results_metrics/{res_id}"
    result_folder = args.output_folder
    os.makedirs(result_folder, exist_ok=True)

    out_file = f"{res_id}.json"
    out_path = os.path.join(result_folder, out_file)
    rename_if_exists(out_path)

    with open(out_path, "w") as f:
        json.dump(score_dict, f, indent=2)
    print("Metrics saved to", out_path)

    return result_folder


def save_summary(args, scores: dict[str, dict[str, float]]):
    def update_cf_results(cf_folder) -> dict[str, dict[str, float]]:
        """Load the results from the counterfactual folder to add to reference results"""
        cf_results = dict[str, dict[str, float]]()
        summary_path = os.path.abspath(
            os.path.join(cf_folder, "../results_summary.csv")
        )
        if os.path.exists(summary_path):
            print("Including counterfactual results found in", summary_path)
            with open(summary_path, mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metric = row["metric"]
                    mean = float(row["mean"])
                    scores[cf_folder][metric] = mean
        return cf_results

    base_name_real = base_name(args.real_folder)
    res_id = f"real={base_name_real}"

    summaries_folder = f"{args.output_folder}/summaries"
    os.makedirs(summaries_folder, exist_ok=True)

    out_file = f"{res_id}_summary.csv"
    out_path = os.path.join(summaries_folder, out_file)
    rename_if_exists(out_path)

    for fake_folder in args.fake_folder:
        # Add the counterfactual results to the reference results
        update_cf_results(fake_folder)

    with open(out_path, mode="w") as f:
        writer = csv.writer(f)
        included_metrics = list(next(iter(scores.values())).keys())
        header = ["path"] + included_metrics
        writer.writerow(header)

        # Write the scores
        for folder, metrics in scores.items():
            row = [folder] + [metrics[metric] for metric in included_metrics]
            writer.writerow(row)
    print("Summary saved to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Calculate image metrics between two folders of images."
            "When calculating reference metrics, the images in each folder should have the same names."
        )
    )
    parser.add_argument(
        "--real_folder", type=str, required=True, help="Folder with real images"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Limit number of real images for FID calculation.",
    )
    parser.add_argument(
        "--fake_folder",
        type=str,
        required=True,
        help="Folder with fake images",
        action="append",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size during processing.",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Folder to save the results",
    )

    args = parser.parse_args()

    # Check all dirs exist
    for folder in [args.real_folder] + args.fake_folder:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)  # Globally disable grads

    background_mae = run_background_mae(args)
    scores = background_mae
    # fid_scores = run_fid(args)
    # for folder, fid_score in fid_scores.items():
    #     scores[folder]["FID"] = fid_score

    save_summary(args, scores)
    print("Done!")
