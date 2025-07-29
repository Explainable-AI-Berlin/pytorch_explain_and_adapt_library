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
import torch
import argparse
import os
import matplotlib.pyplot as plt
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
        paired_dataset = PairedImageFolderDataset(
            args.real_folder, fake_folder, args.size
        )
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


def init_scoring_methods(args):
    scoring_methods: list[ReferenceScorer] = [
        # PeakSNR(),
        LPIPS(),
        # SSIM(),
    ]

    # Check if we need to include optional reference metrics
    args_dict = vars(args)

    ace_fr_key = "ace_fr_classifier_path"
    if args_dict[ace_fr_key]:
        print("Including ACE Flip Rate metric")
        scoring_methods.append(AceFlipRateYoungOld(args_dict[ace_fr_key]))

    ace_fva_key = "ace_fva_classifier_path"
    if args_dict[ace_fva_key]:
        print("Including ACE FVA metric")
        scoring_methods.append(AceFVA(args_dict[ace_fva_key]))

    ace_mnac_key = "ace_mnac_classifier_path"
    if args_dict[ace_mnac_key]:
        print("Including ACE MNAC metric")
        scoring_methods.append(AceMNAC(args_dict[ace_mnac_key]))
    return scoring_methods


def run_reference(args) -> dict[str, dict[str, float]]:
    # TODO: if cf_results.json or whatever is in the folder also write it into summary file
    scoring_methods: list[ReferenceScorer] = init_scoring_methods(args)

    scores: list[tuple[str, str, float]] = []
    for fake_folder in tqdm(args.fake_folder, "Refs: Fake folders"):
        paired_dataset = PairedImageFolderDataset(
            args.real_folder, fake_folder, args.size
        )

        score_dict = {}

        for scorer in scoring_methods:
            scorer_name = type(scorer).__name__
            dataloader = get_dataloader(paired_dataset, args.batch_size)
            score = run_metric(dataloader, scorer)
            score_dict[scorer_name] = score

            scores.append((fake_folder, scorer_name + "_MEAN", score["MEAN"]))
            scores.append((fake_folder, scorer_name + "_STD", score["STD"]))

        save_ref_result(args, fake_folder, score_dict)

        # print("Creating Diff Plot")
        # plot_differences(paired_dataset, result_folder)
    scores_dict: dict[str, dict[str, float]] = {}
    for folder, metric, score in scores:
        scores_dict.setdefault(folder, {})[metric] = score

    return scores_dict


def run_metric(
    paired_dataloader: torch.utils.data.DataLoader, scorer: ReferenceScorer
) -> dict[str, float]:
    names: list[str] = []
    scores: list[torch.Tensor] = []
    for real_imgs, fake_imgs, names_batch in tqdm(
        paired_dataloader, desc=f"Ref {type(scorer).__name__}"
    ):
        real_imgs_pt = real_imgs.to(device)
        fake_imgs_pt = fake_imgs.to(device)

        batch_scores = scorer.compute_counterfactual_score(real_imgs_pt, fake_imgs_pt)

        # Collect scores for mean and std later, can stay in batch
        scores.append(batch_scores)
        names.extend(names_batch)

    scores_tensor = torch.cat(scores)
    result_dict: dict = {name: s for name, s in zip(names, scores_tensor.tolist())}
    mean_score = scores_tensor.mean(dim=0).tolist()
    std_score = scores_tensor.std(dim=0).tolist()
    result_dict["MEAN"] = mean_score
    result_dict["STD"] = std_score
    return result_dict


def plot_differences(paired_dataset, result_folder):
    dataset_length = min(len(paired_dataset), 10)  # Limit to 10 images
    fig, axes = plt.subplots(dataset_length, 3, figsize=(3 * 3, 3 * dataset_length))
    fig.subplots_adjust(top=0.9)

    for i, (real_img, fake_img, img_name) in enumerate(paired_dataset):
        real_img_np = np.array(real_img.permute(1, 2, 0))
        fake_img_np = np.array(fake_img.permute(1, 2, 0))
        diff_img_np = np.mean(fake_img_np - real_img_np, axis=2)

        axes[i, 0].imshow(real_img_np)
        axes[i, 0].set_title(f"Real Image: {img_name}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(fake_img_np)
        axes[i, 1].set_title(f"Fake Image: {img_name}")
        axes[i, 1].axis("off")

        im = axes[i, 2].imshow(diff_img_np, cmap="bwr", vmin=-1, vmax=1)
        axes[i, 2].set_title(f"Difference Total: {np.sum(diff_img_np):.2f}")
        axes[i, 2].axis("off")

        if i == dataset_length - 1:
            break

    cbar_ax = fig.add_axes([0.15, 0.95, 0.75, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    plt.tight_layout()
    out_path = os.path.join(result_folder, "reference_differences.png")
    plt.savefig(out_path, dpi=200)
    plt.close("all")


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
        "--size", type=int, default=256, help="Images are resized to this size"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size during processing.",
    )
    parser.add_argument(
        "--ace_fr_classifier_path",
        type=str,
        help="Path to the ACE classifier for the ACE Flip Rate metric (young vs old). Only used for reference metrics.",
    )
    parser.add_argument(
        "--ace_fva_classifier_path",
        type=str,
        help="Path to the ACE classifier for the FVA metric (face similarity). Only used for reference metrics.",
    )
    parser.add_argument(
        "--ace_mnac_classifier_path",
        type=str,
        help="Path to the ACE classifier for the MNAC metric. Only used for reference metrics.",
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

    fid_scores = run_fid(args)
    scores = run_reference(args)

    for folder, fid_score in fid_scores.items():
        scores[folder]["FID"] = fid_score

    save_summary(args, scores)
    print("Done!")
