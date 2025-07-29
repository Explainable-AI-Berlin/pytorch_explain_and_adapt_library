from dataclasses import dataclass
import json
import os

import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from diff_cf_ir.file_utils import save_image
from diff_cf_ir.metrics import get_regr_confidence
import numpy as np
import csv


@dataclass
class CFResult:
    image_path: str
    x: torch.Tensor
    x_reconstructed: torch.Tensor
    x_prime: torch.Tensor
    y_target: float
    y_initial_pred: float
    y_final_pred: float
    success: bool
    steps: int

    # Optional fields for if we know true latents (squares) or oracle values for other methods
    y_initial_true: float = -1
    # y_final_latent: float = -1
    # success_latent = False
    # y_pred_oracle: float = -1
    # success_oracle: bool = False

    def __post_init__(self):
        self.y_initial_confidence: float = get_regr_confidence(
            torch.tensor(self.y_initial_pred), torch.tensor(self.y_target)
        ).item()
        self.y_final_confidence: float = get_regr_confidence(
            torch.tensor(self.y_final_pred), torch.tensor(self.y_target)
        ).item()
        self.confidence_reduction: float = (
            self.y_initial_confidence - self.y_final_confidence
        )
        self.image_name = os.path.basename(self.image_path)
        self.image_name_base = os.path.splitext(self.image_name)[0]

    def result_dict(self) -> dict:
        result = vars(self).copy()

        # Dont save the tensors
        del result["x"]
        del result["x_reconstructed"]
        del result["x_prime"]

        result = dict(sorted(result.items()))
        return result

    def save_cf(self, folder: str):
        os.makedirs(folder, exist_ok=True)

        cf_dir = os.path.join(folder, "cf")
        os.makedirs(cf_dir, exist_ok=True)

        save_image(self.x_prime, os.path.join(cf_dir, self.image_name))

    def update_y_true_initial(self, y_true_initial: float):
        self.y_initial_true = y_true_initial
        self.y_initial_true_mae = abs(self.y_initial_true - self.y_initial_pred)

    def update_y_final_latent(self, y_final_latent: float, confidence_threshold: float):
        self.y_final_latent = y_final_latent
        self.y_final_latent_mae = abs(y_final_latent - self.y_target)
        self.success_latent = self.y_final_latent_mae <= confidence_threshold

    def update_y_oracle_initial(self, y_oracle_initial: float):
        self.y_initial_pred_oracle = y_oracle_initial
        self.y_initial_pred_oracle_mae = abs(y_oracle_initial - self.y_initial_pred)
        self.y_initial_confidence_oracle = get_regr_confidence(
            torch.tensor(y_oracle_initial), torch.tensor(self.y_target)
        ).item()

    def update_y_oracle_final(self, y_oracle_final: float, confidence_threshold: float):
        self.y_final_pred_oracle = y_oracle_final
        self.y_final_pred_oracle_mae = abs(y_oracle_final - self.y_final_pred)
        self.y_final_confidence_oracle = get_regr_confidence(
            torch.tensor(y_oracle_final), torch.tensor(self.y_target)
        ).item()
        self.success_oracle = self.y_final_confidence_oracle <= confidence_threshold
        self.confidence_reduction_oracle: float = (
            self.y_initial_confidence_oracle - self.y_final_confidence_oracle
        )

    def update_fva(self, fva_score: float, fs_score: float):
        self.x_prime_fva = fva_score
        self.x_prime_fs = fs_score


def save_cf_results(
    cf_results: list[CFResult],
    result_dir: str,
):
    def calculate_success_rates(
        cf_results: list[CFResult],
    ) -> dict[str, dict[str, float]]:
        successes = [res.success for res in cf_results]

        results_dict = {
            "success_rate": {
                "mean": np.mean(successes),
                "std": np.std(successes),
            }
        }

        if hasattr(cf_results[0], "y_final_latent"):
            success_latent = [res.success_latent for res in cf_results]
            results_dict["success_rate_latent"] = {
                "mean": np.mean(success_latent),
                "std": np.std(success_latent),
            }

        if hasattr(cf_results[0], "y_final_pred_oracle"):
            success_oracle = [res.success_oracle for res in cf_results]
            results_dict["success_rate_oracle"] = {
                "mean": np.mean(success_oracle),
                "std": np.std(success_oracle),
            }
        return results_dict

    def calculate_statistics(cf_results: list[CFResult]) -> dict[str, dict[str, float]]:
        y_initial_confidences = [res.y_initial_confidence for res in cf_results]
        y_final_confidences = [res.y_final_confidence for res in cf_results]
        confidence_reductions = [res.confidence_reduction for res in cf_results]
        steps = [res.steps for res in cf_results]

        statistics_dict = {
            "y_initial_confidence": {
                "mean": np.mean(y_initial_confidences),
                "std": np.std(y_initial_confidences),
            },
            "y_final_confidence": {
                "mean": np.mean(y_final_confidences),
                "std": np.std(y_final_confidences),
            },
            "steps": {
                "mean": np.mean(steps),
                "std": np.std(steps),
            },
            "confidence_reduction": {
                "mean": np.mean(confidence_reductions),
                "std": np.std(confidence_reductions),
            },
        }

        # Only either one of them
        if hasattr(cf_results[0], "y_initial_true_mae"):
            # Has initial true values
            y_initial_true_mae = [res.y_initial_true_mae for res in cf_results]
            statistics_dict["y_initial_true_mae"] = {
                "mean": np.mean(y_initial_true_mae),
                "std": np.std(y_initial_true_mae),
            }

        if hasattr(cf_results[0], "y_final_latent_mae"):
            # Has final latent values
            y_final_latent_mae = [res.y_final_latent_mae for res in cf_results]
            statistics_dict["y_final_latent_mae"] = {
                "mean": np.mean(y_final_latent_mae),
                "std": np.std(y_final_latent_mae),
            }

        if hasattr(cf_results[0], "y_final_pred_oracle_mae"):
            # Has oracle values
            oracle_properties = [
                "y_initial_pred_oracle_mae",
                "y_final_pred_oracle_mae",
                "y_initial_confidence_oracle",
                "y_final_confidence_oracle",
                "confidence_reduction_oracle",
            ]
            for prop in oracle_properties:
                values = [getattr(res, prop) for res in cf_results]
                statistics_dict[prop] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        return statistics_dict

    os.makedirs(result_dir, exist_ok=True)

    # Save individual results as JSON
    results = [res.result_dict() for res in cf_results]
    with open(os.path.join(result_dir, "cf_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Calculate success rates and statistics
    success_rates = calculate_success_rates(cf_results)
    statistics = calculate_statistics(cf_results)

    # Save success rates and statistics as CSV
    sorted_rows = sorted((success_rates | statistics).items())
    with open(os.path.join(result_dir, "results_summary.csv"), "w") as csvfile:
        writer = csv.writer(csvfile)

        # Write success rates
        writer.writerow(["metric", "mean", "std"])
        for key, value in sorted_rows:
            writer.writerow([key, value["mean"], value["std"]])

    torch.save(cf_results, os.path.join(result_dir, "cf_results.pt"))

    generate_contrastive_collage_regr(result_dir, cf_results)

    for res in cf_results:
        res.save_cf(result_dir)


def update_results_true_latents(
    true_latents: list[float],
    cf_results: list[CFResult],
    confidence_threshold: float,
):
    for res, y_true_latent in zip(cf_results, true_latents):
        res.update_y_final_latent(y_true_latent, confidence_threshold)


@torch.no_grad()
def update_results_oracle(
    oracle: torch.nn.Module,
    cf_results: list[CFResult],
    confidence_threshold: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for res in tqdm(cf_results, desc="Adding Oracle results"):
        x_init_and_prime = torch.vstack([res.x, res.x_prime]).to(device)
        y_pred_oracle = oracle(x_init_and_prime)
        res.update_y_oracle_initial(y_pred_oracle[0].item())
        res.update_y_oracle_final(y_pred_oracle[1].item(), confidence_threshold)


def generate_contrastive_collage_regr(
    output_folder: str,
    cf_results: list[CFResult],
    collage_folder_name: str = "collages",
    save_heatmaps=False,
) -> None:
    plt.close("all")
    # Set to serif font
    plt.rcParams["font.family"] = "serif"

    # <--------------- Helper Functions ----------------
    def bwr_heatmap(x, counterfactual):
        # Calculate the difference between x and counterfactual
        difference = (counterfactual - x).mean(dim=0)

        # Normalize the difference to the range [-1, 1]
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        colormap = mpl.colormaps["bwr"]
        heatmap = colormap(norm(difference.cpu().numpy()))

        # Convert the heatmap to a tensor and permute dimensions to match the expected output
        heatmap_high_contrast = torch.tensor(heatmap).permute(2, 0, 1)[:3]

        # # Prepare x_in and counterfactual_rgb for output
        # if x.shape[0] == 3:
        #     x_in = torch.clone(x)
        #     counterfactual_rgb = torch.clone(counterfactual)
        # else:  # Case: x is a grayscale
        #     x_in = torch.tile(x, [3, 1, 1])
        #     counterfactual_rgb = torch.tile(torch.clone(counterfactual), [3, 1, 1])

        # ScalarMappable to plot colorbar
        sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
        return heatmap_high_contrast, sm

    def format_title_string(cf_result: CFResult):
        """Formats the title string for the collage.

        Format:
                  (Label, Initial Prediction, Initial Oracle) → Target: (..., ...) → ...
                            (Final Prediction, Oracle): (..., ...)
                   Confidence (Initial, Final, Oracle): (..., ..., ...)
                             (Success, Success Oracle): (..., ...)
        """
        # If true label exists
        label_target_title = "(Label, Initial, Oracle Initial) → Target:"
        label = cf_result.y_initial_true
        y_initial_pred = cf_result.y_initial_pred
        y_initial_pred_oracle = (
            cf_result.y_initial_pred_oracle
            if hasattr(cf_result, "y_initial_pred_oracle")
            else -1
        )
        y_target = cf_result.y_target
        label_target_line = f"{label_target_title:>38} ({label:.2f}, {y_initial_pred:.2f}, {y_initial_pred_oracle:.2f}) → {y_target:.2f}"

        # If we have latent values, take them instead of the oracle
        latent_available = hasattr(cf_result, "y_final_latent")  # Only for squares
        oracle_str = "Latent" if latent_available else "Oracle"
        prediction_oracle_title = f"(Final Prediction, {oracle_str}):"
        y_final_pred = cf_result.y_final_pred
        y_final_pred_oracle = (
            cf_result.y_final_latent
            if latent_available
            else cf_result.y_final_pred_oracle
        )
        prediction_oracle_line = f"{prediction_oracle_title:>38} ({y_final_pred:.2f}, {y_final_pred_oracle:.2f})"

        confidence_title = f"Confidence (Initial, Final, {oracle_str}):"
        confidence_oracle = (
            cf_result.y_final_latent_mae
            if latent_available
            else cf_result.y_final_pred_oracle_mae
        )
        confidence_line = f"{confidence_title:>38} ({cf_result.y_initial_confidence:.2f}, {cf_result.y_final_confidence:.2f}, {confidence_oracle:.2f})"
        success_title = f"(Success, Success {oracle_str}):"
        success_oracle = (
            cf_result.success_latent if latent_available else cf_result.success_oracle
        )
        success_line = f"{success_title:>38} ({cf_result.success}, {success_oracle})"
        title_string = (
            "\n".join(
                [
                    label_target_line,
                    prediction_oracle_line,
                    success_line,
                    confidence_line,
                ]
            )
            + "\n"
        )

        return title_string

    # ---------------- Helper Functions --------------->

    collages_folder = os.path.join(output_folder, collage_folder_name)
    # collages_folder_svg = os.path.join(output_folder, collage_folder_name + "_svg")
    os.makedirs(collages_folder, exist_ok=True)
    # os.makedirs(collages_folder_svg, exist_ok=True)

    if save_heatmaps:
        heatmaps_folder = os.path.join(output_folder, "heatmaps")
        os.makedirs(heatmaps_folder, exist_ok=True)

    # collage_paths = []
    # heatmap_list = []
    for cf_result in cf_results:
        x = cf_result.x.squeeze(0)
        x_reconstr = cf_result.x_reconstructed.squeeze(0)
        counterfactual = cf_result.x_prime.squeeze(0)

        heatmap_bwr, sm = bwr_heatmap(x_reconstr, counterfactual)

        if save_heatmaps:
            heatmap_path = os.path.join(
                heatmaps_folder,
                f"{cf_result.image_name_base}_heatmap.png",
            )
            plt.imsave(heatmap_path, heatmap_bwr.permute(1, 2, 0).cpu().numpy())

        plt.figure()
        subplots_fig, axes = plt.subplots(1, 4, figsize=(8, 3), layout="constrained")
        images = [x, x_reconstr, counterfactual, heatmap_bwr]
        titles = [
            "Original",
            "Reconstructed",
            "Counterfactual",
            "Mean diff. to Original",
        ]

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(title)
            ax.axis("off")

        # Add colorbar to the right side of the diff plot
        cax = axes[-1].inset_axes((1.05, 0, 0.08, 1.0))
        plt.colorbar(sm, cax=cax)

        # # Save as svg first
        # collage_path = os.path.join(
        #     collages_folder_svg,
        #     f"{cf_result.image_name_base}_collage.svg",
        # )
        # plt.savefig(collage_path, dpi=300)
        # print("Saved regr collage to " + collage_path)

        # Save again for preview
        title_string = format_title_string(cf_result)
        plt.suptitle(title_string)
        collage_path = os.path.join(
            collages_folder,
            f"{cf_result.image_name_base}_collage.png",
        )
        plt.savefig(collage_path, dpi=150)  # Preview image only

        print("Saved regr collage to " + collage_path)
        plt.close("all")

        # collage_paths.append(collage_path)

    # return heatmap_list, collage_paths


def generate_collage_multitarget(
    output_folder: str,
    cf_results_multitarget: list[CFResult],
) -> None:
    collage_folder_name: str = "collages_multitarget"
    plt.close("all")
    # Set to serif font
    plt.rcParams["font.family"] = "serif"

    assert len(cf_results_multitarget) == 5, "Only 5 targets supported for now."

    # <--------------- Helper Functions ----------------
    def bwr_heatmap(x, counterfactual):
        # Calculate the difference between x and counterfactual
        difference = (counterfactual - x).mean(dim=0)

        # Normalize the difference to the range [-1, 1]
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        colormap = mpl.colormaps["bwr"]
        heatmap = colormap(norm(difference.cpu().numpy()))

        # Convert the heatmap to a tensor and permute dimensions to match the expected output
        heatmap_out = torch.tensor(heatmap).permute(2, 0, 1)[:3]

        # ScalarMappable to plot colorbar
        sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
        return heatmap_out, sm

    # ---------------- Helper Functions --------------->

    collages_folder = os.path.join(output_folder, collage_folder_name)
    heatmaps_folder = os.path.join(output_folder, "heatmaps")
    reconstruction_folder = os.path.join(output_folder, "x_reconstruction")
    os.makedirs(collages_folder, exist_ok=True)
    os.makedirs(heatmaps_folder, exist_ok=True)
    os.makedirs(reconstruction_folder, exist_ok=True)

    # Create Figure:
    # First row: Reconstruction + all the CF
    # Second Row: Original + All the heatmaps
    plt.figure()
    rows, cols = 2, 6
    subplots_fig, axes = plt.subplots(rows, cols, figsize=(12, 4), layout="constrained")

    x_hat_str = r"\hat{x}"
    y_hat_str = r"\hat{y}"
    y_tilde_str = r"\tilde{y}"
    for col in range(cols):
        if col == 0:  # First column: Reconstruction and Original
            result = cf_results_multitarget[col]
            y = result.y_initial_true
            y_hat = f"{result.y_initial_pred:.2f}"

            title_upper = f"${x_hat_str}: {y_hat_str}={y_hat}$"
            img_upper = cf_results_multitarget[0].x_reconstructed[0]

            # Save the reconstruction once
            plt.imsave(
                os.path.join(reconstruction_folder, result.image_name),
                img_upper.permute(1, 2, 0).cpu().numpy(),
            )

            title_lower = f"$x: y={y}$"
            img_lower = cf_results_multitarget[0].x[0]
        else:  # Other columns: Counterfactuals And heatmaps
            result = cf_results_multitarget[col - 1]
            y_hat = f"{result.y_final_pred:.2f}"
            y_target = f"{result.y_target:.2f}"

            title_upper = f"${y_tilde_str} = {y_target}, {y_hat_str} = {y_hat}$"
            img_upper = result.x_prime[0]

            title_lower = ""  # TODO: maybe something else?
            heatmap, sm = bwr_heatmap(
                result.x_reconstructed.squeeze(0), result.x_prime.squeeze(0)
            )
            # Save the heatmap first
            heatmap_path = os.path.join(
                heatmaps_folder,
                f"{result.image_name_base}_heatmap.png",
            )
            plt.imsave(heatmap_path, heatmap.permute(1, 2, 0).cpu().numpy())
            img_lower = heatmap

        # Do the Plotting
        for row in range(rows):
            if row == 0:
                title = title_upper
                img = img_upper
            else:
                title = title_lower
                img = img_lower

            ax = axes[row, col]
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(title)
            ax.axis("off")

    # Add colorbar to the right side of the last ax (last heatmap)
    cax = ax.inset_axes((1.10, 0, 0.08, 1.0))
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Mean Difference", rotation=90, labelpad=5)

    image_name_base = cf_results_multitarget[0].image_name_base
    # Save as svg first
    # collage_path = os.path.join(
    #     collages_folder,
    #     f"{image_name_base}_multitarget.svg",
    # )
    # plt.savefig(collage_path, dpi=300)
    # print("Saved regr collage to " + collage_path)

    # Save again for preview
    # title_string = format_title_string(cf_result)
    # plt.suptitle(title_string)
    collage_path = os.path.join(
        collages_folder,
        f"{image_name_base}_multitarget.png",
    )
    plt.savefig(collage_path, dpi=150)  # Preview image only

    print("Saved regr collage to " + collage_path)
    plt.close("all")

    if not os.path.exists(os.path.join(output_folder, "colorbar.svg")):
        save_only_colorbar(output_folder, sm)


def save_only_colorbar(folder: str, sm: ScalarMappable):
    # Create a figure
    fig = plt.figure(figsize=(1, 2))  # Set figure size for colorbar
    ax = fig.add_axes((1.20, 0, 0.1, 1.0))  # [left, bottom, width, height]

    # Create and add colorbar to the separate axis
    cb = (plt.colorbar(sm, cax=ax),)

    # Save the colorbar only
    fig.savefig(
        os.path.join(folder, "colorbar.svg"), dpi=300, bbox_inches="tight", pad_inches=0
    )

    # Close figure to free memory
    plt.close(fig)
