import argparse
from typing import List, Set
import os

from diff_cf_ir.counterfactuals import CFResult, generate_contrastive_collage_regr
import torch
import matplotlib.pyplot as plt


def main(parent_folder: str, output_folder: str, filters: Set[str] = set()):
    cf_results_path = os.path.join(parent_folder, "cf_results.pt")

    diffeocf_results_list: List[CFResult] = torch.load(
        cf_results_path, map_location="cpu"
    )

    if filters:
        diffeocf_results_list = [
            result
            for result in diffeocf_results_list
            if result.image_name_base in filters
        ]

    os.makedirs(output_folder, exist_ok=True)
    generate_contrastive_collage_regr(
        output_folder, diffeocf_results_list, save_heatmaps=True
    )

    # Save x, x' as image
    x_folder = os.path.join(output_folder, "x")
    x_prime_folder = os.path.join(output_folder, "x_prime")
    os.makedirs(x_folder, exist_ok=True)
    os.makedirs(x_prime_folder, exist_ok=True)
    for result in diffeocf_results_list:
        plt.imsave(
            os.path.join(x_folder, result.image_name),
            result.x[0].permute(1, 2, 0).cpu().numpy(),
        )
        plt.imsave(
            os.path.join(x_prime_folder, result.image_name),
            result.x_prime[0].permute(1, 2, 0).cpu().numpy(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates collages for the results of the multitarget attack again."
    )
    parser.add_argument(
        "result_folder",
        type=str,
        help="Parent folder of the results. Each subfolder should contain a result for a cf run.",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Where to save the results",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="Filter the results by the given string. Can be separated by commas for multiple filters.",
    )
    args = parser.parse_args()

    filters: Set[str] = set(args.filters.split(",")) if args.filters else None

    main(args.result_folder, args.output_folder, filters=filters)
