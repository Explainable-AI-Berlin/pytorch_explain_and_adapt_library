import argparse
from typing import List
import os

from diff_cf_ir.counterfactuals import CFResult, generate_contrastive_collage_regr
import torch
from tqdm import tqdm


def main(parent_folder: str):
    for subdir in tqdm(os.listdir(parent_folder)):
        results_folder = os.path.join(parent_folder, subdir)
        if not os.path.isdir(results_folder):
            continue

        cf_results: List[CFResult] = torch.load(
            os.path.join(results_folder, "cf_results.pt"), map_location="cpu"
        )
        generate_contrastive_collage_regr(
            results_folder, cf_results, collage_folder_name="collages_pub"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts the collages from the results to SVG and ready for publication."
    )
    parser.add_argument(
        "result_folder",
        type=str,
        help="Parent folder of the results. Each subfolder should contain a result for a cf run.",
    )
    args = parser.parse_args()

    main(args.result_folder)
