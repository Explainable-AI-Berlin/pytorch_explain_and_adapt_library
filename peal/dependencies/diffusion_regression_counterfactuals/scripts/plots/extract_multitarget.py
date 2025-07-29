import argparse
from typing import List, Set
import os

from diff_cf_ir.counterfactuals import CFResult, generate_collage_multitarget
import torch
import matplotlib.pyplot as plt


def main(parent_folder: str, output_folder: str):
    cf_results_path = os.path.join(parent_folder, "cf_results.pt")

    diffeocf_results_list: List[CFResult] = torch.load(
        cf_results_path, map_location="cpu"
    )

    os.makedirs(output_folder, exist_ok=True)
    num_targets = 5
    num_batches = len(diffeocf_results_list) // num_targets
    for b_i in range(num_batches):
        cur_results = diffeocf_results_list[b_i * num_targets : (b_i + 1) * num_targets]
        generate_collage_multitarget(output_folder, cur_results)

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
    args = parser.parse_args()

    main(args.result_folder, args.output_folder)
