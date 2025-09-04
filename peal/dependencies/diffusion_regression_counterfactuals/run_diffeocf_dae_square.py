import torch
from tqdm import tqdm
from diff_cf_ir.diffeocf import (
    DiffeoCF,
)
from diff_cf_ir.counterfactuals import (
    CFResult,
    save_cf_results,
    update_results_true_latents,
)
from diff_cf_ir.file_utils import (
    assert_paths_exist,
    create_result_dir,
    deterministic_run,
    dump_args,
)
from diff_cf_ir.generators import DAE
import argparse

from diff_cf_ir.models import load_resnet
from diff_cf_ir.squares_dataset import (
    SquaresDataset,
    get_experiment_targets,
    inner_square_color,
)
from diff_cf_ir.image_folder_dataset import default_transforms

import os


def init_args():
    parser = argparse.ArgumentParser(description="Run adversarial attack.")
    parser.add_argument(
        "--gmodel_path",
        type=str,
        required=True,
        help="Path to the generative model.",
    )
    parser.add_argument(
        "--rmodel_path",
        type=str,
        required=True,
        help="Path to the regression model.",
    )
    parser.add_argument(
        "--backward_t",
        type=int,
        default=10,
        help="Backward steps for the DAE.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Number of steps for the attack.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for the optimizer.",
        default=5e-2,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Size of the input image.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="Target confidence to stop the attack",
        default=0.05,
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )
    parser.add_argument(
        "--limit_samples", type=int, help="Limit for the dataset.", default=None
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory to save the results.",
    )

    args = parser.parse_args()
    print("Running with args:", args)
    return args


if __name__ == "__main__":
    args = init_args()

    # Check if all files exist
    assert_paths_exist(
        [
            args.gmodel_path,
            args.rmodel_path,
            args.image_folder,
        ]
    )
    # Load models and data info
    # Load generative model
    gmodel = DAE("square", args.gmodel_path, backward_t=args.backward_t).to("cuda")

    # Load regression model
    print("Loading regression model from path:", args.rmodel_path)
    regressor = load_resnet(args.rmodel_path)

    # Load Square Dataset
    compose = default_transforms(64, ddpm=True)
    dataset = SquaresDataset(root=args.image_folder, mask_mode=True, transform=compose)

    diffeocf_results: list[CFResult] = []

    diffeo_cf = DiffeoCF(
        gmodel=gmodel,
        rmodel=regressor,
        data_shape=(3, 64, 64),
        result_dir=os.path.join(args.result_dir),
    )

    # Save args to a config txt file
    create_result_dir(args.result_dir)
    dump_args(args, args.result_dir)

    num_samples = (
        len(dataset)
        if args.limit_samples is None
        else min(args.limit_samples, len(dataset))
    )
    dataset = torch.utils.data.Subset(dataset, range(num_samples))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    deterministic_run(seed=0)

    masks_result = []
    diffeocf_results: list[CFResult] = []
    with tqdm(dataloader, desc="Running DiffeoCF") as pbar:
        for fs, xs, ys, masks in pbar:
            pbar.set_postfix_str(f"Processing: {fs}")

            targets = get_experiment_targets(ys)
            diffeocf_result = diffeo_cf.adv_attack_dae_batch(
                xs=xs,
                targets=targets,
                stop_ats=targets,
                image_paths=fs,
                confidence_threshold=args.confidence_threshold,
                num_steps=args.num_steps,
                lr=args.lr,
            )
            for i in range(len(diffeocf_result)):
                diffeocf_result[i].update_y_true_initial(ys[i].item())

            diffeocf_results.extend(diffeocf_result)
            masks_result.append(masks.cpu())

    xs = [res.x[0] for res in diffeocf_results]
    xs_reconstructed = [res.x_reconstructed[0] for res in diffeocf_results]
    x_cfs = [res.x_prime[0] for res in diffeocf_results]
    y_initials = torch.Tensor([res.y_initial_pred for res in diffeocf_results])
    y_ends = torch.Tensor([res.y_final_pred for res in diffeocf_results])

    # Update with true final predictions
    masks = torch.cat(masks_result)
    update_results_true_latents(
        [inner_square_color(x_cf, mask).item() for x_cf, mask in zip(x_cfs, masks)],
        diffeocf_results,
        args.confidence_threshold,
    )
    save_cf_results(diffeocf_results, args.result_dir)
