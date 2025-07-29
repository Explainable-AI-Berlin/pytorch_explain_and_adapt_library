import os
from lightning import seed_everything
from tqdm import tqdm
from diff_cf_ir.diffeocf import (
    DiffeoCF,
)

from diff_cf_ir.counterfactuals import (
    CFResult,
    save_cf_results,
    update_results_oracle,
)
import torch
import argparse

from diff_cf_ir.file_utils import (
    assert_paths_exist,
    create_result_dir,
    deterministic_run,
    dump_args,
)
from diff_cf_ir.generators import DAE
from diff_cf_ir.image_folder_dataset import ImageFolderDataset, default_transforms
from diff_cf_ir.models import load_model, load_resnet
import time


def init_args():
    parser = argparse.ArgumentParser(description="Run adversarial attack.")
    parser.add_argument(
        "--gmodel_path",
        type=str,
        required=True,
        help="Path to the generative model checkpoint.",
    )
    parser.add_argument(
        "--rmodel_path",
        type=str,
        required=True,
        help="Path to the regression model.",
    )
    parser.add_argument(
        "--roracle_path",
        type=str,
        required=True,
        help="Path to the oracle model.",
    )

    def parse_target(value):
        if value == "-inf":
            return float("-inf")
        elif value == "inf":
            return float("inf")
        else:
            return float(value)

    parser.add_argument(
        "--target",
        type=parse_target,
        required=True,
        help="Target class for the attack. Can be a specific value or -inf or inf for untargeted attacks.",
    )
    parser.add_argument(
        "--stop_at",
        type=float,
        required=True,
        help="Target goal for the attack. Will stop the attack if the value is reached.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        required=False,
        help="Confidence to goal to stop the attack",
        default=0.05,
    )
    parser.add_argument(
        "--dist",
        type=lambda x: x.lower() if x.lower() != "none" else None,
        help="Distance function to use for the attack.",
        choices=["l1", "l2", None],
    )
    parser.add_argument(
        "--dist_type",
        type=str,
        help="Which distance to use for the attack.",
        default="latent",
        choices=["latent", "pixel"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer to use for the attack.",
        choices=["adam", "sgd", "sgd_momentum"],
        default="adam",
    )
    parser.add_argument(
        "--forward_t",
        type=int,
        default=250,
        help="Number of steps for forward diffusion.",
        required=False,
    )
    parser.add_argument(
        "--backward_t",
        type=int,
        default=20,
        help="Number of steps for backwards diffusion.",
        required=False,
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        required=False,
        default=100,
        help="Number of steps for the attack.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        help="Learning rate for the optimizer.",
        default=5e-2,
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help=(
            "Path to the samples folder. "
            "The folder should contain images and an optional data.csv file with two "
            "fields for filename and age."
            " If available, it will update true labels in the result dict."
        ),
    )
    parser.add_argument(
        "--size", type=int, required=True, help="Target size of the input image."
    )
    parser.add_argument(
        "--limit_samples", type=int, help="Limit for the dataset.", default=None
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for the attack", default=1
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        default="diffeocf_results_default",
        help="Directory to save the results.",
    )

    args = parser.parse_args()
    print("Running with args:", args)

    return args


def load_meta(meta_path: str):
    meta = {}
    with open(meta_path) as f:
        for line in f.readlines():
            filename, age = line.strip().split(",")
            meta[filename] = float(age) / 100

    return meta


if __name__ == "__main__":
    args = init_args()
    # Check if all files exist
    assert_paths_exist(
        [
            args.gmodel_path,
            args.rmodel_path,
            args.roracle_path,
            args.image_folder,
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generative model
    gmodel = DAE(
        "ffhq",
        checkpoint_path=args.gmodel_path,
        forward_t=args.forward_t,
        backward_t=args.backward_t,
    ).to(device)

    # Load regression model
    regressor = load_model(args.rmodel_path)

    diffeo_cf = DiffeoCF(
        gmodel=gmodel,
        rmodel=regressor,
        data_shape=(3, args.size, args.size),
        result_dir=args.result_dir,
        dist=args.dist,
        optimizer=args.optimizer,
        dist_type=args.dist_type,
    )

    # Save args to a config txt file
    create_result_dir(args.result_dir)
    dump_args(args, args.result_dir)

    compose = default_transforms(args.size, ddpm=True)
    dataset = ImageFolderDataset(
        folder=args.image_folder, size=args.size, transform=compose
    )
    meta_path = os.path.join(args.image_folder, "data.csv")
    meta = None
    if os.path.exists(meta_path):
        meta = load_meta(meta_path)

    num_samples = (
        len(dataset)
        if args.limit_samples is None
        else min(args.limit_samples, len(dataset))
    )
    dataset = torch.utils.data.Subset(dataset, range(num_samples))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    deterministic_run(0)

    targets = (
        torch.Tensor([[args.target]]).to(device).broadcast_to([args.batch_size, 1])
    )
    stop_ats = (
        torch.Tensor([[args.stop_at]]).to(device).broadcast_to([args.batch_size, 1])
    )
    diffeocf_results_list: list[CFResult] = []
    with tqdm(dataloader, desc="Running DiffeoCF") as pbar:
        for fs, xs in pbar:
            fs_basenames = [os.path.basename(f) for f in fs]
            pbar.set_postfix_str(f"Processing: {fs_basenames}")

            diffeocf_results = diffeo_cf.adv_attack_dae_batch(
                xs=xs,
                targets=targets[: len(xs)],
                stop_ats=stop_ats[: len(xs)],
                image_paths=fs,
                confidence_threshold=args.confidence_threshold,
                num_steps=args.num_steps,
                lr=args.lr,
            )

            for b, diffeocf_result in enumerate(diffeocf_results):
                if meta is not None:
                    f = fs[b]
                    f_basename = os.path.basename(f)
                    y_true = meta[f_basename]
                    diffeocf_result.update_y_true_initial(y_true)

                diffeocf_results_list.append(diffeocf_result)

    # Save the results for the diffeo_cf attacks
    del regressor
    del gmodel
    oracle = load_resnet(args.roracle_path).to(device)
    update_results_oracle(oracle, diffeocf_results_list, args.confidence_threshold)

    save_cf_results(diffeocf_results_list, args.result_dir)
