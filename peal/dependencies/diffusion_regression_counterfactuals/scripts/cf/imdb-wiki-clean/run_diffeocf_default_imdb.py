import os
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

from counterfactuals.generative_models import factory
from counterfactuals.data import get_data_info
from counterfactuals.utils import load_checkpoint

from diff_cf_ir.file_utils import assert_paths_exist, create_result_dir, dump_args
from diff_cf_ir.image_folder_dataset import ImageFolderDataset
from diff_cf_ir.models import load_resnet


def init_args():
    parser = argparse.ArgumentParser(description="Run adversarial attack.")
    parser.add_argument(
        "--gmodel_path", type=str, required=True, help="Path to the generative model."
    )
    parser.add_argument(
        "--gmodel_type",
        type=str,
        required=True,
        help="Type to the generative model.",
        default="Flow",
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
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset for the generative model",
        default="CelebA",
    )
    parser.add_argument(
        "--attack_style",
        type=str,
        choices=["x", "z"],
        required=True,
        default="z",
        help="Attack style: 'x' or 'z'.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        required=True,
        default=5000,
        help="Number of steps for the attack.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="Learning rate for the optimizer.",
        default=5e-2,
    )
    parser.add_argument(
        "--target",
        type=float,
        required=True,
        help="Target class for the attack.",
        default=1.0,
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        required=True,
        help="Target confidence to stop the attack",
        default=0.05,
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help=(
            "Path to the samples folder. "
            "The folder should contain images and a data.csv file with two "
            "fields for filename and age."
        ),
    )
    parser.add_argument(
        "--size", type=int, required=True, help="Target size of the input image."
    )
    parser.add_argument(
        "--limit_samples", type=int, help="Limit for the dataset.", default=None
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
    if not os.path.exists(meta_path):
        return None

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

    # Load models and data info
    # Load generative model
    data_info = get_data_info(args.dataset)
    gmodel, _ = factory.get_generative_model(args.gmodel_type, data_info, device)
    load_checkpoint(args.gmodel_path, gmodel, device)
    gmodel = gmodel.to(device).eval()

    # Load regression model
    regressor = load_resnet(args.rmodel_path)

    diffeo_cf = DiffeoCF(
        gmodel=gmodel,
        rmodel=regressor,
        data_shape=(3, args.size, args.size),
        result_dir=args.result_dir,
    )

    # Save args to a config txt file
    create_result_dir(args.result_dir)
    dump_args(args, args.result_dir)

    dataset = ImageFolderDataset(folder=args.image_folder, size=args.size)
    meta = load_meta(os.path.join(args.image_folder, "data.csv"))

    num_samples = (
        len(dataset)
        if args.limit_samples is None
        else min(args.limit_samples, len(dataset))
    )
    target = torch.Tensor([[args.target]]).to(device)

    diffeocf_results: list[CFResult] = []
    with tqdm(range(num_samples), desc="Running DiffeoCF") as pbar:
        for i in pbar:
            f, x = dataset[i]
            x = x.unsqueeze(0)

            f_basename = os.path.basename(f)
            y_true = meta[f_basename] if meta else -1

            pbar.set_postfix_str(f"Processing: {f}")

            diffeocf_result = diffeo_cf.adv_attack(
                x=x,
                attack_style=args.attack_style,
                num_steps=args.num_steps,
                lr=args.lr,
                target=args.target,
                confidence_threshold=args.confidence_threshold,
                image_path=f,
            )

            diffeocf_result.update_y_true_initial(y_true)

            diffeocf_results.append(diffeocf_result)

    # Save the results for the diffeo_cf attacks
    del regressor
    oracle = load_resnet(args.roracle_path).to(device)
    update_results_oracle(oracle, diffeocf_results, args.confidence_threshold)

    save_cf_results(diffeocf_results, args.result_dir)
