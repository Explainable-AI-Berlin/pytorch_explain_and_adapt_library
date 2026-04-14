import argparse

from src.calculate_square_regression_counterfactuals import calculate_square_regression_counterfactuals


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
    calculate_square_regression_counterfactuals(args)

