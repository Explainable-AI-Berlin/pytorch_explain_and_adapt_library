import sys

from templates import *
from templates_latent import *
import argparse

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description="Run square64 training")
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the model",
    )
    args = parser.parse_args()

    # train the autoenc moodel
    # this can be run on 2080Ti's.
    conf = square64_autoenc()
    conf.base_dir = args.output_path

    train(conf)
