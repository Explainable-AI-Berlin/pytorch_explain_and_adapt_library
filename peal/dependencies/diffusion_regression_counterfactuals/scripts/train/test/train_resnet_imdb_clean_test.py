import json
import sys
import os

from diff_cf_ir.file_utils import dump_args


import torch
import torchvision.transforms as T

from diff_cf_ir.imdb_clean_dataset import ImdbCleanDataModule
from diff_cf_ir.models import ResNetRegression
from diff_cf_ir.train import setup_trainer
import argparse


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train ResNet on IMDB Clean Dataset")
    parser.add_argument(
        "--folder_path",
        type=str,
        default="/data/imdb-clean/imdb-clean-1024-cropped",
        help="Path to the dataset folder",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="imdb_clean",
        help="Name for the output folder",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Size to resize images to"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Set seed to be for oracle model (1).",
    )
    parser.add_argument("--resnet_type", type=str, default=None, help="ResNet type")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="LR")

    args = parser.parse_args()

    # Check directories
    if not os.path.exists(args.folder_path):
        raise FileNotFoundError(f"Folder {args.folder_path} does not exist")

    torch.set_float32_matmul_precision("medium")

    small_images = args.image_size < 128
    model = ResNetRegression(
        args.resnet_type, small_images=small_images, learning_rate=args.learning_rate
    )

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    args.seed = 1 if args.oracle else 0
    trainer = setup_trainer(args.name, seed=args.seed)

    transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
        ]
    )

    data_module = ImdbCleanDataModule(
        folder_path=args.folder_path,
        transform=transforms,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("Running with the following arguments:")
    print(args)

    dump_args(args, trainer.logger.log_dir)

    trainer.fit(model=model, datamodule=data_module)
