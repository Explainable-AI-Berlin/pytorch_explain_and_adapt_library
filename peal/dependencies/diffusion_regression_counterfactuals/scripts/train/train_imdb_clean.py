import os

from diff_cf_ir.file_utils import dump_args

import torch
import torchvision.transforms as T

from diff_cf_ir.imdb_clean_dataset import ImdbCleanDataModule
from diff_cf_ir.models import ResNetRegression, DenseNetRegression
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
        "--densenet_weights",
        type=str,
        default=None,
        help="If provided, uses densenet and pretrained weights instead",
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Whether to fully finetune the densenet model. Only used if densenet_weights is provided.",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Set seed to be for oracle model (1).",
    )

    args = parser.parse_args()

    # Check directories
    if not os.path.exists(args.folder_path):
        raise FileNotFoundError(f"Folder {args.folder_path} does not exist")

    torch.set_float32_matmul_precision("medium")

    if args.densenet_weights:
        print("Using DenseNet model and weights")
        model = DenseNetRegression(
            args.densenet_weights, full_finetune=args.full_finetune
        )
    else:
        print("Using ResNet model")
        small_images = args.image_size < 128
        resnet_type = "resnet18" if small_images else "resnet152"
        model = ResNetRegression(resnet_type, small_images=small_images)

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
