import os
import argparse
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diff_cf_ir.file_utils import assert_paths_exist
from diff_cf_ir.image_folder_dataset import ImageFolderDataset


def find_closest_images(
    sample_img: torch.Tensor, dataset_loader: DataLoader, top_n: int = 5
):
    distances = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_img = sample_img.to(device)
    for img_paths, batch in tqdm(dataset_loader, desc="Processing dataset batches"):

        batch = batch.to(device)
        batch_distances = torch.sum(torch.abs(sample_img - batch), dim=[1, 2, 3])

        for dist, img_path in zip(batch_distances, img_paths):
            distances.append((dist, img_path))

    distances = sorted(distances, key=lambda x: x[0])
    return [img_path for _, img_path in distances[:top_n]]


def main(args):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    sample_datset = ImageFolderDataset(args.sample_folder, size=args.size)
    dataset = ImageFolderDataset(args.dataset_folder, size=args.size)
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    for img_path, img in tqdm(sample_datset, desc="Processing samples"):
        closest_images = find_closest_images(img, dataset_loader)

        sample_output_folder = os.path.join(
            args.output_folder, os.path.splitext(img_path)[0]
        )
        os.makedirs(sample_output_folder, exist_ok=True)

        for i, img_path in enumerate(closest_images):
            img_basename = os.path.basename(img_path)
            output_path = os.path.join(
                sample_output_folder, f"match_{i:02d}={img_basename}"
            )
            shutil.copyfile(img_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find closest images in dataset")
    parser.add_argument(
        "sample_folder", type=str, help="Path to the folder containing sample images"
    )
    parser.add_argument(
        "dataset_folder", type=str, help="Path to the folder containing dataset images"
    )
    parser.add_argument(
        "output_folder", type=str, help="Path to the folder to save output images"
    )
    parser.add_argument(
        "--size", type=int, help="Size to which images are resized", required=True
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for processing", default=1024
    )
    args = parser.parse_args()

    assert_paths_exist([args.sample_folder, args.dataset_folder])

    torch.set_grad_enabled(False)
    main(args)
