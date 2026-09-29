"""
Generate ImageNet binary classification datasets for PEAL experiments.

This script creates data.csv files for binary classification tasks by
extracting images from the original ImageNet dataset for two specified classes.

Usage:
    python tools/generate_imagenet_binary_dataset.py --class1 269 --class2 248 --name wolf_vs_eskimo_dog
    python tools/generate_imagenet_binary_dataset.py --class1 286 --class2 293 --name cougar_vs_cheetah
    python tools/generate_imagenet_binary_dataset.py --class1 285 --class2 283 --name egyptian_cat_vs_persian_cat

ImageNet Class Indices:
    Wolf (grey wolf) = 269
    Eskimo Dog = 248
    Cougar = 286
    Cheetah = 293
    Egyptian Cat = 285
    Persian Cat = 283

Further arguments: ``--imagenet_path`` (the ImageNet ``train_set`` directory
with one synset folder per class), ``--output_path`` (default ``$PEAL_DATA``),
``--max_samples`` per class (default 1300) and ``--symlink`` to link instead
of copying. Only classes listed in ``IMAGENET_SYNSETS`` are supported.

Writes ``<output_path>/<name>/imgs/0/`` and ``imgs/1/`` with the images of
``--class1`` and ``--class2`` and ``<output_path>/<name>/data.csv`` with the
header ``ImgPath,Class``, the layout the PEAL image datasets
(``peal/data/datasets.py``) read through a data config's ``dataset_path``.
"""

import os
import argparse
import shutil
from pathlib import Path


# ImageNet synset to class index mapping for our classes
IMAGENET_SYNSETS = {
    248: "n02109961",  # Eskimo dog, husky
    269: "n02114548",  # white wolf, Arctic wolf, Canis lupus tundrarum
    283: "n02123394",  # Persian cat
    285: "n02124075",  # Egyptian cat
    286: "n02125311",  # cougar, puma, catamount, mountain lion, painter, panther, Felis concolor
    293: "n02130308",  # cheetah, chetah, Acinonyx jubatus
    # Spurious-feature classes of "Spurious Features Everywhere" (arXiv:2212.04871)
    # and their visually similar control classes; see the ImageNet-probe block of
    # reproduction_scripts/reproduce_didae_results.sh.
    94: "n01833805",  # hummingbird            (spurious: red feeder / flower)
    92: "n01828970",  # bee eater              (control)
    565: "n03393912",  # freight car            (spurious: graffiti)
    705: "n03895866",  # passenger car, coach   (control)
    554: "n03344393",  # fireboat               (spurious: water jet)
    625: "n03662601",  # lifeboat               (control)
    537: "n03218198",  # dogsled                (spurious: snow)
    603: "n03538406",  # horse cart             (control)
    105: "n01882714",  # koala                  (spurious: eucalyptus plants)
    106: "n01883070",  # wombat                 (control)
    933: "n07697313",  # cheeseburger           (spurious: fries)
    934: "n07697537",  # hotdog                 (control)
}

# Human-readable names
CLASS_NAMES = {
    248: "eskimo_dog",
    269: "wolf",
    283: "persian_cat",
    285: "egyptian_cat",
    286: "cougar",
    293: "cheetah",
    94: "hummingbird",
    92: "bee_eater",
    565: "freight_car",
    705: "passenger_car",
    554: "fireboat",
    625: "lifeboat",
    537: "dogsled",
    603: "horse_cart",
    105: "koala",
    106: "wombat",
    933: "cheeseburger",
    934: "hotdog",
}


def generate_binary_dataset(
    imagenet_path: str,
    output_path: str,
    class1: int,
    class2: int,
    name: str,
    max_samples_per_class: int = 1300,
    copy_images: bool = True,
):
    """
    Generate a binary classification dataset from ImageNet.

    Args:
        imagenet_path: Path to ImageNet train_set directory
        output_path: Path to output directory (e.g., $PEAL_DATA)
        class1: First ImageNet class index (will be class 0 in binary)
        class2: Second ImageNet class index (will be class 1 in binary)
        name: Name for the output dataset directory
        max_samples_per_class: Maximum number of samples per class
        copy_images: If True, copy images; if False, create symlinks
    """
    # Resolve environment variables
    if "$" in imagenet_path:
        imagenet_path = os.path.expandvars(imagenet_path)
    if "$" in output_path:
        output_path = os.path.expandvars(output_path)

    dataset_dir = os.path.join(output_path, name)
    imgs_dir = os.path.join(dataset_dir, "imgs")

    # Create output directories
    Path(imgs_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(imgs_dir, "0")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(imgs_dir, "1")).mkdir(parents=True, exist_ok=True)

    data_rows = []

    for binary_class, imagenet_class in enumerate([class1, class2]):
        synset = IMAGENET_SYNSETS.get(imagenet_class)
        if synset is None:
            raise ValueError(
                f"Unknown ImageNet class index: {imagenet_class}. "
                f"Please add the synset to IMAGENET_SYNSETS dictionary."
            )

        source_dir = os.path.join(imagenet_path, synset)
        if not os.path.exists(source_dir):
            raise FileNotFoundError(
                f"ImageNet synset directory not found: {source_dir}"
            )

        # Get all images from this class
        # skip dotfiles: the ImageNet folders contain stray rsync temp files
        # (e.g. n03218198/.n03218198_22853.JPEG.TvWQ0c) that sort first
        images = sorted(f for f in os.listdir(source_dir) if not f.startswith("."))[
            :max_samples_per_class
        ]

        print(
            f"Processing class {imagenet_class} ({CLASS_NAMES.get(imagenet_class, 'unknown')}) "
            f"-> binary class {binary_class}: {len(images)} images"
        )

        for img_name in images:
            source_path = os.path.join(source_dir, img_name)
            rel_path = os.path.join(str(binary_class), img_name)
            dest_path = os.path.join(imgs_dir, rel_path)

            if copy_images:
                shutil.copy2(source_path, dest_path)
            else:
                # Create relative symlink
                os.symlink(source_path, dest_path)

            data_rows.append((rel_path, binary_class))

    # Write data.csv
    csv_path = os.path.join(dataset_dir, "data.csv")
    with open(csv_path, "w") as f:
        f.write("ImgPath,Class\n")
        for img_path, label in data_rows:
            f.write(f"{img_path},{label}\n")

    print("\nDataset created successfully!")
    print(f"  Directory: {dataset_dir}")
    print(f"  CSV file: {csv_path}")
    print(f"  Total samples: {len(data_rows)}")
    print(
        f"  Class 0 ({CLASS_NAMES.get(class1, class1)}): {sum(1 for _, c in data_rows if c == 0)} samples"
    )
    print(
        f"  Class 1 ({CLASS_NAMES.get(class2, class2)}): {sum(1 for _, c in data_rows if c == 1)} samples"
    )


def main():
    """Parse the command line and call ``generate_binary_dataset``.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Generate ImageNet binary classification datasets for PEAL"
    )
    parser.add_argument(
        "--imagenet_path",
        type=str,
        default="/home/space/datasets/imagenet/2012/train_set",
        help="Path to ImageNet train_set directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="$PEAL_DATA",
        help="Path to output directory (default: $PEAL_DATA)",
    )
    parser.add_argument(
        "--class1",
        type=int,
        required=True,
        help="First ImageNet class index (will be class 0 in binary)",
    )
    parser.add_argument(
        "--class2",
        type=int,
        required=True,
        help="Second ImageNet class index (will be class 1 in binary)",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the output dataset directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1300,
        help="Maximum number of samples per class (default: 1300)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying images",
    )

    args = parser.parse_args()

    generate_binary_dataset(
        imagenet_path=args.imagenet_path,
        output_path=args.output_path,
        class1=args.class1,
        class2=args.class2,
        name=args.name,
        max_samples_per_class=args.max_samples,
        copy_images=not args.symlink,
    )


if __name__ == "__main__":
    main()
