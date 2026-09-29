"""Generate a synthetic confounded dataset from a ``DataConfig`` YAML.

Dispatches on ``dataset_class`` to the generators in
``peal/data/dataset_generators.py``: ``celeba`` runs
``ConfounderDatasetGenerator``, which stamps a confounder (e.g. a copyright
tag) onto a subset of the source images and writes the poisoned copy, and
``SquareDataset`` runs ``SquareDatasetGenerator``, which draws the synthetic
square images. Any other ``dataset_class`` does nothing.

Invocation::

    python generate_dataset.py --config <data.yaml> \\
        [--<any DataConfig field> value]

Reads the source images under ``dataset_origin_path`` (CelebA) and writes
the new dataset into ``dataset_path``: ``imgs/``, ``masks/`` where the
generator produces them, and the ``data.csv`` label file. An existing
SquareDataset directory is moved aside with an ``_old_<timestamp>`` suffix.
"""

import argparse

from peal.data.interfaces import DataConfig
from peal.global_utils import (
    load_yaml_config,
    add_class_arguments,
    integrate_arguments,
    set_random_seed,
)
from peal.data.dataset_generators import (
    ConfounderDatasetGenerator,
    SquareDatasetGenerator,
)


def main():
    """Load the ``DataConfig`` and run the generator for its dataset class.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    add_class_arguments(parser, DataConfig)
    args = parser.parse_args()

    config = load_yaml_config(args.config, DataConfig)
    integrate_arguments(args, config, exclude=["config"])
    set_random_seed(config.seed)

    if config.dataset_class == "celeba":
        cdg = ConfounderDatasetGenerator(**config.__dict__, data_config=config)
        cdg.generate_dataset()
        print("Dataset generated successfully")

    elif config.dataset_class == "SquareDataset":
        cdg = SquareDatasetGenerator(data_config=config)
        cdg.generate_dataset()


if __name__ == "__main__":
    main()
