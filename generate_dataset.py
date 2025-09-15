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

    elif config.dataset_class == "SquareDataset":
        cdg = SquareDatasetGenerator(data_config=config)
        cdg.generate_dataset()


if __name__ == "__main__":
    main()
