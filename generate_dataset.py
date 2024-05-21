import argparse

from peal.configs.data.data_config import DataConfig
from peal.global_utils import load_yaml_config, add_class_arguments, integrate_arguments
from peal.data.dataset_generators import ConfounderDatasetGenerator, SquareDatasetGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    add_class_arguments(parser, DataConfig)
    args = parser.parse_args()

    config = load_yaml_config(args.config, DataConfig)
    integrate_arguments(args, config, exclude=["config"])

    if config.dataset_class == "celeba":
        cdg = ConfounderDatasetGenerator(**config.__dict__, data_config=config)
        cdg.generate_dataset()

    elif config.dataset_class == "square":
        cdg = SquareDatasetGenerator(data_config=config)
        cdg.generate_dataset()


main()
