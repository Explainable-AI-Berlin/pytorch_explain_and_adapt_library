import argparse

from peal.configs.data.data_template import DataConfig
from peal.global_utils import load_yaml_config, add_class_arguments, integrate_arguments
from peal.data.dataset_generators import ConfounderDatasetGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, required=True)
    add_class_arguments(parser, DataConfig)
    args = parser.parse_args()

    data_config = load_yaml_config(args.data_config, DataConfig)
    integrate_arguments(args, data_config, exclude=["data_config"])

    if data_config.dataset_class == 'celeba':
        cdg = ConfounderDatasetGenerator(**data_config.__dict__)
        cdg.generate_dataset()


main()
