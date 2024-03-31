import argparse

from peal.global_utils import load_yaml_config, add_class_arguments
from peal.generators.generator_factory import get_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    #add_class_arguments(parser, ModelConfig)
    args = parser.parse_args()

    generator_config = load_yaml_config(args.config)

    generator = get_generator(generator_config)
    generator.train_model()

main()