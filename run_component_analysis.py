import argparse

from peal.global_utils import load_yaml_config, set_random_seed
from peal.generators.generator_factory import get_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--is_loaded", type=bool, default=True)
    parser.add_argument("--sd_config", type=str, default=None)
    # add_class_arguments(parser, ModelConfig)
    args = parser.parse_args()

    generator_config = load_yaml_config(args.config)
    if hasattr(generator_config, "is_loaded"):
        generator_config.is_loaded = args.is_loaded

    if not args.sd_config is None:
        sparse_dictionary_config = load_yaml_config(args.sd_config)

    else:
        sparse_dictionary_config = None

    set_random_seed(generator_config.seed)

    generator = get_generator(generator_config)
    generator.explain_all_components(sparse_dictionary_config)


if __name__ == "__main__":
    main()
