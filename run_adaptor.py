import argparse

from peal.global_utils import load_yaml_config, set_random_seed
from peal.adaptors.adaptor_factory import get_adaptor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    adaptor_config = load_yaml_config(args.config)
    set_random_seed(adaptor_config.seed)

    adaptor = get_adaptor(adaptor_config)
    adaptor.run()


if __name__ == "__main__":
    main()
