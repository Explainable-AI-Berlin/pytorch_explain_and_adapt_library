import argparse

from peal.global_utils import load_yaml_config
from peal.adaptors.adaptor_factory import get_adaptor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    adaptor_config = load_yaml_config(args.config)

    adaptor = get_adaptor(adaptor_config)
    adaptor.run()

main()