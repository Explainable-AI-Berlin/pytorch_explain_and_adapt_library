import argparse

from peal.global_utils import load_yaml_config
from peal.teachers.spray_teacher import SprayConfig, Spray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    virelay_config = load_yaml_config(args.config, config_model=SprayConfig)
    virelay_teacher = Spray(virelay_config)
    virelay_teacher.get_feedback()

if __name__ == "__main__":
    main()