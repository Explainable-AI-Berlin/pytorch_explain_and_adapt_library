import argparse
import sys

from peal.global_utils import load_yaml_config, add_class_arguments, integrate_arguments, set_random_seed
from peal.training.trainers import ModelTrainer
from peal.training.interfaces import PredictorConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    add_class_arguments(parser, PredictorConfig)
    args = parser.parse_args()

    if hasattr(args, "config"):
        config = args.config

    else:
        config = sys.argv[-1]

    config = load_yaml_config(config, PredictorConfig)
    integrate_arguments(args, config, exclude=["config"])
    set_random_seed(config.seed)

    model_trainer = ModelTrainer(config)
    model_trainer.fit(
        continue_training=config.is_loaded, is_initialized=config.is_loaded
    )

if __name__ == '__main__':
    main()
