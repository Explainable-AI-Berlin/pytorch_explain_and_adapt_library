import argparse

from peal.configs.models.model_config import ModelConfig
from peal.global_utils import load_yaml_config, add_class_arguments, integrate_arguments
from peal.training.trainers import ModelTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    add_class_arguments(parser, ModelConfig)
    args = parser.parse_args()

    model_config = load_yaml_config(args.model_config, ModelConfig)
    integrate_arguments(args, model_config, exclude=["model_config"])

    model_trainer = ModelTrainer(model_config)
    model_trainer.fit(
        continue_training=model_config.is_loaded, is_initialized=model_config.is_loaded
    )


main()
