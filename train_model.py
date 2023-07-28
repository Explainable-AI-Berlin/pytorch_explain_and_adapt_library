import argparse

from peal.configs.models.template import ModelConfig
from peal.utils import load_yaml_config
from peal.training.trainers import ModelTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    args = parser.parse_args()

    model_config = load_yaml_config(args.model_config, ModelConfig)

    ModelTrainer(model_config).fit()