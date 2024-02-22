import argparse
import torch
import os

from peal.configs.data.data_config import DataConfig
from peal.configs.models.model_config import ModelConfig, TaskConfig
from peal.configs.training.training_template import TrainingConfig
from peal.data.dataloaders import create_dataloaders_from_datasource
from peal.global_utils import load_yaml_config, add_class_arguments, integrate_arguments
from peal.training.trainers import calculate_test_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_config", type=str, default=None)
    args = parser.parse_args()

    # TODO this can't be done properly before bug is fixed...
    model_config = load_yaml_config(args.model_config) #, ModelConfig)
    if not isinstance(model_config.data, DataConfig):
        model_config.data = DataConfig(**model_config.data)
        model_config.training = TrainingConfig(**model_config.training)
        model_config.task = TaskConfig(**model_config.task)

    if not args.data_config is None:
        model_config.data = load_yaml_config(args.data_config, DataConfig)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.model_path is None:
        model_path = args.model_path

    else:
        model_path = os.path.join(model_config.base_path, "model.cpl")

    model = torch.load(model_path, map_location=device)
    model.eval()
    test_dataloader = create_dataloaders_from_datasource(model_config)[-1]
    correct, group_accuracies, group_distribution, worst_group_accuracy = calculate_test_accuracy(
        model, test_dataloader, device, True
    )
    print("Test accuracy: ", correct)
    print("Group accuracies: ", group_accuracies)
    print("Group distribution: ", group_distribution)
    print("Worst group accuracy: ", worst_group_accuracy)

main()
