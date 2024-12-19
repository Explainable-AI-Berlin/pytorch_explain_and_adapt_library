import argparse
import torch
import os

from peal.architectures.predictors import TaskConfig
from peal.data.datasets import DataConfig
from peal.training.trainers import TrainingConfig, ModelTrainer, PredictorConfig
from peal.data.dataloaders import create_dataloaders_from_datasource
from peal.global_utils import load_yaml_config, set_random_seed
from peal.training.trainers import calculate_test_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_config", type=str, default=None)
    parser.add_argument("--partition", type=int, default=-1)
    args = parser.parse_args()

    # TODO this can't be done properly before bug is fixed...
    model_config = load_yaml_config(args.model_config)
    if not isinstance(model_config.data, DataConfig):
        model_config.data = DataConfig(**model_config.data)

    if not isinstance(model_config.training, TrainingConfig):
        model_config.training = TrainingConfig(**model_config.training)

    if not isinstance(model_config.task, TaskConfig):
        model_config.task = TaskConfig(**model_config.task)

    if not args.data_config is None:
        model_config.data = load_yaml_config(args.data_config, DataConfig)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.model_path is None:
        model_path = args.model_path

    else:
        model_path = os.path.join(model_config.model_path, "model.cpl")

    model = torch.load(model_path, map_location=device)
    if not isinstance(model, torch.nn.Module):
        predictor_config = load_yaml_config(args.model_config, PredictorConfig)
        model_weights = model
        model = ModelTrainer(predictor_config).model
        model.load_state_dict(model_weights)

    set_random_seed(model_config.seed)
    model.eval()
    test_dataloader = create_dataloaders_from_datasource(model_config)[args.partition]
    correct, group_accuracies, group_distribution, groups, worst_group_accuracy = calculate_test_accuracy(
        model, test_dataloader, device, True
    )
    partitions = ['Training', 'Validation', 'Test']
    print(partitions[args.partition] + " accuracy: " + str(correct))
    print("Group accuracies: " + str(group_accuracies))
    print("Group distribution: " + str(group_distribution))
    print("Groups: " + str(groups))
    print("Worst group accuracy: " + str(worst_group_accuracy))

main()
