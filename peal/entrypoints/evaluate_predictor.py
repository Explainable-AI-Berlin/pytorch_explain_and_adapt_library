"""Print the accuracy and group accuracies of a trained predictor.

Loads the pickled model (``model.cpl``, or a state dict that is rebuilt into
the architecture through ``ModelTrainer``) and scores it with
``calculate_test_accuracy`` from ``peal/training/trainers.py`` on the
dataloaders of the predictor's data config.

Invocation::

    python evaluate_predictor.py --model_config <predictor.yaml> \\
        [--model_path <model.cpl>] [--data_config <data.yaml>] [--partition 2]

``--model_path`` defaults to ``<model_path>/model.cpl`` of the predictor
config; ``--data_config`` swaps in another dataset (e.g. a poisoned test
set); ``--partition`` selects one split (0 Training, 1 Validation, 2 Test),
the default ``-1`` evaluates all three. ``<PEAL_BASE>`` prefixes are resolved.

Prints, per split: accuracy, per-group accuracies, group distribution, worst
group accuracy and the group average, which always divides by four. Nothing is
written to disk; the reproduction scripts capture the output.
"""

import argparse
import types

import torch
import os
import numpy as np

from peal.architectures.interfaces import TaskConfig
from peal.data.interfaces import DataConfig
from peal.training.trainers import ModelTrainer
from peal.training.interfaces import TrainingConfig, PredictorConfig
from peal.data.dataloaders import create_dataloaders_from_datasource
from peal.global_utils import (
    load_yaml_config,
    set_random_seed,
    get_project_resource_dir,
)
from peal.training.trainers import calculate_test_accuracy


def main():
    """Load the model and data config, then print the split accuracies.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If ``--partition`` is outside ``0..2``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_config", type=str, default=None)
    parser.add_argument("--partition", type=int, default=-1)
    args = parser.parse_args()

    # TODO this can't be done properly before bug is fixed...
    model_config = load_yaml_config(args.model_config, PredictorConfig)
    if not isinstance(model_config.data, DataConfig):
        model_config.data = DataConfig(**model_config.data)

    if not isinstance(model_config.training, TrainingConfig):
        model_config.training = TrainingConfig(**model_config.training)

    if not isinstance(model_config.task, TaskConfig):
        model_config.task = TaskConfig(**model_config.task)

    if not args.data_config is None:
        model_config.data = load_yaml_config(args.data_config)

    if not isinstance(model_config.data, DataConfig):
        if isinstance(model_config.data, types.SimpleNamespace):
            model_config.data = vars(model_config.data)
        model_config.data = DataConfig(**model_config.data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.model_path is None:
        model_path = args.model_path

    else:
        model_path = os.path.join(model_config.model_path, "model.cpl")
    if model_path.startswith("<PEAL_BASE>"):
        model_path = model_path.replace("<PEAL_BASE>", get_project_resource_dir())
    # torch >= 2.6 defaults to weights_only=True, which cannot unpickle a whole nn.Module
    model = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(model, torch.nn.Module):
        predictor_config = load_yaml_config(args.model_config, PredictorConfig)
        model_weights = model
        model = ModelTrainer(predictor_config).model
        model.load_state_dict(model_weights)
    set_random_seed(0)
    model.eval()
    train_dataloader, val_dataloader, test_dataloader = (
        create_dataloaders_from_datasource(
            config=model_config, test_config=model_config.data
        )
    )
    partitions = ["Training", "Validation", "Test"]
    # --partition selects one split (0 Training, 1 Validation, 2 Test); the
    # default of -1 keeps the original behaviour of printing all three. It used
    # to be parsed and then ignored, so the `--partition 1` in several predictor
    # configs silently evaluated everything, which on CelebA is four times the
    # work of the split actually being reported.
    if args.partition >= 0:
        if args.partition >= len(partitions):
            raise SystemExit(
                f"--partition must be one of 0..{len(partitions) - 1} "
                f"({', '.join(f'{n}={p}' for n, p in enumerate(partitions))})"
            )
        partitions = [partitions[args.partition]]
    for i in partitions:
        if i == "Training":
            dataloader = train_dataloader
        elif i == "Validation":
            dataloader = val_dataloader
        else:
            dataloader = test_dataloader

        correct, group_accuracies, group_distribution, groups, worst_group_accuracy = (
            calculate_test_accuracy(model, dataloader, device, True)
        )
        print(i + " accuracy: " + str(correct))
        print("Group accuracies: " + str(group_accuracies))
        print("Group distribution: " + str(group_distribution))
        print("Groups: " + str(groups))
        print("Worst group accuracy: " + str(worst_group_accuracy))
        # Average over the groups this dataset actually has. This divided by a
        # hardcoded 4 before, which is right only for the 2-class x 2-confounder
        # tasks and silently halved the value for the 2-group ImageNet pairs. The
        # adaptor already used np.mean, so the numbers in the papers, which come
        # from its `test_avg_group_accuracy`, were never affected by this.
        print(
            "Average group accuracy: " + str(float(np.mean(np.array(group_accuracies))))
        )

    # correct, group_accuracies, group_distribution, groups, worst_group_accuracy = (
    #     calculate_test_accuracy(model, test_dataloader, device, True)
    # )

    # print(partitions[args.partition] + " accuracy: " + str(correct))
    # print("Group accuracies: " + str(group_accuracies))
    # print("Group distribution: " + str(group_distribution))
    # print("Groups: " + str(groups))
    # print("Worst group accuracy: " + str(worst_group_accuracy))


if __name__ == "__main__":
    main()
