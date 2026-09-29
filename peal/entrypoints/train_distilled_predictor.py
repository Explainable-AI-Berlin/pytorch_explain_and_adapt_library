"""Distil a trained predictor into a smoother surrogate for counterfactuals.

Drives ``distill_predictor`` in ``peal/training/trainers.py``: the predictor
of ``--config_predictor`` is loaded together with its training and
validation dataloaders, and a second network described by
``--config_distilled_predictor`` is fitted to mimic it with
``leakysoftplus`` activations in place of ReLU, so that gradients through it
are informative. The explainers perform the same step on the fly (the
``distilled_predictor`` field of the explainer config); this script runs it
in isolation.

Invocation::

    python train_distilled_predictor.py --config_predictor <predictor.yaml> \\
        --config_distilled_predictor <distilled_predictor.yaml>

Both arguments are required, so the ``sys.argv[-1]`` fallback and the
commented-out ``ModelTrainer`` block are dead code. A ``<PEAL_BASE>`` prefix
in the distilled config's ``model_path`` is resolved to the project
directory; the distilled model (``model.cpl`` and ``config.yaml``) is written
there.
"""

import argparse
import sys
import os
from peal.global_utils import (
    load_yaml_config,
    set_random_seed,
    get_project_resource_dir,
)
from peal.training.interfaces import PredictorConfig
from peal.training.trainers import get_predictor, distill_predictor
from peal.data.dataloaders import create_dataloaders_from_datasource
import torch


def main():
    """Load both predictor configs and distil the first into the second.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_predictor",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the main predictor.",
    )
    parser.add_argument(
        "--config_distilled_predictor",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the distilled predictor.",
    )

    # Add arguments from PredictorConfig class (if applicable)
    # add_class_arguments(parser, PredictorConfig)

    args = parser.parse_args()

    if hasattr(args, "config_predictor") and hasattr(
        args, "config_distilled_predictor"
    ):
        config_predictor = args.config_predictor
        distilled_predictor_config = args.config_distilled_predictor

    else:
        config = sys.argv[-1]

    config_predictor = load_yaml_config(config_predictor, PredictorConfig)
    distilled_predictor_config = load_yaml_config(
        distilled_predictor_config, PredictorConfig
    )
    # integrate_arguments(args, config, exclude=["config"])
    set_random_seed(config_predictor.seed)

    predictor = get_predictor(config_predictor)
    training_dataset, val_dataset, test_dataset = create_dataloaders_from_datasource(
        config=config_predictor, datasource=None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.to(device)
    base_path = distilled_predictor_config.model_path

    split_path = base_path.split("/")
    if split_path[0] == "<PEAL_BASE>":
        base_path = os.path.join(get_project_resource_dir(), *split_path[1:])
    gradient_predictor = distill_predictor(
        distilled_predictor_config,
        base_path,
        predictor,
        [training_dataset, val_dataset],
        replace_with_activation="leakysoftplus",
    )
    # model_trainer = ModelTrainer(config)
    # model_trainer.fit(
    #     continue_training=distilled_predictor_config.continue_training,
    #     is_initialized=distilled_predictor_config.is_loaded,
    # )


if __name__ == "__main__":
    main()
