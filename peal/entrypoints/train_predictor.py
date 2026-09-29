"""Train a predictor (student or test model) from a ``PredictorConfig`` YAML.

Drives ``ModelTrainer`` in ``peal/training/trainers.py``, which builds the
architecture named in the config, fits it on the dataset of the referenced
data config and tracks accuracy and group accuracies on the validation split.

Invocation::

    python train_predictor.py --config <predictor.yaml> \\
        [--<any PredictorConfig field> value] [--wandb_project NAME]

Every ``PredictorConfig`` field is exposed as an optional ``--flag`` that
overrides the YAML value, e.g. ``--continue_training True`` to resume from
``model_path`` or ``--is_loaded True`` to start from its saved weights.
Unknown flags are ignored. A ``<PEAL_BASE>`` prefix in ``model_path`` is
resolved to the project directory. A W&B run named
``predictor_<dataset_class>`` is opened when the package and an API key are
available.

Writes the run directory ``model_path`` with ``config.yaml`` and the trained
model pickled as ``model.cpl``.
"""

import argparse
import os

from peal.global_utils import (
    load_yaml_config,
    add_class_arguments,
    integrate_arguments,
    set_random_seed,
    get_project_resource_dir,
)
from peal.training.trainers import ModelTrainer
from peal.training.interfaces import PredictorConfig


def main():
    """Load the ``PredictorConfig``, apply overrides and fit the model.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="peal_sae_analysis")
    add_class_arguments(parser, PredictorConfig)
    args, unknown = parser.parse_known_args()

    config = load_yaml_config(args.config, PredictorConfig)
    integrate_arguments(args, config, exclude=["config", "wandb_project"])
    set_random_seed(config.seed)

    split_path = config.model_path.split("/")
    if split_path[0] == "<PEAL_BASE>":
        base_path = os.path.join(get_project_resource_dir(), *split_path[1:])
        config.model_path = base_path

    print(f"[Train Predictor] Model path: {config.model_path}")

    # W&B Grouping
    data_cfg = (
        load_yaml_config(config.data)
        if hasattr(config, "data") and isinstance(config.data, str)
        else getattr(config, "data", None)
    )
    dataset_variant = (
        getattr(data_cfg, "dataset_class", "dataset") if data_cfg else "dataset"
    )
    group_name = f"foundation/{dataset_variant}"

    try:
        import wandb

        if wandb.run is None:
            run_name = f"predictor_{dataset_variant}"
            wandb.init(project=args.wandb_project, name=run_name, group=group_name)
    except ImportError:
        pass
    except Exception as exc:  # e.g. no API key configured on the cluster
        print(
            f"[Train Predictor] W&B disabled ({exc.__class__.__name__}); training anyway."
        )

    model_trainer = ModelTrainer(config)
    model_trainer.fit(
        continue_training=config.continue_training, is_initialized=config.is_loaded
    )


if __name__ == "__main__":
    main()
