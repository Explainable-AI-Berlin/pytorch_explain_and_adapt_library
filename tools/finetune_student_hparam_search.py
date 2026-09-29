"""Grid-search the finetuning hyper-parameters of a CFKD student.

Uses the ``CFKD`` adaptor in
``peal/adaptors/counterfactual_knowledge_distillation.py`` to produce (or
reuse) the counterfactual training and validation datasets of one finetune
iteration, then trains a fresh copy of the original student for every
combination of learning rate, optimizer, mixing ratio and batch
concatenation with ``ModelTrainer`` (``peal/training/trainers.py``) and
scores each on the CFKD test loader.

Invocation::

    python tools/finetune_student_hparam_search.py --search_config <search.yaml> \\
        [--config <cfkd_adaptor.yaml>] [--learning_rates 1e-5,3e-5] \\
        [--optimizers adamw,sgd] [--mixing_ratios 0.05,0.1] \\
        [--concatenate_batches false,true] \\
        [--continuous_learning finetune|deep_feature_reweighting] \\
        [--selection_metric worst_group_accuracy|avg_group_accuracy|accuracy]

Every setting may come from the search YAML or from the matching ``--flag``;
the flag wins. Comma-separated strings are split into lists. ``--student``,
``--teacher`` and ``--base_dir`` override the corresponding CFKD settings.

Writes ``<output_dir>`` (default ``<base_dir>/finetune_hparam_search``) with
one sub-directory per trial holding ``model.cpl``, ``metrics.json`` and
``trial_config.yaml``, plus ``results.csv`` over all trials and ``best.json``
for the trial with the highest selection metric.
"""

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

import sys

# Runnable from a clone without installing PEAL: put the repository root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peal.adaptors.counterfactual_knowledge_distillation import CFKD, CFKDConfig
from peal.architectures.predictors import get_predictor
from peal.data.dataloaders import (
    DataloaderMixer,
    WeightedDataloaderList,
    create_dataloaders_from_datasource,
)
from peal.global_utils import load_yaml_config, save_yaml_config, set_random_seed
from peal.training.trainers import ModelTrainer, calculate_test_accuracy


def parse_csv_values(value, cast):
    """Split a comma-separated string and cast every non-empty item.

    Parameters
    ----------
    value : str
        Comma-separated values, e.g. ``"1e-5, 3e-5"``.
    cast : callable
        Applied to each stripped item.

    Returns
    -------
    list
        The cast items, in order.
    """
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_bool_values(value):
    """Split a comma-separated string of booleans.

    Parameters
    ----------
    value : str
        Items such as ``"true,false"``; accepted spellings are 1/true/yes/y
        and 0/false/no/n, case-insensitive.

    Returns
    -------
    list of bool
        The parsed values.

    Raises
    ------
    ValueError
        If an item is not a recognised spelling.
    """
    bools = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"1", "true", "yes", "y"}:
            bools.append(True)
        elif item in {"0", "false", "no", "n"}:
            bools.append(False)
        else:
            raise ValueError(f"Cannot parse boolean value: {item}")
    return bools


def load_search_config(path):
    """Read the search YAML as a plain dictionary.

    Parameters
    ----------
    path : str
        Path to the search config.

    Returns
    -------
    dict
        The YAML content, or an empty dict for an empty file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_setting(args, search_config, name, default=None):
    """Resolve one setting: command-line flag first, then YAML, then default.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line; ``getattr(args, name)`` must exist.
    search_config : dict
        The search YAML.
    name : str
        Setting name (identical on the command line and in the YAML).
    default : object, optional
        Returned when neither source provides the setting.

    Returns
    -------
    object
        The resolved value.
    """
    value = getattr(args, name)
    if value is not None:
        return value
    return search_config.get(name, default)


def as_list(value, cast=None):
    """Coerce a setting to a list, parsing comma-separated strings.

    Parameters
    ----------
    value : None, str, list, tuple or scalar
        The raw setting.
    cast : callable, optional
        Applied to each item; ``bool`` triggers ``parse_bool_values``.

    Returns
    -------
    list
        The items (empty for ``None``).
    """
    if value is None:
        return []
    if isinstance(value, str):
        if cast is bool:
            return parse_bool_values(value)
        return parse_csv_values(value, cast or str)
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if cast is None:
        return values
    return [cast(v) for v in values]


def as_bool(value):
    """Coerce a scalar setting to a boolean.

    Parameters
    ----------
    value : bool, str or int
        The raw setting.

    Returns
    -------
    bool
        The parsed value.
    """
    if isinstance(value, bool):
        return value
    return parse_bool_values(str(value))[0]


def load_student(student_arg, device):
    """Load the optional student override through ``get_predictor``.

    Parameters
    ----------
    student_arg : str or None
        Path to a ``model.cpl`` or a predictor config; ``None`` keeps the
        student of the CFKD config.
    device : str
        Device the model is moved to.

    Returns
    -------
    torch.nn.Module or None
        The student in eval mode, or ``None``.

    Raises
    ------
    ValueError
        If the argument does not resolve to an ``nn.Module``.
    """
    if student_arg is None:
        return None
    student, _ = get_predictor(student_arg, device=device)
    if not isinstance(student, torch.nn.Module):
        raise ValueError("--student must resolve to a torch.nn.Module.")
    student.eval()
    return student


def prepare_cfkd_datasets(cfkd, finetune_iteration, regenerate):
    """Run the CFKD pipeline up to the counterfactual dataset of one iteration.

    Initialises the run, retrieves the counterfactuals and the feedback for
    ``finetune_iteration`` and creates the training dataset from them. The
    adaptor's ``overwrite`` flag is temporarily set to ``regenerate`` so that
    cached counterfactuals are reused unless regeneration is requested.

    Parameters
    ----------
    cfkd : CFKD
        The initialised adaptor.
    finetune_iteration : int
        The iteration whose datasets are prepared.
    regenerate : bool
        Recompute the counterfactuals instead of reusing cached ones.

    Returns
    -------
    tuple
        ``(train_dataset_path, validation_dataset_path, validation_stats)``;
        the validation path is ``<base_dir>/<iteration>/validation_dataset``
        and may not exist.
    """
    overwrite_buffer = cfkd.overwrite
    cfkd.overwrite = bool(regenerate)
    validation_prestats, validation_tracked_values, writer = cfkd.initialize_run()

    tracked_values = cfkd.retrieve_counterfactual_list(
        validation_stats=validation_prestats,
        finetune_iteration=finetune_iteration,
    )
    feedback = cfkd.retrieve_feedback(
        tracked_values=tracked_values,
        finetune_iteration=finetune_iteration,
        mode="train",
    )

    validation_stats = cfkd.retrieve_validation_stats(
        finetune_iteration=finetune_iteration - 1,
        validation_prestats=validation_prestats,
        validation_tracked_values=validation_tracked_values,
    )

    train_dataset_path = cfkd.create_dataset(
        feedback=feedback,
        finetune_iteration=finetune_iteration,
        mode="train",
        config=cfkd.data_config,
        **tracked_values,
    )
    validation_dataset_path = os.path.join(
        cfkd.base_dir, str(finetune_iteration), "validation_dataset"
    )

    cfkd.overwrite = overwrite_buffer
    if writer is not None:
        writer.flush()
    return train_dataset_path, validation_dataset_path, validation_stats


def build_train_mixer(cfkd, trial_config, train_dataset_path, mixing_ratio):
    """Mix the counterfactual training set with the original training data.

    Parameters
    ----------
    cfkd : CFKD
        Provides the original ``train_dataloader``, data config and hints.
    trial_config : CFKDConfig
        Its ``training`` block configures the ``DataloaderMixer``.
    train_dataset_path : str
        Directory of the generated counterfactual dataset.
    mixing_ratio : float
        Share of counterfactual samples; the original data gets
        ``1 - mixing_ratio``.

    Returns
    -------
    DataloaderMixer
        The mixed training dataloader, with hints enabled when CFKD uses them.
    """
    generated_dataloader, _, _ = create_dataloaders_from_datasource(
        config=cfkd.data_config,
        datasource=train_dataset_path,
    )
    old_dataloader = DataloaderMixer(trial_config.training, cfkd.train_dataloader)
    mixed_dataloader = DataloaderMixer(trial_config.training, generated_dataloader)
    mixed_dataloader.append(
        old_dataloader,
        weight_added_dataloader=1 - mixing_ratio,
    )
    mixed_dataloader.return_src_internal = True
    if cfkd.hints_enabled:
        mixed_dataloader.enable_hints()
    return mixed_dataloader


def build_validation_dataloaders(cfkd, validation_dataset_path, include_generated):
    """Collect the validation loaders: the original one plus, optionally, CFs.

    Parameters
    ----------
    cfkd : CFKD
        Provides ``val_dataloader`` and ``validation_data_config``.
    validation_dataset_path : str
        Directory of the generated validation counterfactuals.
    include_generated : bool
        Append the generated loader when the directory exists and is
        non-empty.

    Returns
    -------
    WeightedDataloaderList
        The validation loaders.
    """
    validation_dataloaders = WeightedDataloaderList([cfkd.val_dataloader])
    if include_generated and os.path.exists(validation_dataset_path):
        _, generated_val_dataloader, _ = create_dataloaders_from_datasource(
            config=cfkd.validation_data_config,
            datasource=validation_dataset_path,
        )
        if (
            isinstance(generated_val_dataloader, torch.utils.data.DataLoader)
            and len(generated_val_dataloader.dataset) > 0
        ):
            validation_dataloaders.append(generated_val_dataloader)
    return validation_dataloaders


def evaluate_model(model, dataloader, device, max_test_batches=None):
    """Score a model with ``calculate_test_accuracy`` and unpack the result.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate; switched to eval mode.
    dataloader : torch.utils.data.DataLoader
        The evaluation data.
    device : str or torch.device
        Device for the forward passes.
    max_test_batches : int, optional
        Cap on the number of evaluated batches.

    Returns
    -------
    dict
        ``accuracy``, ``worst_group_accuracy``, ``avg_group_accuracy``,
        ``group_accuracies``, ``group_distribution`` and ``groups`` as plain
        Python numbers and lists.
    """
    model.eval()
    result = calculate_test_accuracy(
        model,
        dataloader,
        device,
        calculate_group_accuracies=True,
        max_test_batches=max_test_batches,
        tracking_level=0,
    )
    accuracy, group_accuracies, group_distribution, groups, worst_group_accuracy = (
        result
    )
    return {
        "accuracy": float(accuracy),
        "worst_group_accuracy": float(worst_group_accuracy),
        "avg_group_accuracy": float(np.mean(group_accuracies)),
        "group_accuracies": [float(x) for x in group_accuracies],
        "group_distribution": [float(x) for x in group_distribution],
        "groups": groups.tolist() if hasattr(groups, "tolist") else groups,
    }


def metric_value(metrics, metric_name):
    """Pick the selection metric out of an ``evaluate_model`` result.

    Parameters
    ----------
    metrics : dict
        Output of ``evaluate_model``.
    metric_name : str
        ``"accuracy"``, ``"avg_group_accuracy"`` or
        ``"worst_group_accuracy"``.

    Returns
    -------
    float
        The metric value.

    Raises
    ------
    ValueError
        For an unknown metric name.
    """
    if metric_name == "accuracy":
        return metrics["accuracy"]
    if metric_name == "avg_group_accuracy":
        return metrics["avg_group_accuracy"]
    if metric_name == "worst_group_accuracy":
        return metrics["worst_group_accuracy"]
    raise ValueError(f"Unknown metric: {metric_name}")


def build_trial_grid(learning_rates, optimizers, mixing_ratios, concatenate_options):
    """Enumerate the trials of the grid.

    Concatenated batches ignore the mixing ratio, so those trials use only
    the first ratio; non-concatenated trials are expanded over all ratios.

    Parameters
    ----------
    learning_rates : list of float
    optimizers : list of str
    mixing_ratios : list of float
    concatenate_options : list of bool

    Returns
    -------
    list of dict
        Each with ``learning_rate``, ``optimizer``, ``mixing_ratio`` and
        ``concatenate_batches``.
    """
    trials = []
    concat_mixing_ratio = mixing_ratios[0]
    for learning_rate in learning_rates:
        for optimizer in optimizers:
            for concatenate_batches in concatenate_options:
                if concatenate_batches:
                    trials.append(
                        {
                            "learning_rate": learning_rate,
                            "optimizer": optimizer,
                            "mixing_ratio": concat_mixing_ratio,
                            "concatenate_batches": True,
                        }
                    )
                else:
                    for mixing_ratio in mixing_ratios:
                        trials.append(
                            {
                                "learning_rate": learning_rate,
                                "optimizer": optimizer,
                                "mixing_ratio": mixing_ratio,
                                "concatenate_batches": False,
                            }
                        )
    return trials


def main():
    """Prepare the CFKD datasets, run every trial and write the summary files.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If neither the search YAML nor ``--config`` names the CFKD config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_config", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="CFKD adaptor config.")
    parser.add_argument(
        "--student", type=str, default=None, help="Optional student model override."
    )
    parser.add_argument(
        "--teacher", type=str, default=None, help="Optional teacher override."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Optional CFKD run directory override.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--finetune_iteration", type=int, default=None)
    parser.add_argument("--learning_rates", type=str, default=None)
    parser.add_argument("--optimizers", type=str, default=None)
    parser.add_argument("--mixing_ratios", type=str, default=None)
    parser.add_argument("--concatenate_batches", type=str, default=None)
    parser.add_argument(
        "--continuous_learning",
        type=str,
        default=None,
        choices=["finetune", "deep_feature_reweighting"],
    )
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--skip_generated_validation", type=as_bool, default=None)
    parser.add_argument("--regenerate_counterfactuals", type=as_bool, default=None)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default=None,
        choices=["worst_group_accuracy", "avg_group_accuracy", "accuracy"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    search_config = load_search_config(args.search_config)

    config_path = get_setting(args, search_config, "config")
    if config_path is None:
        raise ValueError("Provide `config` in the search YAML or pass --config.")

    seed = get_setting(args, search_config, "seed", 0)
    set_random_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adaptor_config = load_yaml_config(config_path, CFKDConfig)
    base_dir = get_setting(args, search_config, "base_dir")
    if base_dir is not None:
        adaptor_config.base_dir = base_dir
    output_dir = get_setting(
        args,
        search_config,
        "output_dir",
        os.path.join(adaptor_config.base_dir, "finetune_hparam_search"),
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    student = load_student(get_setting(args, search_config, "student"), device)
    cfkd = CFKD(
        adaptor_config=adaptor_config,
        student=student,
        teacher=get_setting(args, search_config, "teacher"),
        base_dir=adaptor_config.base_dir,
    )

    finetune_iteration = get_setting(args, search_config, "finetune_iteration", 1)
    regenerate_counterfactuals = get_setting(
        args, search_config, "regenerate_counterfactuals", False
    )
    regenerate_counterfactuals = as_bool(regenerate_counterfactuals)
    train_dataset_path, validation_dataset_path, _ = prepare_cfkd_datasets(
        cfkd,
        finetune_iteration,
        regenerate_counterfactuals,
    )

    learning_rates = as_list(
        get_setting(args, search_config, "learning_rates", [1e-5, 3e-5, 1e-4]),
        float,
    )
    optimizers = as_list(get_setting(args, search_config, "optimizers", ["adamw"]), str)
    mixing_ratios = as_list(
        get_setting(args, search_config, "mixing_ratios", [0.05, 0.1, 0.2]),
        float,
    )
    concatenate_options = as_list(
        get_setting(args, search_config, "concatenate_batches", [False, True]),
        bool,
    )
    continuous_learning = get_setting(
        args, search_config, "continuous_learning", "deep_feature_reweighting"
    )
    max_epochs = get_setting(args, search_config, "max_epochs")
    steps_per_epoch = get_setting(args, search_config, "steps_per_epoch")
    train_batch_size = get_setting(args, search_config, "train_batch_size")
    val_batch_size = get_setting(args, search_config, "val_batch_size")
    skip_generated_validation = get_setting(
        args, search_config, "skip_generated_validation", False
    )
    skip_generated_validation = as_bool(skip_generated_validation)
    selection_metric = get_setting(
        args, search_config, "selection_metric", "worst_group_accuracy"
    )

    rows = []
    best = None
    original_student = copy.deepcopy(cfkd.original_student).cpu()
    trials = build_trial_grid(
        learning_rates,
        optimizers,
        mixing_ratios,
        concatenate_options,
    )

    for trial in trials:
        learning_rate = trial["learning_rate"]
        optimizer = trial["optimizer"]
        mixing_ratio = trial["mixing_ratio"]
        concatenate_batches = trial["concatenate_batches"]
        trial_name = (
            f"lr_{learning_rate:g}_opt_{optimizer}_mix_{mixing_ratio:g}"
            f"_concat_{int(concatenate_batches)}"
        ).replace(".", "p")
        trial_dir = os.path.join(output_dir, trial_name)
        Path(trial_dir).mkdir(parents=True, exist_ok=True)

        set_random_seed(seed)
        trial_config = copy.deepcopy(cfkd.adaptor_config)
        trial_config.model_path = trial_dir
        trial_config.training.learning_rate = learning_rate
        trial_config.training.optimizer = optimizer
        trial_config.training.concatenate_batches = concatenate_batches
        trial_config.continuous_learning = continuous_learning
        if max_epochs is not None:
            trial_config.training.max_epochs = max_epochs
        if steps_per_epoch is not None:
            trial_config.training.steps_per_epoch = steps_per_epoch
        if train_batch_size is not None:
            trial_config.training.train_batch_size = train_batch_size
        if val_batch_size is not None:
            trial_config.training.val_batch_size = val_batch_size

        train_mixer = build_train_mixer(
            cfkd,
            trial_config,
            train_dataset_path,
            mixing_ratio,
        )
        val_dataloaders = build_validation_dataloaders(
            cfkd,
            validation_dataset_path,
            not skip_generated_validation,
        )

        model = copy.deepcopy(original_student)
        trainer = ModelTrainer(
            config=trial_config,
            model=model,
            datasource=(train_mixer, val_dataloaders),
            model_path=trial_dir,
            only_last_layer=continuous_learning == "deep_feature_reweighting",
        )
        trainer.fit(continue_training=True)

        metrics = evaluate_model(
            trainer.model,
            cfkd.test_dataloader,
            trainer.device,
            max_test_batches=trial_config.max_test_batches,
        )
        score = metric_value(metrics, selection_metric)
        row = {
            "trial": trial_name,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "mixing_ratio": mixing_ratio,
            "concatenate_batches": concatenate_batches,
            "continuous_learning": continuous_learning,
            "score": score,
            **metrics,
            "model_path": os.path.join(trial_dir, "model.cpl"),
        }
        rows.append(row)
        with open(os.path.join(trial_dir, "metrics.json"), "w") as f:
            json.dump(row, f, indent=2)
        save_yaml_config(trial_config, os.path.join(trial_dir, "trial_config.yaml"))

        if best is None or row["score"] > best["score"]:
            best = row
        print(json.dumps(row, indent=2))

    results_path = os.path.join(output_dir, "results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(output_dir, "best.json"), "w") as f:
        json.dump(best, f, indent=2)

    print("Best trial:")
    print(json.dumps(best, indent=2))
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
