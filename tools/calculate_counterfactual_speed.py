#!/usr/bin/env python3
"""Benchmark how many counterfactuals per second a CFKD config generates.

Builds the student, generator and explainer of a CFKD adaptor config the way
``CFKD`` in ``peal/adaptors/counterfactual_knowledge_distillation.py`` does
(``get_predictor``, ``get_generator``, ``get_explainer``), then calls
``explainer.explain_batch`` on validation batches and reads the
``counterfactuals_per_second`` that the explainer reports. Feedback,
validation statistics and finetuning are skipped, and no distilled predictor
is trained: a config whose explainer needs one must already have it saved
under ``<base_dir>/<iteration>/explainer/distilled_predictor/model.cpl``.

Invocation::

    python tools/calculate_counterfactual_speed.py --config <cfkd_adaptor.yaml> \\
        [--num-batches 1] [--batch-size N] [--output <result.json>]

Collages are rendered into a temporary directory and discarded. The result
(speed, counts, explainer type, device, host, timestamp) is written as JSON
to ``--output``, or next to the config as
``<config stem>_counterfactual_speed.json``.
"""

import argparse
import copy
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch

import sys

# Runnable from a clone without installing PEAL: put the repository root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peal.adaptors.counterfactual_knowledge_distillation import CFKDConfig
from peal.architectures.predictors import get_predictor
from peal.data.dataloaders import (
    DataloaderMixer,
    WeightedDataloaderList,
    create_dataloaders_from_datasource,
)
from peal.explainers.explainer_factory import get_explainer
from peal.generators.generator_factory import get_generator
from peal.global_utils import load_yaml_config, set_random_seed
from peal.sparse_dictionaries.sparse_dictionary_factory import (
    get_sparse_dictionary,
)


def parse_args():
    """Parse ``--config``, ``--num-batches``, ``--batch-size`` and ``--output``.

    Returns
    -------
    argparse.Namespace
        The parsed arguments; ``config`` and ``output`` are ``Path`` objects.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark counterfactual generation speed for a CFKD config without "
            "running feedback, validation statistics, or finetuning."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--num-batches",
        default=1,
        type=int,
        help="Number of batches to benchmark (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help="Override the adaptor batch size.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output JSON path. Defaults to beside the config file.",
    )
    return parser.parse_args()


def get_timestep_respacing(explainer_config):
    """Derive the generator's timestep respacing from the explainer config.

    Parameters
    ----------
    explainer_config : ExplainerConfig
        Uses ``num_discretization_steps / sampling_time_fraction`` when both
        exist, otherwise ``timestep_respacing``.

    Returns
    -------
    int or None
        The number of diffusion steps, or ``None`` when the config has
        neither.
    """
    if hasattr(explainer_config, "num_discretization_steps") and hasattr(
        explainer_config, "sampling_time_fraction"
    ):
        return int(
            explainer_config.num_discretization_steps
            / explainer_config.sampling_time_fraction
        )

    return getattr(explainer_config, "timestep_respacing", None)


def unpack_batch(sample):
    """Return ``(x, y)`` from a dict batch or an ``(x, y)`` tuple batch.

    Parameters
    ----------
    sample : dict or tuple
        A dataloader batch; a list-valued ``y`` is reduced to its first
        entry.

    Returns
    -------
    tuple of torch.Tensor
        Inputs and labels.
    """
    if isinstance(sample, dict):
        return sample["x"], sample["y"]

    x, y = sample
    if isinstance(y, (list, tuple)):
        y = y[0]
    return x, y


def create_explainer(config, device):
    """Instantiate the predictor, dataloaders, generator and explainer.

    Mirrors the setup in ``CFKD.__init__`` with tracking, generator
    validation and image saving switched off.

    Parameters
    ----------
    config : CFKDConfig
        The adaptor config; its ``training.val_batch_size`` is set to
        ``config.batch_size``.
    device : str
        Device for the predictor and generator.

    Returns
    -------
    tuple
        ``(predictor, explainer, val_dataloader)``.
    """
    predictor, _ = get_predictor(config.student, device=device)
    predictor.eval()

    config.training.val_batch_size = config.batch_size
    train_dataloader, val_dataloader, _ = create_dataloaders_from_datasource(
        config=config,
        test_config=config.test_data if config.test_data is not None else config.data,
        enable_hints=False,
    )
    dataloader_mixer = DataloaderMixer(config.training, train_dataloader)
    validation_dataloaders = WeightedDataloaderList([val_dataloader])

    generator = get_generator(
        generator=config.generator,
        device=device,
        predictor_dataset=val_dataloader.dataset,
        timestep_respacing=get_timestep_respacing(config.explainer),
    )
    if (
        config.sparse_dictionary is not None
        and getattr(generator, "sparse_dictionary", None) is None
    ):
        generator.sparse_dictionary = get_sparse_dictionary(config.sparse_dictionary)

    explainer_config = copy.deepcopy(config.explainer)
    explainer_config.tracking_level = 0
    explainer_config.validate_generator = False
    if hasattr(explainer_config, "save_images"):
        explainer_config.save_images = False

    explainer = get_explainer(
        explainer=explainer_config,
        predictor=predictor,
        generator=generator,
        input_type=config.data.input_type,
        datasource=[dataloader_mixer, validation_dataloaders],
        tracking_level=0,
    )
    return predictor, explainer, val_dataloader


def find_saved_explainer_distilled_predictor(config):
    """Locate a distilled predictor saved by an earlier CFKD run.

    Looks at ``<base_dir>/<current_iteration>/`` and ``<base_dir>/0/`` first,
    then at every numbered iteration directory, newest first.

    Parameters
    ----------
    config : CFKDConfig
        The adaptor config.

    Returns
    -------
    pathlib.Path or None
        Path to ``explainer/distilled_predictor/model.cpl``, or ``None`` when
        the explainer config needs no distilled predictor.

    Raises
    ------
    FileNotFoundError
        If a distilled predictor is required but none is saved.
    """
    if getattr(config.explainer, "distilled_predictor", None) is None:
        return None

    base_dir = Path(config.base_dir).expanduser().resolve()
    preferred_iterations = [int(config.current_iteration), 0]
    checked_paths = []

    for iteration in dict.fromkeys(preferred_iterations):
        candidate = (
            base_dir
            / str(iteration)
            / "explainer"
            / "distilled_predictor"
            / "model.cpl"
        )
        checked_paths.append(candidate)
        if candidate.is_file():
            return candidate

    for iteration_dir in sorted(base_dir.glob("[0-9]*"), reverse=True):
        candidate = iteration_dir / "explainer" / "distilled_predictor" / "model.cpl"
        if candidate in checked_paths:
            continue
        checked_paths.append(candidate)
        if candidate.is_file():
            return candidate

    checked = "\n".join(f"  - {path}" for path in checked_paths)
    raise FileNotFoundError(
        "The config requires a distilled predictor for counterfactual generation, "
        "but no saved model was found. The speed benchmark will not train one.\n"
        f"Checked:\n{checked}"
    )


def stage_saved_distilled_predictor(model_path, temp_dir):
    """Symlink the saved distilled predictor into the temporary run directory.

    The explainer looks for ``<explainer_path>/explainer/distilled_predictor/
    model.cpl``; linking it there stops the explainer from training one.

    Parameters
    ----------
    model_path : pathlib.Path or None
        The saved ``model.cpl``; nothing happens for ``None``.
    temp_dir : str
        The temporary directory used as ``explainer_path``.

    Returns
    -------
    None
    """
    if model_path is None:
        return

    staged_path = Path(temp_dir) / "explainer" / "distilled_predictor" / "model.cpl"
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.symlink_to(model_path)


def create_counterfactual_batch(
    sample, predictor, explainer, output_size, batch_size, device
):
    """Turn a validation batch into the dict ``explain_batch`` expects.

    The source class is the predictor's argmax and the target class is the
    next class modulo ``output_size``; the start confidence of the target is
    the temperature-scaled softmax probability.

    Parameters
    ----------
    sample : dict or tuple
        A dataloader batch.
    predictor : torch.nn.Module
        The student.
    explainer : ExplainerInterface
        Provides ``explainer_config.temperature``.
    output_size : int
        Number of classes.
    batch_size : int
        The batch is truncated to this many samples.
    device : str
        Device for the predictor forward pass.

    Returns
    -------
    dict
        ``x_list``, ``y_list``, ``y_source_list``, ``y_target_list``,
        ``y_target_start_confidence_list`` and ``idx_list``.
    """
    x, y = unpack_batch(sample)
    x = x[:batch_size]
    y = y[: len(x)]

    with torch.no_grad():
        logits = predictor(x.to(device)).detach().cpu()
        y_source = logits.argmax(dim=-1)
        y_target = (y_source + 1) % output_size
        y_target_start_confidence = (
            torch.softmax(logits / explainer.explainer_config.temperature, dim=-1)
            .gather(1, y_target.unsqueeze(1))
            .squeeze(1)
        )

    return {
        "x_list": x,
        "y_list": y,
        "y_source_list": y_source,
        "y_target_list": y_target,
        "y_target_start_confidence_list": y_target_start_confidence,
        "idx_list": [0] * len(x),
    }


def main():
    """Run the benchmark and write the JSON result.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        For a non-positive ``--num-batches`` or ``--batch-size``.
    FileNotFoundError
        If the config file does not exist.
    RuntimeError
        If the explainer does not report a generation speed.
    """
    args = parse_args()
    if args.num_batches < 1:
        raise ValueError("--num-batches must be at least 1")

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_yaml_config(str(config_path), CFKDConfig)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        config.batch_size = args.batch_size
    if config.seed is not None:
        set_random_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor, explainer, val_dataloader = create_explainer(config, device)
    distilled_predictor_path = find_saved_explainer_distilled_predictor(config)
    output_size = (
        config.task.output_channels
        if config.task.output_channels is not None
        else config.data.output_size[0]
    )

    generated_count = 0
    elapsed_seconds = 0.0
    batch_speeds = []
    dataloader_iterator = iter(val_dataloader)

    with tempfile.TemporaryDirectory(prefix="peal-cf-speed-") as temp_dir:
        stage_saved_distilled_predictor(distilled_predictor_path, temp_dir)
        for batch_idx in range(args.num_batches):
            try:
                sample = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(val_dataloader)
                sample = next(dataloader_iterator)

            batch = create_counterfactual_batch(
                sample,
                predictor,
                explainer,
                output_size,
                config.batch_size,
                device,
            )
            explainer.counterfactuals_per_second = None
            result = explainer.explain_batch(
                batch=batch,
                base_path=str(Path(temp_dir) / "collages"),
                explainer_path=temp_dir,
                start_idx=(batch_idx + 1) * config.batch_size,
                mode="validation",
            )
            batch_generated_count = len(result["x_counterfactual_list"])
            batch_speed = explainer.counterfactuals_per_second
            if batch_speed is None:
                raise RuntimeError(
                    "The explainer did not report counterfactual generation speed."
                )

            batch_speeds.append(float(batch_speed))
            generated_count += batch_generated_count
            elapsed_seconds += batch_generated_count / batch_speed

    counterfactuals_per_second = generated_count / elapsed_seconds
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else config_path.with_name(f"{config_path.stem}_counterfactual_speed.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "config": str(config_path),
        "counterfactuals_per_second": counterfactuals_per_second,
        "generated_counterfactuals": generated_count,
        "elapsed_seconds": elapsed_seconds,
        "num_batches": args.num_batches,
        "batch_size": config.batch_size,
        "batch_counterfactuals_per_second": batch_speeds,
        "explainer_type": config.explainer.explainer_type,
        "distilled_predictor": (
            str(distilled_predictor_path)
            if distilled_predictor_path is not None
            else None
        ),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "host": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"Counterfactuals per second: {counterfactuals_per_second:.6f}")
    print(f"Generated counterfactuals: {generated_count}")
    print(f"Elapsed seconds: {elapsed_seconds:.6f}")
    print(f"Saved result: {output_path}")


if __name__ == "__main__":
    main()
