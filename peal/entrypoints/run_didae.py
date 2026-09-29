"""DiDAE: Dictionary-based Interpretable Diffusion Autoencoder Explanations.

Entry point for the automated counterfactual workflow. Drives the ``DiDAE``
adaptor in ``peal/adaptors/didae.py``: sweep the sparse-dictionary directions
of a diffusion autoencoder for ones that flip the student predictor, render
and rank the candidate counterfactuals, collect feedback on the false
directions and finetune the student (optionally through CFKD) on them.

Usage::

    python run_didae.py \\
        --config configs/didae_experiments/adaptors/<run>_didae.yaml \\
        [--<any DiDAEConfig field> value]

Every ``DiDAEConfig`` field is exposed as an optional ``--flag`` overriding
the YAML value. ``PEAL_CUDA_SYNC=1`` makes CUDA launches synchronous for
debugging.

Reads the adaptor config plus the predictor, data, generator and sparse
dictionary configs it references. Writes into ``base_dir``: ``config.yaml``,
``logs/``, ``sweep_results.pt``, ``direction_collages/``,
``direction_feedback.txt``, ``cfkd_finetuning/`` and the corrected student as
``model.cpl`` (and ``model.onnx``). The exit status is 1 when the sweep found
no direction at all.
"""

import argparse
import os

# Synchronous CUDA launches make every kernel error point at its real call site,
# at the cost of serializing every kernel launch. Keep it opt-in: run with
# PEAL_CUDA_SYNC=1 when debugging a CUDA fault.
if os.environ.get("PEAL_CUDA_SYNC", "0") == "1":
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from peal.adaptors.didae import DiDAEConfig, DiDAE
from peal.global_utils import (
    load_yaml_config,
    add_class_arguments,
    integrate_arguments,
    set_random_seed,
)


def main():
    """Load the ``DiDAEConfig``, apply command-line overrides and run DiDAE.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        With status 1 when the direction sweep found nothing.
    """
    parser = argparse.ArgumentParser(
        description="DiDAE: Automated counterfactual discovery and correction"
    )
    parser.add_argument("--config", type=str, required=True)
    add_class_arguments(parser, DiDAEConfig)
    args = parser.parse_args()

    adaptor_config = load_yaml_config(args.config, DiDAEConfig)
    integrate_arguments(args, adaptor_config, exclude=["config"])

    if adaptor_config.seed is not None:
        set_random_seed(adaptor_config.seed)

    didae = DiDAE(adaptor_config=adaptor_config)
    didae.run()

    # A sweep that found no direction at all is a failure. Returning 0 for it
    # made "0 total latent flips across 1000 samples x 1536 directions" look
    # like a completed run in reproduction_scripts/reproduce_didae_results.sh.
    if getattr(didae, "sweep_found_nothing", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
