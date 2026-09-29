"""Explain a trained predictor with a stand-alone explainer run.

Builds the explainer that ``explainer_type`` in the config selects through
``peal.explainers.explainer_factory.get_explainer`` (chiefly the
counterfactual explainer in ``peal/explainers/counterfactual_explainer.py``),
runs it over the dataset it is configured with, has the configured teacher
annotate the explanations and renders the interpretations.

Invocation::

    python run_explainer.py --config <explainer.yaml> \\
        [--oracle_path <model.cpl>] [--confounder_oracle_path <model.cpl>]

Explainer configs live in ``configs/*/explainers``. The explanations,
collages and interpretation figures are written into the config's
``explanations_dir``. The two oracle paths are forwarded to
``explainer.run`` and the models are loaded afterwards, but the evaluation
against them that the trailing branches announce was never implemented: the
loaded oracles are unused.
"""

import argparse
import torch

from peal.explainers.explainer_factory import get_explainer
from peal.global_utils import load_yaml_config, set_random_seed


def main():
    """Build the explainer from ``--config`` and run, annotate and visualise.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--oracle_path", type=str, default=None)
    parser.add_argument("--confounder_oracle_path", type=str, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_yaml_config(args.config)
    set_random_seed(config.seed)
    explainer = get_explainer(config)
    explanations_dict = explainer.run(args.oracle_path, args.confounder_oracle_path)
    feedback = explainer.human_annotate_explanations(**explanations_dict)
    feedback = explainer.visualize_interpretations(
        feedback, explanations_dict["y_source_list"], explanations_dict["y_target_list"]
    )

    if not args.oracle_path is None:
        # evaluate model
        oracle = torch.load(args.oracle_path, map_location=device)
        # evaluate the explanations with the oracle

    if not args.confounder_oracle_path is None:
        # evaluate model when confounder is predicted by changing task config
        confounder_oracle = torch.load(args.confounder_oracle_path, map_location=device)
        # evaluate the explanations with the confounder oracle


if __name__ == "__main__":
    main()
