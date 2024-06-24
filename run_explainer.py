import argparse
import torch

# from peal.explainers.explainer_factory import get_explainer

from peal.data.dataset_factory import get_datasets
from peal.explainers.explainer_factory import get_explainer
from peal.global_utils import load_yaml_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--oracle_path", type=str, default=None)
    parser.add_argument("--confounder_oracle_path", type=str, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_yaml_config(args.config)

    explainer = get_explainer(config)
    explanations_dict = explainer.run(args.oracle_path, args.confounder_oracle_path)
    feedback = explainer.human_annotate_counterfactuals(**explanations_dict)
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


main()
