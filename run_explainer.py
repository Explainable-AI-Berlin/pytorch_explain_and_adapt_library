import argparse
import torch

from peal.explainers.explainer_factory import get_explainer

from peal.data.dataset_factory import get_datasets
from peal.global_utils import load_yaml_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--oracle_path", type=str, default=None)
    parser.add_argument("--confounder_oracle_path", type=str, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_yaml_config(args.config)
    dataset_config = load_yaml_config(args.dataset_config)

    explainer = get_explainer(config)
    dataset = get_datasets(dataset_config)
    explanations_dict = explainer.run(dataset, args.oracle_path, args.confounder_oracle_path)

    if not args.oracle_path is None:
        # evaluate model
        oracle = torch.load(args.oracle_path, map_location=device)
        # evaluate the explanations with the oracle

    if not args.confounder_oracle_path is None:
        # evaluate model when confounder is predicted by changing task config
        confounder_oracle = torch.load(args.confounder_oracle_path, map_location=device)
        # evaluate the explanations with the confounder oracle

main()
