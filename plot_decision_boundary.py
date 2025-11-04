import argparse
import torch
import os
import pathlib

from torch.utils.data import DataLoader

from peal.data.custom_datasets import ColoredMnistConfig
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig
from peal.global_utils import load_yaml_config


def main(
    dataset,
    file_suffix,
    original_model=None,
    corrected_model=None,
    save_plots_to="./plots",
):
    pathlib.Path(save_plots_to).mkdir(exist_ok=True, parents=True)
    if original_model is not None:
        dataset.visualize_decision_boundary(
            original_model,
            32,
            device,
            os.path.join(save_plots_to, f"decision_boundary_original{file_suffix}.png"),
        )
    if corrected_model is not None:
        dataset.visualize_decision_boundary(
            corrected_model,
            32,
            device,
            os.path.join(
                save_plots_to, f"decision_boundary_corrected{file_suffix}.png"
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=str, required=True)
    parser.add_argument("--model-uncorrected", type=str, default=None)
    parser.add_argument("--model-corrected", type=str, default=None)
    parser.add_argument("--file-suffix", type=str, default="")
    parser.add_argument("--save-to", type=str, default="./plots")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config = load_yaml_config(args.data_config)
    dataset = get_datasets(data_config)[2]

    original_model = None
    if args.model_uncorrected is not None:
        original_model = torch.load(args.model_uncorrected, map_location=device)
        if hasattr(original_model, "model"):
            original_model = original_model.model

    corrected_model = None
    if args.model_corrected is not None:
        corrected_model = torch.load(args.model_corrected, map_location=device)
        if hasattr(corrected_model, "model"):
            corrected_model = corrected_model.model

    if original_model is None and corrected_model is None:
        raise ValueError(
            "Specify at least one of --model-corrected or --model-uncorrected"
        )

    main(
        dataset,
        args.file_suffix,
        original_model=original_model,
        corrected_model=corrected_model,
        save_plots_to=args.save_to,
    )
