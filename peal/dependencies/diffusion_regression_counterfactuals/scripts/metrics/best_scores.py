import pandas as pd
import argparse
from typing import List, Dict, Callable

# Define the sorting functions for each metric
# The callable takes the key and the bool indicates if the sorting is ascending
# By default, the sorting is descending
SORTING_KEYS: Dict[str, tuple[Callable[[pd.Series], pd.Series], bool]] = {
    "AceFVA_MEAN": (lambda x: x.apply(lambda r: eval(r)[1]), False),
    "AceFlipRateYoungOld_MEAN": (lambda x: x, False),
    "AceMNAC_MEAN": (lambda x: x, True),
    "FID": (lambda x: x, True),
    "LPIPS_MEAN": (lambda x: x, False),
    "steps": (lambda x: x, True),
    "success_rate": (lambda x: x, False),
    "success_rate_oracle": (lambda x: x, False),
    "y_initial_confidence": (lambda x: x, True),
    "y_initial_confidence_oracle": (lambda x: x, True),
    "y_final_confidence": (lambda x: x, True),
    "y_final_confidence_oracle": (lambda x: x, True),
    "y_final_pred_oracle_mae": (lambda x: x, True),
    "y_initial_confidences": (lambda x: x, True),
    "y_initial_pred_oracle_mae": (lambda x: x, True),
    "confidence_reduction": (lambda x: x, True),
    "confidence_reduction_oracle": (lambda x: x, True),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the best score for each metric.")
    parser.add_argument(
        "csv_files", type=str, nargs="+", help="Paths to the input CSV files"
    )
    return parser.parse_args()


def extract_best_scores(
    csv_files: List[str],
) -> dict[str, pd.DataFrame]:
    # Read and merge the CSV files
    df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    # Remove all columns that have STD in the name
    df.drop(columns=[col for col in df.columns if "STD" in col], inplace=True)
    metrics = df.columns[1:]  # First column is the path

    # Get the basename of the path (by default the counterfactuals are saved to /cf)
    df["path"] = df["path"].apply(lambda x: x.replace("/cf", "").split("/")[-1])

    best_scores: Dict[str, pd.DataFrame] = {}

    # Group by metric and find the best scores
    for metric in metrics:
        key_func, ascending = SORTING_KEYS[metric]
        best_scores[metric] = df.sort_values(
            by=metric, key=key_func, ascending=ascending
        )

    return best_scores


def main() -> None:
    args = parse_arguments()
    # Process the file and extract the best scores
    best_scores_df = extract_best_scores(args.csv_files)

    # Display the results
    print("Best scores for each metric:")
    for metric, df in best_scores_df.items():
        print(f"\n{metric}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
