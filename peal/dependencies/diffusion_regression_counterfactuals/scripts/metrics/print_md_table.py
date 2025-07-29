import pandas as pd
import argparse
import os


def preprocess(df):
    df = df.drop(columns=[c for c in df.columns if "STD" in c])

    def cleanup_path(s):
        s = s.replace("/cf", "")
        s = os.path.basename(s)
        # s = s.replace("CelebaHQ_FR-method=PGD-step=1.0-", "")
        # s = s.replace("CelebaHQ_FR-lr=0.002-bt=10-", "")
        return s

    df["path"] = df["path"].apply(cleanup_path)

    selected_columns = [
        "path",
        "y_final_confidence",
        "y_final_confidence_oracle",
        "y_final_pred_oracle_mae",
        "success_rate",
        "AceFVA_MEAN",
        "AceMNAC_MEAN",
        "FID",
    ]
    df = df[selected_columns].sort_values("y_final_confidence")

    for col in selected_columns:
        if col != "path" and col != "AceFVA_MEAN":
            df[col] = df[col].astype(float)

    return df


def print_rows(df):
    print(
        "| Path | Final Confidence | Final Confidence Oracle | Oracle MAE | Success Rate | FVA | AceMNAC | FID |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, row in df.iterrows():
        fva = [round(v, 3) for v in eval(row["AceFVA_MEAN"])]
        row_data = [
            row["path"],
            f"{row['y_final_confidence']:.3f}",
            f"{row['y_final_confidence_oracle']:.3f}",
            f"{row['y_final_pred_oracle_mae']:.3f}",
            f"{row['success_rate']:.3f}",
            f"{fva}",
            f"{row['AceMNAC_MEAN']:.3f}",
            f"{row['FID']:.3f}",
        ]
        print("| " + " | ".join(row_data) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and print markdown table from CSV summary file."
    )
    parser.add_argument("csv_file", type=str, help="Path to the CSV file to process")
    args = parser.parse_args()
    assert os.path.exists(args.csv_file), f"File {args.csv_file} does not exist."

    df = pd.read_csv(args.csv_file)
    df = preprocess(df)
    print_rows(df)
