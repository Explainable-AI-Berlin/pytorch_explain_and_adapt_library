from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join VAE kNN support scores with posterior metric sidecars.")
    parser.add_argument("--support-csv", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--strict", action="store_true", help="Require every support row to have posterior metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = join_vae_support_metrics(
        support_csv=args.support_csv,
        metrics_csv=args.metrics_csv,
        output_csv=args.output_csv,
        strict=bool(args.strict),
    )
    print(json.dumps({"output_csv": str(output)}, indent=2, sort_keys=True))


def join_vae_support_metrics(
    *,
    support_csv: str | Path,
    metrics_csv: str | Path,
    output_csv: str | Path,
    strict: bool = False,
) -> Path:
    support_path = Path(support_csv)
    metrics_path = Path(metrics_csv)
    output = Path(output_csv)
    metrics = _read_keyed(metrics_path)
    with support_path.open("r", newline="") as f:
        support_reader = csv.DictReader(f)
        support_rows = list(support_reader)
        support_columns = list(support_reader.fieldnames or [])
    metric_columns = [
        "posterior_kl_raw",
        "posterior_kl",
        "latent_norm_raw",
        "latent_norm_scaled_mean",
        "latent_norm_unscaled_mean",
        "latent_mean_scalar",
        "latent_std_scalar",
        "reconstruction_mse",
    ]
    columns = support_columns + [column for column in metric_columns if column not in support_columns]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in support_rows:
            sample_id = str(row["sample_id"])
            metric = metrics.get(sample_id)
            if metric is None:
                if strict:
                    raise ValueError(f"missing VAE metrics for sample_id={sample_id}")
                metric = {}
            out = dict(row)
            for column in metric_columns:
                out[column] = metric.get(column, "")
            writer.writerow(out)
    return output


def _read_keyed(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate metrics row for sample_id={sample_id}")
            rows[sample_id] = dict(row)
    return rows


if __name__ == "__main__":
    main()
