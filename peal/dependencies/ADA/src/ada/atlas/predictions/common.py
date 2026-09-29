from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence


PREDICTION_COLUMNS = [
    "sample_id",
    "true_label",
    "model_id",
    "predicted_label",
    "correct",
    "top1_logit",
    "top2_logit",
    "logit_margin",
    "max_probability_raw",
    "entropy_raw",
    "max_probability_calibrated",
    "nll",
]


def entropy_from_probs(probs: Sequence[float], *, eps: float = 1.0e-12) -> float:
    return -sum(float(p) * math.log(max(float(p), eps)) for p in probs)


def write_prediction_rows(rows: Iterable[dict], output_csv: str | Path) -> Path:
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PREDICTION_COLUMNS})
    return output
