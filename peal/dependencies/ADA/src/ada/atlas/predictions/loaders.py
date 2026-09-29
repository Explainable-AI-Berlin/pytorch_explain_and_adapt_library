from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence


REQUIRED_PREDICTION_COLUMNS = {"sample_id", "predicted_label", "confidence_raw"}


def load_prediction_csv(path: str | Path, *, required_columns: set[str] | None = None) -> list[dict[str, str]]:
    required = required_columns or REQUIRED_PREDICTION_COLUMNS
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise ValueError(f"Prediction table missing required columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    assert_unique_sample_ids(rows)
    return rows


def assert_unique_sample_ids(rows: Sequence[Mapping[str, str]]) -> None:
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError("Prediction row has empty sample_id")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in prediction table: {sample_id}")
        seen.add(sample_id)


def strict_join_by_sample_id(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    *,
    right_name: str = "prediction",
) -> list[dict[str, object]]:
    right_by_id: dict[str, Mapping[str, object]] = {}
    for row in right:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"{right_name} row has empty sample_id")
        if sample_id in right_by_id:
            raise ValueError(f"Duplicate sample_id in {right_name}: {sample_id}")
        right_by_id[sample_id] = row

    joined: list[dict[str, object]] = []
    missing: list[str] = []
    for row in left:
        sample_id = str(row.get("sample_id", ""))
        match = right_by_id.get(sample_id)
        if match is None:
            missing.append(sample_id)
            continue
        merged = dict(row)
        for key, value in match.items():
            if key != "sample_id":
                merged[key] = value
        joined.append(merged)

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing {right_name} rows for {len(missing)} sample IDs; first: {preview}")
    return joined
