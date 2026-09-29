from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EligibilityThresholds:
    min_train_count: int = 100
    min_val_count: int = 10
    same_class_purity: float = 0.90
    min_positive_class_margin: float = 0.0
    max_duplicate_fraction: float = 0.10


def parse_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def is_region_eligible(row: Mapping[str, object], thresholds: EligibilityThresholds) -> bool:
    train_count = int(row.get("train_count", 0))
    val_count = int(row.get("validation_count", 0))
    purity = parse_float(row.get("same_class_purity"))
    margin = parse_float(row.get("class_margin"))
    duplicate_fraction = parse_float(row.get("duplicate_fraction"))
    return (
        train_count >= int(thresholds.min_train_count)
        and val_count >= int(thresholds.min_val_count)
        and purity is not None
        and purity >= float(thresholds.same_class_purity)
        and margin is not None
        and margin > float(thresholds.min_positive_class_margin)
        and duplicate_fraction is not None
        and duplicate_fraction <= float(thresholds.max_duplicate_fraction)
    )


def annotate_eligibility(
    rows: Sequence[dict[str, object]],
    thresholds: EligibilityThresholds,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item["eligible_primary"] = int(is_region_eligible(item, thresholds))
        out.append(item)
    return out


def select_eligible_regions(
    rows: Sequence[Mapping[str, object]],
    thresholds: EligibilityThresholds,
    *,
    max_regions: int | None = None,
) -> list[dict[str, object]]:
    eligible = [dict(row) for row in rows if is_region_eligible(row, thresholds)]
    eligible.sort(
        key=lambda row: (
            -int(row.get("validation_count", 0)),
            -int(row.get("train_count", 0)),
            -(parse_float(row.get("class_margin")) or 0.0),
            str(row.get("region_id", "")),
        )
    )
    if max_regions is not None:
        return eligible[: int(max_regions)]
    return eligible
