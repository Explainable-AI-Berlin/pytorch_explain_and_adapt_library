from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ExposureRow:
    sample_id: str
    relative_path: str
    class_id: int
    class_name: str
    region_id: str
    multiplicity: int
    source_condition: str
    seed: int


def exposure_totals(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    unique_active = 0
    total_exposures = 0
    zero_multiplicity = 0
    for row in rows:
        multiplicity = int(row["multiplicity"])
        total_exposures += multiplicity
        if multiplicity > 0:
            unique_active += 1
        else:
            zero_multiplicity += 1
    return {
        "unique_active_count": int(unique_active),
        "total_exposure_count": int(total_exposures),
        "zero_multiplicity_count": int(zero_multiplicity),
    }
