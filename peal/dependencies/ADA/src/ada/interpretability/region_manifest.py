from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ada.atlas.data.manifests import ManifestRow, load_manifest_csv
from ada.atlas.hashing import hash_rows, stable_hash


@dataclass(frozen=True)
class RegionInterpretabilityManifestConfig:
    train_cache: Path
    validation_cache: Path
    enriched_regions_dir: Path
    pilot_regions_dir: Path
    deletion_controls_dir: Path
    output_dir: Path
    baseline_predictions_csv: Path | None = None
    control_count: int = 256
    seed: int = 0
    overwrite: bool = False


def build_region_interpretability_manifest(config: RegionInterpretabilityManifestConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"interpretability manifest already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest_csv(Path(config.train_cache) / "manifest.csv")
    val_rows = load_manifest_csv(Path(config.validation_cache) / "manifest.csv")
    train_by_id = {row.sample_id: row for row in train_rows}
    val_by_id = {row.sample_id: row for row in val_rows}
    selected_regions = _read_csv(Path(config.pilot_regions_dir) / "selected_pilot_regions.csv")
    membership = _read_csv(Path(config.enriched_regions_dir) / "region_membership_k10.csv")
    assignments = _read_csv(Path(config.enriched_regions_dir) / "validation_assignments_k10.csv")
    deletion_index = _read_csv(Path(config.deletion_controls_dir) / "deletion_control_manifests.csv")
    prediction_by_id = _predictions_by_id(config.baseline_predictions_csv)

    members_by_region: dict[str, list[str]] = defaultdict(list)
    region_by_train_id: dict[str, str] = {}
    train_ids_by_class: dict[int, list[str]] = defaultdict(list)
    for row in membership:
        sid = str(row["sample_id"])
        rid = str(row["region_id"])
        members_by_region[rid].append(sid)
        region_by_train_id[sid] = rid
        train_ids_by_class[int(row["class_id"])].append(sid)
    val_ids_by_region: dict[str, list[str]] = defaultdict(list)
    for row in assignments:
        val_ids_by_region[str(row["region_id"])].append(str(row["sample_id"]))

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(int(config.seed))
    for region in selected_regions:
        region_id = str(region["region_id"])
        class_id = int(region["class_id"])
        region_member_ids = [sid for sid in members_by_region.get(region_id, []) if sid in train_by_id]
        region_member_set = set(region_member_ids)
        for sid in region_member_ids:
            rows.append(_sample_row(train_by_id[sid], region, "region_train_member", region_by_train_id.get(sid, ""), deletion_seed=""))

        outside_ids = [sid for sid in train_ids_by_class[class_id] if sid not in region_member_set and sid in train_by_id]
        high_support_ids = sorted(outside_ids)[: min(int(config.control_count), len(outside_ids))]
        random_ids = list(outside_ids)
        if len(random_ids) > int(config.control_count):
            random_ids = [str(x) for x in rng.choice(np.asarray(random_ids, dtype=object), size=int(config.control_count), replace=False).tolist()]
        for sid in high_support_ids:
            rows.append(_sample_row(train_by_id[sid], region, "same_class_high_support_control", region_by_train_id.get(sid, ""), deletion_seed=""))
        for sid in random_ids:
            rows.append(_sample_row(train_by_id[sid], region, "same_class_random_outside_region_control", region_by_train_id.get(sid, ""), deletion_seed=""))

        for sid in [sid for sid in val_ids_by_region.get(region_id, []) if sid in val_by_id]:
            rows.append(_sample_row(val_by_id[sid], region, "region_val_member", region_id, deletion_seed=""))
            pred = prediction_by_id.get(sid)
            if pred is not None:
                group = "region_val_correct_all_models" if int(pred.get("correct", "0")) else "region_val_wrong_any_model"
                rows.append(_sample_row(val_by_id[sid], region, group, region_id, deletion_seed=""))

        for manifest in deletion_index:
            if str(manifest.get("region_id", "")) != region_id:
                continue
            if str(manifest.get("control_family", "")) != "regional_drop":
                continue
            if abs(float(manifest.get("retention_level", "nan")) - 0.25) > 1.0e-9:
                continue
            exposure = {str(row["sample_id"]): int(float(row["multiplicity"])) for row in _read_csv(Path(str(manifest["exposure_manifest_csv"])))}
            deletion_seed = str(manifest.get("seed", ""))
            for sid in region_member_ids:
                if exposure.get(sid, 0) > 0:
                    rows.append(_sample_row(train_by_id[sid], region, "q025_retained_anchor", region_id, deletion_seed=deletion_seed, retained=1, deleted=0))
                else:
                    rows.append(_sample_row(train_by_id[sid], region, "q025_deleted_real", region_id, deletion_seed=deletion_seed, retained=0, deleted=1))

    result_hash = hash_rows(rows, prefix="region-interpretability-manifest")
    csv_path = output / "samples.csv"
    parquet_path = output / "samples.parquet"
    _write_csv(csv_path, rows)
    parquet_written = _try_write_parquet(parquet_path, rows)
    metadata = {
        "artifact_id": stable_hash(
            {
                "result_hash": result_hash,
                "train_cache": str(config.train_cache),
                "validation_cache": str(config.validation_cache),
                "pilot_regions_dir": str(config.pilot_regions_dir),
                "deletion_controls_dir": str(config.deletion_controls_dir),
            },
            prefix="region-interpretability-artifact",
        ),
        "result_hash": result_hash,
        "config": {
            "train_cache": str(config.train_cache),
            "validation_cache": str(config.validation_cache),
            "enriched_regions_dir": str(config.enriched_regions_dir),
            "pilot_regions_dir": str(config.pilot_regions_dir),
            "deletion_controls_dir": str(config.deletion_controls_dir),
            "baseline_predictions_csv": "" if config.baseline_predictions_csv is None else str(config.baseline_predictions_csv),
            "control_count": int(config.control_count),
            "seed": int(config.seed),
        },
        "control_selection": {
            "same_class_high_support_control": "deterministic same-class outside-region sample_id order; per-sample support scores are not in region_membership_k10.csv",
            "same_class_random_outside_region_control": "deterministic random same-class outside-region sample from frozen seed",
        },
        "summary": {
            "rows": len(rows),
            "regions": len(selected_regions),
            "groups": {group: sum(1 for row in rows if row["analysis_group"] == group) for group in sorted({str(row["analysis_group"]) for row in rows})},
            "parquet_written": parquet_written,
        },
        "outputs": {
            "samples_csv": str(csv_path),
            "samples_parquet": str(parquet_path) if parquet_written else "",
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def _sample_row(
    row: ManifestRow,
    region: Mapping[str, object],
    group: str,
    assigned_region_id: str,
    *,
    deletion_seed: str,
    retained: int = 0,
    deleted: int = 0,
) -> dict[str, object]:
    payload = asdict(row)
    payload.update(
        {
            "region_id": str(region["region_id"]),
            "assigned_region_id": str(assigned_region_id),
            "pilot_type": str(region.get("pilot_type", "")),
            "region_group": str(region.get("pilot_type", "")),
            "analysis_group": str(group),
            "support_percentile": str(region.get("member_support_pct_median", "")),
            "deletion_seed": str(deletion_seed),
            "retained_at_q025": int(retained),
            "deleted_at_q025": int(deleted),
            "selected_region_class_id": int(region["class_id"]),
            "selected_region_class_name": str(region["class_name"]),
        }
    )
    return payload


def _predictions_by_id(path: Path | None) -> dict[str, Mapping[str, str]]:
    if path is None or not str(path) or not Path(path).exists():
        return {}
    return {str(row["sample_id"]): row for row in _read_csv(Path(path))}


def _try_write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> bool:
    try:
        import pandas as pd

        pd.DataFrame(list(rows)).to_parquet(path, index=False)
        return True
    except Exception:
        return False


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
