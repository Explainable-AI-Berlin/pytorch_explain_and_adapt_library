from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ada.actionability.exposure_manifest import ExposureRow, exposure_totals
from ada.atlas.data.manifests import load_manifest_csv
from ada.atlas.hashing import hash_rows, stable_hash


CONTROL_FAMILIES = (
    "regional_drop",
    "same_class_random_drop",
    "global_random_drop",
    "same_class_count_preserving_replacement",
)


@dataclass(frozen=True)
class DeletionControlConfig:
    train_cache: Path
    enriched_regions_dir: Path
    pilot_regions_dir: Path
    output_dir: Path
    experiment_id: str = "e4a_in100_k10_causal_pilot_deletion_controls"
    superseded_deletion_dir: Path | None = None
    retention_levels: tuple[float, ...] = (1.0, 0.25, 0.0)
    seeds: tuple[int, ...] = (0, 1, 2)
    overwrite: bool = False


def build_deletion_controls(config: DeletionControlConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"deletion-control artifact already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest_csv(Path(config.train_cache) / "manifest.csv")
    sample_ids = [row.sample_id for row in train_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("train manifest contains duplicate sample IDs")
    train_by_id = {row.sample_id: row for row in train_rows}
    original_manifest_hash = hash_rows((asdict(row) for row in train_rows), prefix="manifest")
    class_counts = Counter(int(row.class_id) for row in train_rows)

    enrich_metadata = _load_json(Path(config.enriched_regions_dir) / "metadata.json")
    pilot_metadata = _load_json(Path(config.pilot_regions_dir) / "metadata.json")
    membership = _read_csv(Path(config.enriched_regions_dir) / "region_membership_k10.csv")
    selected_regions = _read_csv(Path(config.pilot_regions_dir) / "selected_pilot_regions.csv")
    if not selected_regions:
        raise ValueError("no selected pilot regions found")

    region_by_sample = {str(row["sample_id"]): str(row["region_id"]) for row in membership}
    members_by_region: dict[str, list[str]] = {}
    for row in membership:
        members_by_region.setdefault(str(row["region_id"]), []).append(str(row["sample_id"]))

    retention_levels = _normalize_retention_levels(config.retention_levels)
    seeds = tuple(int(seed) for seed in config.seeds)
    index_rows: list[dict[str, object]] = []
    for region in selected_regions:
        region_id = str(region["region_id"])
        class_id = int(region["class_id"])
        region_member_ids = sorted(members_by_region.get(region_id, []))
        if not region_member_ids:
            raise ValueError(f"selected region has no members: {region_id}")
        same_class_outside = sorted(
            row.sample_id for row in train_rows if int(row.class_id) == class_id and row.sample_id not in set(region_member_ids)
        )
        for retention in retention_levels:
            retain_count = _retained_count(len(region_member_ids), retention)
            for seed in seeds:
                retained_region, deleted_region = _split_ids(
                    region_member_ids,
                    retain_count=retain_count,
                    seed=_derived_seed(seed, region_id, retention, "regional"),
                )
                if math.isclose(float(retention), 1.0):
                    index_rows.append(
                        _write_manifest(
                            output,
                            train_rows=train_rows,
                            region_by_sample=region_by_sample,
                            region=region,
                            control_family="baseline",
                            retention=retention,
                            seed=seed,
                            deleted_ids=[],
                            replacement_ids=[],
                            original_manifest_hash=original_manifest_hash,
                            source_metadata=(enrich_metadata, pilot_metadata),
                            class_counts=class_counts,
                        )
                    )
                    continue
                deleted_count = len(deleted_region)
                if deleted_count <= 0:
                    raise AssertionError("non-baseline retention produced no deletions")
                for family in CONTROL_FAMILIES:
                    if family == "regional_drop":
                        deleted_ids = deleted_region
                        replacement_ids: list[str] = []
                    elif family == "same_class_random_drop":
                        deleted_ids = _choose_without_replacement(
                            same_class_outside,
                            deleted_count,
                            seed=_derived_seed(seed, region_id, retention, family),
                        )
                        replacement_ids = []
                    elif family == "global_random_drop":
                        deleted_ids = _choose_without_replacement(
                            sample_ids,
                            deleted_count,
                            seed=_derived_seed(seed, region_id, retention, family),
                        )
                        replacement_ids = []
                    elif family == "same_class_count_preserving_replacement":
                        deleted_ids = deleted_region
                        replacement_ids = _choose_with_replacement(
                            same_class_outside,
                            deleted_count,
                            seed=_derived_seed(seed, region_id, retention, family),
                        )
                    else:
                        raise AssertionError(f"unknown control family: {family}")
                    index_rows.append(
                        _write_manifest(
                            output,
                            train_rows=train_rows,
                            region_by_sample=region_by_sample,
                            region=region,
                            control_family=family,
                            retention=retention,
                            seed=seed,
                            deleted_ids=deleted_ids,
                            replacement_ids=replacement_ids,
                            original_manifest_hash=original_manifest_hash,
                            source_metadata=(enrich_metadata, pilot_metadata),
                            class_counts=class_counts,
                        )
                    )

    index_hash = hash_rows(index_rows, prefix="deletion-control-index")
    config_payload = {
        "train_cache": str(config.train_cache),
        "enriched_regions_dir": str(config.enriched_regions_dir),
        "pilot_regions_dir": str(config.pilot_regions_dir),
        "output_dir": str(config.output_dir),
        "experiment_id": config.experiment_id,
        "superseded_deletion_dir": str(config.superseded_deletion_dir or ""),
        "retention_levels": list(retention_levels),
        "seeds": list(seeds),
    }
    config_hash = stable_hash(config_payload, prefix="deletion-control-config")
    artifact_id = stable_hash(
        {
            "config_hash": config_hash,
            "index_hash": index_hash,
            "enrichment_artifact_id": enrich_metadata.get("artifact_id", ""),
            "pilot_artifact_id": pilot_metadata.get("artifact_id", ""),
        },
        prefix="deletion-control-artifact",
    )
    _write_csv(output / "deletion_control_manifests.csv", index_rows)
    metadata = {
        "artifact_id": artifact_id,
        "experiment_id": config.experiment_id,
        "config": config_payload,
        "config_hash": config_hash,
        "index_hash": index_hash,
        "original_manifest_hash": original_manifest_hash,
        "enrichment_artifact_id": enrich_metadata.get("artifact_id", ""),
        "pilot_region_artifact_id": pilot_metadata.get("artifact_id", ""),
        "source_region_artifact_id": enrich_metadata.get("source_region_artifact_id", ""),
        "source_validation_assignment_hash": enrich_metadata.get("source_validation_assignment_hash", ""),
        "outputs": {"deletion_control_manifests_csv": str(output / "deletion_control_manifests.csv")},
        "summary": {
            "selected_regions": len(selected_regions),
            "retention_levels": list(retention_levels),
            "seeds": list(seeds),
            "manifest_count": len(index_rows),
            "by_control_family": _count_by(index_rows, "control_family"),
        },
        "leakage_declaration": (
            "Deletion-control manifests use train sample IDs, train labels, train K=10 memberships, "
            "and selected train-geometry pilot regions. Frozen validation assignments are carried only "
            "by artifact hash for evaluation; no validation errors are used."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    _mark_superseded(config.superseded_deletion_dir, replacement_artifact_id=artifact_id)
    completed.write_text("ok\n")
    return metadata


def _write_manifest(
    output: Path,
    *,
    train_rows: Sequence[object],
    region_by_sample: Mapping[str, str],
    region: Mapping[str, object],
    control_family: str,
    retention: float,
    seed: int,
    deleted_ids: Sequence[str],
    replacement_ids: Sequence[str],
    original_manifest_hash: str,
    source_metadata: tuple[Mapping[str, object], Mapping[str, object]],
    class_counts: Mapping[int, int],
) -> dict[str, object]:
    region_id = str(region["region_id"])
    retention_label = _retention_label(retention)
    seed_label = f"seed_{int(seed):03d}"
    out_dir = output / region_id / str(control_family) / retention_label / seed_label
    out_dir.mkdir(parents=True, exist_ok=True)

    deleted_set = set(str(sid) for sid in deleted_ids)
    multiplicities = {row.sample_id: 0 if row.sample_id in deleted_set else 1 for row in train_rows}
    for sid in replacement_ids:
        sid = str(sid)
        if sid in deleted_set:
            raise AssertionError("replacement pool must not include deleted IDs")
        multiplicities[sid] = int(multiplicities.get(sid, 0)) + 1

    rows = [
        asdict(
            ExposureRow(
                sample_id=row.sample_id,
                relative_path=row.relative_path,
                class_id=int(row.class_id),
                class_name=row.class_name,
                region_id=str(region_by_sample.get(row.sample_id, "")),
                multiplicity=int(multiplicities[row.sample_id]),
                source_condition=str(control_family),
                seed=int(seed),
            )
        )
        for row in train_rows
    ]
    totals = exposure_totals(rows)
    if len(rows) != len({str(row["sample_id"]) for row in rows}):
        raise AssertionError("exposure manifest contains duplicate sample IDs")
    if control_family == "same_class_count_preserving_replacement" and totals["total_exposure_count"] != len(train_rows):
        raise AssertionError("count-preserving replacement must preserve total exposure count")
    if control_family == "baseline" and any(int(row["multiplicity"]) != 1 for row in rows):
        raise AssertionError("baseline must assign multiplicity 1 to every sample")

    active_class_counts = Counter()
    exposure_class_counts = Counter()
    for row in rows:
        multiplicity = int(row["multiplicity"])
        if multiplicity > 0:
            active_class_counts[int(row["class_id"])] += 1
        exposure_class_counts[int(row["class_id"])] += multiplicity
    if control_family == "same_class_count_preserving_replacement":
        expected_class_counts = {int(key): int(value) for key, value in class_counts.items()}
        observed_class_counts = {int(key): int(value) for key, value in exposure_class_counts.items()}
        if observed_class_counts != expected_class_counts:
            raise AssertionError("count-preserving replacement must preserve class exposure counts")

    _write_csv(out_dir / "exposure_manifest.csv", rows)
    _write_lines(out_dir / "deleted_sample_ids.txt", sorted(deleted_set))
    _write_lines(out_dir / "replacement_sample_ids.txt", sorted(str(sid) for sid in replacement_ids))
    enrich_metadata, pilot_metadata = source_metadata
    payload = {
        "manifest_type": "ada_counted_exposure_deletion_control",
        "region_id": region_id,
        "class_id": int(region["class_id"]),
        "class_name": str(region["class_name"]),
        "pilot_type": str(region.get("pilot_type", "")),
        "control_family": str(control_family),
        "retention_level": float(retention),
        "retention_label": retention_label,
        "seed": int(seed),
        "original_train_count": len(train_rows),
        "original_manifest_hash": original_manifest_hash,
        "enrichment_artifact_id": enrich_metadata.get("artifact_id", ""),
        "pilot_region_artifact_id": pilot_metadata.get("artifact_id", ""),
        "source_region_artifact_id": enrich_metadata.get("source_region_artifact_id", ""),
        "source_validation_assignment_hash": enrich_metadata.get("source_validation_assignment_hash", ""),
        "deleted_sample_count": len(deleted_set),
        "replacement_exposure_count": len(replacement_ids),
        "totals": totals,
        "active_class_counts": {str(key): int(value) for key, value in sorted(active_class_counts.items())},
        "exposure_class_counts": {str(key): int(value) for key, value in sorted(exposure_class_counts.items())},
        "original_class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
        "exposure_manifest_hash": hash_rows(rows, prefix="exposure"),
        "deleted_sample_ids_hash": hash_rows(({"sample_id": sid} for sid in sorted(deleted_set)), prefix="deleted"),
        "replacement_sample_ids_hash": hash_rows(
            ({"sample_id": sid} for sid in sorted(str(sid) for sid in replacement_ids)),
            prefix="replacement",
        ),
        "exposure_manifest_csv": str(out_dir / "exposure_manifest.csv"),
    }
    payload["manifest_id"] = stable_hash(payload, prefix="exposure-manifest")
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (out_dir / "COMPLETED").write_text("ok\n")
    return {
        "manifest_id": payload["manifest_id"],
        "region_id": region_id,
        "class_id": int(region["class_id"]),
        "class_name": str(region["class_name"]),
        "pilot_type": str(region.get("pilot_type", "")),
        "control_family": str(control_family),
        "retention_level": float(retention),
        "retention_label": retention_label,
        "seed": int(seed),
        "deleted_sample_count": len(deleted_set),
        "replacement_exposure_count": len(replacement_ids),
        "unique_active_count": totals["unique_active_count"],
        "total_exposure_count": totals["total_exposure_count"],
        "zero_multiplicity_count": totals["zero_multiplicity_count"],
        "manifest_json": str(out_dir / "manifest.json"),
        "exposure_manifest_csv": str(out_dir / "exposure_manifest.csv"),
    }


def _normalize_retention_levels(levels: Sequence[float]) -> tuple[float, ...]:
    parsed = tuple(sorted({float(level) for level in levels}, reverse=True))
    if any(level < 0.0 or level > 1.0 for level in parsed):
        raise ValueError("retention levels must be between 0 and 1")
    return parsed


def _retained_count(total: int, level: float) -> int:
    if float(level) >= 1.0:
        return int(total)
    if float(level) <= 0.0:
        return 0
    return int(math.floor(int(total) * float(level)))


def _split_ids(ids: Sequence[str], *, retain_count: int, seed: int) -> tuple[list[str], list[str]]:
    values = np.asarray(sorted(str(sid) for sid in ids), dtype=object)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(values)
    retained = sorted(str(sid) for sid in values[: int(retain_count)].tolist())
    deleted = sorted(str(sid) for sid in values[int(retain_count) :].tolist())
    return retained, deleted


def _choose_without_replacement(candidates: Sequence[str], count: int, *, seed: int) -> list[str]:
    values = np.asarray(sorted(str(sid) for sid in candidates), dtype=object)
    if int(count) > len(values):
        raise ValueError(f"cannot choose {count} IDs without replacement from {len(values)} candidates")
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(values, size=int(count), replace=False)
    return sorted(str(sid) for sid in chosen.tolist())


def _choose_with_replacement(candidates: Sequence[str], count: int, *, seed: int) -> list[str]:
    values = np.asarray(sorted(str(sid) for sid in candidates), dtype=object)
    if len(values) == 0:
        raise ValueError("replacement candidate pool is empty")
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(values, size=int(count), replace=True)
    return [str(sid) for sid in chosen.tolist()]


def _derived_seed(seed: int, region_id: str, retention: float, salt: str) -> int:
    digest = stable_hash(
        {"seed": int(seed), "region_id": str(region_id), "retention": float(retention), "salt": str(salt)}
    )
    return int(digest[:8], 16)


def _retention_label(level: float) -> str:
    return f"retain_{int(round(float(level) * 100)):03d}"


def _count_by(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _mark_superseded(path: Path | None, *, replacement_artifact_id: str) -> None:
    if path is None or not str(path):
        return
    target = Path(path)
    if not target.exists():
        return
    note = target / "SUPERSEDED_FOR_PRIMARY_CAUSAL_ANALYSIS.md"
    note.write_text(
        "\n".join(
            [
                "# Superseded For Primary Causal Analysis",
                "",
                "This deletion-manifest artifact is retained for provenance but superseded for the primary causal pilot.",
                "",
                "Reason:",
                "",
                "- mixed K=5/K=10 region selection;",
                "- duplicate selected classes;",
                "- no count-preserving deletion controls;",
                "- replaced by a K=10 one-region-per-class causal-pilot control artifact.",
                "",
                f"Replacement artifact: `{replacement_artifact_id}`",
                "",
            ]
        )
    )


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
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


def _write_lines(path: Path, values: Sequence[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values))
