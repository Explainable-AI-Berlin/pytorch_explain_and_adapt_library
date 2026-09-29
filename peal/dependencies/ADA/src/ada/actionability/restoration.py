from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ada.actionability.deletion_controls import _derived_seed, _retention_label
from ada.actionability.exposure_manifest import ExposureRow, exposure_totals
from ada.atlas.data.manifests import load_manifest_csv
from ada.atlas.hashing import hash_rows, stable_hash


PRIMARY_RESTORATION_CONDITIONS = (
    "unique_target_region_restoration",
    "unique_same_class_non_target_additions",
    "target_anchor_oversampling",
    "target_anchor_augmented_oversampling",
    "region_weighted_loss",
    "full_real_restoration",
)

SECONDARY_RESTORATION_CONDITIONS = (
    "unique_target_region_restoration",
    "unique_same_class_non_target_additions",
    "full_real_restoration",
)


@dataclass(frozen=True)
class RestorationConfig:
    train_cache: Path
    enriched_regions_dir: Path
    pilot_regions_dir: Path
    deletion_controls_dir: Path
    output_dir: Path
    experiment_id: str = "e5a_in100_real_restoration_pilot_manifests"
    primary_retention: float = 0.25
    secondary_retention: float = 0.0
    budget_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
    seeds: tuple[int, ...] = (0, 1, 2)
    overwrite: bool = False


def build_restoration_manifests(config: RestorationConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"restoration artifact already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest_csv(Path(config.train_cache) / "manifest.csv")
    if len({row.sample_id for row in train_rows}) != len(train_rows):
        raise ValueError("train manifest contains duplicate sample IDs")
    train_by_id = {row.sample_id: row for row in train_rows}
    original_rows = [
        asdict(
            ExposureRow(
                sample_id=row.sample_id,
                relative_path=row.relative_path,
                class_id=int(row.class_id),
                class_name=row.class_name,
                region_id="",
                multiplicity=1,
                source_condition="baseline",
                seed=0,
            )
        )
        for row in train_rows
    ]
    original_manifest_hash = hash_rows((asdict(row) for row in train_rows), prefix="manifest")
    original_class_counts = Counter(int(row.class_id) for row in train_rows)

    enrich_metadata = _load_json(Path(config.enriched_regions_dir) / "metadata.json")
    pilot_metadata = _load_json(Path(config.pilot_regions_dir) / "metadata.json")
    deletion_metadata = _load_json(Path(config.deletion_controls_dir) / "metadata.json")
    membership = _read_csv(Path(config.enriched_regions_dir) / "region_membership_k10.csv")
    selected_regions = _read_csv(Path(config.pilot_regions_dir) / "selected_pilot_regions.csv")
    deletion_index = _read_csv(Path(config.deletion_controls_dir) / "deletion_control_manifests.csv")

    region_by_sample = {str(row["sample_id"]): str(row["region_id"]) for row in membership}
    for row in original_rows:
        row["region_id"] = region_by_sample.get(str(row["sample_id"]), "")

    members_by_region: dict[str, list[str]] = {}
    for row in membership:
        members_by_region.setdefault(str(row["region_id"]), []).append(str(row["sample_id"]))

    deletion_by_key = {
        (str(row["region_id"]), int(row["seed"]), str(row["control_family"]), float(row["retention_level"])): row
        for row in deletion_index
    }
    selected_by_region = {str(row["region_id"]): row for row in selected_regions}
    budget_fractions = _normalize_budget_fractions(config.budget_fractions)
    retentions = (float(config.primary_retention), float(config.secondary_retention))
    seeds = tuple(int(seed) for seed in config.seeds)

    index_rows: list[dict[str, object]] = []
    for region_id, region in selected_by_region.items():
        class_id = int(region["class_id"])
        class_name = str(region["class_name"])
        region_member_ids = sorted(members_by_region.get(region_id, []))
        if not region_member_ids:
            raise ValueError(f"selected region has no members: {region_id}")
        region_member_set = set(region_member_ids)
        same_class_outside = sorted(
            row.sample_id
            for row in train_rows
            if int(row.class_id) == class_id and row.sample_id not in region_member_set
        )
        if not same_class_outside:
            raise ValueError(f"region has no same-class non-target pool: {region_id}")

        for seed in seeds:
            baseline_row = deletion_by_key.get((region_id, seed, "baseline", 1.0))
            if baseline_row is None:
                raise ValueError(f"missing q=1 baseline deletion row for {region_id} seed={seed}")
            baseline_exposure = _read_csv(Path(str(baseline_row["exposure_manifest_csv"])))
            baseline_hash = hash_rows(baseline_exposure, prefix="exposure")
            for retention in retentions:
                deleted_row = deletion_by_key.get((region_id, seed, "regional_drop", retention))
                if deleted_row is None:
                    raise ValueError(f"missing regional_drop row for {region_id} retention={retention} seed={seed}")
                deleted_exposure = _read_csv(Path(str(deleted_row["exposure_manifest_csv"])))
                by_sample = _exposure_by_sample(deleted_exposure)
                deleted_target_ids = [sid for sid in region_member_ids if int(by_sample[sid]["multiplicity"]) == 0]
                retained_target_ids = [sid for sid in region_member_ids if int(by_sample[sid]["multiplicity"]) > 0]
                deleted_count = len(deleted_target_ids)
                if deleted_count <= 0:
                    raise ValueError(f"retention={retention} produced no deleted target samples for {region_id}")
                budget_counts = _budget_counts(deleted_count, budget_fractions)
                target_order = _shuffle_ids(
                    deleted_target_ids,
                    seed=_derived_seed(seed, region_id, retention, "restore-target-order"),
                    replace=False,
                )
                same_class_order = _shuffle_ids(
                    same_class_outside,
                    seed=_derived_seed(seed, region_id, retention, "restore-same-class-order"),
                    replace=False,
                )
                if max(budget_counts.values()) > len(same_class_order):
                    raise ValueError(f"same-class non-target pool too small for {region_id}")

                conditions = PRIMARY_RESTORATION_CONDITIONS if math.isclose(retention, float(config.primary_retention)) else SECONDARY_RESTORATION_CONDITIONS
                for condition in conditions:
                    if condition == "full_real_restoration":
                        index_rows.append(
                            _write_restoration_manifest(
                                output,
                                region=region,
                                retention=retention,
                                seed=seed,
                                condition=condition,
                                budget_fraction=1.0,
                                budget_count=deleted_count,
                                base_exposure=baseline_exposure,
                                restored_target_ids=deleted_target_ids,
                                same_class_added_ids=[],
                                anchor_added_ids=[],
                                deleted_target_ids=deleted_target_ids,
                                retained_target_ids=retained_target_ids,
                                source_deletion_row=deleted_row,
                                source_baseline_row=baseline_row,
                                original_manifest_hash=original_manifest_hash,
                                baseline_exposure_hash=baseline_hash,
                                source_metadata=(enrich_metadata, pilot_metadata, deletion_metadata),
                                original_class_counts=original_class_counts,
                                full_restoration_expected=True,
                            )
                        )
                        continue
                    for fraction, budget_count in budget_counts.items():
                        restored_target_ids: list[str] = []
                        same_class_added_ids: list[str] = []
                        anchor_added_ids: list[str] = []
                        base_exposure = deleted_exposure
                        if condition == "unique_target_region_restoration":
                            restored_target_ids = target_order[:budget_count]
                        elif condition == "unique_same_class_non_target_additions":
                            same_class_added_ids = same_class_order[:budget_count]
                        elif condition == "target_anchor_oversampling":
                            anchor_added_ids = _draw_anchor_additions(
                                retained_target_ids,
                                budget_count,
                                seed=_derived_seed(seed, region_id, retention, f"{condition}-{fraction}"),
                            )
                        elif condition == "target_anchor_augmented_oversampling":
                            anchor_added_ids = _draw_anchor_additions(
                                retained_target_ids,
                                budget_count,
                                seed=_derived_seed(seed, region_id, retention, f"{condition}-{fraction}"),
                            )
                        elif condition == "region_weighted_loss":
                            pass
                        else:
                            raise AssertionError(f"unknown restoration condition: {condition}")
                        index_rows.append(
                            _write_restoration_manifest(
                                output,
                                region=region,
                                retention=retention,
                                seed=seed,
                                condition=condition,
                                budget_fraction=fraction,
                                budget_count=budget_count,
                                base_exposure=base_exposure,
                                restored_target_ids=restored_target_ids,
                                same_class_added_ids=same_class_added_ids,
                                anchor_added_ids=anchor_added_ids,
                                deleted_target_ids=deleted_target_ids,
                                retained_target_ids=retained_target_ids,
                                source_deletion_row=deleted_row,
                                source_baseline_row=baseline_row,
                                original_manifest_hash=original_manifest_hash,
                                baseline_exposure_hash=baseline_hash,
                                source_metadata=(enrich_metadata, pilot_metadata, deletion_metadata),
                                original_class_counts=original_class_counts,
                                full_restoration_expected=False,
                            )
                        )

    index_hash = hash_rows(index_rows, prefix="restoration-index")
    config_payload = {
        "train_cache": str(config.train_cache),
        "enriched_regions_dir": str(config.enriched_regions_dir),
        "pilot_regions_dir": str(config.pilot_regions_dir),
        "deletion_controls_dir": str(config.deletion_controls_dir),
        "output_dir": str(config.output_dir),
        "experiment_id": config.experiment_id,
        "primary_retention": float(config.primary_retention),
        "secondary_retention": float(config.secondary_retention),
        "budget_fractions": list(budget_fractions),
        "seeds": list(seeds),
    }
    config_hash = stable_hash(config_payload, prefix="restoration-config")
    artifact_id = stable_hash(
        {
            "config_hash": config_hash,
            "index_hash": index_hash,
            "deletion_control_artifact_id": deletion_metadata.get("artifact_id", ""),
            "pilot_artifact_id": pilot_metadata.get("artifact_id", ""),
        },
        prefix="restoration-artifact",
    )
    _write_csv(output / "restoration_manifests.csv", index_rows)
    shutil.copyfile(output / "restoration_manifests.csv", output / "deletion_control_manifests.csv")
    metadata = {
        "artifact_id": artifact_id,
        "experiment_id": config.experiment_id,
        "config": config_payload,
        "config_hash": config_hash,
        "index_hash": index_hash,
        "original_manifest_hash": original_manifest_hash,
        "enrichment_artifact_id": enrich_metadata.get("artifact_id", ""),
        "pilot_region_artifact_id": pilot_metadata.get("artifact_id", ""),
        "deletion_control_artifact_id": deletion_metadata.get("artifact_id", ""),
        "source_region_artifact_id": enrich_metadata.get("source_region_artifact_id", ""),
        "source_validation_assignment_hash": enrich_metadata.get("source_validation_assignment_hash", ""),
        "outputs": {
            "restoration_manifests_csv": str(output / "restoration_manifests.csv"),
            "trainer_compat_manifest_csv": str(output / "deletion_control_manifests.csv"),
        },
        "summary": {
            "selected_regions": len(selected_by_region),
            "seeds": list(seeds),
            "retention_levels": list(retentions),
            "budget_fractions": list(budget_fractions),
            "manifest_count": len(index_rows),
            "by_condition": _count_by(index_rows, "restoration_condition"),
            "by_retention": _count_by(index_rows, "retention_level"),
            "full_restoration_rows": sum(1 for row in index_rows if str(row["restoration_condition"]) == "full_real_restoration"),
        },
        "cached_feature_augmented_oversampling_note": (
            "`target_anchor_augmented_oversampling` is represented as a separate exposure condition in this "
            "cached-feature DINO CLS probe. Fresh image augmentations require an image-space or augmented-cache trainer; "
            "this pilot row is therefore an exposure-accounting placeholder unless such augmented features are supplied."
        ),
        "leakage_declaration": (
            "Restoration manifests use only train sample IDs, train labels, train K=10 memberships, and the immutable "
            "deletion-control pilot. Frozen validation assignments are referenced only by artifact hash for evaluation; "
            "validation errors are not used to select restored examples."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def _write_restoration_manifest(
    output: Path,
    *,
    region: Mapping[str, object],
    retention: float,
    seed: int,
    condition: str,
    budget_fraction: float,
    budget_count: int,
    base_exposure: Sequence[Mapping[str, str]],
    restored_target_ids: Sequence[str],
    same_class_added_ids: Sequence[str],
    anchor_added_ids: Sequence[str],
    deleted_target_ids: Sequence[str],
    retained_target_ids: Sequence[str],
    source_deletion_row: Mapping[str, str],
    source_baseline_row: Mapping[str, str],
    original_manifest_hash: str,
    baseline_exposure_hash: str,
    source_metadata: tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]],
    original_class_counts: Mapping[int, int],
    full_restoration_expected: bool,
) -> dict[str, object]:
    region_id = str(region["region_id"])
    retention_label = _retention_label(float(retention))
    budget_label = _budget_label(float(budget_fraction), int(budget_count))
    seed_label = f"seed_{int(seed):03d}"
    out_dir = output / region_id / condition / retention_label / budget_label / seed_label
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [dict(row) for row in base_exposure]
    by_sample = _exposure_by_sample(rows)
    restored_set = set(str(sid) for sid in restored_target_ids)
    same_class_added = [str(sid) for sid in same_class_added_ids]
    anchor_added = [str(sid) for sid in anchor_added_ids]
    retained_target_set = set(str(sid) for sid in retained_target_ids)
    deleted_target_set = set(str(sid) for sid in deleted_target_ids)

    for row in rows:
        sample_id = str(row["sample_id"])
        row["source_condition"] = condition
        row["seed"] = int(seed)
        row["loss_weight"] = 1.0
        row["augmentation_policy"] = "none"
        row["restoration_role"] = "background"
        if sample_id in deleted_target_set:
            row["restoration_role"] = "deleted_target_region"
        if sample_id in retained_target_set:
            row["restoration_role"] = "retained_target_region_anchor"

    if condition == "unique_target_region_restoration":
        for sid in restored_set:
            if sid not in deleted_target_set:
                raise AssertionError("target restoration IDs must come from the deleted target pool")
            by_sample[sid]["multiplicity"] = 1
            by_sample[sid]["restoration_role"] = "restored_target_region"
    elif condition == "unique_same_class_non_target_additions":
        for sid in same_class_added:
            if sid in deleted_target_set:
                raise AssertionError("same-class non-target additions must not be deleted target IDs")
            by_sample[sid]["multiplicity"] = int(by_sample[sid]["multiplicity"]) + 1
            by_sample[sid]["restoration_role"] = "same_class_non_target_extra_exposure"
    elif condition in {"target_anchor_oversampling", "target_anchor_augmented_oversampling"}:
        if not anchor_added:
            raise ValueError(f"{condition} requires at least one retained target anchor")
        for sid in anchor_added:
            if sid not in retained_target_set:
                raise AssertionError("anchor oversampling IDs must come from retained target anchors")
            by_sample[sid]["multiplicity"] = int(by_sample[sid]["multiplicity"]) + 1
            by_sample[sid]["restoration_role"] = "retained_target_region_anchor_extra_exposure"
            if condition == "target_anchor_augmented_oversampling":
                by_sample[sid]["augmentation_policy"] = "standard_image_aug_requested_cached_feature_placeholder"
    elif condition == "region_weighted_loss":
        if not retained_target_set:
            raise ValueError("region_weighted_loss requires retained target anchors")
        extra_loss_per_anchor = float(budget_count) / float(len(retained_target_set))
        for sid in retained_target_set:
            by_sample[sid]["loss_weight"] = 1.0 + extra_loss_per_anchor
            by_sample[sid]["restoration_role"] = "retained_target_region_anchor_loss_weighted"
    elif condition == "full_real_restoration":
        rows = [dict(row) for row in base_exposure]
    else:
        raise AssertionError(f"unknown condition: {condition}")

    if condition != "full_real_restoration":
        rows = [by_sample[str(row["sample_id"])] for row in rows]

    totals = exposure_totals(rows)
    if len(rows) != len({str(row["sample_id"]) for row in rows}):
        raise AssertionError("restoration exposure manifest contains duplicate sample IDs")

    active_class_counts = Counter()
    exposure_class_counts = Counter()
    for row in rows:
        multiplicity = int(row["multiplicity"])
        if multiplicity > 0:
            active_class_counts[int(row["class_id"])] += 1
        exposure_class_counts[int(row["class_id"])] += multiplicity

    exposure_hash = hash_rows(rows, prefix="exposure")
    full_matches = bool(full_restoration_expected and exposure_hash == baseline_exposure_hash)
    if full_restoration_expected and not full_matches:
        raise AssertionError("full restoration exposure manifest must exactly match the original q=1 baseline exposure manifest")

    _write_csv(out_dir / "exposure_manifest.csv", rows)
    _write_lines(out_dir / "restored_target_sample_ids.txt", sorted(restored_set))
    _write_lines(out_dir / "same_class_added_sample_ids.txt", sorted(set(same_class_added)))
    _write_lines(out_dir / "anchor_added_sample_ids.txt", anchor_added)
    _write_lines(out_dir / "deleted_target_sample_ids.txt", sorted(deleted_target_set))
    enrich_metadata, pilot_metadata, deletion_metadata = source_metadata
    payload = {
        "manifest_type": "ada_counted_exposure_real_restoration",
        "region_id": region_id,
        "class_id": int(region["class_id"]),
        "class_name": str(region["class_name"]),
        "pilot_type": str(region.get("pilot_type", "")),
        "control_family": condition,
        "restoration_condition": condition,
        "retention_level": float(retention),
        "retention_label": retention_label,
        "seed": int(seed),
        "restoration_budget_fraction": float(budget_fraction),
        "restoration_budget_label": budget_label,
        "restoration_budget_count": int(budget_count),
        "source_deleted_manifest_id": str(source_deletion_row["manifest_id"]),
        "source_baseline_manifest_id": str(source_baseline_row["manifest_id"]),
        "source_deleted_exposure_manifest_csv": str(source_deletion_row["exposure_manifest_csv"]),
        "source_baseline_exposure_manifest_csv": str(source_baseline_row["exposure_manifest_csv"]),
        "original_train_count": len(rows),
        "original_manifest_hash": original_manifest_hash,
        "baseline_exposure_hash": baseline_exposure_hash,
        "exposure_manifest_hash": exposure_hash,
        "full_restoration_expected": bool(full_restoration_expected),
        "full_restoration_matches_baseline_manifest": full_matches,
        "deleted_target_sample_count": len(deleted_target_set),
        "retained_target_sample_count": len(retained_target_set),
        "restored_target_sample_count": len(restored_set),
        "same_class_added_unique_count": len(set(same_class_added)),
        "anchor_added_exposure_count": len(anchor_added),
        "loss_weighted_anchor_count": len(retained_target_set) if condition == "region_weighted_loss" else 0,
        "totals": totals,
        "active_class_counts": {str(key): int(value) for key, value in sorted(active_class_counts.items())},
        "exposure_class_counts": {str(key): int(value) for key, value in sorted(exposure_class_counts.items())},
        "original_class_counts": {str(key): int(value) for key, value in sorted(original_class_counts.items())},
        "enrichment_artifact_id": enrich_metadata.get("artifact_id", ""),
        "pilot_region_artifact_id": pilot_metadata.get("artifact_id", ""),
        "deletion_control_artifact_id": deletion_metadata.get("artifact_id", ""),
        "source_region_artifact_id": enrich_metadata.get("source_region_artifact_id", ""),
        "source_validation_assignment_hash": enrich_metadata.get("source_validation_assignment_hash", ""),
        "exposure_manifest_csv": str(out_dir / "exposure_manifest.csv"),
    }
    payload["manifest_id"] = stable_hash(payload, prefix="restoration-manifest")
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (out_dir / "COMPLETED").write_text("ok\n")
    return {
        "manifest_id": payload["manifest_id"],
        "region_id": region_id,
        "class_id": int(region["class_id"]),
        "class_name": str(region["class_name"]),
        "pilot_type": str(region.get("pilot_type", "")),
        "control_family": condition,
        "restoration_condition": condition,
        "retention_level": float(retention),
        "retention_label": retention_label,
        "seed": int(seed),
        "restoration_budget_fraction": float(budget_fraction),
        "restoration_budget_label": budget_label,
        "restoration_budget_count": int(budget_count),
        "source_deleted_manifest_id": str(source_deletion_row["manifest_id"]),
        "source_baseline_manifest_id": str(source_baseline_row["manifest_id"]),
        "deleted_sample_count": int(payload["deleted_target_sample_count"]),
        "deleted_target_sample_count": int(payload["deleted_target_sample_count"]),
        "retained_target_sample_count": int(payload["retained_target_sample_count"]),
        "restored_target_sample_count": int(payload["restored_target_sample_count"]),
        "same_class_added_unique_count": int(payload["same_class_added_unique_count"]),
        "anchor_added_exposure_count": int(payload["anchor_added_exposure_count"]),
        "loss_weighted_anchor_count": int(payload["loss_weighted_anchor_count"]),
        "unique_active_count": totals["unique_active_count"],
        "total_exposure_count": totals["total_exposure_count"],
        "zero_multiplicity_count": totals["zero_multiplicity_count"],
        "full_restoration_expected": bool(full_restoration_expected),
        "full_restoration_matches_baseline_manifest": full_matches,
        "manifest_json": str(out_dir / "manifest.json"),
        "exposure_manifest_csv": str(out_dir / "exposure_manifest.csv"),
    }


def _normalize_budget_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    parsed = tuple(sorted({float(value) for value in fractions}))
    if not parsed:
        raise ValueError("at least one budget fraction is required")
    if any(value <= 0.0 or value > 1.0 for value in parsed):
        raise ValueError("budget fractions must be in (0, 1]")
    if not math.isclose(parsed[-1], 1.0):
        raise ValueError("budget fractions must include 1.0 for the full-restoration integrity check")
    return parsed


def _budget_counts(deleted_count: int, fractions: Sequence[float]) -> dict[float, int]:
    out: dict[float, int] = {}
    last = 0
    for fraction in fractions:
        count = int(round(float(deleted_count) * float(fraction)))
        if fraction > 0.0:
            count = max(1, count)
        count = min(int(deleted_count), max(last, count))
        out[float(fraction)] = int(count)
        last = count
    return out


def _budget_label(fraction: float, count: int) -> str:
    return f"budget_{int(round(float(fraction) * 100)):03d}_n{int(count):04d}"


def _draw_anchor_additions(anchor_ids: Sequence[str], count: int, *, seed: int) -> list[str]:
    if not anchor_ids:
        raise ValueError("anchor pool is empty")
    values = np.asarray(sorted(str(sid) for sid in anchor_ids), dtype=object)
    rng = np.random.default_rng(int(seed))
    return [str(sid) for sid in rng.choice(values, size=int(count), replace=True).tolist()]


def _shuffle_ids(ids: Sequence[str], *, seed: int, replace: bool) -> list[str]:
    values = np.asarray(sorted(str(sid) for sid in ids), dtype=object)
    if len(values) == 0:
        return []
    rng = np.random.default_rng(int(seed))
    if replace:
        return [str(sid) for sid in rng.choice(values, size=len(values), replace=True).tolist()]
    rng.shuffle(values)
    return [str(sid) for sid in values.tolist()]


def _exposure_by_sample(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in out:
            raise ValueError(f"duplicate sample_id in exposure rows: {sample_id}")
        out[sample_id] = dict(row)
    return out


def _count_by(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


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
