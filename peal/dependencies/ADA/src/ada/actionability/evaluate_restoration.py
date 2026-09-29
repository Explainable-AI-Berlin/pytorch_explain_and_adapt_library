from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ada.actionability.region_assignment import l2_normalize
from ada.actionability.regions import load_embedding_bank
from ada.actionability.train_deletion_probe import write_probe_index
from ada.atlas.hashing import hash_rows, stable_hash


@dataclass(frozen=True)
class RestorationEvaluationConfig:
    train_cache: Path
    validation_cache: Path
    enriched_regions_dir: Path
    restoration_dir: Path
    restoration_probe_output_dir: Path
    deletion_probe_output_dir: Path
    output_dir: Path
    experiment_id: str = "e5a_in100_real_restoration_pilot_evaluation"
    support_k: int = 50
    overwrite: bool = False


def evaluate_restoration_grid(config: RestorationEvaluationConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"restoration evaluation artifact already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train = load_embedding_bank(config.train_cache, mmap=True)
    val = load_embedding_bank(config.validation_cache, mmap=True)
    train_embeddings = l2_normalize(np.asarray(train.embeddings, dtype=np.float32))
    val_embeddings = l2_normalize(np.asarray(val.embeddings, dtype=np.float32))
    train_by_id = {row.sample_id: idx for idx, row in enumerate(train.rows)}
    val_labels = np.asarray([int(row.class_id) for row in val.rows], dtype=np.int64)
    val_ids = [row.sample_id for row in val.rows]

    assignments = _read_csv(Path(config.enriched_regions_dir) / "validation_assignments_k10.csv")
    target_val_by_region: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        target_val_by_region[str(row["region_id"])].add(str(row["sample_id"]))

    restoration_index = _read_csv(Path(config.restoration_dir) / "restoration_manifests.csv")
    write_probe_index(config.restoration_probe_output_dir)
    write_probe_index(config.deletion_probe_output_dir)
    restoration_probe_index = _load_probe_index(Path(config.restoration_probe_output_dir))
    deletion_probe_index = _load_probe_index(Path(config.deletion_probe_output_dir))

    deletion_probe_by_manifest = {str(row["manifest_id"]): row for row in deletion_probe_index}
    restoration_probe_by_manifest = {str(row["manifest_id"]): row for row in restoration_probe_index}

    prediction_cache: dict[str, dict[str, Mapping[str, str]]] = {}
    result_rows: list[dict[str, object]] = []
    for manifest_row in restoration_index:
        manifest_id = str(manifest_row["manifest_id"])
        probe_row = restoration_probe_by_manifest.get(manifest_id)
        if probe_row is None:
            continue
        baseline_probe_row = deletion_probe_by_manifest.get(str(manifest_row["source_baseline_manifest_id"]))
        deleted_probe_row = deletion_probe_by_manifest.get(str(manifest_row["source_deleted_manifest_id"]))
        if baseline_probe_row is None or deleted_probe_row is None:
            continue

        restored_pred = _predictions_by_id(Path(str(probe_row["predictions_csv"])), prediction_cache)
        baseline_pred = _predictions_by_id(Path(str(baseline_probe_row["predictions_csv"])), prediction_cache)
        deleted_pred = _predictions_by_id(Path(str(deleted_probe_row["predictions_csv"])), prediction_cache)

        region_id = str(manifest_row["region_id"])
        class_id = int(manifest_row["class_id"])
        target_ids = target_val_by_region[region_id]
        same_class_ids = {sid for sid, label in zip(val_ids, val_labels.tolist()) if int(label) == class_id}
        off_target_ids = same_class_ids.difference(target_ids)
        global_ids = set(val_ids)

        exposure_rows = _read_csv(Path(str(manifest_row["exposure_manifest_csv"])))
        active_train_indices_by_class: dict[int, list[int]] = defaultdict(list)
        for row in exposure_rows:
            if int(row["multiplicity"]) <= 0:
                continue
            train_idx = train_by_id[str(row["sample_id"])]
            active_train_indices_by_class[int(row["class_id"])].append(train_idx)

        restored_metrics = _metrics(restored_pred, target_ids)
        restored_offtarget = _metrics(restored_pred, off_target_ids)
        restored_global = _metrics(restored_pred, global_ids)
        baseline_metrics = _metrics(baseline_pred, target_ids)
        deleted_metrics = _metrics(deleted_pred, target_ids)
        budget = int(manifest_row["restoration_budget_count"])
        target_error_gain = float(deleted_metrics["error"]) - float(restored_metrics["error"])
        target_ce_gain = float(deleted_metrics["cross_entropy"]) - float(restored_metrics["cross_entropy"])
        target_margin_gain = float(restored_metrics["true_class_logit_margin"]) - float(deleted_metrics["true_class_logit_margin"])
        denom = float(deleted_metrics["error"]) - float(baseline_metrics["error"])

        result_rows.append(
            {
                "manifest_id": manifest_id,
                "region_id": region_id,
                "class_id": class_id,
                "class_name": manifest_row["class_name"],
                "pilot_type": manifest_row["pilot_type"],
                "restoration_condition": manifest_row["restoration_condition"],
                "control_family": manifest_row["control_family"],
                "retention_level": float(manifest_row["retention_level"]),
                "seed": int(manifest_row["seed"]),
                "restoration_budget_fraction": float(manifest_row["restoration_budget_fraction"]),
                "restoration_budget_count": budget,
                "deleted_target_sample_count": int(manifest_row["deleted_target_sample_count"]),
                "retained_target_sample_count": int(manifest_row["retained_target_sample_count"]),
                "target_val_count": len(target_ids),
                "same_class_offtarget_val_count": len(off_target_ids),
                "global_val_count": len(global_ids),
                "baseline_target_error": baseline_metrics["error"],
                "deleted_target_error": deleted_metrics["error"],
                "target_error": restored_metrics["error"],
                "repair_gain_target_error": target_error_gain,
                "recovery_fraction_target_error": target_error_gain / denom if abs(denom) > 1.0e-12 else math.nan,
                "repair_efficiency_target_error": target_error_gain / float(budget) if budget > 0 else math.nan,
                "baseline_target_cross_entropy": baseline_metrics["cross_entropy"],
                "deleted_target_cross_entropy": deleted_metrics["cross_entropy"],
                "target_cross_entropy": restored_metrics["cross_entropy"],
                "repair_gain_target_cross_entropy": target_ce_gain,
                "repair_efficiency_target_cross_entropy": target_ce_gain / float(budget) if budget > 0 else math.nan,
                "baseline_target_true_class_logit_margin": baseline_metrics["true_class_logit_margin"],
                "deleted_target_true_class_logit_margin": deleted_metrics["true_class_logit_margin"],
                "target_true_class_logit_margin": restored_metrics["true_class_logit_margin"],
                "repair_gain_target_true_class_logit_margin": target_margin_gain,
                "same_class_offtarget_error": restored_offtarget["error"],
                "same_class_offtarget_cross_entropy": restored_offtarget["cross_entropy"],
                "same_class_offtarget_true_class_logit_margin": restored_offtarget["true_class_logit_margin"],
                "global_error": restored_global["error"],
                "global_cross_entropy": restored_global["cross_entropy"],
                "target_support": _mean_support(
                    query_ids=target_ids,
                    val_ids=val_ids,
                    val_embeddings=val_embeddings,
                    train_embeddings=train_embeddings,
                    reference_indices=active_train_indices_by_class[class_id],
                    support_k=int(config.support_k),
                ),
                "same_class_offtarget_support": _mean_support(
                    query_ids=off_target_ids,
                    val_ids=val_ids,
                    val_embeddings=val_embeddings,
                    train_embeddings=train_embeddings,
                    reference_indices=active_train_indices_by_class[class_id],
                    support_k=int(config.support_k),
                ),
                "full_restoration_expected": _boolish(manifest_row.get("full_restoration_expected", "")),
                "full_restoration_matches_baseline_manifest": _boolish(manifest_row.get("full_restoration_matches_baseline_manifest", "")),
                "probe_predictions_csv": probe_row["predictions_csv"],
                "source_deleted_manifest_id": manifest_row["source_deleted_manifest_id"],
                "source_baseline_manifest_id": manifest_row["source_baseline_manifest_id"],
            }
        )

    condition_rows = _condition_summary(result_rows)
    region_bootstrap_rows = _bootstrap_ci(result_rows, group_key="region_id", prefix="region")
    class_bootstrap_rows = _bootstrap_ci(result_rows, group_key="class_id", prefix="class")
    results_hash = hash_rows(result_rows, prefix="restoration-eval")
    condition_hash = hash_rows(condition_rows, prefix="restoration-condition-summary")
    region_bootstrap_hash = hash_rows(region_bootstrap_rows, prefix="restoration-region-bootstrap")
    class_bootstrap_hash = hash_rows(class_bootstrap_rows, prefix="restoration-class-bootstrap")
    metadata = {
        "artifact_id": stable_hash(
            {
                "results_hash": results_hash,
                "condition_hash": condition_hash,
                "region_bootstrap_hash": region_bootstrap_hash,
                "class_bootstrap_hash": class_bootstrap_hash,
                "support_k": int(config.support_k),
            },
            prefix="restoration-eval-artifact",
        ),
        "experiment_id": config.experiment_id,
        "config": {
            "train_cache": str(config.train_cache),
            "validation_cache": str(config.validation_cache),
            "enriched_regions_dir": str(config.enriched_regions_dir),
            "restoration_dir": str(config.restoration_dir),
            "restoration_probe_output_dir": str(config.restoration_probe_output_dir),
            "deletion_probe_output_dir": str(config.deletion_probe_output_dir),
            "output_dir": str(config.output_dir),
            "experiment_id": config.experiment_id,
            "support_k": int(config.support_k),
        },
        "results_hash": results_hash,
        "condition_hash": condition_hash,
        "region_bootstrap_hash": region_bootstrap_hash,
        "class_bootstrap_hash": class_bootstrap_hash,
        "outputs": {
            "restoration_evaluation_csv": str(output / "restoration_evaluation.csv"),
            "restoration_condition_summary_csv": str(output / "restoration_condition_summary.csv"),
            "region_bootstrap_ci_csv": str(output / "region_bootstrap_ci.csv"),
            "class_bootstrap_ci_csv": str(output / "class_bootstrap_ci.csv"),
        },
        "summary": {
            "rows": len(result_rows),
            "condition_summary_rows": len(condition_rows),
            "region_bootstrap_rows": len(region_bootstrap_rows),
            "class_bootstrap_rows": len(class_bootstrap_rows),
            "unique_regions": len({row["region_id"] for row in result_rows}),
            "support_k": int(config.support_k),
        },
    }
    _write_csv(output / "restoration_evaluation.csv", result_rows)
    _write_csv(output / "restoration_condition_summary.csv", condition_rows)
    _write_csv(output / "region_bootstrap_ci.csv", region_bootstrap_rows)
    _write_csv(output / "class_bootstrap_ci.csv", class_bootstrap_rows)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def _load_probe_index(path: Path) -> list[dict[str, str]]:
    probe_index_path = path / "probe_runs.csv"
    if not probe_index_path.exists():
        raise FileNotFoundError(f"missing probe index: {probe_index_path}")
    return _read_csv(probe_index_path)


def _predictions_by_id(path: Path, cache: dict[str, dict[str, Mapping[str, str]]]) -> dict[str, Mapping[str, str]]:
    key = str(path)
    if key not in cache:
        cache[key] = {str(row["sample_id"]): row for row in _read_csv(path)}
    return cache[key]


def _metrics(pred_by_id: Mapping[str, Mapping[str, str]], sample_ids: set[str]) -> dict[str, float]:
    rows = [pred_by_id[sid] for sid in sorted(sample_ids) if sid in pred_by_id]
    if not rows:
        return {"error": math.nan, "cross_entropy": math.nan, "true_class_logit_margin": math.nan}
    return {
        "error": float(np.mean([1.0 - int(row["correct"]) for row in rows])),
        "cross_entropy": _mean_float(rows, "nll"),
        "true_class_logit_margin": _mean_true_class_margin(rows),
    }


def _mean_true_class_margin(rows: Sequence[Mapping[str, str]]) -> float:
    if rows and "true_class_logit_margin" in rows[0] and rows[0].get("true_class_logit_margin", "") != "":
        return _mean_float(rows, "true_class_logit_margin")
    return _mean_float(rows, "logit_margin")


def _mean_float(rows: Sequence[Mapping[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return float(np.mean(values)) if values else math.nan


def _mean_support(
    *,
    query_ids: set[str],
    val_ids: Sequence[str],
    val_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    reference_indices: Sequence[int],
    support_k: int,
) -> float:
    if not query_ids or not reference_indices:
        return math.nan
    val_index = {sid: idx for idx, sid in enumerate(val_ids)}
    qidx = [val_index[sid] for sid in sorted(query_ids) if sid in val_index]
    if not qidx:
        return math.nan
    refs = np.asarray(reference_indices, dtype=np.int64)
    ref = np.asarray(train_embeddings[refs], dtype=np.float32)
    k_eff = min(int(support_k), int(ref.shape[0]))
    vals: list[float] = []
    for start in range(0, len(qidx), 512):
        query = np.asarray(val_embeddings[qidx[start:start + 512]], dtype=np.float32)
        distances = 1.0 - query @ ref.T
        kth = np.partition(distances, kth=k_eff - 1, axis=1)[:, k_eff - 1]
        vals.extend(float(x) for x in kth.tolist())
    return float(np.mean(vals)) if vals else math.nan


def _condition_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["pilot_type"]),
                str(row["restoration_condition"]),
                float(row["retention_level"]),
                float(row["restoration_budget_fraction"]),
            )
        ].append(row)
    out: list[dict[str, object]] = []
    for (pilot_type, condition, retention, budget_fraction), group_rows in sorted(groups.items()):
        out.append(
            {
                "pilot_type": pilot_type,
                "restoration_condition": condition,
                "retention_level": retention,
                "restoration_budget_fraction": budget_fraction,
                "row_count": len(group_rows),
                "region_count": len({str(row["region_id"]) for row in group_rows}),
                "mean_budget_count": float(np.mean([int(row["restoration_budget_count"]) for row in group_rows])),
                "mean_target_error": _mean_object_float(group_rows, "target_error"),
                "mean_repair_gain_target_error": _mean_object_float(group_rows, "repair_gain_target_error"),
                "mean_recovery_fraction_target_error": _mean_object_float(group_rows, "recovery_fraction_target_error"),
                "mean_repair_efficiency_target_error": _mean_object_float(group_rows, "repair_efficiency_target_error"),
                "mean_target_cross_entropy": _mean_object_float(group_rows, "target_cross_entropy"),
                "mean_repair_gain_target_cross_entropy": _mean_object_float(group_rows, "repair_gain_target_cross_entropy"),
                "mean_target_true_class_logit_margin": _mean_object_float(group_rows, "target_true_class_logit_margin"),
                "mean_repair_gain_target_true_class_logit_margin": _mean_object_float(group_rows, "repair_gain_target_true_class_logit_margin"),
                "mean_same_class_offtarget_error": _mean_object_float(group_rows, "same_class_offtarget_error"),
                "mean_global_error": _mean_object_float(group_rows, "global_error"),
                "mean_target_support": _mean_object_float(group_rows, "target_support"),
                "mean_same_class_offtarget_support": _mean_object_float(group_rows, "same_class_offtarget_support"),
            }
        )
    return out


def _bootstrap_ci(
    rows: Sequence[Mapping[str, object]],
    *,
    group_key: str,
    prefix: str,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> list[dict[str, object]]:
    value_keys = (
        "repair_gain_target_error",
        "recovery_fraction_target_error",
        "repair_efficiency_target_error",
        "repair_gain_target_cross_entropy",
        "repair_gain_target_true_class_logit_margin",
    )
    groups: dict[tuple[str, str, float, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["pilot_type"]),
                str(row["restoration_condition"]),
                float(row["retention_level"]),
                float(row["restoration_budget_fraction"]),
            )
        ].append(row)
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, object]] = []
    for (pilot_type, condition, retention, budget_fraction), group_rows in sorted(groups.items()):
        keys = sorted({str(row[group_key]) for row in group_rows})
        if not keys:
            continue
        rows_by_key: dict[str, list[Mapping[str, object]]] = {
            key: [row for row in group_rows if str(row[group_key]) == key] for key in keys
        }
        for value_key in value_keys:
            values_by_key = {
                key: _mean_object_float(key_rows, value_key)
                for key, key_rows in rows_by_key.items()
            }
            values_by_key = {key: value for key, value in values_by_key.items() if not math.isnan(value)}
            if not values_by_key:
                continue
            observed = float(np.mean(list(values_by_key.values())))
            draws = []
            valid_keys = list(values_by_key.keys())
            for _ in range(int(n_bootstrap)):
                sampled = rng.choice(valid_keys, size=len(valid_keys), replace=True)
                draws.append(float(np.mean([values_by_key[str(key)] for key in sampled])))
            out.append(
                {
                    "bootstrap_type": f"{prefix}_bootstrap",
                    "pilot_type": pilot_type,
                    "restoration_condition": condition,
                    "retention_level": float(retention),
                    "restoration_budget_fraction": float(budget_fraction),
                    "value": value_key,
                    f"{prefix}_count": len(valid_keys),
                    "mean": observed,
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "bootstrap_samples": int(n_bootstrap),
                }
            )
    return out


def _mean_object_float(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key, math.nan)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value_float):
            values.append(value_float)
    return float(np.mean(values)) if values else math.nan


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
