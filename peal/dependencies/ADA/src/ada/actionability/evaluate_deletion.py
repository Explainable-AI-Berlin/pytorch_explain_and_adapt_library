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
from ada.atlas.hashing import hash_rows, stable_hash


@dataclass(frozen=True)
class DeletionEvaluationConfig:
    train_cache: Path
    validation_cache: Path
    enriched_regions_dir: Path
    deletion_controls_dir: Path
    probe_output_dir: Path
    output_dir: Path
    experiment_id: str = "e4a_in100_k10_causal_pilot_deletion_evaluation"
    support_k: int = 50
    overwrite: bool = False


def evaluate_deletion_grid(config: DeletionEvaluationConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"deletion evaluation artifact already exists: {output}")
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

    manifest_index = _read_csv(Path(config.deletion_controls_dir) / "deletion_control_manifests.csv")
    probe_index_path = Path(config.probe_output_dir) / "probe_runs.csv"
    if not probe_index_path.exists():
        raise FileNotFoundError(f"missing probe index: {probe_index_path}")
    probe_index = _read_csv(probe_index_path)
    probe_by_manifest = {str(row["manifest_id"]): row for row in probe_index}

    result_rows: list[dict[str, object]] = []
    for manifest_row in manifest_index:
        manifest_id = str(manifest_row["manifest_id"])
        probe_row = probe_by_manifest.get(manifest_id)
        if probe_row is None:
            continue
        predictions = _read_csv(Path(str(probe_row["predictions_csv"])))
        pred_by_id = {str(row["sample_id"]): row for row in predictions}
        region_id = str(manifest_row["region_id"])
        class_id = int(manifest_row["class_id"])
        target_ids = target_val_by_region[region_id]
        same_class_ids = {sid for sid, label in zip(val_ids, val_labels.tolist()) if int(label) == class_id}
        off_target_ids = same_class_ids.difference(target_ids)

        exposure_rows = _read_csv(Path(str(manifest_row["exposure_manifest_csv"])))
        active_train_indices_by_class: dict[int, list[int]] = defaultdict(list)
        for row in exposure_rows:
            if int(row["multiplicity"]) <= 0:
                continue
            train_idx = train_by_id[str(row["sample_id"])]
            active_train_indices_by_class[int(row["class_id"])].append(train_idx)

        target_support = _mean_support(
            query_ids=target_ids,
            val_ids=val_ids,
            val_embeddings=val_embeddings,
            train_embeddings=train_embeddings,
            reference_indices=active_train_indices_by_class[class_id],
            support_k=int(config.support_k),
        )
        off_target_support = _mean_support(
            query_ids=off_target_ids,
            val_ids=val_ids,
            val_embeddings=val_embeddings,
            train_embeddings=train_embeddings,
            reference_indices=active_train_indices_by_class[class_id],
            support_k=int(config.support_k),
        )
        target_metrics = _metrics(pred_by_id, target_ids)
        off_target_metrics = _metrics(pred_by_id, off_target_ids)
        global_metrics = _metrics(pred_by_id, set(val_ids))
        result_rows.append(
            {
                "manifest_id": manifest_id,
                "region_id": region_id,
                "class_id": class_id,
                "class_name": manifest_row["class_name"],
                "pilot_type": manifest_row["pilot_type"],
                "control_family": manifest_row["control_family"],
                "retention_level": float(manifest_row["retention_level"]),
                "seed": int(manifest_row["seed"]),
                "target_val_count": len(target_ids),
                "same_class_offtarget_val_count": len(off_target_ids),
                "global_val_count": len(val_ids),
                "target_error": target_metrics["error"],
                "target_cross_entropy": target_metrics["cross_entropy"],
                "target_true_class_logit_margin": target_metrics["true_class_logit_margin"],
                "same_class_offtarget_error": off_target_metrics["error"],
                "same_class_offtarget_cross_entropy": off_target_metrics["cross_entropy"],
                "same_class_offtarget_true_class_logit_margin": off_target_metrics["true_class_logit_margin"],
                "global_error": global_metrics["error"],
                "global_cross_entropy": global_metrics["cross_entropy"],
                "global_true_class_logit_margin": global_metrics["true_class_logit_margin"],
                "target_support": target_support,
                "same_class_offtarget_support": off_target_support,
                "probe_predictions_csv": probe_row["predictions_csv"],
            }
        )

    effect_rows = _dose_response_effects(result_rows)
    paired_rows = _paired_control_effects(effect_rows)
    bootstrap_rows = _bootstrap_ci(
        effect_rows,
        value_keys=(
            "delta_target_error",
            "delta_target_cross_entropy",
            "delta_target_true_class_logit_margin",
            "delta_same_class_offtarget_error",
            "delta_same_class_offtarget_cross_entropy",
            "delta_global_error",
            "delta_global_cross_entropy",
        ),
    )
    results_hash = hash_rows(result_rows, prefix="deletion-eval")
    effect_hash = hash_rows(effect_rows, prefix="deletion-effects")
    paired_hash = hash_rows(paired_rows, prefix="deletion-paired")
    bootstrap_hash = hash_rows(bootstrap_rows, prefix="deletion-bootstrap")
    metadata = {
        "artifact_id": stable_hash(
            {
                "results_hash": results_hash,
                "effect_hash": effect_hash,
                "paired_hash": paired_hash,
                "bootstrap_hash": bootstrap_hash,
                "support_k": int(config.support_k),
            },
            prefix="deletion-eval-artifact",
        ),
        "experiment_id": config.experiment_id,
        "config": {
            "train_cache": str(config.train_cache),
            "validation_cache": str(config.validation_cache),
            "enriched_regions_dir": str(config.enriched_regions_dir),
            "deletion_controls_dir": str(config.deletion_controls_dir),
            "probe_output_dir": str(config.probe_output_dir),
            "output_dir": str(config.output_dir),
            "experiment_id": config.experiment_id,
            "support_k": int(config.support_k),
        },
        "results_hash": results_hash,
        "effect_hash": effect_hash,
        "paired_effect_hash": paired_hash,
        "bootstrap_hash": bootstrap_hash,
        "outputs": {
            "deletion_evaluation_csv": str(output / "deletion_evaluation.csv"),
            "dose_response_effects_csv": str(output / "dose_response_effects.csv"),
            "paired_control_effects_csv": str(output / "paired_control_effects.csv"),
            "region_bootstrap_ci_csv": str(output / "region_bootstrap_ci.csv"),
        },
        "summary": {
            "rows": len(result_rows),
            "effect_rows": len(effect_rows),
            "paired_effect_rows": len(paired_rows),
            "bootstrap_rows": len(bootstrap_rows),
            "unique_regions": len({row["region_id"] for row in result_rows}),
            "support_k": int(config.support_k),
        },
    }
    _write_csv(output / "deletion_evaluation.csv", result_rows)
    _write_csv(output / "dose_response_effects.csv", effect_rows)
    _write_csv(output / "paired_control_effects.csv", paired_rows)
    _write_csv(output / "region_bootstrap_ci.csv", bootstrap_rows)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


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


def _dose_response_effects(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    baseline_by_region_seed: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in rows:
        if str(row["control_family"]) == "baseline":
            baseline_by_region_seed[(str(row["region_id"]), int(row["seed"]))] = row
    out: list[dict[str, object]] = []
    for row in rows:
        base = baseline_by_region_seed.get((str(row["region_id"]), int(row["seed"])))
        if base is None:
            continue
        item = dict(row)
        for key in (
            "target_error",
            "target_cross_entropy",
            "target_true_class_logit_margin",
            "same_class_offtarget_error",
            "same_class_offtarget_cross_entropy",
            "same_class_offtarget_true_class_logit_margin",
            "global_error",
            "global_cross_entropy",
            "global_true_class_logit_margin",
            "target_support",
            "same_class_offtarget_support",
        ):
            item[f"baseline_{key}"] = float(base[key])
            item[f"delta_{key}"] = float(row[key]) - float(base[key])
        out.append(item)
    return out


def _paired_control_effects(effect_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, int, float], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in effect_rows:
        if str(row["control_family"]) == "baseline":
            continue
        key = (str(row["region_id"]), int(row["seed"]), float(row["retention_level"]))
        by_key[key][str(row["control_family"])] = row
    out: list[dict[str, object]] = []
    for (region_id, seed, retention), family_rows in sorted(by_key.items()):
        regional = family_rows.get("regional_drop")
        if regional is None:
            continue
        for family, control in sorted(family_rows.items()):
            if family == "regional_drop":
                continue
            out.append(
                {
                    "region_id": region_id,
                    "class_id": int(regional["class_id"]),
                    "pilot_type": regional["pilot_type"],
                    "seed": int(seed),
                    "retention_level": float(retention),
                    "control_family": family,
                    "regional_minus_control_target_error": float(regional["target_error"]) - float(control["target_error"]),
                    "regional_minus_control_target_cross_entropy": float(regional["target_cross_entropy"])
                    - float(control["target_cross_entropy"]),
                    "regional_minus_control_target_true_class_logit_margin": float(regional["target_true_class_logit_margin"])
                    - float(control["target_true_class_logit_margin"]),
                    "regional_minus_control_offtarget_error": float(regional["same_class_offtarget_error"])
                    - float(control["same_class_offtarget_error"]),
                    "regional_minus_control_offtarget_cross_entropy": float(regional["same_class_offtarget_cross_entropy"])
                    - float(control["same_class_offtarget_cross_entropy"]),
                    "regional_minus_control_global_error": float(regional["global_error"]) - float(control["global_error"]),
                    "regional_minus_control_global_cross_entropy": float(regional["global_cross_entropy"])
                    - float(control["global_cross_entropy"]),
                    "regional_minus_control_target_support": float(regional["target_support"]) - float(control["target_support"]),
                }
            )
    return out


def _bootstrap_ci(
    rows: Sequence[Mapping[str, object]],
    *,
    value_keys: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["pilot_type"]), str(row["control_family"]), float(row["retention_level"]))].append(row)
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, object]] = []
    for (pilot_type, control_family, retention), group_rows in sorted(groups.items()):
        region_ids = sorted({str(row["region_id"]) for row in group_rows})
        if not region_ids:
            continue
        by_region: dict[str, list[Mapping[str, object]]] = {
            region_id: [row for row in group_rows if str(row["region_id"]) == region_id] for region_id in region_ids
        }
        for value_key in value_keys:
            values_by_region = {
                region_id: float(np.mean([float(row[value_key]) for row in region_rows]))
                for region_id, region_rows in by_region.items()
            }
            observed = float(np.mean(list(values_by_region.values())))
            draws = []
            for _ in range(int(n_bootstrap)):
                sampled = rng.choice(region_ids, size=len(region_ids), replace=True)
                draws.append(float(np.mean([values_by_region[str(region_id)] for region_id in sampled])))
            out.append(
                {
                    "pilot_type": pilot_type,
                    "control_family": control_family,
                    "retention_level": float(retention),
                    "value": value_key,
                    "region_count": len(region_ids),
                    "mean": observed,
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "bootstrap_samples": int(n_bootstrap),
                }
            )
    return out


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
