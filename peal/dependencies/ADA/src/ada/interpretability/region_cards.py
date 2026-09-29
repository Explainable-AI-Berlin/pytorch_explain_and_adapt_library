from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ada.atlas.data.manifests import load_manifest_csv
from ada.atlas.hashing import hash_rows, stable_hash
from ada.interpretability.contrastive_concepts import (
    bootstrap_contrast_stability,
    bootstrap_top_frequency,
    concept_scores,
    format_top_phrases,
    permutation_null_p_values,
)
from ada.interpretability.sparse_concepts import sparse_positive_decomposition
from ada.interpretability.text_ambiguity import calibrate_temperature, class_prompt_matrix, text_class_metrics, within_class_standardize


@dataclass(frozen=True)
class RegionLanguageCardConfig:
    vlm_shared_dir: Path
    enriched_regions_dir: Path
    pilot_regions_dir: Path
    output_dir: Path
    interpretability_manifest_dir: Path | None = None
    train_cache: Path | None = None
    validation_cache: Path | None = None
    deletion_evaluation_csv: Path | None = None
    restoration_evaluation_csv: Path | None = None
    baseline_predictions_csv: Path | None = None
    control_max_count: int = 256
    top_k_phrases: int = 12
    bootstrap_samples: int = 100
    bootstrap_top_k: int = 20
    permutation_samples: int = 100
    min_bootstrap_frequency: float = 0.0
    min_supporting_images: int = 5
    sparse_enabled: bool = True
    sparse_max_terms: int = 8
    text_temperature: float | None = None
    text_logit_scale: float | None = None
    temperature_calibration_samples: int = 2048
    seed: int = 0
    overwrite: bool = False


def build_region_language_cards(config: RegionLanguageCardConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"region language cards already exist: {output}")
    output.mkdir(parents=True, exist_ok=True)

    shared = Path(config.vlm_shared_dir)
    train_embeddings = np.load(shared / "train_image_embeddings.npy", mmap_mode="r")
    val_embeddings = np.load(shared / "val_image_embeddings.npy", mmap_mode="r")
    phrase_embeddings = np.load(shared / "phrase_embeddings.npy", mmap_mode="r")
    phrase_rows = _read_csv(shared / "phrase_bank.csv")
    train_manifest_path = shared / "train_manifest.csv"
    val_manifest_path = shared / "val_manifest.csv"
    train_rows = load_manifest_csv(train_manifest_path if train_manifest_path.exists() else Path(config.train_cache or "") / "manifest.csv")
    val_rows = load_manifest_csv(val_manifest_path if val_manifest_path.exists() else Path(config.validation_cache or "") / "manifest.csv")
    train_idx = {row.sample_id: idx for idx, row in enumerate(train_rows)}
    val_idx = {row.sample_id: idx for idx, row in enumerate(val_rows)}

    selected_regions = _read_csv(Path(config.pilot_regions_dir) / "selected_pilot_regions.csv")
    membership = _read_csv(Path(config.enriched_regions_dir) / "region_membership_k10.csv")
    assignments = _read_csv(Path(config.enriched_regions_dir) / "validation_assignments_k10.csv")
    members_by_region: dict[str, list[str]] = defaultdict(list)
    train_ids_by_class: dict[int, list[str]] = defaultdict(list)
    for row in membership:
        sid = str(row["sample_id"])
        members_by_region[str(row["region_id"])].append(sid)
        train_ids_by_class[int(row["class_id"])].append(sid)
    val_ids_by_region: dict[str, list[str]] = defaultdict(list)
    for row in assignments:
        val_ids_by_region[str(row["region_id"])].append(str(row["sample_id"]))

    prediction_by_id = _predictions_by_id(config.baseline_predictions_csv)
    deletion_summary = _region_effect_summary(config.deletion_evaluation_csv)
    restoration_summary = _region_restoration_summary(config.restoration_evaluation_csv)
    class_embeddings, class_ids, class_display_names = class_prompt_matrix(phrase_rows, np.asarray(phrase_embeddings, dtype=np.float32))
    text_temperature = (
        float(config.text_temperature)
        if config.text_temperature is not None
        else calibrate_temperature(
            np.asarray(val_embeddings[: min(int(config.temperature_calibration_samples), len(val_rows))], dtype=np.float32),
            [int(row.class_id) for row in val_rows[: min(int(config.temperature_calibration_samples), len(val_rows))]],
            class_embeddings,
            class_ids,
        )
    )
    frozen_manifest = _manifest_groups(config.interpretability_manifest_dir)

    rng = np.random.default_rng(int(config.seed))
    card_rows: list[dict[str, object]] = []
    concept_rows: list[dict[str, object]] = []
    ambiguity_rows: list[dict[str, object]] = []
    sparse_rows: list[dict[str, object]] = []
    deleted_retained_rows: list[dict[str, object]] = []
    for region in selected_regions:
        region_id = str(region["region_id"])
        class_id = int(region["class_id"])
        manifest_groups = frozen_manifest.get(region_id, {})
        region_member_ids = _manifest_ids(manifest_groups, "region_train_member", train_idx) or [sid for sid in members_by_region.get(region_id, []) if sid in train_idx]
        if not region_member_ids:
            continue
        same_class_outside = [sid for sid in train_ids_by_class[class_id] if sid not in set(region_member_ids) and sid in train_idx]
        manifest_control_ids = _manifest_ids(manifest_groups, "same_class_high_support_control", train_idx)
        if manifest_control_ids:
            control_ids = manifest_control_ids[: int(config.control_max_count)]
        elif len(same_class_outside) > int(config.control_max_count):
            choice = rng.choice(np.asarray(same_class_outside, dtype=object), size=int(config.control_max_count), replace=False)
            control_ids = [str(x) for x in choice.tolist()]
        else:
            control_ids = same_class_outside

        region_emb = np.asarray(train_embeddings[[train_idx[sid] for sid in region_member_ids]], dtype=np.float32)
        control_emb = np.asarray(train_embeddings[[train_idx[sid] for sid in control_ids]], dtype=np.float32) if control_ids else np.empty((0, phrase_embeddings.shape[1]), dtype=np.float32)

        target_val_ids = _manifest_ids(manifest_groups, "region_val_member", val_idx) or [sid for sid in val_ids_by_region.get(region_id, []) if sid in val_idx]
        failure_ids = _manifest_ids(manifest_groups, "region_val_wrong_any_model", val_idx)
        correct_ids = _manifest_ids(manifest_groups, "region_val_correct_all_models", val_idx)
        if not failure_ids and not correct_ids:
            failure_ids, correct_ids = _split_failures(target_val_ids, prediction_by_id)
        failure_emb = np.asarray(val_embeddings[[val_idx[sid] for sid in failure_ids]], dtype=np.float32) if failure_ids else None
        correct_emb = np.asarray(val_embeddings[[val_idx[sid] for sid in correct_ids]], dtype=np.float32) if correct_ids else None

        scores = concept_scores(
            region_emb,
            np.asarray(phrase_embeddings, dtype=np.float32),
            control_embeddings=control_emb if len(control_emb) else None,
            failure_embeddings=failure_emb,
            correct_region_embeddings=correct_emb,
        )
        stability = bootstrap_top_frequency(
            region_emb,
            control_emb,
            np.asarray(phrase_embeddings, dtype=np.float32),
            top_k=int(config.bootstrap_top_k),
            n_bootstrap=int(config.bootstrap_samples),
            seed=int(stable_hash({"region_id": region_id, "seed": int(config.seed)})[-8:], 16),
        ) if len(control_emb) else np.zeros(len(phrase_rows), dtype=np.float32)
        region_stability = (
            bootstrap_contrast_stability(
                region_emb,
                control_emb,
                np.asarray(phrase_embeddings, dtype=np.float32),
                top_k=int(config.bootstrap_top_k),
                n_bootstrap=int(config.bootstrap_samples),
                seed=int(stable_hash({"region_id": region_id, "kind": "region", "seed": int(config.seed)})[-8:], 16),
            )
            if len(control_emb)
            else {}
        )
        region_perm_p = (
            permutation_null_p_values(
                region_emb,
                control_emb,
                np.asarray(phrase_embeddings, dtype=np.float32),
                n_permutations=int(config.permutation_samples),
                seed=int(stable_hash({"region_id": region_id, "kind": "perm", "seed": int(config.seed)})[-8:], 16),
            )
            if len(control_emb)
            else np.full(len(phrase_rows), math.nan, dtype=np.float32)
        )

        for phrase_idx, phrase in enumerate(phrase_rows):
            concept_rows.append(
                {
                    "region_id": region_id,
                    "class_id": class_id,
                    "class_name": region["class_name"],
                    "pilot_type": region.get("pilot_type", ""),
                    "phrase_id": phrase["phrase_id"],
                    "phrase": phrase["phrase"],
                    "phrase_type": phrase["phrase_type"],
                    "group": phrase.get("group", ""),
                    "absolute_score": float(scores["absolute_score"][phrase_idx]),
                    "region_contrast_score": _score_or_nan(scores, "region_contrast_score", phrase_idx),
                    "region_contrast_standardized_effect": _score_or_nan(scores, "region_contrast_standardized_effect", phrase_idx),
                    "failure_contrast_score": _score_or_nan(scores, "failure_contrast_score", phrase_idx),
                    "failure_contrast_standardized_effect": _score_or_nan(scores, "failure_contrast_standardized_effect", phrase_idx),
                    "region_contrast_bootstrap_top_frequency": float(stability[phrase_idx]),
                    "region_contrast_sign_consistency": _stability_or_nan(region_stability, "sign_consistency", phrase_idx),
                    "region_contrast_ci_low": _stability_or_nan(region_stability, "effect_ci_low", phrase_idx),
                    "region_contrast_ci_high": _stability_or_nan(region_stability, "effect_ci_high", phrase_idx),
                    "region_contrast_permutation_p": float(region_perm_p[phrase_idx]),
                }
            )

        ambiguity, region_ambiguity_rows = _region_text_ambiguity(
            target_val_ids=target_val_ids,
            val_idx=val_idx,
            val_embeddings=val_embeddings,
            labels=[int(row.class_id) for row in val_rows],
            class_embeddings=class_embeddings,
            class_ids=class_ids,
            class_display_names=class_display_names,
            temperature=float(text_temperature),
        )
        for row in region_ambiguity_rows:
            ambiguity_rows.append({"region_id": region_id, "class_id": class_id, "class_name": region["class_name"], **row})

        deleted_summary = _deleted_vs_retained_concepts(
            region=region,
            manifest_groups=manifest_groups,
            train_idx=train_idx,
            train_embeddings=train_embeddings,
            phrase_rows=phrase_rows,
            phrase_embeddings=np.asarray(phrase_embeddings, dtype=np.float32),
            config=config,
        )
        deleted_retained_rows.extend(deleted_summary["rows"])
        deleted_top = deleted_summary["top_phrases"]

        sparse_top = ""
        if config.sparse_enabled:
            coeff_region = sparse_positive_decomposition(region_emb, np.asarray(phrase_embeddings, dtype=np.float32), max_terms=int(config.sparse_max_terms))
            coeff_control = (
                sparse_positive_decomposition(control_emb, np.asarray(phrase_embeddings, dtype=np.float32), max_terms=int(config.sparse_max_terms))
                if len(control_emb)
                else np.zeros_like(coeff_region[:0])
            )
            sparse_delta = coeff_region.mean(axis=0) - (coeff_control.mean(axis=0) if len(coeff_control) else 0.0)
            sparse_top = format_top_phrases(phrase_rows, sparse_delta, top_k=int(config.top_k_phrases))
            for phrase_idx in np.argsort(-sparse_delta)[: int(config.top_k_phrases)]:
                sparse_rows.append(
                    {
                        "region_id": region_id,
                        "class_id": class_id,
                        "class_name": region["class_name"],
                        "phrase_id": phrase_rows[int(phrase_idx)]["phrase_id"],
                        "phrase": phrase_rows[int(phrase_idx)]["phrase"],
                        "sparse_region_minus_control": float(sparse_delta[int(phrase_idx)]),
                    }
                )

        card_rows.append(
            {
                "region_id": region_id,
                "class_id": class_id,
                "class_name": region["class_name"],
                "pilot_type": region.get("pilot_type", ""),
                "train_count": int(float(region.get("train_count", len(region_member_ids)))),
                "validation_count": int(float(region.get("validation_count", len(target_val_ids)))),
                "member_support_pct_median": _float_or_nan(region.get("member_support_pct_median", "")),
                "global_neighbor_purity": _float_or_nan(region.get("global_neighbor_purity", "")),
                "local_label_entropy": _float_or_nan(region.get("local_label_entropy", "")),
                "robust_class_margin": _float_or_nan(region.get("robust_class_margin", "")),
                "q25_regional_delete_delta_target_error": deletion_summary.get(region_id, {}).get("q25_regional_delete_delta_target_error", math.nan),
                "q25_unique_target_repair_gain": restoration_summary.get(region_id, {}).get("q25_unique_target_repair_gain", math.nan),
                "q25_same_class_nontarget_repair_gain": restoration_summary.get(region_id, {}).get("q25_same_class_nontarget_repair_gain", math.nan),
                "region_member_count_encoded": len(region_member_ids),
                "control_count_encoded": len(control_ids),
                "target_validation_count_encoded": len(target_val_ids),
                "failure_count": len(failure_ids),
                "correct_region_count": len(correct_ids),
                "mean_text_class_margin": ambiguity["mean_text_class_margin"],
                "mean_text_class_entropy": ambiguity["mean_text_class_entropy"],
                "top_competing_classes": ambiguity["top_competing_classes"],
                "top_absolute_concepts": format_top_phrases(phrase_rows, scores["absolute_score"], top_k=int(config.top_k_phrases)),
                "top_region_vs_control_concepts": format_top_phrases(
                    phrase_rows,
                    scores.get("region_contrast_score", np.zeros(len(phrase_rows), dtype=np.float32)),
                    top_k=int(config.top_k_phrases),
                    stability=stability,
                    min_stability=float(config.min_bootstrap_frequency),
                ),
                "top_failure_vs_correct_concepts": format_top_phrases(
                    phrase_rows,
                    scores.get("failure_contrast_score", np.zeros(len(phrase_rows), dtype=np.float32)),
                    top_k=int(config.top_k_phrases),
                ),
                "top_deleted_vs_retained_concepts": deleted_top,
                "top_sparse_region_vs_control_concepts": sparse_top,
            }
        )

    _write_csv(output / "region_language_cards.csv", card_rows)
    _write_csv(output / "region_concept_scores.csv", concept_rows)
    _write_csv(output / "region_text_ambiguity.csv", ambiguity_rows)
    _write_csv(output / "region_sparse_concepts.csv", sparse_rows)
    _write_csv(output / "region_deleted_vs_retained_concepts.csv", deleted_retained_rows)
    result_hash = hash_rows(card_rows, prefix="region-language-cards")
    metadata = {
        "artifact_id": stable_hash(
            {
                "result_hash": result_hash,
                "vlm_shared_dir": str(config.vlm_shared_dir),
                "enriched_regions_dir": str(config.enriched_regions_dir),
                "pilot_regions_dir": str(config.pilot_regions_dir),
            },
            prefix="region-language-artifact",
        ),
        "result_hash": result_hash,
        "config": {
            "vlm_shared_dir": str(config.vlm_shared_dir),
            "enriched_regions_dir": str(config.enriched_regions_dir),
            "pilot_regions_dir": str(config.pilot_regions_dir),
            "output_dir": str(config.output_dir),
            "interpretability_manifest_dir": "" if config.interpretability_manifest_dir is None else str(config.interpretability_manifest_dir),
            "deletion_evaluation_csv": None if config.deletion_evaluation_csv is None else str(config.deletion_evaluation_csv),
            "restoration_evaluation_csv": None if config.restoration_evaluation_csv is None else str(config.restoration_evaluation_csv),
            "baseline_predictions_csv": None if config.baseline_predictions_csv is None else str(config.baseline_predictions_csv),
            "control_max_count": int(config.control_max_count),
            "top_k_phrases": int(config.top_k_phrases),
            "bootstrap_samples": int(config.bootstrap_samples),
            "bootstrap_top_k": int(config.bootstrap_top_k),
            "permutation_samples": int(config.permutation_samples),
            "min_bootstrap_frequency": float(config.min_bootstrap_frequency),
            "min_supporting_images": int(config.min_supporting_images),
            "sparse_enabled": bool(config.sparse_enabled),
            "text_temperature": float(text_temperature),
            "text_logit_scale": None if config.text_logit_scale is None else float(config.text_logit_scale),
            "seed": int(config.seed),
        },
        "summary": {
            "regions": len(card_rows),
            "concept_rows": len(concept_rows),
            "ambiguity_rows": len(ambiguity_rows),
            "sparse_rows": len(sparse_rows),
            "deleted_vs_retained_rows": len(deleted_retained_rows),
        },
        "outputs": {
            "region_language_cards_csv": str(output / "region_language_cards.csv"),
            "region_concept_scores_csv": str(output / "region_concept_scores.csv"),
            "region_text_ambiguity_csv": str(output / "region_text_ambiguity.csv"),
            "region_sparse_concepts_csv": str(output / "region_sparse_concepts.csv"),
            "region_deleted_vs_retained_concepts_csv": str(output / "region_deleted_vs_retained_concepts.csv"),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def _region_text_ambiguity(
    *,
    target_val_ids: Sequence[str],
    val_idx: Mapping[str, int],
    val_embeddings: np.ndarray,
    labels: Sequence[int],
    class_embeddings: np.ndarray,
    class_ids: Sequence[int],
    class_display_names: Sequence[str],
    temperature: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    idx = [val_idx[sid] for sid in target_val_ids if sid in val_idx]
    if not idx:
        return (
            {
                "mean_text_class_margin": math.nan,
                "mean_text_class_entropy": math.nan,
                "mean_text_true_class_probability": math.nan,
                "top_competing_classes": "",
            },
            [],
        )
    region_labels = [int(labels[i]) for i in idx]
    metrics = text_class_metrics(
        np.asarray(val_embeddings[idx], dtype=np.float32),
        region_labels,
        class_embeddings,
        class_ids,
        temperature=float(temperature),
    )
    competing = metrics["text_competing_class_id"]
    names_by_id = {int(cid): name for cid, name in zip(class_ids, class_display_names)}
    counts: dict[int, int] = defaultdict(int)
    for class_id in competing:
        counts[int(class_id)] += 1
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    ambiguity_z = within_class_standardize(-np.asarray(metrics["text_class_margin"], dtype=np.float32), region_labels)
    rows: list[dict[str, object]] = []
    for local_idx, sample_id in enumerate([sid for sid in target_val_ids if sid in val_idx]):
        rows.append(
            {
                "sample_id": sample_id,
                "true_class_text_similarity": float(metrics["true_class_text_similarity"][local_idx]),
                "true_class_text_rank": int(metrics["true_class_text_rank"][local_idx]),
                "top_text_class_id": int(metrics["top_text_class_id"][local_idx]),
                "top_text_class_name": names_by_id.get(int(metrics["top_text_class_id"][local_idx]), str(metrics["top_text_class_id"][local_idx])),
                "top_competing_class_id": int(metrics["text_competing_class_id"][local_idx]),
                "top_competing_class_name": names_by_id.get(int(metrics["text_competing_class_id"][local_idx]), str(metrics["text_competing_class_id"][local_idx])),
                "top_competing_similarity": float(metrics["text_competing_score"][local_idx]),
                "text_margin": float(metrics["text_class_margin"][local_idx]),
                "top_two_text_margin": float(metrics["top_two_text_margin"][local_idx]),
                "text_entropy": float(metrics["text_class_entropy"][local_idx]),
                "text_true_class_probability": float(metrics["text_true_class_probability"][local_idx]),
                "within_class_standardized_ambiguity": float(ambiguity_z[local_idx]),
            }
        )
    return (
        {
            "mean_text_class_margin": float(np.mean(metrics["text_class_margin"])),
            "mean_text_class_entropy": float(np.mean(metrics["text_class_entropy"])),
            "mean_text_true_class_probability": float(np.mean(metrics["text_true_class_probability"])),
            "top_competing_classes": ";".join(f"{names_by_id.get(cid, str(cid))}:{count}" for cid, count in top),
        },
        rows,
    )


def _deleted_vs_retained_concepts(
    *,
    region: Mapping[str, str],
    manifest_groups: Mapping[str, Sequence[Mapping[str, str]]],
    train_idx: Mapping[str, int],
    train_embeddings: np.ndarray,
    phrase_rows: Sequence[dict[str, str]],
    phrase_embeddings: np.ndarray,
    config: RegionLanguageCardConfig,
) -> dict[str, object]:
    region_id = str(region["region_id"])
    rows: list[dict[str, object]] = []
    top_by_seed: list[str] = []
    retained_by_seed = _manifest_ids_by_seed(manifest_groups, "q025_retained_anchor", train_idx)
    deleted_by_seed = _manifest_ids_by_seed(manifest_groups, "q025_deleted_real", train_idx)
    for deletion_seed in sorted(set(retained_by_seed) | set(deleted_by_seed)):
        retained_ids = retained_by_seed.get(deletion_seed, [])
        deleted_ids = deleted_by_seed.get(deletion_seed, [])
        if len(retained_ids) < int(config.min_supporting_images) or len(deleted_ids) < int(config.min_supporting_images):
            continue
        retained_emb = np.asarray(train_embeddings[[train_idx[sid] for sid in retained_ids]], dtype=np.float32)
        deleted_emb = np.asarray(train_embeddings[[train_idx[sid] for sid in deleted_ids]], dtype=np.float32)
        scores = concept_scores(
            deleted_emb,
            phrase_embeddings,
            deleted_embeddings=deleted_emb,
            retained_embeddings=retained_emb,
        )
        stability = bootstrap_contrast_stability(
            deleted_emb,
            retained_emb,
            phrase_embeddings,
            top_k=int(config.bootstrap_top_k),
            n_bootstrap=int(config.bootstrap_samples),
            seed=int(stable_hash({"region_id": region_id, "kind": "deleted_retained", "seed": deletion_seed})[-8:], 16),
        )
        perm_p = permutation_null_p_values(
            deleted_emb,
            retained_emb,
            phrase_embeddings,
            n_permutations=int(config.permutation_samples),
            seed=int(stable_hash({"region_id": region_id, "kind": "deleted_retained_perm", "seed": deletion_seed})[-8:], 16),
        )
        score = scores["deleted_vs_retained_score"]
        top_by_seed.append(format_top_phrases(phrase_rows, score, top_k=int(config.top_k_phrases), stability=stability["top_frequency"], min_stability=float(config.min_bootstrap_frequency)))
        for phrase_idx, phrase in enumerate(phrase_rows):
            rows.append(
                {
                    "region_id": region_id,
                    "class_id": int(region["class_id"]),
                    "class_name": region["class_name"],
                    "pilot_type": region.get("pilot_type", ""),
                    "deletion_seed": deletion_seed,
                    "retained_count": len(retained_ids),
                    "deleted_count": len(deleted_ids),
                    "phrase_id": phrase["phrase_id"],
                    "phrase": phrase["phrase"],
                    "phrase_type": phrase["phrase_type"],
                    "deleted_vs_retained_score": float(score[phrase_idx]),
                    "deleted_vs_retained_standardized_effect": float(scores["deleted_vs_retained_standardized_effect"][phrase_idx]),
                    "bootstrap_top_frequency": float(stability["top_frequency"][phrase_idx]),
                    "sign_consistency": float(stability["sign_consistency"][phrase_idx]),
                    "effect_ci_low": float(stability["effect_ci_low"][phrase_idx]),
                    "effect_ci_high": float(stability["effect_ci_high"][phrase_idx]),
                    "permutation_p": float(perm_p[phrase_idx]),
                }
            )
    return {"rows": rows, "top_phrases": " | ".join(text for text in top_by_seed if text)}


def _split_failures(sample_ids: Sequence[str], prediction_by_id: Mapping[str, Mapping[str, str]]) -> tuple[list[str], list[str]]:
    if not prediction_by_id:
        return [], []
    failure_ids: list[str] = []
    correct_ids: list[str] = []
    for sid in sample_ids:
        row = prediction_by_id.get(sid)
        if row is None:
            continue
        if int(row.get("correct", "0")):
            correct_ids.append(sid)
        else:
            failure_ids.append(sid)
    return failure_ids, correct_ids


def _predictions_by_id(path: Path | None) -> dict[str, Mapping[str, str]]:
    if path is None or not str(path):
        return {}
    return {str(row["sample_id"]): row for row in _read_csv(Path(path))}


def _manifest_groups(path: Path | None) -> dict[str, dict[str, list[dict[str, str]]]]:
    if path is None or not str(path):
        return {}
    csv_path = Path(path) / "samples.csv"
    if not csv_path.exists():
        return {}
    out: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in _read_csv(csv_path):
        out[str(row["region_id"])][str(row["analysis_group"])].append(row)
    return {region_id: dict(groups) for region_id, groups in out.items()}


def _manifest_ids(groups: Mapping[str, Sequence[Mapping[str, str]]], group: str, index: Mapping[str, int]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in groups.get(group, []):
        sid = str(row["sample_id"])
        if sid not in index or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _manifest_ids_by_seed(groups: Mapping[str, Sequence[Mapping[str, str]]], group: str, index: Mapping[str, int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in groups.get(group, []):
        sid = str(row["sample_id"])
        seed = str(row.get("deletion_seed", ""))
        key = (seed, sid)
        if sid not in index or key in seen:
            continue
        seen.add(key)
        out[seed].append(sid)
    return dict(out)


def _region_effect_summary(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not str(path) or not Path(path).exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    for row in _read_csv(Path(path)):
        if str(row.get("control_family", "")) == "regional_drop" and abs(float(row.get("retention_level", "nan")) - 0.25) < 1.0e-9:
            vals = out.setdefault(str(row["region_id"]), {"q25_regional_delete_delta_target_error": []})
            vals["q25_regional_delete_delta_target_error"].append(float(row.get("delta_target_error", math.nan)))
    return {region_id: {key: float(np.nanmean(value)) for key, value in vals.items()} for region_id, vals in out.items()}


def _region_restoration_summary(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not str(path) or not Path(path).exists():
        return {}
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in _read_csv(Path(path)):
        if abs(float(row.get("retention_level", "nan")) - 0.25) > 1.0e-9:
            continue
        if abs(float(row.get("restoration_budget_fraction", "nan")) - 1.0) > 1.0e-9:
            continue
        condition = str(row.get("restoration_condition", ""))
        if condition == "unique_target_region_restoration":
            out[str(row["region_id"])]["q25_unique_target_repair_gain"].append(float(row.get("repair_gain_target_error", math.nan)))
        elif condition == "unique_same_class_non_target_additions":
            out[str(row["region_id"])]["q25_same_class_nontarget_repair_gain"].append(float(row.get("repair_gain_target_error", math.nan)))
    return {region_id: {key: float(np.nanmean(value)) for key, value in vals.items()} for region_id, vals in out.items()}


def _score_or_nan(scores: Mapping[str, np.ndarray], key: str, idx: int) -> float:
    value = scores.get(key)
    if value is None:
        return math.nan
    return float(value[int(idx)])


def _stability_or_nan(stability: Mapping[str, np.ndarray], key: str, idx: int) -> float:
    value = stability.get(key)
    if value is None:
        return math.nan
    return float(value[int(idx)])


def _float_or_nan(value: object) -> float:
    if value in ("", None):
        return math.nan
    return float(value)


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
