from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ada.atlas.hashing import hash_rows, stable_hash


@dataclass(frozen=True)
class OutlierImageReportConfig:
    manifest_dir: Path
    region_cards_csv: Path
    text_ambiguity_csv: Path
    output_dir: Path
    train_image_root: Path
    validation_image_root: Path
    max_images: int = 16
    top_concepts: int = 4
    overwrite: bool = False


def build_outlier_image_report(config: OutlierImageReportConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"outlier image report already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_csv(Path(config.manifest_dir) / "samples.csv")
    cards = {str(row["region_id"]): row for row in _read_csv(config.region_cards_csv)}
    ambiguity_rows = _read_csv(config.text_ambiguity_csv)
    manifest_by_id = {
        str(row["sample_id"]): row
        for row in manifest_rows
        if str(row.get("analysis_group", "")) == "region_val_member"
    }

    joined: list[dict[str, object]] = []
    for row in ambiguity_rows:
        sample_id = str(row["sample_id"])
        sample = manifest_by_id.get(sample_id)
        if sample is None:
            continue
        region = cards.get(str(row["region_id"]), {})
        image_path = Path(config.validation_image_root) / str(sample["relative_path"])
        concepts = _top_phrases(str(region.get("top_sparse_region_vs_control_concepts", "")), int(config.top_concepts))
        if not concepts:
            concepts = _top_phrases(str(region.get("top_region_vs_control_concepts", "")), int(config.top_concepts))
        deleted_retained = _top_phrases(str(region.get("top_deleted_vs_retained_concepts", "")), int(config.top_concepts))
        description = _description(row, region, concepts, deleted_retained)
        joined.append(
            {
                "sample_id": sample_id,
                "region_id": str(row["region_id"]),
                "pilot_type": str(region.get("pilot_type", sample.get("pilot_type", ""))),
                "class_id": int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "relative_path": str(sample["relative_path"]),
                "image_path": str(image_path),
                "true_class_text_similarity": _float(row.get("true_class_text_similarity")),
                "true_class_text_rank": int(float(row.get("true_class_text_rank", "0") or "0")),
                "top_competing_class_name": str(row.get("top_competing_class_name", "")),
                "top_competing_similarity": _float(row.get("top_competing_similarity")),
                "text_margin": _float(row.get("text_margin")),
                "text_entropy": _float(row.get("text_entropy")),
                "within_class_standardized_ambiguity": _float(row.get("within_class_standardized_ambiguity")),
                "q25_regional_delete_delta_target_error": _float(region.get("q25_regional_delete_delta_target_error")),
                "q25_unique_target_repair_gain": _float(region.get("q25_unique_target_repair_gain")),
                "q25_same_class_nontarget_repair_gain": _float(region.get("q25_same_class_nontarget_repair_gain")),
                "region_descriptors": "; ".join(concepts),
                "deleted_vs_retained_descriptors": "; ".join(deleted_retained),
                "description": description,
            }
        )

    joined.sort(key=lambda row: (float(row["text_margin"]), -float(row["text_entropy"])))
    selected = joined[: int(config.max_images)]
    rows_csv = output / "outlier_image_descriptions.csv"
    report_md = output / "outlier_image_descriptions.md"
    _write_csv(rows_csv, selected)
    report_md.write_text(_markdown_report(selected, config), encoding="utf-8")
    result_hash = hash_rows(selected, prefix="outlier-image-descriptions")
    metadata = {
        "artifact_id": stable_hash(
            {
                "result_hash": result_hash,
                "manifest_dir": str(config.manifest_dir),
                "region_cards_csv": str(config.region_cards_csv),
                "text_ambiguity_csv": str(config.text_ambiguity_csv),
            },
            prefix="outlier-image-artifact",
        ),
        "result_hash": result_hash,
        "summary": {
            "candidate_validation_images": len(joined),
            "reported_images": len(selected),
            "selection": "lowest SigLIP2 true-class text margin among validation members of the eight causal DINO regions",
        },
        "outputs": {
            "outlier_image_descriptions_csv": str(rows_csv),
            "outlier_image_descriptions_md": str(report_md),
        },
        "config": {
            "manifest_dir": str(config.manifest_dir),
            "region_cards_csv": str(config.region_cards_csv),
            "text_ambiguity_csv": str(config.text_ambiguity_csv),
            "train_image_root": str(config.train_image_root),
            "validation_image_root": str(config.validation_image_root),
            "max_images": int(config.max_images),
            "top_concepts": int(config.top_concepts),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    completed.write_text("ok\n", encoding="utf-8")
    return metadata


def _description(
    row: Mapping[str, object],
    region: Mapping[str, object],
    concepts: Sequence[str],
    deleted_retained: Sequence[str],
) -> str:
    true_name = str(row.get("class_name", ""))
    competitor = str(row.get("top_competing_class_name", ""))
    margin = _float(row.get("text_margin"))
    entropy = _float(row.get("text_entropy"))
    pilot_type = str(region.get("pilot_type", ""))
    repair = _float(region.get("q25_unique_target_repair_gain"))
    nontarget = _float(region.get("q25_same_class_nontarget_repair_gain"))
    concept_text = ", ".join(concepts[:3]) if concepts else "no stable phrase-bank descriptor"
    missing_text = ", ".join(deleted_retained[:2]) if deleted_retained else "no deleted-vs-retained descriptor"
    return (
        f"{pilot_type} region for {true_name}; SigLIP2 text margin {margin:.4f} "
        f"against top competitor {competitor} with entropy {entropy:.4f}. "
        f"Region-level descriptors: {concept_text}. "
        f"Deleted-vs-retained hints: {missing_text}. "
        f"Unique target repair {repair:.4f}; same-class non-target repair {nontarget:.4f}."
    )


def _markdown_report(rows: Sequence[Mapping[str, object]], config: OutlierImageReportConfig) -> str:
    lines = [
        "# Outlier Image Descriptions",
        "",
        "These are validation images from the eight causally tested DINOv2 regions, selected by lowest SigLIP2 shared image-text true-class margin.",
        "Descriptions are phrase-retrieval summaries from the completed SigLIP2 language-card artifact, not free-form Qwen3-VL output.",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        path = str(row["image_path"])
        lines.extend(
            [
                f"## {idx}. {row['class_name']} / {row['pilot_type']}",
                "",
                f"- image: [{row['relative_path']}](<{path}>)",
                f"- sample: `{row['sample_id']}`",
                f"- region: `{row['region_id']}`",
                f"- text margin: `{float(row['text_margin']):.4f}`",
                f"- true-class rank: `{row['true_class_text_rank']}`",
                f"- top competing text class: `{row['top_competing_class_name']}`",
                f"- text entropy: `{float(row['text_entropy']):.4f}`",
                f"- region descriptors: {row['region_descriptors']}",
                f"- deleted-vs-retained hints: {row['deleted_vs_retained_descriptors']}",
                f"- causal context: q=0.25 deletion delta `{float(row['q25_regional_delete_delta_target_error']):.4f}`, unique repair `{float(row['q25_unique_target_repair_gain']):.4f}`, non-target repair `{float(row['q25_same_class_nontarget_repair_gain']):.4f}`",
                "",
                str(row["description"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Provenance",
            "",
            f"- manifest: `{config.manifest_dir}`",
            f"- region cards: `{config.region_cards_csv}`",
            f"- text ambiguity: `{config.text_ambiguity_csv}`",
            "",
        ]
    )
    return "\n".join(lines)


def _top_phrases(text: str, k: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for block in str(text).split("|"):
        for item in block.split(";"):
            item = item.strip()
            if not item:
                continue
            phrase = item.split(":")[0].strip()
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            out.append(phrase)
            if len(out) >= int(k):
                return out
    return out


def _float(value: object) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


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
