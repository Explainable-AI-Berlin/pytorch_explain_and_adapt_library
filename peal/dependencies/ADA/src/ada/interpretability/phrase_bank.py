from __future__ import annotations

import os

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ada.atlas.data.imagenet_meta import class_display_names
from ada.atlas.data.manifests import load_manifest_csv
from ada.atlas.hashing import hash_rows, stable_hash


DEFAULT_GENERAL_PHRASES = (
    ("viewpoint", "a side view"),
    ("viewpoint", "a rear view"),
    ("viewpoint", "viewed from above"),
    ("viewpoint", "viewed from below"),
    ("viewpoint", "a front view"),
    ("viewpoint", "a profile view"),
    ("viewpoint", "an unusual viewpoint"),
    ("pose", "an unusual pose"),
    ("pose", "a typical pose"),
    ("pose", "the object is lying down"),
    ("pose", "the object is standing upright"),
    ("scale", "a close-up photo"),
    ("scale", "a small object in a large scene"),
    ("scale", "an object filling the frame"),
    ("scale", "a distant object"),
    ("scale", "a tiny object"),
    ("scale", "a large object"),
    ("visibility", "partially hidden"),
    ("visibility", "heavily occluded"),
    ("visibility", "partly outside the frame"),
    ("visibility", "the whole object is visible"),
    ("visibility", "only part of the object is visible"),
    ("visibility", "the object is blocked by another object"),
    ("count", "multiple objects"),
    ("count", "a single object"),
    ("count", "a crowd of objects"),
    ("count", "two objects"),
    ("background", "a cluttered background"),
    ("background", "a plain background"),
    ("background", "an outdoor scene"),
    ("background", "an indoor scene"),
    ("background", "a natural background"),
    ("background", "an artificial background"),
    ("background", "a messy scene"),
    ("background", "a clean simple scene"),
    ("lighting", "low light"),
    ("lighting", "bright sunlight"),
    ("lighting", "strong shadows"),
    ("lighting", "dim lighting"),
    ("lighting", "backlit"),
    ("lighting", "overexposed"),
    ("lighting", "underexposed"),
    ("quality", "motion blur"),
    ("quality", "out of focus"),
    ("quality", "low resolution"),
    ("quality", "sharp focus"),
    ("quality", "noisy image"),
    ("quality", "compressed image"),
    ("medium", "a drawing"),
    ("medium", "an illustration"),
    ("medium", "a sculpture"),
    ("medium", "a black and white image"),
    ("medium", "a painting"),
    ("medium", "a cartoon"),
    ("medium", "a photo of a toy"),
    ("medium", "a real photograph"),
    ("color", "unusual color"),
    ("color", "bright colors"),
    ("color", "dark colors"),
    ("color", "muted colors"),
    ("color", "high contrast"),
    ("color", "low contrast"),
    ("texture", "unusual texture"),
    ("texture", "smooth texture"),
    ("texture", "rough texture"),
    ("texture", "striped texture"),
    ("texture", "spotted texture"),
    ("composition", "cropped composition"),
    ("composition", "object against a busy scene"),
    ("composition", "centered object"),
    ("composition", "off-center object"),
    ("composition", "object near the edge of the frame"),
    ("context", "unusual context"),
    ("context", "typical context"),
    ("context", "object in use"),
    ("context", "object on the ground"),
    ("context", "object in water"),
    ("context", "object in the sky"),
)

CLASS_TEMPLATES = (
    "a photo of a {}",
    "an image of a {}",
    "a photograph of the {}",
    "the object is a {}",
)

CLASS_ATTRIBUTE_TEMPLATES = (
    ("viewpoint", "a side-view photo of a {}"),
    ("viewpoint", "a front-view photo of a {}"),
    ("viewpoint", "a rear-view photo of a {}"),
    ("viewpoint", "a {} viewed from above"),
    ("scale", "a close-up photo of a {}"),
    ("scale", "a small {} in a large scene"),
    ("scale", "a distant {}"),
    ("visibility", "a partially occluded {}"),
    ("visibility", "a mostly hidden {}"),
    ("visibility", "a cropped {}"),
    ("count", "multiple {} objects"),
    ("background", "a {} against a cluttered background"),
    ("background", "a {} against a plain background"),
    ("lighting", "a {} in dark lighting"),
    ("lighting", "a backlit {}"),
    ("quality", "a blurry photo of a {}"),
    ("medium", "a drawing of a {}"),
    ("medium", "a painting of a {}"),
    ("color", "an unusually colored {}"),
    ("context", "a {} in an unusual context"),
)


@dataclass(frozen=True)
class PhraseBankConfig:
    train_cache: Path
    output_dir: Path
    imagenet_meta: Path | None = Path(os.path.join(os.environ.get("PEAL_DATA", "datasets"), "imagenet_torchvision/data/meta.bin"))
    extra_phrases_csv: Path | None = None
    confusion_prompts_csv: Path | None = None
    include_general_phrases: bool = True
    include_class_prompts: bool = True
    include_class_attribute_prompts: bool = True
    overwrite: bool = False


def build_phrase_bank(config: PhraseBankConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"phrase bank already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    phrase_bank_version = "siglip2_region_language_v2"
    if config.include_general_phrases:
        for group, phrase in DEFAULT_GENERAL_PHRASES:
            rows.append(_phrase_row(phrase=phrase, phrase_type="general", group=group, source="default", phrase_bank_version=phrase_bank_version))

    train_rows = load_manifest_csv(Path(config.train_cache) / "manifest.csv")
    class_names_by_id = [None] * (max(row.class_id for row in train_rows) + 1)
    for row in train_rows:
        class_names_by_id[int(row.class_id)] = str(row.class_name)
    class_names = [str(name) for name in class_names_by_id]
    display_names = class_display_names(class_names, config.imagenet_meta)
    if config.include_class_prompts:
        for class_id, (class_name, display_name) in enumerate(zip(class_names, display_names)):
            for synonym in _synonyms(display_name):
                for template_id, template in enumerate(CLASS_TEMPLATES):
                    rows.append(
                        _phrase_row(
                            phrase=template.format(synonym),
                            phrase_type="class_prompt",
                            group=f"class_template_{template_id}",
                            source="imagenet_meta",
                            class_id=class_id,
                            class_name=class_name,
                            display_name=display_name,
                            synonym=synonym,
                            phrase_bank_version=phrase_bank_version,
                        )
                    )
    if config.include_class_attribute_prompts:
        for class_id, (class_name, display_name) in enumerate(zip(class_names, display_names)):
            synonym = _synonyms(display_name)[0]
            for group, template in CLASS_ATTRIBUTE_TEMPLATES:
                rows.append(
                    _phrase_row(
                        phrase=template.format(synonym),
                        phrase_type="class_attribute",
                        group=group,
                        source="imagenet_meta",
                        class_id=class_id,
                        class_name=class_name,
                        display_name=display_name,
                        synonym=synonym,
                        class_conditional=True,
                        phrase_bank_version=phrase_bank_version,
                    )
                )

    if config.extra_phrases_csv is not None and str(config.extra_phrases_csv):
        for row in _read_csv(Path(config.extra_phrases_csv)):
            phrase = str(row.get("phrase", "")).strip()
            if not phrase:
                continue
            rows.append(
                _phrase_row(
                    phrase=phrase,
                    phrase_type=str(row.get("phrase_type", "extra")),
                    group=str(row.get("group", "extra")),
                    source=str(config.extra_phrases_csv),
                    class_id=_optional_int(row.get("class_id", "")),
                    class_name=str(row.get("class_name", "")),
                    display_name=str(row.get("display_name", "")),
                    phrase_bank_version=phrase_bank_version,
                )
            )

    if config.confusion_prompts_csv is not None and str(config.confusion_prompts_csv):
        for row in _read_csv(Path(config.confusion_prompts_csv)):
            class_id = _optional_int(row.get("class_id", ""))
            phrase = str(row.get("phrase", "")).strip()
            if not phrase:
                target = str(row.get("target_class_name", "")).strip()
                competitor = str(row.get("confusion_class_name", "")).strip()
                if target and competitor:
                    phrase = f"an image that could be a {target} or a {competitor}"
            if not phrase:
                continue
            rows.append(
                _phrase_row(
                    phrase=phrase,
                    phrase_type="confusion_prompt",
                    group=str(row.get("group", "known_confusion")),
                    source=str(config.confusion_prompts_csv),
                    class_id=class_id,
                    class_name=str(row.get("class_name", "")),
                    display_name=str(row.get("display_name", row.get("target_class_name", ""))),
                    phrase_bank_version=phrase_bank_version,
                )
            )

    rows = _dedupe_phrases(rows)
    for idx, row in enumerate(rows):
        row["phrase_id"] = stable_hash({"idx": idx, "phrase": row["phrase"], "type": row["phrase_type"]}, prefix="phrase")
    phrase_hash = hash_rows(rows, prefix="phrase-bank")
    metadata = {
        "artifact_id": stable_hash(
            {
                "phrase_hash": phrase_hash,
                "train_cache": str(config.train_cache),
                "imagenet_meta": None if config.imagenet_meta is None else str(config.imagenet_meta),
            },
            prefix="phrase-bank-artifact",
        ),
        "phrase_hash": phrase_hash,
        "train_cache": str(config.train_cache),
        "imagenet_meta": None if config.imagenet_meta is None else str(config.imagenet_meta),
            "summary": {
                "phrases": len(rows),
                "general_phrases": sum(1 for row in rows if row["phrase_type"] == "general"),
                "class_prompts": sum(1 for row in rows if row["phrase_type"] == "class_prompt"),
                "class_attribute_prompts": sum(1 for row in rows if row["phrase_type"] == "class_attribute"),
                "confusion_prompts": sum(1 for row in rows if row["phrase_type"] == "confusion_prompt"),
                "classes": len(class_names),
            },
        "outputs": {"phrase_bank_csv": str(output / "phrase_bank.csv")},
    }
    _write_csv(output / "phrase_bank.csv", rows)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def _phrase_row(
    *,
    phrase: str,
    phrase_type: str,
    group: str,
    source: str,
    class_id: int | None = None,
    class_name: str = "",
    display_name: str = "",
    synonym: str = "",
    class_conditional: bool = False,
    phrase_bank_version: str = "",
) -> dict[str, object]:
    return {
        "phrase_id": "",
        "concept_id": "",
        "phrase": str(phrase),
        "phrase_type": str(phrase_type),
        "concept_category": str(group),
        "group": str(group),
        "source": str(source),
        "class_id": "" if class_id is None else int(class_id),
        "class_name": str(class_name),
        "display_name": str(display_name),
        "synonym": str(synonym),
        "class_conditional": int(bool(class_conditional)),
        "templates": str(phrase),
        "phrase_bank_version": str(phrase_bank_version),
    }


def _dedupe_phrases(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, object]] = []
    for row in rows:
        key = (str(row["phrase"]).strip().lower(), str(row["phrase_type"]), str(row.get("class_id", "")))
        if key in seen:
            continue
        seen.add(key)
        new_row = dict(row)
        new_row["concept_id"] = stable_hash(
            {
                "phrase": new_row["phrase"],
                "phrase_type": new_row["phrase_type"],
                "class_id": new_row.get("class_id", ""),
                "group": new_row.get("group", ""),
            },
            prefix="concept",
        )
        out.append(new_row)
    return out


def _synonyms(display_name: str) -> list[str]:
    raw = str(display_name).replace(" or ", ",")
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        values = [str(display_name).strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def _optional_int(value: object) -> int | None:
    if value in ("", None):
        return None
    return int(value)


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
