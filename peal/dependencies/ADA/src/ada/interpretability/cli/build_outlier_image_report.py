from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.atlas.hashing import file_sha1
from ada.interpretability.outlier_image_report import OutlierImageReportConfig, build_outlier_image_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build linked outlier-image descriptions from region language cards.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--region-cards-csv", type=Path)
    parser.add_argument("--text-ambiguity-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-image-root", type=Path)
    parser.add_argument("--validation-image-root", type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--top-concepts", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = _build_config(parse_args())
    metadata = build_outlier_image_report(config)
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), **metadata["outputs"]}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> OutlierImageReportConfig:
    raw = _load_config(args.config) if args.config else {}
    cards_dir = Path(raw.get("region_cards_dir", ""))
    cfg = OutlierImageReportConfig(
        manifest_dir=Path(raw.get("interpretability_manifest_dir", raw.get("manifest_dir", ""))),
        region_cards_csv=Path(raw.get("region_cards_csv", cards_dir / "region_language_cards.csv")),
        text_ambiguity_csv=Path(raw.get("text_ambiguity_csv", cards_dir / "region_text_ambiguity.csv")),
        output_dir=Path(raw.get("outlier_report_dir", raw.get("output_dir", ""))),
        train_image_root=Path(raw.get("train_image_root", "")),
        validation_image_root=Path(raw.get("validation_image_root", "")),
        max_images=int(raw.get("max_images", 16)),
        top_concepts=int(raw.get("top_concepts", 4)),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "manifest_dir",
        "region_cards_csv",
        "text_ambiguity_csv",
        "output_dir",
        "train_image_root",
        "validation_image_root",
        "max_images",
        "top_concepts",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    cfg = replace(cfg, **overrides)
    for key in ("manifest_dir", "region_cards_csv", "text_ambiguity_csv", "output_dir", "train_image_root", "validation_image_root"):
        if not str(getattr(cfg, key)):
            raise ValueError(f"{key} is required")
    return cfg


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = Path(path).read_text()
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    data["config_path"] = str(path)
    data["config_file_sha1"] = file_sha1(path)
    return data


if __name__ == "__main__":
    main()
