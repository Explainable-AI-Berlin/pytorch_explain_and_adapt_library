from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an immutable development-WNID exclusion artifact.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path, help="CSV manifest containing class_name/WNID rows.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--source-name")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = _load_config(args.config) if args.config else {}
    manifest = args.manifest or Path(raw.get("manifest", ""))
    output = args.output_dir or Path(raw.get("output_dir", ""))
    experiment_id = args.experiment_id or str(raw.get("experiment_id", "e7a_in1k_excluded_development_wnids_in100"))
    source_name = args.source_name or str(raw.get("source_name", "imagenet100_development_set"))
    overwrite = bool(args.overwrite or raw.get("overwrite", False))
    if not str(manifest):
        raise ValueError("manifest is required")
    if not str(output):
        raise ValueError("output_dir is required")
    output = Path(output)
    completed = output / "COMPLETED"
    if completed.exists() and not overwrite:
        raise FileExistsError(f"exclusion artifact already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(Path(manifest))
    if not rows:
        raise ValueError(f"manifest has no rows: {manifest}")
    if "class_name" not in rows[0]:
        raise ValueError(f"manifest must contain class_name column: {manifest}")

    wnids = sorted({str(row["class_name"]) for row in rows if str(row.get("class_name", "")).strip()})
    if not wnids:
        raise ValueError(f"manifest yielded zero WNIDs: {args.manifest}")

    wnid_path = output / "excluded_development_wnids.txt"
    wnid_path.write_text("".join(f"{wnid}\n" for wnid in wnids))
    digest = hashlib.sha256(wnid_path.read_bytes()).hexdigest()
    (output / "excluded_development_wnids.sha256").write_text(f"{digest}  {wnid_path.name}\n")

    metadata = {
        "experiment_id": experiment_id,
        "source_name": source_name,
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256_file(Path(manifest)),
        "wnid_count": len(wnids),
        "wnid_sha256": digest,
        "outputs": {
            "excluded_development_wnids": str(wnid_path),
            "excluded_development_wnids_sha256": str(output / "excluded_development_wnids.sha256"),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    return data


if __name__ == "__main__":
    main()
