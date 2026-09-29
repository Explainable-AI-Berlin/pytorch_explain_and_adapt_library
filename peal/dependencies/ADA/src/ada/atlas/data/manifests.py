from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ada.atlas.hashing import hash_rows, stable_hash


DEFAULT_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    relative_path: str
    source_dataset: str
    split: str
    class_id: int
    class_name: str
    patient_id: str = ""
    slide_id: str = ""
    site_id: str = ""
    hidden_group_id: str = ""


@dataclass(frozen=True)
class ImageFolderManifest:
    dataset: str
    split: str
    root: str
    rows: tuple[ManifestRow, ...]
    class_names: tuple[str, ...]
    skipped_dirs: tuple[str, ...]
    skipped_files: tuple[str, ...]
    manifest_hash: str

    @property
    def sample_count(self) -> int:
        return len(self.rows)

    @property
    def class_count(self) -> int:
        return len(self.class_names)


def _is_private_name(path: Path) -> bool:
    return path.name.startswith(".") or path.name.startswith("_")


def _sample_id(dataset: str, split: str, relative_path: str) -> str:
    return stable_hash(
        {
            "dataset": dataset,
            "split": split,
            "relative_path": relative_path.replace("\\", "/"),
        },
        prefix="sample",
    )


def _visible_class_dirs(root: Path, skip_private: bool) -> tuple[list[Path], list[str]]:
    skipped: list[str] = []
    classes: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if skip_private and _is_private_name(child):
            skipped.append(child.name)
            continue
        classes.append(child)
    return classes, skipped


def build_imagefolder_manifest(
    root: str | Path,
    *,
    dataset: str,
    split: str,
    extensions: Iterable[str] | None = None,
    skip_private_dirs: bool = True,
) -> ImageFolderManifest:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root_path}")

    exts = {x.lower() for x in (extensions or DEFAULT_IMAGE_EXTENSIONS)}
    class_dirs, skipped_dirs = _visible_class_dirs(root_path, skip_private_dirs)
    rows: list[ManifestRow] = []
    skipped_files: list[str] = []

    for class_id, class_dir in enumerate(class_dirs):
        files = sorted((p for p in class_dir.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root_path).as_posix())
        for file_path in files:
            rel = file_path.relative_to(root_path).as_posix()
            if file_path.suffix.lower() not in exts:
                skipped_files.append(rel)
                continue
            rows.append(
                ManifestRow(
                    sample_id=_sample_id(dataset, split, rel),
                    relative_path=rel,
                    source_dataset=dataset,
                    split=split,
                    class_id=class_id,
                    class_name=class_dir.name,
                )
            )

    row_dicts = [asdict(row) for row in rows]
    manifest_hash = hash_rows(row_dicts, prefix="manifest")
    return ImageFolderManifest(
        dataset=dataset,
        split=split,
        root=str(root_path),
        rows=tuple(rows),
        class_names=tuple(path.name for path in class_dirs),
        skipped_dirs=tuple(skipped_dirs),
        skipped_files=tuple(skipped_files),
        manifest_hash=manifest_hash,
    )


def manifest_summary(manifest: ImageFolderManifest) -> dict:
    return {
        "dataset": manifest.dataset,
        "split": manifest.split,
        "root": manifest.root,
        "sample_count": manifest.sample_count,
        "class_count": manifest.class_count,
        "manifest_hash": manifest.manifest_hash,
        "skipped_dirs": list(manifest.skipped_dirs),
        "skipped_file_count": len(manifest.skipped_files),
    }


def write_manifest(manifest: ImageFolderManifest, output_dir: str | Path) -> Path:
    output = Path(output_dir) / manifest.dataset / manifest.split / manifest.manifest_hash
    output.mkdir(parents=True, exist_ok=True)
    write_manifest_files(manifest, output)
    return output


def write_manifest_files(manifest: ImageFolderManifest, output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_csv = output / "manifest.csv"
    with manifest_csv.open("w", newline="") as f:
        fieldnames = list(asdict(manifest.rows[0]).keys()) if manifest.rows else list(ManifestRow.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest.rows:
            writer.writerow(asdict(row))

    metadata = manifest_summary(manifest)
    metadata["class_names"] = list(manifest.class_names)
    metadata["skipped_files"] = list(manifest.skipped_files)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (output / "COMPLETED").write_text("ok\n")


def load_manifest_csv(path: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            raw = dict(raw)
            raw["class_id"] = int(raw["class_id"])
            rows.append(ManifestRow(**raw))
    return rows


def replace_manifest_rows(manifest: ImageFolderManifest, rows: Sequence[ManifestRow]) -> ImageFolderManifest:
    row_dicts = [asdict(row) for row in rows]
    return ImageFolderManifest(
        dataset=manifest.dataset,
        split=manifest.split,
        root=manifest.root,
        rows=rows,
        class_names=manifest.class_names,
        skipped_dirs=manifest.skipped_dirs,
        skipped_files=manifest.skipped_files,
        manifest_hash=hash_rows(row_dicts, prefix="manifest"),
    )


def subset_manifest(manifest: ImageFolderManifest, max_samples: int | None) -> ImageFolderManifest:
    if max_samples is None or max_samples >= manifest.sample_count:
        return manifest
    return replace_manifest_rows(manifest, tuple(manifest.rows[: int(max_samples)]))


def slice_manifest(manifest: ImageFolderManifest, start_index: int, end_index: int | None) -> ImageFolderManifest:
    start = int(start_index)
    end = manifest.sample_count if end_index is None else int(end_index)
    if start < 0:
        raise ValueError("start_index must be non-negative")
    if end < start:
        raise ValueError("end_index must be greater than or equal to start_index")
    if start == 0 and end >= manifest.sample_count:
        return manifest
    return replace_manifest_rows(manifest, tuple(manifest.rows[start:end]))


def shard_manifest(manifest: ImageFolderManifest, shard_index: int, num_shards: int) -> ImageFolderManifest:
    shard = int(shard_index)
    total = int(num_shards)
    if total < 1:
        raise ValueError("num_shards must be positive")
    if shard < 0 or shard >= total:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    n = manifest.sample_count
    start = (shard * n) // total
    end = ((shard + 1) * n) // total
    return slice_manifest(manifest, start, end)


def assert_disjoint_sample_ids(named_manifests: Sequence[tuple[str, Sequence[ManifestRow]]]) -> None:
    seen: dict[str, str] = {}
    for name, rows in named_manifests:
        for row in rows:
            previous = seen.get(row.sample_id)
            if previous is not None:
                raise ValueError(f"Sample ID {row.sample_id} appears in both {previous} and {name}")
            seen[row.sample_id] = name
