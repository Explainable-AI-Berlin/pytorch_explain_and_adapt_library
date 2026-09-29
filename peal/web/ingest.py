"""
Dataset ingest for the web demo: a zip of one folder per class becomes the
``imgs/<0|1>/...`` + ``data.csv`` layout ``Image2MixedDataset`` reads (the same
layout tools/generate_imagenet_binary_dataset.py writes).

Uploads come from strangers, so extraction refuses path traversal, absolute
paths, symlinks, too many files, too many bytes and zip bombs.
"""

import os
import shutil
import stat
import zipfile
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IGNORED_DIRS = {"__MACOSX", ".git", "__pycache__"}
DEFAULT_MAX_FILES = int(os.environ.get("PEAL_WEB_MAX_FILES", "20000"))
DEFAULT_MAX_BYTES = int(
    float(os.environ.get("PEAL_WEB_MAX_ZIP_MB", "4096")) * 1024 * 1024
)
# a member that expands to more than this ratio AND this size is a zip bomb
BOMB_RATIO = 200
BOMB_MIN_BYTES = 10 * 1024 * 1024


class IngestError(ValueError):
    """Raised for any rejected or malformed upload (bad zip, unsafe member,
    unknown class folder, ...). The message is meant to be shown to the
    uploader."""


def _is_symlink(info):
    """Whether a ``ZipInfo`` entry is a symlink (unix mode bits in
    ``external_attr``)."""
    return stat.S_ISLNK((info.external_attr >> 16) & 0o170000)


def safe_extract(
    zip_path, dest, max_files=DEFAULT_MAX_FILES, max_bytes=DEFAULT_MAX_BYTES
):
    """
    Extract ``zip_path`` under ``dest`` after validating every member.

    All members are checked before anything is written: absolute paths,
    ``..`` components, symlinks, more than ``max_files`` files, a total
    uncompressed size above ``max_bytes`` and members with a suspicious
    compression ratio (``BOMB_RATIO`` above ``BOMB_MIN_BYTES``) are rejected.

    Parameters
    ----------
    zip_path : str
        Path of the uploaded zip archive.
    dest : str
        Directory to extract into; created if missing.
    max_files : int
        Maximum number of (non-directory) members.
    max_bytes : int
        Maximum total uncompressed size in bytes.

    Returns
    -------
    int
        Number of files written.

    Raises
    ------
    IngestError
        If the archive is not a zip file or any check fails.
    """
    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise IngestError(f"not a zip file: {exc}") from exc
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > max_files:
            raise IngestError(f"zip holds {len(infos)} files, the limit is {max_files}")
        total = 0
        for info in infos:
            name = info.filename
            if name.startswith(("/", "\\")) or os.path.isabs(name):
                raise IngestError(f"absolute path in zip: {name}")
            parts = Path(name.replace("\\", "/")).parts
            if ".." in parts:
                raise IngestError(f"path traversal in zip: {name}")
            if _is_symlink(info):
                raise IngestError(f"symlink in zip: {name}")
            total += info.file_size
            if total > max_bytes:
                raise IngestError(
                    f"zip expands to more than {max_bytes // (1024 * 1024)} MB"
                )
            if (
                info.compress_size > 0
                and info.file_size > BOMB_MIN_BYTES
                and info.file_size / info.compress_size > BOMB_RATIO
            ):
                raise IngestError(f"suspicious compression ratio for {name}")
        n = 0
        for info in infos:
            target = os.path.abspath(
                os.path.join(dest, info.filename.replace("\\", "/"))
            )
            if not target.startswith(dest + os.sep):
                raise IngestError(f"path escapes destination: {info.filename}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    return n


def _image_files(folder):
    """Sorted image paths below ``folder`` (relative to it), skipping hidden
    entries and ``IGNORED_DIRS``."""
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        for f in files:
            if f.startswith("."):
                continue
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                out.append(os.path.relpath(os.path.join(root, f), folder))
    return sorted(out)


def discover_classes(root, max_depth=3):
    """
    Find the folder level that holds one sub-folder per class.

    A zip that wraps everything in a single top-level folder is descended,
    up to ``max_depth`` levels.

    Parameters
    ----------
    root : str
        Directory the upload was extracted to.
    max_depth : int
        How many single-folder wrappers may be descended.

    Returns
    -------
    class_root : str
        The directory whose sub-folders are the classes.
    classes : dict
        ``{class_name: [image paths relative to that class folder]}`` with at
        least two entries.

    Raises
    ------
    IngestError
        If no level with at least two image-bearing class folders is found.
    """
    root = os.path.abspath(root)
    level = root
    for _ in range(max_depth + 1):
        subdirs = sorted(
            d
            for d in os.listdir(level)
            if os.path.isdir(os.path.join(level, d))
            and d not in IGNORED_DIRS
            and not d.startswith(".")
        )
        classes = {}
        for d in subdirs:
            imgs = _image_files(os.path.join(level, d))
            if imgs:
                classes[d] = imgs
        if len(classes) >= 2:
            return level, classes
        if len(subdirs) == 1 and not classes:
            level = os.path.join(level, subdirs[0])
            continue
        if len(subdirs) == 1 and len(classes) == 1:
            level = os.path.join(level, subdirs[0])
            continue
        break
    raise IngestError(
        "could not find at least two class folders with images; expected "
        "<zip>/<class_a>/*.jpg and <zip>/<class_b>/*.jpg (one wrapping folder is fine)"
    )


def build_pair_dataset(
    class_root, classes, class_a, class_b, out_dir, max_per_class=None, link=True
):
    """
    Write the binary ``imgs/<0|1>`` + ``data.csv`` layout for two classes.

    ``class_a`` becomes label 0 and ``class_b`` label 1. Images are renamed
    to ``<label>/<index:06d><ext>`` and ``data.csv`` gets the header
    ``ImgPath,Class``. Symlinks are used by default (the extracted upload
    stays the owner of the bytes); ``link=False`` copies.

    Parameters
    ----------
    class_root : str
        Directory returned by ``discover_classes``.
    classes : dict
        ``{class_name: [relative image paths]}`` from ``discover_classes``.
    class_a, class_b : str
        The two class folders to use; must differ and exist in ``classes``.
    out_dir : str
        Dataset root that receives ``imgs/`` and ``data.csv``.
    max_per_class : int, optional
        Truncate each class to its first ``max_per_class`` images.
    link : bool
        Symlink (True) or copy (False) the images.

    Returns
    -------
    dict
        ``{class_name: number of images used}`` for the two classes.

    Raises
    ------
    IngestError
        If the classes are equal or unknown.
    """
    if class_a == class_b:
        raise IngestError("the two classes must differ")
    for c in (class_a, class_b):
        if c not in classes:
            raise IngestError(
                f"unknown class folder {c!r}; available: {sorted(classes)}"
            )
    imgs_dir = os.path.join(out_dir, "imgs")
    rows = []
    counts = {}
    for label, cls in enumerate((class_a, class_b)):
        Path(os.path.join(imgs_dir, str(label))).mkdir(parents=True, exist_ok=True)
        files = classes[cls]
        if max_per_class is not None:
            files = files[: int(max_per_class)]
        counts[cls] = len(files)
        for i, rel in enumerate(files):
            src = os.path.join(class_root, cls, rel)
            ext = os.path.splitext(rel)[1].lower()
            dst_rel = os.path.join(str(label), f"{i:06d}{ext}")
            dst = os.path.join(imgs_dir, dst_rel)
            if link:
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
            rows.append((dst_rel, label))
    with open(os.path.join(out_dir, "data.csv"), "w") as f:
        f.write("ImgPath,Class\n")
        for rel, label in rows:
            f.write(f"{rel},{label}\n")
    return counts


def ingest_zip(zip_path, work_dir):
    """
    Extract an upload into ``work_dir/extracted`` and discover its classes.

    Parameters
    ----------
    zip_path : str
        The uploaded zip archive.
    work_dir : str
        Working directory; extraction happens in its ``extracted`` sub-dir.

    Returns
    -------
    class_root : str
        Directory whose sub-folders are the classes.
    classes : dict
        ``{class_name: [relative image paths]}`` as from ``discover_classes``.
    """
    extract_dir = os.path.join(work_dir, "extracted")
    safe_extract(zip_path, extract_dir)
    class_root, classes = discover_classes(extract_dir)
    return class_root, classes
