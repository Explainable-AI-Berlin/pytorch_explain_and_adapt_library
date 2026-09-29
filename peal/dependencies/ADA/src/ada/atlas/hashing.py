from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(obj: Any, *, prefix: str | None = None) -> str:
    digest = hashlib.sha1(stable_json_dumps(obj).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}" if prefix else digest


def hash_rows(rows: Iterable[Mapping[str, Any]], *, prefix: str | None = None) -> str:
    h = hashlib.sha1()
    for row in rows:
        h.update(stable_json_dumps(dict(row)).encode("utf-8"))
        h.update(b"\n")
    digest = h.hexdigest()
    return f"{prefix}-{digest[:16]}" if prefix else digest


def file_sha1(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
