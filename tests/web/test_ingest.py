import os
import zipfile

import pytest

from peal.web import ingest


def _png_bytes():
    # 1x1 PNG
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360f8cfc00000030101003fc5c6a70000000049454e44ae426082"
    )


def make_zip(path, layout, wrapper=None, extra=()):
    with zipfile.ZipFile(path, "w") as zf:
        for cls, n in layout.items():
            for i in range(n):
                name = f"{cls}/img_{i}.png"
                if wrapper:
                    name = f"{wrapper}/{name}"
                zf.writestr(name, _png_bytes())
        for name, data in extra:
            zf.writestr(name, data)
    return path


def test_discover_and_build(tmp_path):
    z = make_zip(
        tmp_path / "d.zip",
        {"cat": 3, "dog": 2, "bird": 1},
        wrapper="mydata",
        extra=[
            ("__MACOSX/mydata/cat/._img_0.png", b"junk"),
            ("mydata/readme.txt", b"hi"),
        ],
    )
    class_root, classes = ingest.ingest_zip(str(z), str(tmp_path / "work"))
    assert os.path.basename(class_root) == "mydata"
    assert {k: len(v) for k, v in classes.items()} == {"cat": 3, "dog": 2, "bird": 1}
    counts = ingest.build_pair_dataset(
        class_root, classes, "dog", "cat", str(tmp_path / "ds"), link=False
    )
    assert counts == {"dog": 2, "cat": 3}
    rows = open(tmp_path / "ds" / "data.csv").read().strip().splitlines()
    assert rows[0] == "ImgPath,Class"
    assert len(rows) == 6
    assert sum(r.endswith(",0") for r in rows[1:]) == 2
    assert os.path.isfile(tmp_path / "ds" / "imgs" / "1" / "000002.png")
    with pytest.raises(ingest.IngestError):
        ingest.build_pair_dataset(
            class_root, classes, "dog", "dog", str(tmp_path / "ds2")
        )
    with pytest.raises(ingest.IngestError):
        ingest.build_pair_dataset(
            class_root, classes, "dog", "fish", str(tmp_path / "ds2")
        )


def test_max_per_class(tmp_path):
    z = make_zip(tmp_path / "d.zip", {"a": 5, "b": 5})
    class_root, classes = ingest.ingest_zip(str(z), str(tmp_path / "work"))
    counts = ingest.build_pair_dataset(
        class_root, classes, "a", "b", str(tmp_path / "ds"), max_per_class=2, link=False
    )
    assert counts == {"a": 2, "b": 2}


def test_rejects_traversal_and_absolute(tmp_path):
    z = make_zip(
        tmp_path / "bad.zip", {"a": 1, "b": 1}, extra=[("../evil.png", _png_bytes())]
    )
    with pytest.raises(ingest.IngestError):
        ingest.ingest_zip(str(z), str(tmp_path / "w1"))
    z = make_zip(
        tmp_path / "bad2.zip", {"a": 1, "b": 1}, extra=[("/etc/evil.png", _png_bytes())]
    )
    with pytest.raises(ingest.IngestError):
        ingest.ingest_zip(str(z), str(tmp_path / "w2"))


def test_rejects_too_many_files(tmp_path, monkeypatch):
    z = make_zip(tmp_path / "many.zip", {"a": 3, "b": 3})
    with pytest.raises(ingest.IngestError):
        ingest.safe_extract(str(z), str(tmp_path / "w"), max_files=4)


def test_rejects_symlink_member(tmp_path):
    z = tmp_path / "s.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("a/x.png", _png_bytes())
        zf.writestr("b/x.png", _png_bytes())
        info = zipfile.ZipInfo("a/link.png")
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(ingest.IngestError):
        ingest.ingest_zip(str(z), str(tmp_path / "w"))


def test_single_class_fails(tmp_path):
    z = make_zip(tmp_path / "one.zip", {"a": 2})
    with pytest.raises(ingest.IngestError):
        ingest.ingest_zip(str(z), str(tmp_path / "w"))
