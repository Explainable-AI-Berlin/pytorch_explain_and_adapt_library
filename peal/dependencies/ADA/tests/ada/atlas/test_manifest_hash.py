from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ada.atlas.data.manifests import build_imagefolder_manifest


class ManifestHashTests(unittest.TestCase):
    def test_manifest_is_deterministic_and_skips_private_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "n0002").mkdir()
            (root / "n0001").mkdir()
            (root / "_dino_latents").mkdir()
            (root / "n0002" / "b.jpg").write_bytes(b"b")
            (root / "n0001" / "a.jpg").write_bytes(b"a")
            (root / "_dino_latents" / "cache_config.json").write_text("{}")

            first = build_imagefolder_manifest(root, dataset="toy", split="train")
            second = build_imagefolder_manifest(root, dataset="toy", split="train")

        self.assertEqual(first.manifest_hash, second.manifest_hash)
        self.assertEqual(first.class_names, ("n0001", "n0002"))
        self.assertEqual(first.sample_count, 2)
        self.assertEqual(first.skipped_dirs, ("_dino_latents",))
        self.assertEqual([row.relative_path for row in first.rows], ["n0001/a.jpg", "n0002/b.jpg"])


if __name__ == "__main__":
    unittest.main()
