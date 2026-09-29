from __future__ import annotations

import unittest

from ada.atlas.data.manifests import ManifestRow, assert_disjoint_sample_ids


class SplitOverlapTests(unittest.TestCase):
    def test_disjoint_sample_ids_pass(self) -> None:
        train = [ManifestRow("a", "a.jpg", "toy", "train", 0, "c")]
        val = [ManifestRow("b", "b.jpg", "toy", "val", 0, "c")]
        assert_disjoint_sample_ids([("train", train), ("val", val)])

    def test_overlap_raises(self) -> None:
        train = [ManifestRow("a", "a.jpg", "toy", "train", 0, "c")]
        val = [ManifestRow("a", "a.jpg", "toy", "val", 0, "c")]
        with self.assertRaises(ValueError):
            assert_disjoint_sample_ids([("train", train), ("val", val)])


if __name__ == "__main__":
    unittest.main()
