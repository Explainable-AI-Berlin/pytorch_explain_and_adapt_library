from __future__ import annotations

import unittest

from ada.atlas.support.class_conditional import exact_class_conditional_support
from ada.atlas.support.knn import exact_cosine_knn


class ExactKNNTests(unittest.TestCase):
    def test_exact_cosine_knn_matches_simple_geometry(self) -> None:
        query = [[1.0, 0.0]]
        reference = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        result = exact_cosine_knn(query, reference, k=2)
        self.assertEqual(result.indices[0], [0, 1])
        self.assertAlmostEqual(result.distances[0][0], 0.0)
        self.assertAlmostEqual(result.distances[0][1], 1.0)

    def test_leave_one_out_removes_matching_id_not_first_rank(self) -> None:
        query = [[1.0, 0.0]]
        reference = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        result = exact_cosine_knn(
            query,
            reference,
            k=1,
            query_ids=["same"],
            reference_ids=["other_same_vector", "same", "far"],
            leave_one_out=True,
        )
        self.assertEqual(result.reference_ids[0], ["other_same_vector"])
        self.assertAlmostEqual(result.distances[0][0], 0.0)

    def test_class_conditional_support_uses_requested_label(self) -> None:
        query = [[1.0, 0.0], [0.0, 1.0]]
        reference = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]
        support = exact_class_conditional_support(
            query,
            [0, 1],
            reference,
            [0, 0, 1],
            k=1,
        )
        self.assertEqual(support.labels, [0, 1])
        self.assertIsNotNone(support.kth_distance[0])
        self.assertIsNotNone(support.kth_distance[1])
        self.assertAlmostEqual(support.kth_distance[0] or 0.0, 0.0)
        self.assertAlmostEqual(support.kth_distance[1] or 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
