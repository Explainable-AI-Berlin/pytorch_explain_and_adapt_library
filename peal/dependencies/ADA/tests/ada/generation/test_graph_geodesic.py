from __future__ import annotations

import unittest

import numpy as np

from ada.generation.graph_geodesic import (
    piecewise_slerp_path,
    shortest_path,
    symmetric_knn_adjacency,
)


class GraphGeodesicTests(unittest.TestCase):
    def test_shortest_path_follows_local_arc(self) -> None:
        theta = np.linspace(0.0, np.pi / 2.0, 9)
        values = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        graph = symmetric_knn_adjacency(values, k=2)
        path, distance = shortest_path(graph, 0, 8)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 8)
        self.assertGreaterEqual(len(path), 5)
        self.assertAlmostEqual(distance, np.pi / 2.0, places=5)

    def test_piecewise_path_preserves_endpoints(self) -> None:
        vertices = np.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        path = piecewise_slerp_path(vertices, np.linspace(0.0, 1.0, 7))
        np.testing.assert_allclose(path[0], vertices[0])
        np.testing.assert_allclose(path[-1], vertices[-1])
        self.assertEqual(path.shape, (7, 2))


if __name__ == "__main__":
    unittest.main()
