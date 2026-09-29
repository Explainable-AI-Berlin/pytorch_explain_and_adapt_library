from __future__ import annotations

import unittest

import numpy as np

from ada.generation.conditional_memorization import (
    calibrated_nearest_train_ratio,
    conditional_path_diagnostics,
    effective_rank,
    slerp,
)


class ConditionalMemorizationTests(unittest.TestCase):
    def test_slerp_preserves_endpoints_and_unit_norm(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        values = slerp(a, b, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(values[0], a, atol=1.0e-10)
        np.testing.assert_allclose(values[-1], b, atol=1.0e-10)
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, atol=1.0e-10)

    def test_effective_rank_distinguishes_line_and_plane(self) -> None:
        line = np.stack([np.linspace(-1, 1, 21), np.zeros(21)], axis=1)
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        self.assertAlmostEqual(effective_rank(line), 1.0, places=6)
        self.assertGreater(effective_rank(circle), 1.9)

    def test_path_diagnostics_detect_piecewise_source_locking(self) -> None:
        progress = np.linspace(0.0, 1.0, 41)
        conditions = np.stack([progress, np.zeros_like(progress)], axis=1)
        smooth = np.stack([progress, progress**2], axis=1)
        snapped_x = np.round(progress * 4.0) / 4.0
        snapped = np.stack([snapped_x, snapped_x**2], axis=1)
        train = smooth[::10]
        smooth_metrics = conditional_path_diagnostics(conditions, smooth, train, progress=progress)
        snapped_metrics = conditional_path_diagnostics(conditions, snapped, train, progress=progress)
        self.assertLess(smooth_metrics.stationary_step_fraction, 0.1)
        self.assertGreater(snapped_metrics.stationary_step_fraction, 0.6)
        self.assertGreater(snapped_metrics.max_to_mean_step_ratio, smooth_metrics.max_to_mean_step_ratio)

    def test_calibrated_ratio_is_small_for_exact_training_copies(self) -> None:
        train = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        generated = train.copy()
        fresh = train + np.array([[0.2, 0.1], [-0.1, 0.2], [0.15, -0.2]])
        self.assertAlmostEqual(calibrated_nearest_train_ratio(generated, fresh, train), 0.0)


if __name__ == "__main__":
    unittest.main()
