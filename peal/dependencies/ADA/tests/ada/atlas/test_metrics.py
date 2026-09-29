from __future__ import annotations

import unittest

from ada.atlas.metrics.deciles import quantile_summary
from ada.atlas.metrics.error_detection import average_precision_score, confidence_to_error_risk, roc_auc_score


class ErrorDetectionMetricTests(unittest.TestCase):
    def test_roc_auc_and_average_precision(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(roc_auc_score(scores, labels), 1.0)
        self.assertAlmostEqual(average_precision_score(scores, labels), 1.0)

    def test_confidence_to_error_risk(self) -> None:
        self.assertEqual(confidence_to_error_risk([0.9, 0.2]), [0.09999999999999998, 0.8])

    def test_quantile_summary(self) -> None:
        rows = [
            {"support_distance": 0.9, "is_error": 1, "confidence_raw": 0.6},
            {"support_distance": 0.8, "is_error": 1, "confidence_raw": 0.7},
            {"support_distance": 0.1, "is_error": 0, "confidence_raw": 0.9},
            {"support_distance": 0.0, "is_error": 0, "confidence_raw": 0.8},
        ]
        summary = quantile_summary(rows, score_column="support_distance", bins=2, high_score_is_risky=True)
        self.assertEqual(summary[0]["count"], 2)
        self.assertEqual(summary[0]["error_rate"], 1.0)
        self.assertEqual(summary[1]["error_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
