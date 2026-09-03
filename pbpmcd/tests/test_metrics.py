import unittest

import numpy as np

from pbpmcd.metrics import (
    class_performance_distance,
    classification_metrics,
    evaluate_radius_free,
    merge_alarm_bursts,
    resolve_alarm_merge_windows,
)


class MetricTests(unittest.TestCase):
    def test_classification_metrics(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        np.testing.assert_allclose(result["class_recall"], [0.5, 1.0])

    def test_class_distance(self):
        self.assertAlmostEqual(
            class_performance_distance(np.array([1.0, 0.0]), np.array([0.0, 0.0])),
            1.0,
        )

    def test_radius_free_and_mean_delay(self):
        result = evaluate_radius_free([100, 200], [120, 230], log_length=300)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 0)
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertAlmostEqual(result["mean_delay"], 25.0)

    def test_no_true_positive_delay_is_none(self):
        result = evaluate_radius_free([100], [], log_length=200)
        self.assertIsNone(result["mean_delay"])

    def test_final_log_boundary_is_included(self):
        result = evaluate_radius_free([100], [200], log_length=200)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["mean_delay"], 100.0)

    def test_alarm_burst_merging_matches_cm10k_raw_output(self):
        raw = [
            10, 20, 21, 22, 30, 31, 32, 33, 40, 50, 60,
            70, 71, 72, 73, 80, 90, 91, 93, 94, 95,
        ]
        consolidated, clusters = merge_alarm_bursts(raw, 4)
        self.assertEqual(consolidated, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        self.assertEqual(len(clusters), 9)
        self.assertEqual(clusters[-1], [90, 91, 93, 94, 95])

    def test_scale_normalized_alarm_merge_windows(self):
        self.assertEqual(resolve_alarm_merge_windows(25, 0.02), 1)
        self.assertEqual(resolve_alarm_merge_windows(100, 0.02), 2)
        self.assertEqual(resolve_alarm_merge_windows(200, 0.02), 4)
        self.assertEqual(resolve_alarm_merge_windows(100, 0.0), 0)

    def test_scale_normalization_separates_cf_and_merges_cm_bursts(self):
        cf_raw = [2, 3, 5, 7]
        cm_raw = [
            10, 20, 21, 22, 30, 31, 32, 33, 40, 50, 60,
            70, 71, 72, 73, 80, 90, 91, 93, 94, 95,
        ]
        cf_gap = resolve_alarm_merge_windows(25, 0.02)
        cm_gap = resolve_alarm_merge_windows(100, 0.02)
        self.assertEqual(merge_alarm_bursts(cf_raw, cf_gap)[0], [2, 5, 7])
        self.assertEqual(
            merge_alarm_bursts(cm_raw, cm_gap)[0],
            [10, 20, 30, 40, 50, 60, 70, 80, 90],
        )


if __name__ == "__main__":
    unittest.main()
