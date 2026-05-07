import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import submission_preflight as preflight


class SubmissionPreflightTest(unittest.TestCase):
    def test_shift_metrics_count_large_candidate_moves(self):
        anchor = np.array([4.0, 4.1, 4.2, 4.3, 4.4])
        candidate = np.array([4.0, 4.25, 4.45, 4.12, 4.9])

        metrics = preflight.compute_shift_metrics(anchor, candidate)

        self.assertEqual(metrics.n_abs_gt_005, 4)
        self.assertEqual(metrics.n_abs_gt_010, 4)
        self.assertEqual(metrics.n_abs_gt_020, 2)
        self.assertAlmostEqual(metrics.max_abs_shift, 0.5)

    def test_verdict_holds_large_unanchored_shift(self):
        metrics = preflight.PreflightMetrics(
            pearson=0.997,
            spearman=0.996,
            mean_shift=0.0,
            mean_abs_shift=0.12,
            p90_abs_shift=0.23,
            max_abs_shift=0.50,
            n_abs_gt_005=210,
            n_abs_gt_010=130,
            n_abs_gt_020=30,
            candidate_mean=4.8,
            candidate_std=0.78,
            anchor_mean=4.8,
            anchor_std=0.76,
        )

        verdict = preflight.classify_risk(metrics, bad_axis_rows=[])

        self.assertEqual(verdict.level, "HOLD")
        self.assertIn("large_anchor_shift", verdict.reasons)

    def test_verdict_cautions_on_known_bad_axis_alignment(self):
        metrics = preflight.PreflightMetrics(
            pearson=0.999,
            spearman=0.999,
            mean_shift=0.0,
            mean_abs_shift=0.02,
            p90_abs_shift=0.04,
            max_abs_shift=0.08,
            n_abs_gt_005=10,
            n_abs_gt_010=0,
            n_abs_gt_020=0,
            candidate_mean=4.8,
            candidate_std=0.76,
            anchor_mean=4.8,
            anchor_std=0.76,
        )

        verdict = preflight.classify_risk(
            metrics,
            bad_axis_rows=[
                preflight.BadAxisResult(
                    label="id56_minus_id55",
                    pearson=0.72,
                    spearman=0.71,
                    candidate_projection=0.05,
                )
            ],
        )

        self.assertEqual(verdict.level, "CAUTION")
        self.assertIn("aligned_with_known_bad_axis", verdict.reasons)

    def test_verdict_passes_small_anchor_move(self):
        metrics = preflight.PreflightMetrics(
            pearson=0.9999,
            spearman=0.9999,
            mean_shift=0.001,
            mean_abs_shift=0.01,
            p90_abs_shift=0.03,
            max_abs_shift=0.06,
            n_abs_gt_005=3,
            n_abs_gt_010=0,
            n_abs_gt_020=0,
            candidate_mean=4.801,
            candidate_std=0.761,
            anchor_mean=4.800,
            anchor_std=0.760,
        )

        verdict = preflight.classify_risk(metrics, bad_axis_rows=[])

        self.assertEqual(verdict.level, "PASS")
        self.assertEqual(verdict.reasons, ["small_anchor_shift"])


if __name__ == "__main__":
    unittest.main()
