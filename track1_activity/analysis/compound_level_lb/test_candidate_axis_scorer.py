import math
import unittest

import numpy as np

import score_candidate_axes as scorer


class CandidateAxisScorerTest(unittest.TestCase):
    def test_prediction_metrics_reports_shift_and_rank_correlation(self):
        reference = np.array([1.0, 2.0, 3.0, 4.0])
        candidate = np.array([1.1, 1.9, 3.2, 3.8])

        metrics = scorer.prediction_metrics(reference, candidate)

        self.assertAlmostEqual(metrics["mean_delta"], 0.0)
        self.assertAlmostEqual(metrics["mean_abs_delta"], 0.15)
        self.assertAlmostEqual(metrics["max_abs_delta"], 0.2)
        self.assertAlmostEqual(metrics["pearson"], 0.990847, places=6)
        self.assertAlmostEqual(metrics["spearman"], 1.0)

    def test_residual_correlation_uses_target_centered_errors(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        reference = np.array([1.1, 2.1, 2.9, 3.9])
        candidate = np.array([0.9, 1.9, 3.1, 4.1])

        corr = scorer.residual_correlation(y, reference, candidate)

        self.assertAlmostEqual(corr, -1.0)

    def test_recommendation_closes_weak_single_even_when_axis_is_different(self):
        decision = scorer.recommend_candidate(
            single_mae=0.53,
            pearson_vs_reference=0.91,
            mean_abs_shift=0.12,
            has_test_predictions=True,
        )

        self.assertEqual(decision["decision"], "close")
        self.assertIn("weak_single", decision["reasons"])

    def test_recommendation_allows_blend_only_for_strong_but_correlated_axis(self):
        decision = scorer.recommend_candidate(
            single_mae=0.472,
            pearson_vs_reference=0.999,
            mean_abs_shift=0.02,
            has_test_predictions=True,
        )

        self.assertEqual(decision["decision"], "blend_only")
        self.assertIn("high_correlation", decision["reasons"])

    def test_recommendation_marks_promising_axis_for_review(self):
        decision = scorer.recommend_candidate(
            single_mae=0.472,
            pearson_vs_reference=0.94,
            mean_abs_shift=0.03,
            has_test_predictions=True,
        )

        self.assertEqual(decision["decision"], "review")
        self.assertEqual(decision["reasons"], ["passes_cheap_axis_gate"])

    def test_recommendation_handles_missing_test_predictions(self):
        decision = scorer.recommend_candidate(
            single_mae=0.47,
            pearson_vs_reference=math.nan,
            mean_abs_shift=math.nan,
            has_test_predictions=False,
        )

        self.assertEqual(decision["decision"], "needs_test_predictions")
        self.assertIn("missing_test_predictions", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
