import unittest

import numpy as np
import pandas as pd

import error_anatomy as ea


class ErrorAnatomyTest(unittest.TestCase):
    def test_safe_qcut_returns_ordered_bin_labels(self):
        values = pd.Series([0.0, 1.0, 2.0, 3.0])

        got = ea.safe_qcut(values, q=2)

        self.assertEqual(got.astype(str).tolist(), ["Q1", "Q1", "Q2", "Q2"])

    def test_safe_qcut_handles_constant_values(self):
        values = pd.Series([1.0, 1.0, 1.0])

        got = ea.safe_qcut(values, q=4)

        self.assertEqual(got.astype(str).tolist(), ["all", "all", "all"])

    def test_summarize_binary_slice_reports_delta_vs_background(self):
        df = pd.DataFrame(
            {
                "flag": [True, True, False, False],
                "abs_error": [2.0, 4.0, 1.0, 1.0],
                "residual": [2.0, -4.0, 1.0, -1.0],
            }
        )

        got = ea.summarize_binary_slice(df, "flag")

        self.assertEqual(got["slice"].iloc[0], "flag")
        self.assertEqual(got["n_true"].iloc[0], 2)
        self.assertAlmostEqual(got["mae_true"].iloc[0], 3.0)
        self.assertAlmostEqual(got["mae_false"].iloc[0], 1.0)
        self.assertAlmostEqual(got["delta_mae_true_minus_false"].iloc[0], 2.0)
        self.assertAlmostEqual(got["mean_residual_true"].iloc[0], -1.0)

    def test_mean_abs_error_ignores_nan(self):
        got = ea.mean_abs_error(np.array([1.0, -2.0, np.nan]))

        self.assertAlmostEqual(got, 1.5)


if __name__ == "__main__":
    unittest.main()
