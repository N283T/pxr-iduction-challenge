import unittest

import numpy as np
import pandas as pd

import residual_model_probe as probe


class ResidualModelProbeTest(unittest.TestCase):
    def test_build_lowd_features_encodes_potent_relu_and_no_aux_terms(self):
        df = pd.DataFrame(
            {
                "pred": [4.0, 5.0, 6.0],
                "nn_potent46_tanimoto": [0.2, 0.5, 0.8],
                "has_counter": [True, False, False],
                "has_single_conc_hi": [True, False, True],
                "has_single_conc_lo": [True, False, False],
                "member_std": [0.1, 0.2, 0.3],
                "family_gap": [-0.2, 0.0, 0.2],
            }
        )

        got = probe.build_lowd_features(df)

        self.assertEqual(
            got.columns.tolist(),
            [
                "potent_relu03",
                "potent_relu04",
                "no_aux",
                "no_aux_pred_centered",
                "member_std",
                "family_gap",
            ],
        )
        np.testing.assert_allclose(got["potent_relu03"], [0.0, 0.2, 0.5])
        np.testing.assert_allclose(got["potent_relu04"], [0.0, 0.1, 0.4])
        np.testing.assert_allclose(got["no_aux"], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(got["no_aux_pred_centered"].iloc[1], 0.0)

    def test_fit_predict_ridge_residual_recovers_linear_signal(self):
        X_train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        residual = np.array([0.0, 1.0, 2.0, 3.0])
        X_valid = pd.DataFrame({"x": [4.0]})

        pred = probe.fit_predict_ridge_residual(X_train, residual, X_valid, alpha=1e-9)

        self.assertAlmostEqual(pred[0], 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
