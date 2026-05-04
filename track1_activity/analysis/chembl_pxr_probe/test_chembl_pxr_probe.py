import unittest

import numpy as np
import pandas as pd

import chembl_pxr_activation_probe as probe


class ChemblPxrProbeTest(unittest.TestCase):
    def test_weighted_topk_returns_similarity_weighted_mean(self):
        sim = np.array([[0.8, 0.2, 0.0], [0.1, 0.4, 0.3]], dtype=np.float32)
        values = np.array([6.0, 4.0, 5.0], dtype=np.float32)

        got = probe.topk_weighted_values(sim, values, k=2)

        self.assertAlmostEqual(got[0], (0.8 * 6.0 + 0.2 * 4.0) / 1.0)
        self.assertAlmostEqual(got[1], (0.4 * 4.0 + 0.3 * 5.0) / 0.7)

    def test_build_nn_features_can_exclude_exact_matches(self):
        sim = np.array(
            [
                [1.0, 0.4, 0.2],
                [0.3, 0.9, 0.1],
            ],
            dtype=np.float32,
        )
        values = np.array([7.0, 5.0, 4.0], dtype=np.float32)
        exact = np.array(
            [
                [True, False, False],
                [False, False, False],
            ]
        )

        features = probe.build_nn_features_from_similarity(sim, values, exact, k=2)

        self.assertAlmostEqual(features.loc[0, "chembl_pxr_nn_tanimoto"], 0.4)
        self.assertAlmostEqual(features.loc[0, "chembl_pxr_nn_pchembl"], 5.0)
        self.assertTrue(features.loc[0, "chembl_pxr_has_exact_match"])
        self.assertAlmostEqual(features.loc[1, "chembl_pxr_nn_tanimoto"], 0.9)
        self.assertAlmostEqual(features.loc[1, "chembl_pxr_nn_pchembl"], 5.0)
        self.assertFalse(features.loc[1, "chembl_pxr_has_exact_match"])

    def test_assay_filter_keeps_activation_ec50_and_drops_antagonist(self):
        df = pd.DataFrame(
            {
                "assay_type": ["A", "A", "B"],
                "standard_type": ["EC50", "EC50", "EC50"],
                "confidence_score": [9, 9, 9],
                "pchembl_value": [5.1, 6.2, 7.3],
                "description": [
                    "Activation of human PXR by luciferase reporter assay",
                    "Antagonist activity at human PXR in presence of rifampicin",
                    "Activation of human PXR binding assay",
                ],
            }
        )

        filtered = probe.filter_activation_ec50(df)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["pchembl_value"], 5.1)


if __name__ == "__main__":
    unittest.main()
