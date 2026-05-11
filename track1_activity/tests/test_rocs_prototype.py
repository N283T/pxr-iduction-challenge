import unittest

import numpy as np

from track1_activity.src.rocs_prototype import (
    build_prototype_features,
    select_extreme_prototypes,
)


class TestRocsPrototype(unittest.TestCase):
    def test_select_extreme_prototypes_uses_training_indices_only(self):
        compound_ids = np.array([10, 11, 12, 13, 14])
        y = np.array([7.0, 6.5, 5.1, 4.2, 3.8], dtype=np.float32)
        train_idx = np.array([1, 2, 3, 4])

        active, inactive = select_extreme_prototypes(
            compound_ids=compound_ids,
            y=y,
            train_idx=train_idx,
            n_active=2,
            n_inactive=2,
        )

        self.assertEqual(active, [11, 12])
        self.assertEqual(inactive, [14, 13])

    def test_build_prototype_features_summarizes_scores_and_knn(self):
        target_ids = [100, 101]
        query_ids = [10, 11, 12]
        query_targets = {10: 7.0, 11: 6.0, 12: 4.0}
        score_maps = {
            100: {
                "10": [0.4, 0.2, 0.6],
                "11": [0.2, 0.1, 0.3],
                "12": [0.1, 0.1, 0.2],
            },
            101: {"11": [0.5, 0.4, 0.9]},
        }

        X, names = build_prototype_features(
            target_ids=target_ids,
            score_maps=score_maps,
            query_ids=query_ids,
            query_targets=query_targets,
            prefix="active",
            top_ks=(1, 2),
        )

        self.assertEqual(X.shape, (2, len(names)))
        self.assertIn("active_combo_top1_mean", names)
        self.assertIn("active_combo_top2_mean", names)
        self.assertIn("active_knn_k2_pec50", names)
        combo_top1 = X[0, names.index("active_combo_top1_mean")]
        combo_top2 = X[0, names.index("active_combo_top2_mean")]
        knn2 = X[0, names.index("active_knn_k2_pec50")]
        coverage = X[1, names.index("active_query_coverage")]

        self.assertAlmostEqual(combo_top1, 0.6, places=6)
        self.assertAlmostEqual(combo_top2, 0.45, places=6)
        self.assertAlmostEqual(knn2, (7.0 * 0.6 + 6.0 * 0.3) / 0.9, places=6)
        self.assertAlmostEqual(coverage, 1.0 / 3.0, places=6)

    def test_complete_score_maps_adds_empty_maps_for_missing_targets(self):
        from track1_activity.src.rocs_prototype import complete_score_maps

        completed = complete_score_maps([1, 2, 3], {1: {"9": [0.1, 0.2, 0.3]}})

        self.assertEqual(completed[1], {"9": [0.1, 0.2, 0.3]})
        self.assertEqual(completed[2], {})
        self.assertEqual(completed[3], {})

    def test_build_dense_query_features_keeps_query_columns(self):
        from track1_activity.src.rocs_prototype import build_dense_query_features

        X, names = build_dense_query_features(
            target_ids=[100, 101],
            score_maps={
                100: {"10": [0.1, 0.2, 0.3], "11": [0.4, 0.5, 0.9]},
                101: {"11": [0.7, 0.8, 1.5]},
            },
            query_ids=[10, 11],
            prefix="active",
        )

        self.assertEqual(X.shape, (2, 6))
        self.assertEqual(
            names,
            [
                "active_q10_shape",
                "active_q10_color",
                "active_q10_combo",
                "active_q11_shape",
                "active_q11_color",
                "active_q11_combo",
            ],
        )
        self.assertTrue(np.allclose(X[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]))
        self.assertTrue(np.allclose(X[1], [0.0, 0.0, 0.0, 0.7, 0.8, 1.5]))


if __name__ == "__main__":
    unittest.main()
