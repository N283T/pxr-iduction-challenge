import unittest

import numpy as np

from track1_activity.src.kan_embed import (
    FoldStandardizer,
    build_kan_width,
    fit_pca_if_needed,
)


class TestKanEmbed(unittest.TestCase):
    def test_fold_standardizer_uses_train_statistics_and_inverts_targets(self):
        X_train = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]], dtype=np.float32)
        y_train = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        X_val = np.array([[7.0, 10.0]], dtype=np.float32)

        scaler = FoldStandardizer.fit(X_train, y_train)
        X_train_z = scaler.transform_x(X_train)
        X_val_z = scaler.transform_x(X_val)
        y_train_z = scaler.transform_y(y_train)

        np.testing.assert_allclose(X_train_z[:, 0].mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(X_train_z[:, 0].std(), 1.0, atol=1e-6)
        np.testing.assert_allclose(X_train_z[:, 1], 0.0, atol=1e-6)
        self.assertGreater(float(X_val_z[0, 0]), 1.0)
        np.testing.assert_allclose(scaler.inverse_y(y_train_z), y_train, atol=1e-6)

    def test_build_kan_width(self):
        self.assertEqual(build_kan_width(input_dim=16, hidden_dim=8), [16, 8, 1])
        self.assertEqual(
            build_kan_width(input_dim=16, hidden_dim=8, second_hidden_dim=4),
            [16, 8, 4, 1],
        )

    def test_fit_pca_if_needed_skips_when_dimension_is_small(self):
        X_train = np.eye(4, dtype=np.float32)
        X_test = np.ones((2, 4), dtype=np.float32)

        X_train_out, X_test_out, pca = fit_pca_if_needed(X_train, X_test, max_dim=8)

        self.assertIsNone(pca)
        np.testing.assert_allclose(X_train_out, X_train)
        np.testing.assert_allclose(X_test_out, X_test)

    def test_fit_pca_if_needed_reduces_dimension(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(20, 6)).astype(np.float32)
        X_test = rng.normal(size=(3, 6)).astype(np.float32)

        X_train_out, X_test_out, pca = fit_pca_if_needed(X_train, X_test, max_dim=3)

        self.assertIsNotNone(pca)
        self.assertEqual(X_train_out.shape, (20, 3))
        self.assertEqual(X_test_out.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
