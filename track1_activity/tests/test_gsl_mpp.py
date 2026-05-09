from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from gsl_mpp import (  # noqa: E402
    apply_residual_correction,
    propagate_residuals,
    tanimoto_similarity,
    topk_row_normalized_adjacency,
)


class TestGslMpp(unittest.TestCase):
    def test_tanimoto_similarity_handles_binary_bit_matrices(self):
        query = np.array([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)
        anchor = np.array([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=np.uint8)

        sim = tanimoto_similarity(query, anchor)

        np.testing.assert_allclose(sim, [[1.0, 1.0 / 3.0], [0.0, 0.5]])

    def test_topk_adjacency_row_normalizes_and_removes_self_edges(self):
        sim = np.array(
            [
                [1.0, 0.8, 0.1],
                [0.8, 1.0, 0.4],
                [0.1, 0.4, 1.0],
            ],
            dtype=np.float64,
        )

        adj = topk_row_normalized_adjacency(sim, k=1, include_self=False)

        np.testing.assert_allclose(np.diag(adj), 0.0)
        np.testing.assert_allclose(adj.sum(axis=1), 1.0)
        self.assertEqual(np.count_nonzero(adj[0]), 1)
        self.assertEqual(adj[0, 1], 1.0)
        self.assertEqual(adj[1, 0], 1.0)
        self.assertEqual(adj[2, 1], 1.0)

    def test_propagate_residuals_keeps_labeled_seeds_and_moves_unlabeled_node(self):
        adj = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        residual_seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        labeled_mask = np.array([True, False, False])

        propagated = propagate_residuals(
            adj,
            residual_seed,
            labeled_mask,
            alpha=0.8,
            n_iter=20,
            clamp_labeled=True,
        )

        self.assertEqual(propagated[0], 1.0)
        self.assertGreater(propagated[1], 0.4)
        self.assertGreater(propagated[2], 0.2)

    def test_apply_residual_correction_clips_shift(self):
        anchor = np.array([4.0, 5.0], dtype=np.float64)
        residual = np.array([1.0, -1.0], dtype=np.float64)

        corrected = apply_residual_correction(anchor, residual, gamma=0.5, clip=0.2)

        np.testing.assert_allclose(corrected, [4.2, 4.8])


if __name__ == "__main__":
    unittest.main()
