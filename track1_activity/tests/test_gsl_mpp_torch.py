from pathlib import Path
import sys
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from gsl_mpp_torch import (  # noqa: E402
    DenseGslMppRegressor,
    build_topk_adjacency_torch,
    fit_dense_gsl_mpp,
    masked_mae_loss,
)


class TestGslMppTorch(unittest.TestCase):
    def test_topk_adjacency_is_row_normalized_without_self_edges(self):
        sim = torch.tensor(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.3],
                [0.1, 0.3, 1.0],
            ]
        )

        adj = build_topk_adjacency_torch(sim, k=1, include_self=False)

        torch.testing.assert_close(torch.diag(adj), torch.zeros(3))
        torch.testing.assert_close(adj.sum(dim=1), torch.ones(3))
        self.assertEqual(float(adj[0, 1]), 1.0)
        self.assertEqual(float(adj[1, 0]), 1.0)
        self.assertEqual(float(adj[2, 1]), 1.0)

    def test_masked_mae_loss_uses_only_masked_entries(self):
        pred = torch.tensor([1.0, 10.0, 3.0])
        target = torch.tensor([2.0, -100.0, 1.0])
        mask = torch.tensor([True, False, True])

        loss = masked_mae_loss(pred, target, mask)

        self.assertAlmostEqual(float(loss), 1.5)

    def test_dense_model_forward_shape_and_tiny_fit(self):
        torch.manual_seed(0)
        features = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        init_adj = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        target = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
        mask = np.array([True, True, True, True])

        model = DenseGslMppRegressor(input_dim=2, hidden_dim=8, learned_k=2)
        with torch.no_grad():
            out = model(torch.from_numpy(features), torch.from_numpy(init_adj))
        self.assertEqual(tuple(out.shape), (4,))

        pred, history = fit_dense_gsl_mpp(
            features,
            init_adj,
            target,
            mask,
            epochs=120,
            hidden_dim=8,
            learned_k=2,
            lr=0.03,
            weight_decay=0.0,
            graph_skip=0.5,
            seed=0,
            device="cpu",
        )

        self.assertLess(history[-1], history[0])
        self.assertLess(np.mean(np.abs(pred - target)), 0.4)


if __name__ == "__main__":
    unittest.main()
