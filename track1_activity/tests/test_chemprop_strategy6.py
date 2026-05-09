import unittest

import torch

from track1_activity.src.chemprop_strategy6 import (
    ChempropStrategy6Regressor,
    freeze_all,
    pad_node_embeddings,
)


class DummyNodeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, bmg, V_d=None):
        del V_d
        return self.linear(bmg.V)


class DummyBatchMolGraph:
    def __init__(self):
        self.V = torch.randn(5, 3)
        self.batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)


class TestChempropStrategy6(unittest.TestCase):
    def test_pad_node_embeddings_builds_masked_graph_tensor(self):
        nodes = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

        padded, mask = pad_node_embeddings(nodes, batch)

        self.assertEqual(tuple(padded.shape), (2, 3, 3))
        self.assertEqual(mask.tolist(), [[True, True, False], [True, True, True]])
        torch.testing.assert_close(padded[0, 0], nodes[0])
        torch.testing.assert_close(padded[0, 1], nodes[1])
        torch.testing.assert_close(padded[0, 2], torch.zeros(3))
        torch.testing.assert_close(padded[1], nodes[2:5])

    def test_freeze_all_disables_gradients(self):
        module = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))

        frozen_count = freeze_all(module)

        self.assertEqual(frozen_count, 4)
        self.assertTrue(all(not p.requires_grad for p in module.parameters()))

    def test_strategy6_regressor_returns_graph_level_predictions(self):
        batch = DummyBatchMolGraph()
        node_encoder = DummyNodeEncoder()
        model = ChempropStrategy6Regressor(
            node_encoder=node_encoder,
            input_dim=5,
            hidden_dim=8,
            num_heads=2,
            num_blocks=0,
            num_seeds=1,
            dropout=0.0,
        )

        pred = model(batch)

        self.assertEqual(tuple(pred.shape), (2,))
        self.assertTrue(torch.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
