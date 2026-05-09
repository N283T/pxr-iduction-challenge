import unittest

import torch
from torch_geometric.data import Batch, Data

from track1_activity.src.ka_gnn import (
    FourierKAGNNModel,
    FourierKANLinear,
    PykanSAGEModel,
    augment_node_features_with_edge_mean,
    load_pretrained_encoder_state,
)


class TestKAGNN(unittest.TestCase):
    def test_fourier_kan_linear_shape_and_finite_output(self):
        layer = FourierKANLinear(input_dim=4, output_dim=3, grid_size=2, add_bias=True)
        x = torch.randn(5, 4)

        out = layer(x)

        self.assertEqual(tuple(out.shape), (5, 3))
        self.assertTrue(torch.isfinite(out).all())

    def test_augment_node_features_with_incoming_edge_mean(self):
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
        edge_attr = torch.tensor([[2.0, 4.0], [6.0, 8.0]])

        out = augment_node_features_with_edge_mean(x, edge_index, edge_attr)

        self.assertEqual(tuple(out.shape), (3, 4))
        torch.testing.assert_close(out[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(out[1], torch.tensor([0.0, 1.0, 4.0, 6.0]))
        torch.testing.assert_close(out[2], torch.tensor([2.0, 2.0, 0.0, 0.0]))

    def test_fourier_ka_gnn_returns_one_prediction_per_graph(self):
        g1 = Data(
            x=torch.randn(3, 4),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
            edge_attr=torch.randn(4, 2),
        )
        g2 = Data(
            x=torch.randn(2, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.randn(2, 2),
        )
        batch = Batch.from_data_list([g1, g2])
        model = FourierKAGNNModel(
            in_dim=4,
            edge_dim=2,
            hidden_dim=8,
            out_dim=4,
            grid_size=2,
            num_layers=2,
            pooling="mean",
            dropout=0.0,
        )

        emb = model.encode_graph(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        self.assertEqual(tuple(emb.shape), (2, 8))
        self.assertEqual(tuple(pred.shape), (2, 1))
        self.assertTrue(torch.isfinite(pred).all())

    def test_pykan_sage_returns_one_prediction_per_graph(self):
        g1 = Data(
            x=torch.randn(3, 4),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
            edge_attr=torch.randn(4, 2),
        )
        g2 = Data(
            x=torch.randn(2, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.randn(2, 2),
        )
        batch = Batch.from_data_list([g1, g2])
        model = PykanSAGEModel(
            in_dim=4,
            edge_dim=2,
            hidden_dim=8,
            out_dim=4,
            grid_size=2,
            num_layers=2,
            pooling="mean",
            dropout=0.0,
        )

        emb = model.encode_graph(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        self.assertEqual(tuple(emb.shape), (2, 8))
        self.assertEqual(tuple(pred.shape), (2, 1))
        self.assertTrue(torch.isfinite(pred).all())

    def test_load_pretrained_encoder_state_skips_readout_and_standardization(self):
        model = FourierKAGNNModel(
            in_dim=4,
            edge_dim=2,
            hidden_dim=8,
            out_dim=4,
            grid_size=2,
            num_layers=1,
            pooling="mean",
            dropout=0.0,
        )
        state = {key: torch.full_like(value, 0.25) for key, value in model.state_dict().items()}

        loaded = load_pretrained_encoder_state(model, state)

        self.assertGreater(loaded, 0)
        self.assertTrue(
            torch.allclose(model.input.fourier_coeffs, torch.full_like(model.input.fourier_coeffs, 0.25))
        )
        self.assertFalse(torch.allclose(model.readout_2.bias, torch.full_like(model.readout_2.bias, 0.25)))
        self.assertFalse(torch.allclose(model.x_std, torch.full_like(model.x_std, 0.25)))


if __name__ == "__main__":
    unittest.main()
