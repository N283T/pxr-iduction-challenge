from pathlib import Path
import sys
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from adaptive_readout import AdaptiveReadoutRegressor, SetTransformerReadout  # noqa: E402


class TestAdaptiveReadout(unittest.TestCase):
    def test_readout_is_permutation_invariant_without_dropout(self):
        torch.manual_seed(0)
        readout = SetTransformerReadout(
            input_dim=4, hidden_dim=8, num_heads=2, dropout=0.0
        )
        readout.eval()
        x = torch.randn(1, 5, 4)
        mask = torch.ones(1, 5, dtype=torch.bool)
        perm = torch.tensor([2, 4, 1, 3, 0])

        y1 = readout(x, mask)
        y2 = readout(x[:, perm], mask[:, perm])

        torch.testing.assert_close(y1, y2, atol=1e-6, rtol=1e-6)

    def test_padding_mask_ignores_padded_rows(self):
        torch.manual_seed(0)
        readout = SetTransformerReadout(
            input_dim=3, hidden_dim=8, num_heads=2, dropout=0.0
        )
        readout.eval()
        valid = torch.randn(1, 3, 3)
        padded_a = torch.cat([valid, torch.zeros(1, 2, 3)], dim=1)
        padded_b = torch.cat([valid, torch.randn(1, 2, 3) * 100.0], dim=1)
        mask = torch.tensor([[True, True, True, False, False]])

        y1 = readout(padded_a, mask)
        y2 = readout(padded_b, mask)

        torch.testing.assert_close(y1, y2, atol=1e-6, rtol=1e-6)

    def test_regressor_returns_one_value_per_graph(self):
        model = AdaptiveReadoutRegressor(
            input_dim=5, hidden_dim=8, num_heads=2, dropout=0.0
        )
        x = torch.randn(4, 7, 5)
        mask = torch.ones(4, 7, dtype=torch.bool)

        out = model(x, mask)

        self.assertEqual(tuple(out.shape), (4,))


if __name__ == "__main__":
    unittest.main()
