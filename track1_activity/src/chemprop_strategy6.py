"""Utilities for Buterez 2024 Strategy 6 with ChemProp encoders.

Strategy 6 keeps the low-fidelity-pretrained message-passing encoder frozen and
learns an adaptive high-fidelity graph readout on pEC50.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:  # script-style import when track1_activity/src is on sys.path
    from adaptive_readout import AdaptiveReadoutRegressor
except ModuleNotFoundError:  # package-style import from tests
    from track1_activity.src.adaptive_readout import AdaptiveReadoutRegressor


def freeze_all(module: nn.Module) -> int:
    """Set all module parameters to frozen and return the parameter-tensor count."""
    count = 0
    for param in module.parameters():
        param.requires_grad = False
        count += 1
    return count


def pad_node_embeddings(
    node_embeddings: torch.Tensor, batch_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert concatenated node embeddings into padded graph batches.

    Parameters
    ----------
    node_embeddings:
        Tensor with shape ``(total_nodes, hidden_dim)``.
    batch_index:
        Long tensor with shape ``(total_nodes,)`` mapping each node to a graph id.

    Returns
    -------
    padded:
        Tensor with shape ``(num_graphs, max_nodes, hidden_dim)``.
    mask:
        Boolean tensor with shape ``(num_graphs, max_nodes)``. True entries are
        real atoms; False entries are padding.
    """
    if node_embeddings.ndim != 2:
        raise ValueError("node_embeddings must be a 2D tensor")
    if batch_index.ndim != 1:
        raise ValueError("batch_index must be a 1D tensor")
    if node_embeddings.shape[0] != batch_index.shape[0]:
        raise ValueError("node_embeddings and batch_index length mismatch")
    if batch_index.numel() == 0:
        raise ValueError("batch_index must contain at least one node")

    batch_index = batch_index.to(device=node_embeddings.device, dtype=torch.long)
    n_graphs = int(batch_index.max().item()) + 1
    counts = torch.bincount(batch_index, minlength=n_graphs)
    max_nodes = int(counts.max().item())
    hidden_dim = node_embeddings.shape[1]

    padded = node_embeddings.new_zeros((n_graphs, max_nodes, hidden_dim))
    mask = torch.zeros(
        (n_graphs, max_nodes), dtype=torch.bool, device=node_embeddings.device
    )
    for graph_id in range(n_graphs):
        node_sel = batch_index == graph_id
        n_nodes = int(node_sel.sum().item())
        if n_nodes == 0:
            continue
        padded[graph_id, :n_nodes] = node_embeddings[node_sel]
        mask[graph_id, :n_nodes] = True
    return padded, mask


class ChempropNodeEncoder(nn.Module):
    """Expose atom-level hidden states from a ChemProp MPNN message-passing block."""

    def __init__(self, message_passing: nn.Module):
        super().__init__()
        self.message_passing = message_passing

    def forward(self, bmg: Any, V_d: torch.Tensor | None = None) -> torch.Tensor:
        return self.message_passing(bmg, V_d)


class ChempropStrategy6Regressor(nn.Module):
    """Frozen ChemProp node encoder plus trainable adaptive graph readout."""

    def __init__(
        self,
        node_encoder: nn.Module,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 1,
        num_seeds: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_encoder = node_encoder
        self.readout = AdaptiveReadoutRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            num_seeds=num_seeds,
            dropout=dropout,
        )

    def forward(self, bmg: Any, V_d: torch.Tensor | None = None) -> torch.Tensor:
        node_embeddings = self.node_encoder(bmg, V_d)
        padded, mask = pad_node_embeddings(node_embeddings, bmg.batch)
        return self.readout(padded, mask)
