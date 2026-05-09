"""Learned GSL-MPP-style dense molecule graph model.

The upstream GSL-MPP model learns graph structure over molecule nodes. This
module keeps that central mechanism while using compact node features prepared
by project scripts instead of vendoring the full upstream data framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VERY_SMALL = 1e-12


def build_topk_adjacency_torch(
    similarity: torch.Tensor,
    k: int,
    include_self: bool = False,
) -> torch.Tensor:
    """Top-k row-normalized adjacency from a dense similarity matrix."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    sim = similarity.float().clone()
    if not include_self and sim.shape[0] == sim.shape[1]:
        sim.fill_diagonal_(0.0)
    sim = torch.clamp(sim, min=0.0)
    k_eff = min(k, sim.shape[1])
    if k_eff < sim.shape[1]:
        values, indices = torch.topk(sim, k=k_eff, dim=1)
        adj = torch.zeros_like(sim)
        adj.scatter_(1, indices, values)
    else:
        adj = sim
    row_sum = adj.sum(dim=1, keepdim=True).clamp_min(VERY_SMALL)
    return adj / row_sum


def masked_mae_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error over entries where ``mask`` is true."""
    mask_bool = mask.bool()
    if not bool(mask_bool.any()):
        raise ValueError("masked_mae_loss received an empty mask")
    return torch.mean(torch.abs(pred[mask_bool] - target[mask_bool]))


class DenseGraphConvolution(nn.Module):
    """Dense GCN layer using a pre-normalized adjacency matrix."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return adjacency @ self.linear(features)


class WeightedCosineGraphLearner(nn.Module):
    """Multi-perspective weighted cosine graph learner."""

    def __init__(self, input_dim: int, num_perspectives: int = 8):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_perspectives, input_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weighted = features.unsqueeze(0) * self.weight.unsqueeze(1)
        normalized = F.normalize(weighted, p=2, dim=-1, eps=1e-8)
        sim = normalized @ normalized.transpose(-1, -2)
        return sim.mean(dim=0)


class DenseGslMppRegressor(nn.Module):
    """Dense learned molecule-graph residual regressor."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        learned_k: int = 32,
        num_perspectives: int = 8,
        graph_skip: float = 0.8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not 0 <= graph_skip <= 1:
            raise ValueError(f"graph_skip must be in [0, 1], got {graph_skip}")
        self.learned_k = learned_k
        self.graph_skip = graph_skip
        self.graph_learner = WeightedCosineGraphLearner(input_dim, num_perspectives)
        self.gcn1 = DenseGraphConvolution(input_dim, hidden_dim)
        self.gcn2 = DenseGraphConvolution(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, init_adj: torch.Tensor) -> torch.Tensor:
        learned_sim = self.graph_learner(features)
        learned_adj = build_topk_adjacency_torch(
            learned_sim, k=self.learned_k, include_self=False
        )
        adj = self.graph_skip * init_adj + (1.0 - self.graph_skip) * learned_adj
        adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(VERY_SMALL)
        hidden = F.relu(self.gcn1(features, adj))
        hidden = self.dropout(hidden)
        hidden = F.relu(self.gcn2(hidden, adj))
        joined = torch.cat([hidden, features], dim=1)
        return self.head(joined).squeeze(-1)


@dataclass
class DenseGslMppFitResult:
    predictions: np.ndarray
    history: list[float]


def fit_dense_gsl_mpp(
    features: np.ndarray,
    init_adj: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    epochs: int = 600,
    hidden_dim: int = 128,
    learned_k: int = 32,
    num_perspectives: int = 8,
    graph_skip: float = 0.8,
    dropout: float = 0.1,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, list[float]]:
    """Fit a full-batch dense GSL-MPP regressor and return all-node predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")

    x = torch.as_tensor(features, dtype=torch.float32, device=requested_device)
    adj = torch.as_tensor(init_adj, dtype=torch.float32, device=requested_device)
    y = torch.as_tensor(target, dtype=torch.float32, device=requested_device)
    mask = torch.as_tensor(train_mask, dtype=torch.bool, device=requested_device)

    model = DenseGslMppRegressor(
        input_dim=x.shape[1],
        hidden_dim=hidden_dim,
        learned_k=learned_k,
        num_perspectives=num_perspectives,
        graph_skip=graph_skip,
        dropout=dropout,
    ).to(requested_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, adj)
        loss = masked_mae_loss(pred, y, mask)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        pred = model(x, adj).detach().cpu().numpy().astype(np.float64)
    return pred, history
