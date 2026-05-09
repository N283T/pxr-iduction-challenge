"""PyG-native KA-GNN components for molecular property probes.

This module ports the KA-GNN idea from LongLee220/KA-GNN without depending on
DGL. The main layer uses Fourier KAN edge/message functions, residual graph
message passing, graph pooling, and a KAN readout.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_add_pool, global_max_pool, global_mean_pool
try:
    from kan import KAN
except ModuleNotFoundError:  # pykan is optional outside KA-GNN experiments
    KAN = None
from torch_geometric.utils import scatter


class FourierKANLinear(nn.Module):
    """KAN-style linear map with Fourier basis functions on each input edge."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: int,
        add_bias: bool = True,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or grid_size <= 0:
            raise ValueError("input_dim, output_dim, and grid_size must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.grid_size = int(grid_size)
        scale = math.sqrt(self.input_dim) * math.sqrt(self.grid_size)
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, self.output_dim, self.input_dim, self.grid_size) / scale
        )
        self.bias = nn.Parameter(torch.zeros(self.output_dim)) if add_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)
        k = torch.arange(
            1, self.grid_size + 1, device=x.device, dtype=x.dtype
        ).reshape(1, 1, 1, self.grid_size)
        x_basis = x_flat.reshape(x_flat.shape[0], 1, self.input_dim, 1)
        cos = torch.cos(k * x_basis).reshape(
            1, x_flat.shape[0], self.input_dim, self.grid_size
        )
        sin = torch.sin(k * x_basis).reshape(
            1, x_flat.shape[0], self.input_dim, self.grid_size
        )
        basis = torch.cat([cos, sin], dim=0)
        out = torch.einsum("dbik,djik->bj", basis, self.fourier_coeffs)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x_shape[:-1], self.output_dim)


class FourierKANMessagePassing(nn.Module):
    """Neighbor aggregation where each source feature passes through a KAN map."""

    def __init__(
        self,
        hidden_dim: int,
        grid_size: int,
        add_bias: bool = True,
        aggr: str = "sum",
    ):
        super().__init__()
        if aggr not in {"sum", "mean"}:
            raise ValueError("aggr must be 'sum' or 'mean'")
        self.message = FourierKANLinear(hidden_dim, hidden_dim, grid_size, add_bias)
        self.aggr = aggr

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        messages = self.message(x[src])
        return scatter(messages, dst, dim=0, dim_size=x.shape[0], reduce=self.aggr)


def augment_node_features_with_edge_mean(
    x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
) -> torch.Tensor:
    """Concatenate each node with mean incoming edge features.

    The upstream DGL script updates node features by concatenating aggregated edge
    features before feeding the KA-GNN. This is a PyG equivalent.
    """
    if edge_attr is None:
        return x.float()
    edge_attr_f = edge_attr.float()
    dst = edge_index[1]
    edge_mean = scatter(
        edge_attr_f, dst, dim=0, dim_size=x.shape[0], reduce="mean"
    )
    return torch.cat([x.float(), edge_mean], dim=1)


class FourierKAGNNModel(nn.Module):
    """Fourier KA-GNN for graph-level regression."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        out_dim: int,
        grid_size: int,
        num_layers: int,
        pooling: str = "mean",
        dropout: float = 0.1,
        use_bias: bool = True,
        aggr: str = "sum",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if pooling not in {"mean", "sum", "max"}:
            raise ValueError("pooling must be one of mean/sum/max")
        self.pooling = pooling
        self.input = FourierKANLinear(
            in_dim + edge_dim, hidden_dim, grid_size, add_bias=use_bias
        )
        self.layers = nn.ModuleList(
            [
                FourierKANMessagePassing(
                    hidden_dim, grid_size, add_bias=use_bias, aggr=aggr
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.readout_1 = FourierKANLinear(hidden_dim, out_dim, grid_size, use_bias)
        self.readout_2 = FourierKANLinear(out_dim, 1, grid_size, use_bias)
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("edge_mean", torch.zeros(edge_dim))
        self.register_buffer("edge_std", torch.ones(edge_dim))

    def set_feature_standardization(
        self,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        edge_mean: torch.Tensor,
        edge_std: torch.Tensor,
    ) -> None:
        """Store fold-local feature normalization statistics as buffers."""
        self.x_mean.copy_(x_mean.float())
        self.x_std.copy_(torch.clamp(x_std.float(), min=1e-6))
        self.edge_mean.copy_(edge_mean.float())
        self.edge_std.copy_(torch.clamp(edge_std.float(), min=1e-6))

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        if self.pooling == "sum":
            return global_add_pool(x, batch)
        return global_max_pool(x, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = (x.float() - self.x_mean) / self.x_std
        edge_attr = (edge_attr.float() - self.edge_mean) / self.edge_std
        h = augment_node_features_with_edge_mean(x, edge_index, edge_attr)
        h = self.input(h)
        for layer in self.layers:
            msg = layer(h, edge_index)
            h = F.leaky_relu(h + msg, negative_slope=0.1)
            h = self.dropout(h)
        pooled = self._pool(h, batch)
        out = self.readout_2(F.leaky_relu(self.readout_1(pooled), negative_slope=0.1))
        return out


class PykanSAGEModel(nn.Module):
    """GraphSAGE body with pykan B-spline KAN input and readout maps."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        out_dim: int,
        grid_size: int,
        num_layers: int,
        pooling: str = "mean",
        dropout: float = 0.1,
        kan_bottleneck: int = 5,
        spline_order: int = 3,
        seed: int = 0,
    ):
        super().__init__()
        if KAN is None:
            raise ModuleNotFoundError("pykan is required for PykanSAGEModel")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if pooling not in {"mean", "sum", "max"}:
            raise ValueError("pooling must be one of mean/sum/max")
        self.pooling = pooling
        self.input = KAN(
            width=[in_dim + edge_dim, kan_bottleneck, hidden_dim],
            grid=grid_size,
            k=spline_order,
            seed=seed,
            symbolic_enabled=False,
            auto_save=False,
            save_act=False,
        )
        self.layers = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim, aggr="mean") for _ in range(num_layers - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.readout_1 = KAN(
            width=[hidden_dim, kan_bottleneck, out_dim],
            grid=grid_size,
            k=spline_order,
            seed=seed + 1000,
            symbolic_enabled=False,
            auto_save=False,
            save_act=False,
        )
        self.readout_2 = KAN(
            width=[out_dim, kan_bottleneck, 1],
            grid=grid_size,
            k=spline_order,
            seed=seed + 2000,
            symbolic_enabled=False,
            auto_save=False,
            save_act=False,
        )
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("edge_mean", torch.zeros(edge_dim))
        self.register_buffer("edge_std", torch.ones(edge_dim))

    def set_feature_standardization(
        self,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        edge_mean: torch.Tensor,
        edge_std: torch.Tensor,
    ) -> None:
        self.x_mean.copy_(x_mean.float())
        self.x_std.copy_(torch.clamp(x_std.float(), min=1e-6))
        self.edge_mean.copy_(edge_mean.float())
        self.edge_std.copy_(torch.clamp(edge_std.float(), min=1e-6))

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        if self.pooling == "sum":
            return global_add_pool(x, batch)
        return global_max_pool(x, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = (x.float() - self.x_mean) / self.x_std
        edge_attr = (edge_attr.float() - self.edge_mean) / self.edge_std
        h = augment_node_features_with_edge_mean(x, edge_index, edge_attr)
        h = self.input(h)
        for layer in self.layers:
            msg = layer(h, edge_index)
            h = F.leaky_relu(h + msg, negative_slope=0.1)
            h = self.dropout(h)
        pooled = self._pool(h, batch)
        return self.readout_2(F.leaky_relu(self.readout_1(pooled), negative_slope=0.1))
