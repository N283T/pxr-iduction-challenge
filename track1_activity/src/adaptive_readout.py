"""Adaptive Set Transformer readouts for frozen molecular graph encoders."""

from __future__ import annotations

import torch
from torch import nn


class SetAttentionBlock(nn.Module):
    """Permutation-equivariant multi-head attention block for sets."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        key_padding_mask = None if mask is None else ~mask.bool()
        attended, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ff(x))
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


class PoolingByMultiheadAttention(nn.Module):
    """Set Transformer PMA with learnable seed vectors."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_seeds: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(num_seeds, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        seed = self.seed.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = None if mask is None else ~mask.bool()
        pooled, _ = self.attn(
            seed,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = self.norm1(seed + pooled)
        pooled = self.norm2(pooled + self.ff(pooled))
        return pooled


class SetTransformerReadout(nn.Module):
    """Permutation-invariant adaptive graph readout over node embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 1,
        num_seeds: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                SetAttentionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.pool = PoolingByMultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_seeds=num_seeds,
            dropout=dropout,
        )
        self.out_dim = hidden_dim * num_seeds

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.input_proj(x)
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(dtype=h.dtype)
        for block in self.blocks:
            h = block(h, mask)
        pooled = self.pool(h, mask)
        return pooled.flatten(start_dim=1)


class AdaptiveReadoutRegressor(nn.Module):
    """Set Transformer readout plus scalar regression head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 1,
        num_seeds: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.readout = SetTransformerReadout(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            num_seeds=num_seeds,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(self.readout.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.head(self.readout(x, mask)).squeeze(-1)
