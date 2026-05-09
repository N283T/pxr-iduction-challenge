"""GSL-MPP-inspired molecule-graph utilities for Track 1.

This module intentionally implements a compact, dependency-light probe rather
than vendoring the full upstream GSL-MPP training framework. The shared idea is
an inter-molecule graph: molecules are nodes, Morgan/Tanimoto similarities form
an initial graph, and labels/residuals are propagated across that graph.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

VERY_SMALL = 1e-12


def morgan_bit_matrix(
    smiles_list: list[str], radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """Return a binary Morgan fingerprint matrix for ``smiles_list``.

    Invalid SMILES are encoded as all-zero rows instead of failing the whole
    probe. The project stores standardized SMILES, so invalid rows should not
    occur in normal operation; the defensive behavior keeps diagnostics robust.
    """
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        out[idx] = np.asarray(generator.GetFingerprint(mol), dtype=np.uint8)
    return out


def tanimoto_similarity(query_bits: np.ndarray, anchor_bits: np.ndarray) -> np.ndarray:
    """Compute dense Tanimoto similarities between binary bit matrices."""
    query = np.asarray(query_bits, dtype=np.uint8)
    anchor = np.asarray(anchor_bits, dtype=np.uint8)
    query_pop = query.sum(axis=1, dtype=np.int32)
    anchor_pop = anchor.sum(axis=1, dtype=np.int32)
    intersection = query.astype(np.int32) @ anchor.T.astype(np.int32)
    union = query_pop[:, None] + anchor_pop[None, :] - intersection
    return np.divide(
        intersection,
        np.maximum(union, 1),
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def topk_row_normalized_adjacency(
    similarity: np.ndarray,
    k: int,
    include_self: bool = False,
) -> np.ndarray:
    """Keep top-k neighbors per row and row-normalize edge weights.

    Args:
        similarity: Square or rectangular similarity matrix.
        k: Number of non-zero outgoing edges per row before normalization.
        include_self: If false and the matrix is square, diagonal edges are
            removed before top-k selection.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    sim = np.asarray(similarity, dtype=np.float64).copy()
    if not include_self and sim.shape[0] == sim.shape[1]:
        np.fill_diagonal(sim, 0.0)

    n_rows, n_cols = sim.shape
    k_eff = min(k, n_cols)
    adjacency = np.zeros_like(sim, dtype=np.float64)
    if k_eff == n_cols:
        adjacency = np.maximum(sim, 0.0)
    else:
        top_idx = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
        row_idx = np.arange(n_rows)[:, None]
        adjacency[row_idx, top_idx] = np.maximum(sim[row_idx, top_idx], 0.0)

    row_sum = adjacency.sum(axis=1, keepdims=True)
    non_empty = row_sum[:, 0] > VERY_SMALL
    adjacency[non_empty] = adjacency[non_empty] / row_sum[non_empty]
    return adjacency


def propagate_residuals(
    adjacency: np.ndarray,
    residual_seed: np.ndarray,
    labeled_mask: np.ndarray,
    alpha: float = 0.85,
    n_iter: int = 50,
    clamp_labeled: bool = True,
) -> np.ndarray:
    """Diffuse residual seeds over an inter-molecule graph.

    ``residual_seed`` should be zero for unlabeled rows. When
    ``clamp_labeled`` is true, labeled rows are reset to their original seed
    after every iteration, matching classic label propagation behavior.
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")

    adj = np.asarray(adjacency, dtype=np.float64)
    residual0 = np.asarray(residual_seed, dtype=np.float64)
    labeled = np.asarray(labeled_mask, dtype=bool)
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be square, got {adj.shape}")
    if adj.shape[0] != residual0.shape[0] or residual0.shape[0] != labeled.shape[0]:
        raise ValueError(
            "adjacency, residual_seed, and labeled_mask lengths do not match: "
            f"{adj.shape}, {residual0.shape}, {labeled.shape}"
        )

    residual = residual0.copy()
    for _ in range(n_iter):
        residual = alpha * (adj @ residual) + (1.0 - alpha) * residual0
        if clamp_labeled:
            residual[labeled] = residual0[labeled]
    return residual


def apply_residual_correction(
    anchor_pred: np.ndarray,
    propagated_residual: np.ndarray,
    gamma: float,
    clip: float,
) -> np.ndarray:
    """Apply a clipped residual correction to anchor predictions."""
    if clip <= 0:
        raise ValueError(f"clip must be positive, got {clip}")
    anchor = np.asarray(anchor_pred, dtype=np.float64)
    residual = np.asarray(propagated_residual, dtype=np.float64)
    shift = np.clip(gamma * residual, -clip, clip)
    return anchor + shift
