#!/usr/bin/env -S pixi run python
"""Build compact PCA/Kronecker interaction features for Boltz distance experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

POOLED_ALLPAIRS_PATH = REPO_ROOT.joinpath(
    "data", "boltz_affhead", "pooled_allpairs.parquet"
)
TOKEN_DIST_PATH = REPO_ROOT.joinpath("data", "boltz2_token_distogram_features.parquet")
WEIGHTED_Z_PATH = REPO_ROOT.joinpath(
    "data", "boltz_affhead", "dist_weighted_z_pool.parquet"
)
OUT_PATH = REPO_ROOT.joinpath("data", "boltz_affhead", "dist_interactions.parquet")


def _compound_ids() -> list[int]:
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
        train_ids = [int(r[0]) for r in cur.fetchall()]
        cur.execute("SELECT compound_id FROM test_activity ORDER BY id")
        test_ids = [int(r[0]) for r in cur.fetchall()]
    return train_ids + test_ids


def _load_matrix(path: Path, ids: list[int]) -> np.ndarray:
    df = pd.read_parquet(path)
    if "compound_id" in df.columns:
        df = df.set_index("compound_id")
    df = df.reindex(ids)
    X = df.to_numpy(dtype=np.float32).copy()
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X[~np.isfinite(X)] = np.broadcast_to(col_mean, X.shape)[~np.isfinite(X)]
    return X


def _pca_block(
    X: np.ndarray, n_components: int, prefix: str
) -> tuple[np.ndarray, list[str]]:
    Xs = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=n_components, random_state=42).fit_transform(Xs)
    cols = [f"{prefix}_pc{i:03d}" for i in range(n_components)]
    return Xp.astype(np.float32), cols


def _kron_block(
    A: np.ndarray,
    B: np.ndarray,
    a_dim: int,
    b_dim: int,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    A = A[:, :a_dim]
    B = B[:, :b_dim]
    X = (A[:, :, None] * B[:, None, :]).reshape(A.shape[0], a_dim * b_dim)
    cols = [f"{prefix}_{i:02d}_{j:02d}" for i in range(a_dim) for j in range(b_dim)]
    return X.astype(np.float32), cols


def _hadamard_block(
    A: np.ndarray,
    B: np.ndarray,
    dim: int,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    X = A[:, :dim] * B[:, :dim]
    cols = [f"{prefix}_{i:03d}" for i in range(dim)]
    return X.astype(np.float32), cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    ids = _compound_ids()
    pooled = _load_matrix(POOLED_ALLPAIRS_PATH, ids)
    token = _load_matrix(TOKEN_DIST_PATH, ids)
    weighted = _load_matrix(WEIGHTED_Z_PATH, ids)
    print(f"pooled={pooled.shape} token={token.shape} weighted={weighted.shape}")

    pooled_pca, pooled_cols = _pca_block(pooled, 64, "pooled_ap")
    token_pca, token_cols = _pca_block(token, 48, "token_dist")
    weighted_pca, weighted_cols = _pca_block(weighted, 96, "weighted_z")

    blocks = [
        (pooled_pca, pooled_cols),
        (token_pca, token_cols),
        (weighted_pca, weighted_cols),
        _kron_block(pooled_pca, token_pca, 16, 16, "kron_pooled_token"),
        _kron_block(pooled_pca, weighted_pca, 16, 16, "kron_pooled_weighted"),
        _kron_block(token_pca, weighted_pca, 16, 16, "kron_token_weighted"),
        _hadamard_block(pooled_pca, weighted_pca, 64, "mul_pooled_weighted"),
        _hadamard_block(token_pca, weighted_pca, 48, "mul_token_weighted"),
    ]

    X = np.concatenate([block for block, _ in blocks], axis=1).astype(np.float32)
    cols = [col for _, block_cols in blocks for col in block_cols]
    out_df = pd.DataFrame(X, columns=cols)
    out_df.insert(0, "compound_id", ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False, compression="zstd")
    print(f"Wrote {args.out}: {out_df.shape}")


if __name__ == "__main__":
    main()
