"""Utilities for KAN regressors on frozen molecular embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class FoldStandardizer:
    """Train-fold-only standardization for tabular features and targets."""

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, X_train: np.ndarray, y_train: np.ndarray) -> "FoldStandardizer":
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X_train must be 2D")
        if y.ndim != 1:
            raise ValueError("y_train must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_train and y_train row counts differ")
        x_mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        x_std = X.std(axis=0, dtype=np.float64).astype(np.float32)
        x_std[x_std < 1e-6] = 1.0
        y_mean = float(y.mean(dtype=np.float64))
        y_std = float(y.std(dtype=np.float64))
        if y_std < 1e-6:
            y_std = 1.0
        return cls(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float32) - self.x_mean) / self.x_std).astype(
            np.float32
        )

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=np.float32) - self.y_mean) / self.y_std).astype(
            np.float32
        )

    def inverse_y(self, y_z: np.ndarray) -> np.ndarray:
        return (np.asarray(y_z, dtype=np.float32) * self.y_std + self.y_mean).astype(
            np.float32
        )


def build_kan_width(
    input_dim: int, hidden_dim: int, second_hidden_dim: int | None = None
) -> list[int]:
    """Build a pykan width list for one-output regression."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if second_hidden_dim is None or second_hidden_dim <= 0:
        return [int(input_dim), int(hidden_dim), 1]
    return [int(input_dim), int(hidden_dim), int(second_hidden_dim), 1]


def fit_pca_if_needed(
    X_train: np.ndarray,
    X_other: np.ndarray,
    max_dim: int | None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, PCA | None]:
    """Reduce feature dimension with train-fit PCA when requested.

    pykan scales poorly with hundreds of inputs. This helper lets probes cap the
    embedding dimension without leaking validation/test statistics.
    """
    X_train_arr = np.asarray(X_train, dtype=np.float32)
    X_other_arr = np.asarray(X_other, dtype=np.float32)
    if max_dim is None or max_dim <= 0 or X_train_arr.shape[1] <= max_dim:
        return X_train_arr.copy(), X_other_arr.copy(), None
    n_components = min(int(max_dim), X_train_arr.shape[0], X_train_arr.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_arr).astype(np.float32)
    X_other_pca = pca.transform(X_other_arr).astype(np.float32)
    return X_train_pca, X_other_pca, pca
