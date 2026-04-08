"""Pseudo-label utilities for fold-safe training augmentation.

Currently supports Mordred features only. Pseudo compounds are loaded from a
parquet produced by ``scripts/build_pseudo_labels.py`` (columns:
``compound_id``, ``pseudo_pec50``, ``confidence``, ``n_concs``).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data import load_mordred


def load_pseudo_labels(path: Path) -> pd.DataFrame:
    """Read the pseudo-label parquet."""
    df = pd.read_parquet(path)
    required = {"compound_id", "pseudo_pec50", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pseudo parquet missing columns: {missing}")
    return df


def build_pseudo_feature_matrix(
    feature_name: str,
    pseudo_df: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Build a feature matrix for pseudo compounds matching the train layout.

    Mordred-only for now. ``feature_columns`` MUST be the exact ordered list of
    columns used to build the train matrix (canonical source: the column order
    returned by the same code path that builds ``X_train``). Pseudo Mordred
    rows are reindexed to that order; missing columns are filled with NaN
    (LightGBM tolerates NaN) and the count is logged.
    """
    if feature_name != "mordred":
        raise NotImplementedError(
            f"pseudo labels currently support feature='mordred' only, "
            f"got {feature_name!r}"
        )

    pseudo_ids = pseudo_df["compound_id"].astype(int).tolist()
    mordred_pseudo = load_mordred(pseudo_ids)

    missing_rows = set(pseudo_ids) - set(mordred_pseudo.index)
    if missing_rows:
        raise ValueError(
            f"Mordred missing for {len(missing_rows)} pseudo compounds "
            f"(e.g. {sorted(missing_rows)[:5]})"
        )

    missing_cols = [c for c in feature_columns if c not in mordred_pseudo.columns]
    if missing_cols:
        print(
            f"  [pseudo] {len(missing_cols)} feature columns missing in pseudo "
            f"Mordred (filled with NaN); e.g. {missing_cols[:5]}"
        )

    aligned = mordred_pseudo.reindex(
        index=pseudo_ids, columns=feature_columns, fill_value=np.nan
    )
    X = aligned.values.astype(np.float32)
    assert X.shape[1] == len(feature_columns), (
        f"Pseudo feature column count mismatch: got {X.shape[1]}, "
        f"expected {len(feature_columns)}"
    )
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def augment_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    pseudo_X: np.ndarray,
    pseudo_y: np.ndarray,
    pseudo_w: np.ndarray,
    base_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate real fold-train data with pseudo rows and return weights.

    Real samples receive weight ``base_weight`` (default 1.0). Pseudo sample
    weights are passed in via ``pseudo_w`` (already scaled by the caller).
    """
    X_aug = np.concatenate([X_tr, pseudo_X], axis=0)
    y_aug = np.concatenate([y_tr, pseudo_y], axis=0)
    w_real = np.full(len(X_tr), base_weight, dtype=np.float32)
    w_aug = np.concatenate([w_real, pseudo_w.astype(np.float32)], axis=0)
    return X_aug, y_aug, w_aug
