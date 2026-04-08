"""Pseudo-label utilities for fold-safe training augmentation.

Currently supports Mordred features only. Pseudo compounds are loaded from a
parquet produced by ``scripts/build_pseudo_labels.py`` (columns:
``compound_id``, ``pseudo_pec50``, ``confidence``, ``n_concs``).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data import load_mordred, load_train_mordred

# Re-use the same loaders the main script uses for test compounds
import psycopg2
from data import DB_PARAMS


def load_pseudo_labels(path: Path) -> pd.DataFrame:
    """Read the pseudo-label parquet."""
    df = pd.read_parquet(path)
    required = {"compound_id", "pseudo_pec50", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pseudo parquet missing columns: {missing}")
    return df


def build_pseudo_feature_matrix(
    feature_name: str, pseudo_df: pd.DataFrame
) -> np.ndarray:
    """Build a feature matrix for pseudo compounds matching the train layout.

    Mordred-only for now. The column order is aligned with ``load_train_mordred``
    via the intersection of train Mordred columns and pseudo Mordred columns,
    matching how ``load_features`` builds the train matrix.
    """
    if feature_name != "mordred":
        raise NotImplementedError(
            f"pseudo labels currently support feature='mordred' only, "
            f"got {feature_name!r}"
        )

    pseudo_ids = pseudo_df["compound_id"].astype(int).tolist()
    mordred_train, _ = load_train_mordred()

    # Load test compound_ids to mirror load_features' intersection logic
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM test_activity")
    test_ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    mordred_test = load_mordred(test_ids)
    mordred_pseudo = load_mordred(pseudo_ids)

    missing = set(pseudo_ids) - set(mordred_pseudo.index)
    if missing:
        raise ValueError(
            f"Mordred missing for {len(missing)} pseudo compounds "
            f"(e.g. {sorted(missing)[:5]})"
        )

    # Match the same column subset that load_features uses for train+test.
    # We approximate by intersecting with train columns; downstream the train
    # matrix is built from intersection(train, test), so as long as pseudo
    # contains the same Mordred descriptors (it does, single source table),
    # the intersection is identical to what load_features computed.
    common_cols = mordred_train.columns.intersection(mordred_test.columns).intersection(
        mordred_pseudo.columns
    )
    X = mordred_pseudo.loc[pseudo_ids, common_cols].values.astype(np.float32)
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
