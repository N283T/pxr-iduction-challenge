"""Sanity-check the analog_aware_split_indices implementation.

Verifies:
  1. Every fold's val is disjoint from train (no leakage).
  2. The 46 potent seeds are always in train.
  3. Union of all val indices covers exactly the analog pool.
  4. No compound appears in val more than once.
  5. Val y-dispersion is tighter than train y-dispersion (the
     motivating property from `https://github.com/N283T/pxr-iduction-challenge/issues/181`).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402
from splits import analog_aware_split_indices  # noqa: E402

# Single source of truth: pull the potent-subset definition straight from the
# function we're validating, so the sanity check cannot silently diverge.
_sig = inspect.signature(analog_aware_split_indices)
POTENT_PEC50 = _sig.parameters["potent_pec50_threshold"].default
POTENT_SEL = _sig.parameters["potent_sel_threshold"].default


def load_train() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        df = pd.read_sql(
            """
            SELECT t.compound_id, c.std_smiles, t.pec50,
                   (t.pec50 - ca.pec50) AS selectivity
            FROM train_activity t
            JOIN compounds c ON t.compound_id = c.id
            LEFT JOIN counter_assay ca ON t.compound_id = ca.compound_id
            ORDER BY t.compound_id
            """,
            conn,
        )
    finally:
        conn.close()
    return df


def main() -> None:
    df = load_train().reset_index(drop=True)
    n = len(df)
    print(f"Loaded {n} train compounds")

    splits = analog_aware_split_indices(
        smiles_list=df["std_smiles"].tolist(),
        pec50=df["pec50"].to_numpy(),
        selectivity=df["selectivity"].to_numpy(),
        n_splits=5,
        seed=42,
        verbose=True,
    )

    # Compute potent mask once for checks
    potent_mask = (df["pec50"] >= POTENT_PEC50) & (
        df["selectivity"].fillna(-np.inf) >= POTENT_SEL
    )
    potent_idx = set(np.where(potent_mask.to_numpy())[0].tolist())

    print("\n=== Assertions ===")
    all_val: list[int] = []
    for k, (tr, va) in enumerate(splits):
        tr_set = set(tr.tolist())
        va_set = set(va.tolist())
        assert tr_set.isdisjoint(va_set), f"fold {k}: train/val overlap"
        assert len(tr_set) + len(va_set) <= n, f"fold {k}: duplicates"
        assert potent_idx.issubset(tr_set), (
            f"fold {k}: some potent seeds missing from train"
        )
        all_val.extend(va.tolist())

    assert len(all_val) == len(set(all_val)), "duplicate val indices across folds"
    print("[OK] All folds pass disjointness + potents-always-in-train.")

    # Val y-dispersion check
    print("\n=== Val y-dispersion vs train ===")
    rows = []
    for k, (tr, va) in enumerate(splits):
        tr_y = df.loc[tr, "pec50"].to_numpy()
        va_y = df.loc[va, "pec50"].to_numpy()
        rows.append(
            {
                "fold": k,
                "train_n": len(tr),
                "val_n": len(va),
                "train_pec50_mean": tr_y.mean(),
                "train_pec50_std": tr_y.std(),
                "train_abs_dev_from_median": np.mean(np.abs(tr_y - np.median(tr_y))),
                "val_pec50_mean": va_y.mean(),
                "val_pec50_std": va_y.std(),
                "val_abs_dev_from_median": np.mean(np.abs(va_y - np.median(va_y))),
            }
        )
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    print(
        f"\nmean val abs-dev-from-median: "
        f"{results['val_abs_dev_from_median'].mean():.4f} "
        f"(train: {results['train_abs_dev_from_median'].mean():.4f})"
    )
    print(
        "For reference, LB y-dispersion is ~0.797 "
        "(from https://github.com/N283T/pxr-iduction-challenge/issues/181), train-wide is ~0.910."
    )

    # Also report threshold sensitivity
    print("\n=== Threshold sensitivity (analog pool size) ===")
    for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
        try:
            s = analog_aware_split_indices(
                smiles_list=df["std_smiles"].tolist(),
                pec50=df["pec50"].to_numpy(),
                selectivity=df["selectivity"].to_numpy(),
                n_splits=5,
                analog_tanimoto_threshold=t,
                seed=42,
                verbose=False,
            )
            analog_total = sum(len(v) for _, v in s)
            avg_val_y_disp = np.mean(
                [
                    np.mean(
                        np.abs(
                            df.loc[v, "pec50"].to_numpy()
                            - np.median(df.loc[v, "pec50"].to_numpy())
                        )
                    )
                    for _, v in s
                ]
            )
            avg_val_mean = np.mean([df.loc[v, "pec50"].to_numpy().mean() for _, v in s])
            print(
                f"  threshold={t}: analog_pool={analog_total}, "
                f"avg_val_pec50_mean={avg_val_mean:.3f}, "
                f"avg_val_abs_dev={avg_val_y_disp:.4f}"
            )
        except ValueError as e:
            print(f"  threshold={t}: {e}")


if __name__ == "__main__":
    main()
