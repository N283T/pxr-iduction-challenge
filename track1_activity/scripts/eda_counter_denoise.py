"""Experimental LGBM A/B on counter_assay-based label denoising.

Revisits the "69 doubly-suspicious compound" filter from
https://github.com/N283T/pxr-iduction-challenge/issues/171 (Issue #37, never A/B tested) and related
filters. All runs are rdkit_desc_full + LGBM default, UMAP 5-fold, seed=42.

Not for pool addition -- purely a diagnostic to check whether any
counter-assay-driven label-denoising strategy shifts OOF beyond the
fold-std noise floor (~0.025 MAE).
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402
from splits import umap_split_indices  # noqa: E402


SQL = """
SELECT t.compound_id, t.pec50, c.std_smiles AS smiles,
       ca.pec50 AS counter_pec50,
       agg.log2fc_8p25
FROM train_activity t
JOIN compounds c ON c.id = t.compound_id
LEFT JOIN counter_assay ca ON ca.compound_id = t.compound_id
LEFT JOIN (
  SELECT compound_id,
    AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
             THEN log2_fc_estimate END) AS log2fc_8p25
  FROM single_concentration
  GROUP BY compound_id
) agg ON agg.compound_id = t.compound_id
ORDER BY t.id
"""


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    with psycopg2.connect(**DB_PARAMS) as conn:
        tr = pd.read_sql(SQL, conn)
        ids = tr["compound_id"].tolist()
        rdkit = pd.read_sql(
            "SELECT compound_id, descriptors FROM compound_descriptors_full "
            "WHERE compound_id = ANY(%s)",
            conn,
            params=(ids,),
        ).set_index("compound_id")

    rdkit_df = pd.json_normalize(rdkit["descriptors"]).set_index(rdkit.index)
    X = rdkit_df.reindex(ids).to_numpy(dtype=np.float32)
    col_means = np.nan_to_num(np.nanmean(X, axis=0), nan=0.0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    tr["sel"] = tr["pec50"] - tr["counter_pec50"]
    return tr, X


def run_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    fold_idx: np.ndarray,
    sample_weight: np.ndarray | None,
    keep_mask: np.ndarray,
    tag: str,
) -> dict:
    """Run LGBM 5-fold with optional sample weights and keep_mask.

    keep_mask[i]=False means compound i is excluded from training (but still
    appears in val). For dropped compounds we evaluate them against 0 (they
    contribute to metrics). If we drop during training but keep at val time,
    predictions for val-in-dropped are still made (LGBM trained on the other
    folds without them).

    sample_weight[i] is the loss weight if compound i is in train fold.
    """
    oof = np.zeros_like(y)
    for fold in range(5):
        tr_keep = (fold_idx != fold) & keep_mask
        va = fold_idx == fold
        if sample_weight is not None:
            w = sample_weight[tr_keep]
        else:
            w = None
        m = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            random_state=42,
            verbose=-1,
        )
        m.fit(
            X[tr_keep],
            y[tr_keep],
            sample_weight=w,
            eval_set=[(X[va], y[va])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof[va] = m.predict(X[va])
    mae = float(np.mean(np.abs(oof - y)))
    rae = mae / float(np.mean(np.abs(y - y.mean())))
    sp = float(spearmanr(oof, y).statistic)
    return {"tag": tag, "MAE": mae, "RAE": rae, "Sp": sp}


def main() -> None:
    print("Loading train + counter + single_conc...")
    tr, X = load_data()
    y = tr["pec50"].to_numpy(dtype=np.float32)
    print(f"  n_train={len(tr)}  n_features={X.shape[1]}")

    doubly = (tr["sel"] < 0) & (tr["log2fc_8p25"] < 0.3) & tr["sel"].notna()
    sel_neg = (tr["sel"] < 0) & tr["sel"].notna()
    print(f"  doubly-suspicious (sel<0 AND log2fc<0.3): n={int(doubly.sum())}")
    print(f"  sel<0 (broader filter):                  n={int(sel_neg.sum())}")
    print(
        f"  mean pEC50 doubly={tr.loc[doubly, 'pec50'].mean():.2f} "
        f"sel<0={tr.loc[sel_neg, 'pec50'].mean():.2f} "
        f"all={y.mean():.2f}"
    )

    print("\nComputing UMAP 5-fold split (Morgan+Jaccard, k=50, seed=42)...")
    splits = umap_split_indices(
        tr["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )
    fold_idx = np.zeros(len(y), dtype=int)
    for fold, (_, val_idx) in enumerate(splits):
        fold_idx[val_idx] = fold

    n = len(y)
    all_keep = np.ones(n, dtype=bool)
    ones_w = np.ones(n, dtype=np.float32)

    variants = [
        ("baseline (all 4140)", all_keep, None),
        ("drop doubly (-69)", all_keep & ~doubly.to_numpy(), None),
        ("drop sel<0 (-294)", all_keep & ~sel_neg.to_numpy(), None),
    ]
    # Downweight variants: sample_weight[doubly]=w, others=1
    for w_val in [0.5, 0.1, 0.0]:
        weights = ones_w.copy()
        weights[doubly.to_numpy()] = w_val
        variants.append((f"downweight doubly w={w_val}", all_keep, weights))

    print("\nLGBM 5-fold OOF (UMAP, rdkit_desc_full 217d):")
    print(f"  {'variant':<35s}  {'MAE':<8s}  {'RAE':<8s}  {'Sp':<8s}  Δ MAE")
    print("  " + "-" * 70)
    baseline_mae = None
    for tag, keep, weight in variants:
        r = run_lgbm(X, y, fold_idx, weight, keep, tag)
        if baseline_mae is None:
            baseline_mae = r["MAE"]
            delta = 0.0
        else:
            delta = r["MAE"] - baseline_mae
        print(
            f"  {r['tag']:<35s}  {r['MAE']:.4f}  {r['RAE']:.4f}  "
            f"{r['Sp']:.4f}  {delta:+.4f}"
        )


if __name__ == "__main__":
    main()
