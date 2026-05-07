"""MW-corrected Boltz-2 affinity: fit the paper's ensemble correction
   y_hat = C0 * (y1 + y2) + C1 * MW + C2
on PXR pEC50 via OOF linear regression. Single-model baseline to see if
the Boltz-2 pretrained affinity head + molecular-weight correction alone
can match or beat our pool members.

Features (per-compound, read from compound_boltz2 + compound_descriptors):
  affinity_pred_value_1   (Boltz-2 member 1 prediction, log10(IC50 μM))
  affinity_pred_value_2   (Boltz-2 member 2 prediction)
  amw                     (average molecular weight)

Train: 4,140 pEC50 compounds covered by compound_boltz2 (rcycle=3 main run).
CV: UMAP 5-fold (same as run_train.py default).
Output: experiment row 'lgbm_boltz2_affinity_mw_corrected_umap_default'
(LinearRegression actually, but uses the existing experiment schema).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402
from splits import umap_split_indices  # noqa: E402


def load_features() -> pd.DataFrame:
    sql = """
    SELECT b.compound_id,
           b.affinity_pred_value_1 AS aff1,
           b.affinity_pred_value_2 AS aff2,
           d.amw
      FROM compound_boltz2 b
      JOIN compound_descriptors d ON d.compound_id = b.compound_id
     WHERE b.preprocessing_failed = FALSE
       AND b.affinity_pred_value_1 IS NOT NULL
       AND b.affinity_pred_value_2 IS NOT NULL
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        return pd.read_sql(sql, conn)


def rae(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred))
    mean_true = np.mean(y_true)
    den = np.sum(np.abs(y_true - mean_true))
    return num / den if den > 0 else float("nan")


def main() -> None:
    print("Loading Boltz-2 affinity + MW ...")
    feats = load_features().set_index("compound_id")
    # Pull train compound_ids + y (train_activity is ordered by t.id)
    with psycopg2.connect(**DB_PARAMS) as conn:
        train_df = pd.read_sql(
            """SELECT t.id AS compound_id, t.pec50, c.std_smiles AS smiles
                 FROM train_activity t JOIN compounds c ON c.id = t.compound_id
                 ORDER BY t.id""",
            conn,
        )
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    ids = train_df["compound_id"].to_numpy(dtype=np.int64)

    Xdf = feats.reindex(index=ids)
    mask = Xdf.notna().all(axis=1).to_numpy()
    print(
        f"  train={len(train_df)}, covered={int(mask.sum())}, "
        f"missing={int((~mask).sum())}"
    )

    X = np.stack(
        [
            Xdf["aff1"].to_numpy(dtype=np.float64),
            Xdf["aff2"].to_numpy(dtype=np.float64),
            Xdf["amw"].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    # Paper form: ŷ = C0*(y1+y2) + C1*MW + C2
    # equivalent to LR on [y1+y2, MW]
    X_paper = np.stack([X[:, 0] + X[:, 1], X[:, 2]], axis=1)

    # Row-mean fill any NaN (only a few compounds should be missing MW)
    col_mean = np.nanmean(X_paper, axis=0)
    X_paper = np.where(np.isnan(X_paper), col_mean, X_paper)

    print("Building UMAP 5-fold (seed=42) ...")
    splits = umap_split_indices(train_df["smiles"].tolist(), n_splits=5, seed=42)

    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        reg = LinearRegression()
        reg.fit(X_paper[tr_idx], y[tr_idx])
        oof[va_idx] = reg.predict(X_paper[va_idx])
        print(
            f"  fold {fold}: C0={reg.coef_[0]:.4f}  C1={reg.coef_[1]:.4f}  "
            f"C2={reg.intercept_:.4f}  val MAE={mean_absolute_error(y[va_idx], oof[va_idx]):.4f}"
        )

    oof_mae = float(mean_absolute_error(y, oof))
    oof_rae = float(rae(y, oof))
    oof_r2 = float(r2_score(y, oof))
    oof_sp = float(spearmanr(oof, y).statistic)
    oof_k = float(kendalltau(oof, y).statistic)
    print(
        "\nOverall OOF MW-corrected affinity:\n"
        f"  MAE={oof_mae:.4f}  RAE={oof_rae:.4f}  R2={oof_r2:.4f}  "
        f"Spearman={oof_sp:.4f}  Kendall={oof_k:.4f}"
    )

    # Comparison reference: our best pool members
    print("\nReference (existing pool members):")
    ref = [
        ("tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_default", "top single"),
        ("tabpfn_pooled_boltz_allpairs_umap_default", "raw Boltz"),
        ("tabpfn_boltz_raw_plus_pretrain_concat_umap_default", "new best"),
    ]
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        for name, tag in ref:
            cur.execute(
                "SELECT mae_mean, rae_mean, spearman_mean FROM experiment_summary WHERE name = %s",
                (name,),
            )
            r = cur.fetchone()
            if r:
                print(f"  {tag:12s} {name[:40]:40s} MAE={r[0]:.4f} RAE={r[1]:.4f}")


if __name__ == "__main__":
    main()
