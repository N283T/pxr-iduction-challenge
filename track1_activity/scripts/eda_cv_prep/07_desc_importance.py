"""Feature-importance snapshot for the concatenated 2D descriptor
feature space (Mordred + RDKit 217 + Jazzy), to size the "compress
to TabPFN" idea. No Boltz features at this stage.

Output:
  - total feature count, count with zero gain, count reaching X% of
    cumulative gain importance
  - top-20 features by gain
  - # pairwise |Pearson| >= 0.95 among non-zero-gain features
  - importance CSV to data/eda_cv_prep/desc_importance.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import (  # noqa: E402
    DB_PARAMS,
    JAZZY_FEATURE_COLS,
    load_jazzy,
    load_rdkit_full,
    load_train_mordred,
)

OUT_DIR = REPO_ROOT.joinpath("data", "eda_cv_prep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Loading train compound ids + target...")
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql(
        "SELECT t.compound_id, t.pec50 FROM train_activity t ORDER BY t.id",
        conn,
    )
    conn.close()
    train_ids = df["compound_id"].tolist()
    y = df["pec50"].to_numpy(dtype=np.float64)
    print(f"  n_train = {len(train_ids)}")

    print("Loading Mordred (1460 dim)...")
    mordred_train, _ = load_train_mordred()
    mordred_cols = mordred_train.columns.tolist()
    Xm = mordred_train.loc[train_ids].to_numpy(dtype=np.float32)
    Xm = np.nan_to_num(Xm, nan=0.0, posinf=0.0, neginf=0.0)

    print("Loading RDKit 217 desc...")
    rdkit_full = load_rdkit_full(train_ids)
    rdkit_cols = rdkit_full.columns.tolist()
    Xr = rdkit_full.loc[train_ids].to_numpy(dtype=np.float32)

    print("Loading Jazzy (3 dim)...")
    jazzy = load_jazzy(train_ids).reindex(index=train_ids)
    jazzy_cols = list(JAZZY_FEATURE_COLS)
    Xj = jazzy[jazzy_cols].to_numpy(dtype=np.float32)

    # Prefix to disambiguate cols when Mordred / RDKit share names
    m_names = [f"mordred:{c}" for c in mordred_cols]
    r_names = [f"rdkit:{c}" for c in rdkit_cols]
    j_names = [f"jazzy:{c}" for c in jazzy_cols]
    feature_names = m_names + r_names + j_names

    X = np.concatenate([Xm, Xr, Xj], axis=1)
    print(
        f"Combined X: shape={X.shape}  "
        f"(mordred {Xm.shape[1]} + rdkit {Xr.shape[1]} + jazzy {Xj.shape[1]})"
    )

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(y))
    split = int(0.8 * len(y))
    tr, va = perm[:split], perm[split:]

    print("Fitting LightGBM (1000 rounds, early stop 50)...")
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X[tr], y[tr],
        eval_set=[(X[va], y[va])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iter = model.best_iteration_
    print(f"  best_iter = {best_iter}")

    gain = model.booster_.feature_importance(importance_type="gain")
    split_imp = model.booster_.feature_importance(importance_type="split")
    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": gain,
            "split": split_imp,
        }
    ).sort_values("gain", ascending=False).reset_index(drop=True)

    total = imp_df["gain"].sum()
    imp_df["gain_frac"] = imp_df["gain"] / max(total, 1e-12)
    imp_df["cum_gain"] = imp_df["gain_frac"].cumsum()

    n_used = int((imp_df["gain"] > 0).sum())
    n_total = len(imp_df)
    n_zero = n_total - n_used

    def n_for_cum(pct: float) -> int:
        return int((imp_df["cum_gain"] <= pct).sum()) + 1

    print()
    print(f"  total features: {n_total}")
    print(f"  non-zero gain:  {n_used}")
    print(f"  zero gain:      {n_zero}  ({100*n_zero/n_total:.1f}%)")
    print(f"  feats at 50% cum gain: {n_for_cum(0.5)}")
    print(f"  feats at 80% cum gain: {n_for_cum(0.8)}")
    print(f"  feats at 90% cum gain: {n_for_cum(0.9)}")
    print(f"  feats at 95% cum gain: {n_for_cum(0.95)}")
    print(f"  feats at 99% cum gain: {n_for_cum(0.99)}")
    print()
    print("Top 20 by gain:")
    print(
        imp_df.head(20)[["feature", "gain_frac", "split"]]
        .to_string(index=False, formatters={"gain_frac": "{:.4f}".format})
    )

    # Source-of-gain breakdown by feature group
    def src(name: str) -> str:
        return name.split(":", 1)[0]

    imp_df["src"] = imp_df["feature"].map(src)
    src_stats = imp_df.groupby("src").agg(
        total_feats=("feature", "count"),
        used_feats=("gain", lambda s: int((s > 0).sum())),
        gain_share=("gain_frac", "sum"),
    )
    print()
    print("Source breakdown:")
    print(src_stats.to_string())

    # Multicollinearity: |Pearson| >= 0.95 among used features
    used_cols = imp_df[imp_df["gain"] > 0]["feature"].tolist()
    idx_lookup = {n: i for i, n in enumerate(feature_names)}
    used_idx = np.array([idx_lookup[c] for c in used_cols])
    Xu = X[:, used_idx]
    # Compute in chunks to cap memory
    corr = np.corrcoef(Xu.T)
    np.fill_diagonal(corr, 0.0)
    high = np.abs(corr) >= 0.95
    n_high_pairs = int(high.sum() // 2)
    print()
    print(f"Multicollinearity among {len(used_cols)} used features:")
    print(f"  pairs with |Pearson| >= 0.95: {n_high_pairs}")
    print(f"  pairs with |Pearson| >= 0.99: {int((np.abs(corr) >= 0.99).sum() // 2)}")

    out_path = OUT_DIR.joinpath("desc_importance.csv")
    imp_df[["feature", "src", "gain", "gain_frac", "cum_gain", "split"]].to_csv(
        out_path, index=False
    )
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
