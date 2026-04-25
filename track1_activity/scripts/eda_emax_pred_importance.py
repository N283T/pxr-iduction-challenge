"""Diagnostic: is emax_pred actually used by downstream models?

Sanity check after Phase 1B null (downstream OOF MAE +0.008 with
TabPFN). Question: are the 2 emax_pred columns being used at all, or
ignored / treated as noise?

Method:
  1. Build feature matrix `cheme_2d_full_boltz_log2fc_emax_pred` (2105d).
  2. Run LGBM on (X → y_train) with full-data fit, no CV — single
     fitted model is enough for importance ranking.
  3. Print rank of `emax_estimate_pred` / `emax_vs_pos_ctrl_pred` among
     all 2105 features by gain importance.
  4. Compare to the raw rdkit_desc_full + emax_pred (219d) where
     redundancy with cheme components is gone — does emax become more
     important relative to a simpler feature set?

Interpretation rules:
  - Top-100 with non-trivial gain share → emax IS used; the TabPFN
    regression is TabPFN-specific (Bayesian averaging dilutes weak
    cols), not a structural emax problem.
  - Bottom-50% with near-zero gain → emax info is fully captured by
    cheme already; TabPFN regression is structural; A-2 needed.
  - Top in rdkit-only setting but not cheme → cheme already encodes
    most of the emax signal; emax_pred only helps weaker bases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import load_test_smiles, load_train_smiles_target  # noqa: E402

EMAX_PARQUET = REPO_ROOT.joinpath("data", "emax_predictions.parquet")


def fit_lgbm(X: np.ndarray, y: np.ndarray) -> lgb.Booster:
    params = {
        "objective": "regression",
        "metric": "mae",
        "verbose": -1,
        "seed": 42,
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.01,
        "lambda_l2": 1.0,
    }
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=500)


def importance_summary(
    model: lgb.Booster, feature_names: list[str], target_names: list[str]
) -> None:
    gain = model.feature_importance(importance_type="gain")
    splits = model.feature_importance(importance_type="split")
    df = pd.DataFrame(
        {"feature": feature_names, "gain": gain, "splits": splits}
    ).sort_values("gain", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    df["gain_pct"] = 100.0 * df["gain"] / df["gain"].sum()
    df["cum_gain_pct"] = df["gain_pct"].cumsum()

    print("\nTop 15 features by gain:")
    print(df.head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nTarget features (emax_pred) ranking:")
    for tname in target_names:
        row = df[df["feature"] == tname]
        if row.empty:
            print(f"  {tname:30s} NOT FOUND")
            continue
        r = row.iloc[0]
        print(
            f"  {tname:30s} rank={int(r['rank'])}/{len(df)} "
            f"gain={r['gain']:.2f} ({r['gain_pct']:.3f}%) "
            f"splits={int(r['splits'])} cum_to_here={r['cum_gain_pct']:.2f}%"
        )

    nonzero = (df["gain"] > 0).sum()
    print(f"\nFeatures with non-zero gain: {nonzero}/{len(df)}")


def run_cheme_emax() -> None:
    print("\n========== cheme_2d_full_boltz_log2fc_emax_pred (2105d) ==========")
    from run_train import load_features

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    X_train, _ = load_features(
        "cheme_2d_full_boltz_log2fc_emax_pred", train_df, test_df
    )
    y = train_df["pec50"].to_numpy(dtype=np.float64)

    n = X_train.shape[1]
    feature_names = [f"f_{i}" for i in range(n)]
    feature_names[-2] = "emax_estimate_pred"
    feature_names[-1] = "emax_vs_pos_ctrl_pred"

    model = fit_lgbm(X_train, y)
    importance_summary(
        model, feature_names, ["emax_estimate_pred", "emax_vs_pos_ctrl_pred"]
    )


def run_rdkit_emax() -> None:
    print("\n========== rdkit_desc_full + emax_pred (219d) ==========")
    from data import load_rdkit_full

    train_df = load_train_smiles_target()
    train_ids = list(train_df.index) if train_df.index.name == "compound_id" else None
    if train_ids is None:
        # fetch compound_ids in train_activity.id order
        import psycopg2
        from data import DB_PARAMS

        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
        train_ids = [r[0] for r in cur.fetchall()]
        cur.close()
        conn.close()

    rdf = load_rdkit_full(train_ids).reindex(train_ids)
    X_rdkit = rdf.to_numpy(dtype=np.float64)
    rdkit_cols = list(rdf.columns)
    for j in range(X_rdkit.shape[1]):
        col = X_rdkit[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            col[mask] = float(np.nanmedian(col))
            X_rdkit[:, j] = col

    emax_df = pd.read_parquet(EMAX_PARQUET)
    cols = ["emax_estimate_pred", "emax_vs_pos_ctrl_pred"]
    Xe = emax_df.reindex(index=train_ids)[cols].to_numpy(dtype=np.float64)
    Xe = np.nan_to_num(Xe, nan=0.0, posinf=0.0, neginf=0.0)

    X = np.concatenate([X_rdkit, Xe], axis=1)
    feature_names = rdkit_cols + cols
    y = train_df["pec50"].to_numpy(dtype=np.float64)

    model = fit_lgbm(X, y)
    importance_summary(model, feature_names, cols)


def main() -> None:
    if not EMAX_PARQUET.exists():
        raise SystemExit(
            f"Missing {EMAX_PARQUET}. "
            f"Run track1_activity/scripts/run_emax_predict.py first."
        )
    run_cheme_emax()
    run_rdkit_emax()


if __name__ == "__main__":
    main()
