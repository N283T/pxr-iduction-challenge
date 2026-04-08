#!/usr/bin/env -S pixi run python
"""Build pseudo labels for compounds with single_concentration but no train label.

Task A of the pseudo-labeling plan (#35).

Trains a LightGBM regressor that maps single_concentration features to pEC50,
using the ~2,374 compounds where train_activity and single_concentration overlap.
The fitted model is then applied to compounds that have single_concentration data
but are absent from train_activity.

Usage:
    pixi run python track1_activity/scripts/build_pseudo_labels.py \
        [--out data/pseudo_labels.parquet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))
from data import get_engine  # noqa: E402

CONC_LOW = 8.25e-6
CONC_HIGH = 3.30e-5
SEED = 42
N_SPLITS = 5

FEATURE_COLS = [
    "log2fc_8_25e_6",
    "log2fc_3_30e_5",
    "log2fc_stderr_8_25e_6",
    "cohens_d_8_25e_6",
    "p_value_8_25e_6",
    "n_concs",
]

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": SEED,
}


def load_single_conc_features() -> pd.DataFrame:
    """Aggregate single_concentration into per-compound features."""
    eng = get_engine()
    sql = """
        SELECT compound_id, concentration_m, log2_fc_estimate,
               log2_fc_stderr, cohens_d, p_value
        FROM single_concentration
        ORDER BY compound_id, concentration_m
    """
    sc = pd.read_sql(sql, eng)

    low_mask = sc["concentration_m"].between(CONC_LOW * 0.99, CONC_LOW * 1.01)
    high_mask = sc["concentration_m"].between(CONC_HIGH * 0.99, CONC_HIGH * 1.01)

    low = (
        sc.loc[
            low_mask,
            [
                "compound_id",
                "log2_fc_estimate",
                "log2_fc_stderr",
                "cohens_d",
                "p_value",
            ],
        ]
        .rename(
            columns={
                "log2_fc_estimate": "log2fc_8_25e_6",
                "log2_fc_stderr": "log2fc_stderr_8_25e_6",
                "cohens_d": "cohens_d_8_25e_6",
                "p_value": "p_value_8_25e_6",
            }
        )
        .drop_duplicates(subset="compound_id", keep="first")
    )
    high = (
        sc.loc[high_mask, ["compound_id", "log2_fc_estimate"]]
        .rename(columns={"log2_fc_estimate": "log2fc_3_30e_5"})
        .drop_duplicates(subset="compound_id", keep="first")
    )
    n_concs = (
        sc.groupby("compound_id")["concentration_m"]
        .nunique()
        .rename("n_concs")
        .reset_index()
    )

    feats = low.merge(high, on="compound_id", how="left").merge(
        n_concs, on="compound_id", how="left"
    )
    return feats


def load_train_targets() -> pd.DataFrame:
    eng = get_engine()
    return pd.read_sql(
        "SELECT compound_id, pec50 FROM train_activity ORDER BY compound_id", eng
    )


def train_cv(x: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, list[dict], list[int]]:
    oof = np.full(len(x), np.nan, dtype=float)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_metrics: list[dict] = []
    best_iters: list[int] = []
    for fold, (tr, va) in enumerate(kf.split(x), start=1):
        dt = lgb.Dataset(x.iloc[tr], y.iloc[tr])
        dv = lgb.Dataset(x.iloc[va], y.iloc[va], reference=dt)
        model = lgb.train(
            LGBM_PARAMS,
            dt,
            num_boost_round=500,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        pred = model.predict(x.iloc[va], num_iteration=model.best_iteration)
        oof[va] = pred
        best_iters.append(int(model.best_iteration or 500))
        rmse = float(np.sqrt(mean_squared_error(y.iloc[va], pred)))
        mae = float(mean_absolute_error(y.iloc[va], pred))
        r2 = float(r2_score(y.iloc[va], pred))
        fold_metrics.append({"fold": fold, "rmse": rmse, "mae": mae, "r2": r2})
        print(f"  fold {fold}: rmse={rmse:.4f} mae={mae:.4f} r2={r2:.4f}")
    return oof, fold_metrics, best_iters


def fit_full(x: pd.DataFrame, y: pd.Series, num_boost_round: int) -> lgb.Booster:
    dt = lgb.Dataset(x, y)
    return lgb.train(LGBM_PARAMS, dt, num_boost_round=num_boost_round)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data").joinpath("pseudo_labels.parquet"),
        help="Output parquet path",
    )
    args = parser.parse_args()

    print("[1/6] Loading single_concentration features...")
    feats = load_single_conc_features()
    print(f"  single_conc compounds: {len(feats)}")

    print("[2/6] Loading train_activity targets...")
    train = load_train_targets()
    print(f"  train_activity rows: {len(train)}")

    merged = feats.merge(train, on="compound_id", how="inner")
    merged = merged.dropna(subset=["pec50", "log2fc_8_25e_6"])
    print(f"  overlap with pEC50 and 8.25e-6 log2fc: {len(merged)}")

    median_high = merged["log2fc_3_30e_5"].median()
    print(f"  imputing log2fc_3_30e_5 missing with median={median_high:.4f}")
    merged["log2fc_3_30e_5"] = merged["log2fc_3_30e_5"].fillna(median_high)

    x = merged[FEATURE_COLS]
    y = merged["pec50"]

    print("[3/6] 5-fold KFold CV training...")
    oof, fold_metrics, best_iters = train_cv(x, y)
    oof_rmse = float(np.sqrt(mean_squared_error(y, oof)))
    oof_mae = float(mean_absolute_error(y, oof))
    oof_r2 = float(r2_score(y, oof))
    print(f"  OOF: rmse={oof_rmse:.4f} mae={oof_mae:.4f} r2={oof_r2:.4f}")

    if oof_r2 < 0.45:
        raise SystemExit(f"OOF R²={oof_r2:.4f} below hard gate 0.45 — aborting.")

    print("[4/6] Refitting on full overlap...")
    mean_best_iter = max(50, int(round(float(np.mean(best_iters)))))
    print(f"  mean best_iteration across folds: {mean_best_iter}")
    full_model = fit_full(x, y, num_boost_round=mean_best_iter)

    print("[5/6] Predicting pseudo labels for non-train compounds...")
    train_ids = set(train["compound_id"].tolist())
    target = feats[~feats["compound_id"].isin(train_ids)].copy()
    target = target.dropna(subset=["log2fc_8_25e_6"])
    target["log2fc_3_30e_5"] = target["log2fc_3_30e_5"].fillna(median_high)
    print(f"  target compounds (single_conc only): {len(target)}")

    pseudo = full_model.predict(target[FEATURE_COLS])
    stderr = target["log2fc_stderr_8_25e_6"].to_numpy()
    confidence = np.where(
        np.isnan(stderr),
        0.1,
        1.0 / (1.0 + stderr),
    )
    confidence = np.clip(confidence, 0.1, 1.0)

    out_df = pd.DataFrame(
        {
            "compound_id": target["compound_id"].to_numpy(),
            "pseudo_pec50": pseudo,
            "confidence": confidence,
            "n_concs": target["n_concs"].to_numpy(),
        }
    )

    if out_df["pseudo_pec50"].isna().any():
        raise SystemExit("NaN found in pseudo_pec50 — aborting.")
    if len(out_df) <= 5000:
        raise SystemExit(f"Output row count {len(out_df)} is not > 5000 — aborting.")

    print("[6/6] Saving output...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"  wrote {args.out} ({len(out_df)} rows)")

    print("\n=== Summary ===")
    print(f"  rows: {len(out_df)}")
    print(
        "  pseudo_pec50: "
        f"mean={out_df['pseudo_pec50'].mean():.3f} "
        f"std={out_df['pseudo_pec50'].std():.3f} "
        f"min={out_df['pseudo_pec50'].min():.3f} "
        f"max={out_df['pseudo_pec50'].max():.3f}"
    )
    print(
        "  pseudo_pec50 quantiles: "
        f"q25={out_df['pseudo_pec50'].quantile(0.25):.3f} "
        f"q50={out_df['pseudo_pec50'].quantile(0.50):.3f} "
        f"q75={out_df['pseudo_pec50'].quantile(0.75):.3f}"
    )
    print(f"  mean confidence: {out_df['confidence'].mean():.3f}")
    print(f"  OOF R²: {oof_r2:.4f}  RMSE: {oof_rmse:.4f}  MAE: {oof_mae:.4f}")


if __name__ == "__main__":
    main()
